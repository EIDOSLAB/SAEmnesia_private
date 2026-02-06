import numpy as np
import torch

from SAE.unlearning_utils import get_percentile_threshold


class SAEReconstructHook:
    def __init__(
        self,
        sae,
    ):
        self.sae = sae

    @torch.no_grad()
    def __call__(self, module, input, output):
        output1, output2 = output[0].chunk(2)
        # reshape to SAE input shape
        output1 = output1.permute(0, 2, 3, 1).reshape(
            len(output1), output1.shape[-1] * output1.shape[-2], -1
        )
        output2 = output2.permute(0, 2, 3, 1).reshape(
            len(output2), output2.shape[-1] * output2.shape[-2], -1
        )
        output_cat = torch.cat([output1, output2], dim=0)
        sae_input, _, _ = self.sae.preprocess_input(output_cat)
        pre_acts = self.sae.pre_acts(sae_input)
        top_acts, top_indices = self.sae.select_topk(pre_acts)
        buf = top_acts.new_zeros(top_acts.shape[:-1] + (self.sae.W_dec.mT.shape[-1],))
        latents = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
        sae_out = (latents @ self.sae.W_dec) + self.sae.b_dec
        sae_out1 = sae_out[: output1.shape[1] * len(output1)]
        sae_out2 = sae_out[output1.shape[1] * len(output1) :]
        hook_output = torch.cat(
            [
                sae_out1.reshape(
                    len(output1),
                    int(np.sqrt(output1.shape[-2])),
                    int(np.sqrt(output1.shape[-2])),
                    -1,
                ).permute(0, 3, 1, 2),
                sae_out2.reshape(
                    len(output2),
                    int(np.sqrt(output2.shape[-2])),
                    int(np.sqrt(output2.shape[-2])),
                    -1,
                ).permute(0, 3, 1, 2),
            ],
            dim=0,
        )

        return (hook_output,)


class SAEMaskedUnlearningHook:
    def __init__(
        self,
        concept_to_unlearn,
        percentile,
        multiplier,
        feature_importance_fn,
        concept_latents_dict,
        sae,
        steps=100,
        preserve_error=True,
        seed=None,
        replacement_concept=None,  # NEW
        replacement_params=None,   # NEW
    ):
        self.concept_to_unlearn = concept_to_unlearn
        self.percentile = percentile
        self.multiplier = multiplier
        self.feature_importance_fn = feature_importance_fn
        self.concept_latents_dict = concept_latents_dict
        self.timestep_idx = 0
        self.sae = sae
        self.steps = steps
        self.preserve_error = preserve_error
        self.seed = seed
        self.replacement_concept = replacement_concept  # NEW
        self.replacement_params = replacement_params    # NEW
        
        print('SAE seed: ', str(self.seed))
        if self.replacement_concept:
            print(f'Replacement concept: {self.replacement_concept}')
        
        # Precompute the most important features for concepts to unlearn on every timestep
        self.scaling_factors = []
        self.top_feature_idxs = []
        self.avg_feature_acts = []
        self.all_concept_avg_acts = []
        
        # NEW: Precompute replacement concept features
        self.replacement_feature_idxs = []
        self.replacement_scaling_factors = []
        self.replacement_avg_acts = []
        
        # Compute features for each timestep
        for timestep in range(steps):
            timestep_feature_idxs = []
            timestep_scaling_factors = []
            timestep_all_concept_avg_acts = []
            
            # Process concepts to unlearn
            for concept in self.concept_to_unlearn:
                feature_scores = self.feature_importance_fn(
                    self.concept_latents_dict, concept, timestep, seed=self.seed
                )
                feature_scores = feature_scores.float()
                percentile_threshold = get_percentile_threshold(
                    feature_scores, self.percentile
                )
                top_feature_idxs = torch.where(feature_scores > percentile_threshold)[0]
                timestep_feature_idxs.append(top_feature_idxs)
                concept_acts = self.concept_latents_dict[concept][
                    :, timestep, top_feature_idxs
                ]
                avg_acts = concept_acts.mean(0)
                scaling_factors = avg_acts * self.multiplier
                timestep_scaling_factors.append(scaling_factors)

                # Precompute average activations of features on other concepts
                all_concept_avg_acts = torch.zeros((len(top_feature_idxs)))
                for concept_name in self.concept_latents_dict:
                    all_concept_avg_acts += self.concept_latents_dict[concept_name][
                        :, timestep, top_feature_idxs
                    ].mean(dim=0)
                all_concept_avg_acts /= len(self.concept_latents_dict)
                timestep_all_concept_avg_acts.append(all_concept_avg_acts)
            
            self.top_feature_idxs.append(torch.cat(timestep_feature_idxs))
            self.scaling_factors.append(torch.cat(timestep_scaling_factors))
            self.all_concept_avg_acts.append(torch.cat(timestep_all_concept_avg_acts))
            
            # NEW: Process replacement concept
            if self.replacement_concept and self.replacement_concept in self.concept_latents_dict:
                # Get replacement parameters
                if self.replacement_params:
                    repl_percentile = self.replacement_params.get('percentile', self.percentile)
                    repl_multiplier = abs(self.replacement_params.get('multiplier', abs(self.multiplier)))
                else:
                    repl_percentile = self.percentile
                    repl_multiplier = abs(self.multiplier)  # Use absolute value for positive boost
                
                # Compute replacement features
                repl_feature_scores = self.feature_importance_fn(
                    self.concept_latents_dict, self.replacement_concept, timestep, seed=self.seed
                )
                repl_feature_scores = repl_feature_scores.float()
                repl_percentile_threshold = get_percentile_threshold(
                    repl_feature_scores, repl_percentile
                )
                repl_top_feature_idxs = torch.where(repl_feature_scores > repl_percentile_threshold)[0]
                
                repl_concept_acts = self.concept_latents_dict[self.replacement_concept][
                    :, timestep, repl_top_feature_idxs
                ]
                repl_avg_acts = repl_concept_acts.mean(0)
                repl_scaling_factors = repl_avg_acts * repl_multiplier
                
                self.replacement_feature_idxs.append(repl_top_feature_idxs)
                self.replacement_scaling_factors.append(repl_scaling_factors)
                self.replacement_avg_acts.append(repl_avg_acts)
            else:
                # No replacement for this timestep
                self.replacement_feature_idxs.append(torch.tensor([]))
                self.replacement_scaling_factors.append(torch.tensor([]))
                self.replacement_avg_acts.append(torch.tensor([]))

    @torch.no_grad()
    def __call__(self, module, input, output):
        output1, output2 = output[0].chunk(2)
        # Reshape to SAE input shape
        output1 = output1.permute(0, 2, 3, 1).reshape(
            len(output1), output1.shape[-1] * output1.shape[-2], -1
        )
        output2 = output2.permute(0, 2, 3, 1).reshape(
            len(output2), output2.shape[-1] * output2.shape[-2], -1
        )
        h, w = int(np.sqrt(output2.shape[-2])), int(np.sqrt(output2.shape[-2]))
        output_cat = torch.cat([output1, output2], dim=0)
    
        # Encode activations
        sae_input, _, _ = self.sae.preprocess_input(output_cat)
        pre_acts = self.sae.pre_acts(sae_input)
        top_acts, top_indices = self.sae.select_topk(pre_acts)
        buf = top_acts.new_zeros(top_acts.shape[:-1] + (self.sae.W_dec.mT.shape[-1],))
        latents = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
        recon_acts_original = (latents @ self.sae.W_dec) + self.sae.b_dec
        latents = latents.reshape(len(output_cat), -1, self.sae.num_latents)
        recon_acts_original = recon_acts_original.reshape(
            len(output_cat), -1, self.sae.d_in
        )
    
        if self.preserve_error:
            error_original = (recon_acts_original - output_cat).float()
    
        # === UNLEARNING: Apply negative multiplier ===
        # Mask selecting on which patches to suppress the unlearned concept
        unlearn_mask = latents[
            :, :, self.top_feature_idxs[self.timestep_idx]
        ] > self.all_concept_avg_acts[self.timestep_idx].to(pre_acts.device)
    
        # Expand scaling factors to match mask dimensions
        unlearn_scaling = self.scaling_factors[self.timestep_idx].to(pre_acts.device)
        unlearn_scaling = unlearn_scaling.view(1, 1, -1).expand(unlearn_mask.size(0), unlearn_mask.size(1), -1)
    
        # Apply mask and scaling (negative multiplier to suppress)
        selected_latents = latents[:, :, self.top_feature_idxs[self.timestep_idx]]
        selected_latents = torch.where(
            unlearn_mask, selected_latents * unlearn_scaling, selected_latents
        )
        latents[:, :, self.top_feature_idxs[self.timestep_idx]] = selected_latents
    
        # === NEW: REPLACEMENT with FIXED STRONG ACTIVATION ===
        if (self.replacement_concept and 
            len(self.replacement_feature_idxs[self.timestep_idx]) > 0):
            
            repl_feature_idx = self.replacement_feature_idxs[self.timestep_idx]
            
            # USE A FIXED STRONG VALUE - experiment with different values
            # Start with 50.0, can try 100.0, 200.0, etc.
            FIXED_ACTIVATION_VALUE = 50.0
            
            # Create fixed activation tensor
            fixed_activation = torch.ones(
                latents.size(0), 
                latents.size(1), 
                len(repl_feature_idx),
                device=latents.device,
                dtype=latents.dtype
            ) * FIXED_ACTIVATION_VALUE
            
            # Option 1: SET (replace) the replacement features with fixed strong value
            # This completely overrides whatever was there
            latents[:, :, repl_feature_idx] = fixed_activation
            
            # Option 2 (alternative): ADD fixed value only where unlearned object was
            # Uncomment these lines if you want to add instead of replace:
            # unlearn_patch_mask = unlearn_mask.any(dim=-1, keepdim=True)
            # unlearn_patch_mask = unlearn_patch_mask.expand(-1, -1, len(repl_feature_idx))
            # latents[:, :, repl_feature_idx] = torch.where(
            #     unlearn_patch_mask,
            #     latents[:, :, repl_feature_idx] + fixed_activation,
            #     latents[:, :, repl_feature_idx]
            # )
    
        # Decode back
        recon_acts_ablated = (latents @ self.sae.W_dec) + self.sae.b_dec
        if self.preserve_error:
            recon_acts_ablated = (recon_acts_ablated + error_original).to(output2.dtype)
        else:
            recon_acts_ablated = recon_acts_ablated.to(output_cat.dtype)
    
        hook_output = recon_acts_ablated.reshape(
            len(output_cat),
            h,
            w,
            -1,
        ).permute(0, 3, 1, 2)
        self.timestep_idx += 1
    
        return (hook_output,)