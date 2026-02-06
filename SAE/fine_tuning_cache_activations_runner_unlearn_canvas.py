import io
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

from diffusers.utils.import_utils import is_xformers_available

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import Array2D, Dataset, Features, Value
from datasets.fingerprint import generate_fingerprint
from huggingface_hub import HfApi
from tqdm import tqdm

from SAE.config import FineTuningCacheActivationsRunnerConfig
from UnlearnCanvas_resources.const import class_available, theme_available

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

TORCH_STRING_DTYPE_MAP = {torch.float16: "float16", torch.float32: "float32"}

import json
from collections import defaultdict

class ActivationMetadataGenerator:
    """Generate compact metadata files for querying activations by object and style combinations."""
    
    def __init__(self, cached_activations_path: str, is_sdxl: bool = False):
        self.base_path = Path(cached_activations_path)
        self.is_sdxl = is_sdxl
        
    def generate_metadata(self, hook_names: list = None):
        """Generate compact metadata files for all hook directories."""
        if hook_names is None:
            hook_names = [d.name for d in self.base_path.iterdir() 
                         if d.is_dir() and not d.name.startswith('.')]
        
        for hook_name in hook_names:
            hook_path = self.base_path / hook_name
            if not hook_path.exists():
                continue
                
            print(f"Processing hook: {hook_name}")
            self._generate_hook_metadata(hook_path)
    
    def _generate_hook_metadata(self, hook_path: Path):
        """Generate compact metadata for a specific hook directory."""
        category_dirs = [d for d in hook_path.iterdir() 
                        if d.is_dir() and not d.name.startswith('.') 
                        and d.name != 'metadata']
        
        if not category_dirs:
            print(f"No category directories found in {hook_path}")
            return
        
        metadata_dir = hook_path / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        
        if self.is_sdxl:
            # SDXL: Simple object-only metadata
            object_index = {}
            summary = {"total_samples": 0, "objects": {}}
            
            for category_dir in category_dirs:
                category_name = category_dir.name
                print(f"  Processing object: {category_name}")
                
                try:
                    dataset = Dataset.load_from_disk(str(category_dir))
                    total_samples = len(dataset)
                    print(f"    Found {total_samples} samples")
                    
                    object_index[category_name] = {
                        "sample_count": total_samples,
                        "dataset_path": str(category_dir)
                    }
                    
                    summary["objects"][category_name] = total_samples
                    summary["total_samples"] += total_samples
                    
                except Exception as e:
                    print(f"  Error processing {category_dir}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Save SDXL metadata
            try:
                with open(metadata_dir / "object_index.json", "w") as f:
                    json.dump(object_index, f, indent=2)
                
                with open(metadata_dir / "summary.json", "w") as f:
                    json.dump(summary, f, indent=2)
                
                print(f"  ✅ SDXL metadata saved")
                
            except OSError as e:
                if e.errno == 122:
                    print(f"  ❌ Disk quota exceeded.")
                raise e
        else:
            # SD1.5: Object+Style metadata
            object_style_index = defaultdict(lambda: defaultdict(list))
            style_object_index = defaultdict(lambda: defaultdict(list))
            summary = {"total_samples": 0, "combinations": {}}
            
            for category_dir in category_dirs:
                category_name = category_dir.name
                print(f"  Processing concept: {category_name}")
                
                try:
                    dataset = Dataset.load_from_disk(str(category_dir))
                    total_samples = len(dataset)
                    print(f"    Found {total_samples} samples")
                    
                    # Get all style labels
                    styles = dataset["style_label"]
                    
                    # Group consecutive samples by style to create ranges
                    current_style = None
                    start_idx = 0
                    
                    for idx, style in enumerate(styles):
                        if style != current_style:
                            # Save the previous range if it exists
                            if current_style is not None:
                                entry = {
                                    "sample_range": [start_idx, idx],
                                    "sample_count": idx - start_idx,
                                    "recovery_confidence": "high",
                                    "dataset_path": str(category_dir)
                                }
                                object_style_index[category_name][current_style].append(entry)
                                style_object_index[current_style][category_name].append(entry)
                                
                                combo_key = f"{category_name}+{current_style}"
                                summary["combinations"][combo_key] = summary["combinations"].get(combo_key, 0) + (idx - start_idx)
                                summary["total_samples"] += (idx - start_idx)
                            
                            # Start new range
                            current_style = style
                            start_idx = idx
                    
                    # Don't forget the last range
                    if current_style is not None:
                        entry = {
                            "sample_range": [start_idx, len(styles)],
                            "sample_count": len(styles) - start_idx,
                            "recovery_confidence": "high",
                            "dataset_path": str(category_dir)
                        }
                        object_style_index[category_name][current_style].append(entry)
                        style_object_index[current_style][category_name].append(entry)
                        
                        combo_key = f"{category_name}+{current_style}"
                        summary["combinations"][combo_key] = summary["combinations"].get(combo_key, 0) + (len(styles) - start_idx)
                        summary["total_samples"] += (len(styles) - start_idx)
                    
                    print(f"    Style ranges created for {category_name}")
                    
                except Exception as e:
                    print(f"  Error processing {category_dir}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
            # Save SD1.5 metadata
            try:
                with open(metadata_dir / "recovered_object_to_style_index.json", "w") as f:
                    compact_data = {obj: dict(styles) for obj, styles in object_style_index.items()}
                    json.dump(compact_data, f, indent=2)
                
                with open(metadata_dir / "style_to_object_index.json", "w") as f:
                    compact_data = {style: dict(objects) for style, objects in style_object_index.items()}
                    json.dump(compact_data, f, indent=2)
                
                with open(metadata_dir / "summary.json", "w") as f:
                    json.dump(summary, f, indent=2)
                
                print(f"  ✅ SD1.5 metadata saved with sample_range info")
                
            except OSError as e:
                if e.errno == 122:
                    print(f"  ❌ Disk quota exceeded.")
                raise e

class CacheActivationsRunner:
    def __init__(self, cfg: FineTuningCacheActivationsRunnerConfig):
        self.cfg = cfg
        self.accelerator = Accelerator()

        print(f"DEBUG: organization_type = {self.cfg.organization_type}")
        print(f"DEBUG: Config settings: {vars(self.cfg)}")

        # hacky way to prevent initializing those objects when using only load_and_push_to_hub()
        if self.cfg.hook_names is not None:
            # Detect model type from model_index.json
            model_index_path = os.path.join(self.cfg.model_name, "model_index.json")
            self.is_sdxl = False
            
            if os.path.exists(model_index_path):
                with open(model_index_path, 'r') as f:
                    model_index = json.load(f)
                # SDXL has text_encoder_2, SD1.5 doesn't
                self.is_sdxl = "text_encoder_2" in model_index
                
            if self.is_sdxl:
                from SAE.hooked_sd_noised_pipeline import (
                    HookedStableDiffusionXLPipeline,
                )
                print("🎯 Detected SDXL model - using HookedStableDiffusionXLPipeline")
                print("⚡ SDXL Optimization: No style variants will be generated")
                PipelineClass = HookedStableDiffusionXLPipeline
            else:
                from SAE.hooked_sd_noised_pipeline import (
                    HookedStableDiffusionPipeline,
                )
                print("🎯 Detected SD1.5 model - using HookedStableDiffusionPipeline")
                print("📝 SD1.5: Generating all style variants")
                PipelineClass = HookedStableDiffusionPipeline
        
            self.pipe = PipelineClass.from_pretrained(
                self.cfg.model_name, torch_dtype=self.cfg.dtype, safety_checker=None
            )
            if is_xformers_available():
                print("Enabling xFormers memory efficient attention")
                self.pipe.unet.enable_xformers_memory_efficient_attention()
            self.pipe.to(self.accelerator.device)
            self.pipe.vae.to("cpu")
            self.pipe.set_progress_bar_config(disable=True)

            self.scheduler = self.pipe.scheduler

            # Prepare timesteps
            self.scheduler.set_timesteps(self.cfg.num_inference_steps, device="cpu")
            self.scheduler_timesteps = self.scheduler.timesteps

            self.features_dict = {hookpoint: None for hookpoint in self.cfg.hook_names}

            # Conditional prompt creation based on model type
            all_prompts = []
            all_object_labels = []
            all_style_labels = []
            
            for class_avail in class_available[
                self.cfg.class_start : self.cfg.class_end
            ]:
                with open(
                    os.path.join(
                        "UnlearnCanvas_resources/anchor_prompts/finetune_prompts",
                        f"sd_prompt_{class_avail}.txt",
                    ),
                    "r",
                ) as prompt_file:
                    if self.accelerator.is_main_process:
                        print(f"Preparing prompts for class {class_avail}")
                    for prompt in prompt_file:
                        prompt = prompt.strip()
                        prompt = prompt if not prompt.endswith(".") else prompt[:-1]
                        
                        if self.is_sdxl:
                            # SDXL: Only plain prompts (no style variants for speed)
                            all_prompts.append(prompt + ".")
                            all_object_labels.append(class_avail)
                            # No style_labels for SDXL
                        else:
                            # SD1.5: Generate all style variants
                            for theme in theme_available:
                                theme_prompt = (
                                    f"{prompt} in {theme.replace('_', ' ')} style."
                                )
                                all_prompts.append(theme_prompt)
                                all_object_labels.append(class_avail)
                                all_style_labels.append(theme)
                            # Add plain prompt without style
                            all_prompts.append(prompt + ".")
                            all_object_labels.append(class_avail)
                            all_style_labels.append("none")

            # Create dataset based on model type
            if self.is_sdxl:
                self.dataset = Dataset.from_dict({
                    "caption": all_prompts,
                    "object_label": all_object_labels,
                })
            else:
                self.dataset = Dataset.from_dict({
                    "caption": all_prompts,
                    "object_label": all_object_labels,
                    "style_label": all_style_labels
                })
            
            self.dataset = self.dataset.shuffle(self.cfg.seed)
            if limit := self.cfg.max_num_examples:
                self.dataset = self.dataset.select(range(limit))
            
            if self.accelerator.is_main_process:
                print("\n=== DATASET DEBUG INFO ===")
                unique_objects = set(self.dataset["object_label"])
                print(f"Unique objects in dataset: {sorted(unique_objects)}")
                print(f"Total unique objects: {len(unique_objects)}")
                
                if self.is_sdxl:
                    print(f"Model type: SDXL (no style variants)")
                    print(f"Total dataset size: {len(self.dataset)}")
                    print(f"⚡ Speed: ~51x faster than SD1.5 mode")
                else:
                    unique_styles = set(self.dataset["style_label"])
                    print(f"Model type: SD1.5 (with style variants)")
                    print(f"Unique styles: {sorted(unique_styles)}")
                    print(f"Total dataset size: {len(self.dataset)}")

                # Show first few samples
                print("First 5 samples:")
                for i in range(min(5, len(self.dataset))):
                    sample = self.dataset[i]
                    if self.is_sdxl:
                        print(f"  {i}: object='{sample['object_label']}', caption='{sample['caption'][:50]}...'")
                    else:
                        print(f"  {i}: object='{sample['object_label']}', style='{sample['style_label']}', caption='{sample['caption'][:50]}...'")
                print("=" * 30 + "\n")

            self.num_examples = len(self.dataset)
            self.dataloader = self.get_batches(self.dataset, self.cfg.batch_size)
            self.n_buffers = len(self.dataloader)

    @staticmethod
    def get_batches(items, batch_size):
        num_batches = (len(items) + batch_size - 1) // batch_size
        batches = []

        for i in range(num_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(items))
            batch = items[start_index:end_index]
            batches.append(batch)

        return batches

    @staticmethod
    def _consolidate_shards(
        self,
        source_dir: Path, 
        output_dir: Path, 
        copy_files: bool = True,
        category_name: str = None
    ) -> Dataset:
        """Consolidate sharded datasets into a single directory without rewriting data."""
        first_shard_dir_name = "shard_00000"

        assert source_dir.exists() and source_dir.is_dir()
        assert output_dir.exists() and output_dir.is_dir()
        if not (source_dir / first_shard_dir_name).exists():
            raise Exception(f"No shards in {source_dir} exist!")

        transfer_fn = shutil.copy2 if copy_files else shutil.move

        # Create category subfolder if organizing by category
        final_output_dir = output_dir
        if self.cfg.organization_type != "none" and category_name:
            final_output_dir = output_dir / category_name
            final_output_dir.mkdir(exist_ok=True, parents=True)

        # Move dataset_info.json to the FINAL output directory
        transfer_fn(
            source_dir / first_shard_dir_name / "dataset_info.json",
            final_output_dir / "dataset_info.json",
        )

        arrow_files = []
        file_count = 0

        for shard_dir in sorted(source_dir.iterdir()):
            if not shard_dir.name.startswith("shard_"):
                continue

            # state.json contains arrow filenames
            state = json.loads((shard_dir / "state.json").read_text())

            for data_file in state["_data_files"]:
                src = shard_dir / data_file["filename"]
                new_name = f"data-{file_count:05d}-of-{len(list(source_dir.iterdir())):05d}.arrow"
                dst = final_output_dir / new_name
                transfer_fn(src, dst)
                arrow_files.append({"filename": new_name})
                file_count += 1

        new_state = {
            "_data_files": arrow_files,
            "_fingerprint": None,
            "_format_columns": None,
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": None,
        }

        # Write state.json to the final directory
        with open(final_output_dir / "state.json", "w") as f:
            json.dump(new_state, f, indent=2)

        # Add error handling here
        try:
            ds = Dataset.load_from_disk(str(final_output_dir))
            fingerprint = generate_fingerprint(ds)
            del ds
        except FileNotFoundError as e:
            print(f"ERROR: Could not load dataset from {final_output_dir}")
            print(f"Contents of directory: {list(final_output_dir.iterdir()) if final_output_dir.exists() else 'Directory does not exist'}")
            raise e

        with open(final_output_dir / "state.json", "r+") as f:
            state = json.loads(f.read())
            state["_fingerprint"] = fingerprint
            f.seek(0)
            json.dump(state, f, indent=2)
            f.truncate()

        if not copy_files:  # cleanup source dir
            shutil.rmtree(source_dir)

        return Dataset.load_from_disk(str(final_output_dir))
    
    def generate_activation_metadata(self):
        """Generate metadata files for the cached activations."""
        if not hasattr(self, 'cfg') or self.cfg.new_cached_activations_path is None:
            print("Error: No cached activations path configured")
            return
        
        metadata_generator = ActivationMetadataGenerator(
            self.cfg.new_cached_activations_path,
            is_sdxl=self.is_sdxl
        )
        metadata_generator.generate_metadata(self.cfg.hook_names)
        print(f"✅ Metadata saved to {self.cfg.new_cached_activations_path}/*/metadata/")

    @torch.no_grad()
    def _create_shard(
        self,
        buffer: torch.Tensor,
        hook_name: str,
        object_labels: list = None,
        style_labels: list = None,
    ) -> Dataset:
        batch_size, n_steps, d_sample_size, d_in = buffer.shape

        # Handle case where hook_name is not in features_dict
        if hook_name not in self.features_dict:
            first_hook = next(iter(self.features_dict))
            if self.accelerator.is_main_process:
                print(f"WARNING: Hook name '{hook_name}' not found in features_dict. Using '{first_hook}' instead.")
            hook_name = first_hook

        # Filter buffer based on every N steps
        buffer = buffer[:, :: self.cfg.cache_every_n_timesteps, :, :]

        activations = buffer.reshape(-1, d_sample_size, d_in)
        timesteps = self.scheduler_timesteps[
            :: self.cfg.cache_every_n_timesteps
        ].repeat(batch_size)

        # Create dataset dict
        dataset_dict = {
            "activations": activations,
            "timestep": timesteps,
        }

        # Add labels based on model type
        n_items = batch_size * (len(self.scheduler_timesteps) // self.cfg.cache_every_n_timesteps)
        dataset_dict["object_label"] = object_labels * (n_steps // self.cfg.cache_every_n_timesteps) if object_labels is not None else ["unknown"] * n_items
        
        if not self.is_sdxl:
            # Only include style_label for SD1.5
            dataset_dict["style_label"] = style_labels * (n_steps // self.cfg.cache_every_n_timesteps) if style_labels is not None else ["none"] * n_items

        shard = Dataset.from_dict(
            dataset_dict,
            features=self.features_dict[hook_name],
        )
        return shard

    def create_dataset_feature(self, hook_name, d_in, d_out):
        features = {
            "activations": Array2D(
                shape=(
                    d_in,
                    d_out,
                ),
                dtype=TORCH_STRING_DTYPE_MAP[self.cfg.dtype],
            ),
            "timestep": Value(dtype="uint16"),
        }

        # Always include object_label
        features["object_label"] = Value(dtype="string")
        
        # Only include style_label for SD1.5
        if not self.is_sdxl:
            features["style_label"] = Value(dtype="string")

        self.features_dict[hook_name] = Features(features)

    def _get_category_path(self, hook_name: str, category: str) -> Path:
        """Get the path for a specific category (object or style)."""
        base_path = Path(self.cfg.new_cached_activations_path)
        if self.cfg.organization_type == "none":
            return base_path / hook_name
        else:
            return base_path / hook_name

    @torch.no_grad()
    def run(self) -> dict[str, Dataset]:
        """Run the caching process with optional organization by object or style."""
        assert self.cfg.new_cached_activations_path is not None

        if self.accelerator.is_main_process:
            print(f"DEBUG: Model type = {'SDXL' if self.is_sdxl else 'SD1.5'}")
            print(f"DEBUG: organization_type = {self.cfg.organization_type}")
            print(f"DEBUG: Config settings relevant to organization:")
            print(f"  - organization_type: {self.cfg.organization_type}")
            print(f"  - hook_names: {self.cfg.hook_names}")
            print(f"  - class_start: {self.cfg.class_start}")
            print(f"  - class_end: {self.cfg.class_end}")

        # Create category-specific buffers
        category_buffers = {}
        category_counts = {}
        
        if self.cfg.organization_type == "none":
            # Original behavior - single path per hook
            final_cached_activation_paths = {
                n: Path(os.path.join(self.cfg.new_cached_activations_path, n))
                for n in self.cfg.hook_names
            }
        else:
            # New behavior - organize by category
            final_cached_activation_paths = {}
            for hook_name in self.cfg.hook_names:
                categories = class_available[self.cfg.class_start:self.cfg.class_end] if self.cfg.organization_type == "object" else theme_available
                for category in categories:
                    path = self._get_category_path(hook_name, category)
                    key = f"{hook_name}::{category}"
                    final_cached_activation_paths[key] = path
                    category_buffers[key] = []
                    category_counts[key] = 0
        
        if self.accelerator.is_main_process:
            print("\n=== CATEGORY BUFFERS DEBUG INFO ===")
            print(f"Created category buffers for these keys:")
            for key in category_buffers.keys():
                parts = key.split('::', 1)
                hook_name, category = parts[0], parts[1]
                print(f"  - Hook: {hook_name}, Category: {category}")
            print(f"Total category buffers created: {len(category_buffers)}")
            print("=" * 40 + "\n")

        if self.accelerator.is_main_process:
            for path in final_cached_activation_paths.values():
                path.mkdir(exist_ok=True, parents=True)
                if any(path.iterdir()):
                    print(f"Found existing files in {path}. Will reuse existing shards where possible.")

            tmp_cached_activation_paths = {}
            for n, path in final_cached_activation_paths.items():
                safe_name = n.replace('::', '_').replace('.', '_')
                tmp_cached_activation_paths[n] = path / f".tmp_shards_{safe_name}"

            for path in tmp_cached_activation_paths.values():
                path.mkdir(exist_ok=True, parents=True)

        self.accelerator.wait_for_everyone()

        ### Create temporary sharded datasets
        if self.accelerator.is_main_process:
            print(f"Started caching {self.num_examples} activations")
            print(f"Organization type: {self.cfg.organization_type}")
            print(f"Model type: {'SDXL (fast mode)' if self.is_sdxl else 'SD1.5 (with styles)'}")

        for i, batch in tqdm(
            enumerate(self.dataloader),
            desc="Caching activations",
            total=self.n_buffers,
            disable=not self.accelerator.is_main_process,
        ):
            # Check if this batch's shards already exist
            if self.accelerator.is_main_process:
                skip_batch = True
                if self.cfg.organization_type == "none":
                    for hook_name in self.cfg.hook_names:
                        shard_path = Path(f"{tmp_cached_activation_paths[hook_name]}/shard_{i:05d}")
                        if not shard_path.exists():
                            skip_batch = False
                            break
                else:
                    for key in category_buffers.keys():
                        shard_path = Path(f"{tmp_cached_activation_paths[key]}/shard_{category_counts[key]:05d}")
                        if not shard_path.exists():
                            skip_batch = False
                            break
                        
                if skip_batch:
                    print(f"Skipping batch {i} - shards already exist")
                    continue

            with self.accelerator.split_between_processes(batch) as prompt_batch:
                prompt = prompt_batch[self.cfg.column]
                object_labels = prompt_batch["object_label"]
                style_labels = prompt_batch.get("style_label", None)  # May not exist for SDXL
                
                _, acts_cache = self.pipe.run_with_cache(
                    prompt=prompt,
                    output_type="latent",
                    num_inference_steps=self.cfg.num_inference_steps,
                    save_input=True if self.cfg.output_or_diff == "diff" else False,
                    save_output=True,
                    positions_to_cache=self.cfg.hook_names,
                    guidance_scale=self.cfg.guidance_scale,
                )

            self.accelerator.wait_for_everyone()

            # Gather and process each hook's activations separately
            gathered_buffer = {}
            for hook_name in self.cfg.hook_names:
                if self.cfg.output_or_diff == "diff":
                    gathered_buffer[hook_name] = (
                        acts_cache["output"][hook_name] - acts_cache["input"][hook_name]
                    )
                else:
                    gathered_buffer[hook_name] = acts_cache["output"][hook_name]
            
            # Gather labels along with buffers
            gather_dict = {
                "buffer": gathered_buffer,
                "object_labels": object_labels,
            }
            if style_labels is not None:
                gather_dict["style_labels"] = style_labels
            
            gathered_data = gather_object([gather_dict])

            if self.accelerator.is_main_process:
                for hook_name in self.cfg.hook_names:
                    # Concatenate all gathered buffers
                    gathered_buffer_acts = torch.cat(
                        [data["buffer"][hook_name] for data in gathered_data],
                        dim=0,
                    )
                    
                    # Concatenate all labels
                    all_object_labels = []
                    all_style_labels = []
                    for data in gathered_data:
                        all_object_labels.extend(data["object_labels"])
                        if "style_labels" in data:
                            all_style_labels.extend(data["style_labels"])
                    
                    if self.features_dict[hook_name] is None:
                        self.create_dataset_feature(
                            hook_name,
                            gathered_buffer_acts.shape[-2],
                            gathered_buffer_acts.shape[-1],
                        )

                    if self.cfg.organization_type == "none":
                        # Original behavior - save all activations together
                        shard = self._create_shard(
                            gathered_buffer_acts, 
                            hook_name,
                            object_labels=all_object_labels,
                            style_labels=all_style_labels if not self.is_sdxl else None
                        )
                        shard.save_to_disk(
                            f"{tmp_cached_activation_paths[hook_name]}/shard_{i:05d}",
                            num_shards=1,
                        )
                    else:
                        # New behavior - organize by category
                        for idx, act in enumerate(gathered_buffer_acts):
                            category = (all_object_labels[idx] if self.cfg.organization_type == "object" 
                                      else all_style_labels[idx])
                            key = f"{hook_name}::{category}"

                            if key in category_buffers:
                                category_buffers[key].append(act.unsqueeze(0))
                            else:
                                if i == 0:
                                    print(f"❌ ERROR: Key '{key}' not found in category_buffers!")

                        # Save shards when buffers reach batch size
                        for key, buffer_list in category_buffers.items():
                            if len(buffer_list) >= self.cfg.batch_size:
                                buffer_tensor = torch.cat(buffer_list[:self.cfg.batch_size], dim=0)
                                hook_name_key = key.split('::')[0]
                                category = key.split('::')[1]
                                
                                shard_object_labels = [category] * self.cfg.batch_size
                                shard_style_labels = None
                                
                                if not self.is_sdxl and self.cfg.organization_type == "style":
                                    shard_style_labels = [category] * self.cfg.batch_size
                                elif not self.is_sdxl:
                                    # For object organization in SD1.5, extract actual style labels
                                    shard_style_labels = ["none"] * self.cfg.batch_size  # Default, could be improved
                                
                                shard = self._create_shard(
                                    buffer_tensor, 
                                    hook_name_key,
                                    object_labels=shard_object_labels,
                                    style_labels=shard_style_labels
                                )
                                shard_idx = category_counts[key]
                                shard.save_to_disk(
                                    f"{tmp_cached_activation_paths[key]}/shard_{shard_idx:05d}",
                                    num_shards=1,
                                )
                                category_buffers[key] = buffer_list[self.cfg.batch_size:]
                                category_counts[key] += 1
                    
                    del gathered_buffer_acts
                del gathered_buffer

        # Save remaining activations in buffers
        if self.accelerator.is_main_process and self.cfg.organization_type != "none":
            for key, buffer_list in category_buffers.items():
                if buffer_list:
                    buffer_tensor = torch.cat(buffer_list, dim=0)
                    hook_name_key = key.split('::')[0]
                    category = key.split('::')[1]
                    
                    shard_object_labels = [category] * len(buffer_list)
                    shard_style_labels = None
                    
                    if not self.is_sdxl and self.cfg.organization_type == "style":
                        shard_style_labels = [category] * len(buffer_list)
                    elif not self.is_sdxl:
                        shard_style_labels = ["none"] * len(buffer_list)
                    
                    shard = self._create_shard(
                        buffer_tensor, 
                        hook_name_key,
                        object_labels=shard_object_labels,
                        style_labels=shard_style_labels
                    )
                    shard_idx = category_counts[key]
                    shard.save_to_disk(
                        f"{tmp_cached_activation_paths[key]}/shard_{shard_idx:05d}",
                        num_shards=1,
                    )

        ### Consolidate sharded datasets together
        datasets = {}
        
        if self.accelerator.is_main_process:
            print(f"\n=== PRE-CONSOLIDATION DEBUG ===")
            
            for key, path in tmp_cached_activation_paths.items():
                if not (path.exists() and any(path.iterdir())):
                    continue

                print(f"\n--- CONSOLIDATION: {key} ---")

                # Extract category_name from key if using organization
                category_name = None
                hook_name_key = key
                if self.cfg.organization_type != "none":
                    parts = key.split('::', 1)
                    hook_name_key = parts[0]
                    category_name = parts[1] if len(parts) > 1 else None

                # Determine the correct output directory
                if self.cfg.organization_type == "none":
                    output_dir = final_cached_activation_paths[key]
                else:
                    base_hook_dir = Path(self.cfg.new_cached_activations_path) / hook_name_key
                    base_hook_dir.mkdir(exist_ok=True, parents=True)
                    output_dir = base_hook_dir

                try:
                    datasets[key] = CacheActivationsRunner._consolidate_shards(
                        self,
                        path, 
                        output_dir,
                        copy_files=False,
                        category_name=category_name
                    )
                    print(f"✅ Successfully consolidated {key}")

                except Exception as e:
                    print(f"❌ ERROR consolidating {key}: {e}")
                    import traceback
                    traceback.print_exc()

            if self.cfg.hf_repo_id and self.cfg.organization_type == "none":
                print("Pushing to hub...")
                for hook_name, dataset in datasets.items():
                    dataset.push_to_hub(
                        repo_id=f"{self.cfg.hf_repo_id}_{hook_name}",
                        num_shards=self.cfg.hf_num_shards or self.n_buffers,
                        private=self.cfg.hf_is_private_repo,
                        revision=self.cfg.hf_revision,
                    )

                meta_io = io.BytesIO()
                meta_contents = json.dumps(
                    asdict(self.cfg), indent=2, ensure_ascii=False
                ).encode("utf-8")
                meta_io.write(meta_contents)
                meta_io.seek(0)

                api = HfApi()
                api.upload_file(
                    path_or_fileobj=meta_io,
                    path_in_repo="cache_activations_runner_cfg.json",
                    repo_id=self.cfg.hf_repo_id,
                    repo_type="dataset",
                    commit_message="Add cache_activations_runner metadata",
                )
        
        if self.accelerator.is_main_process:
            print("\n=== FINAL DIRECTORY STRUCTURE CHECK ===")
            base_path = Path(self.cfg.new_cached_activations_path)
            
            def print_directory_tree(path, prefix="", max_depth=3, current_depth=0):
                if current_depth >= max_depth:
                    return
                if not path.exists():
                    print(f"{prefix}❌ {path.name} (does not exist)")
                    return
                
                items = sorted(path.iterdir())
                for i, item in enumerate(items):
                    is_last = i == len(items) - 1
                    current_prefix = "└── " if is_last else "├── "
                    print(f"{prefix}{current_prefix}{item.name}")
                    
                    if item.is_dir() and current_depth < max_depth - 1:
                        next_prefix = prefix + ("    " if is_last else "│   ")
                        print_directory_tree(item, next_prefix, max_depth, current_depth + 1)
            
            print(f"Directory structure under {base_path}:")
            print_directory_tree(base_path)
            print("=" * 50)
        
        return datasets

    def load_and_push_to_hub(self) -> None:
        """Load dataset from disk and push it to the hub."""
        assert self.cfg.new_cached_activations_path is not None
        dataset = Dataset.load_from_disk(self.cfg.new_cached_activations_path)
        if self.accelerator.is_main_process:
            print("Loaded dataset from disk")

            if self.cfg.hf_repo_id:
                print("Pushing to hub...")
                dataset.push_to_hub(
                    repo_id=self.cfg.hf_repo_id,
                    num_shards=self.cfg.hf_num_shards
                    or (len(dataset) // self.cfg.batch_size),
                    private=self.cfg.hf_is_private_repo,
                    revision=self.cfg.hf_revision,
                )

                meta_io = io.BytesIO()
                meta_contents = json.dumps(
                    asdict(self.cfg), indent=2, ensure_ascii=False
                ).encode("utf-8")
                meta_io.write(meta_contents)
                meta_io.seek(0)

                api = HfApi()
                api.upload_file(
                    path_or_fileobj=meta_io,
                    path_in_repo="cache_activations_runner_cfg.json",
                    repo_id=self.cfg.hf_repo_id,
                    repo_type="dataset",
                    commit_message="Add cache_activations_runner metadata",
                )