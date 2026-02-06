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

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

TORCH_STRING_DTYPE_MAP = {torch.float16: "float16", torch.float32: "float32"}

# Nudity-related keywords for classification
NUDITY_KEYWORDS = [
    "naked", "nude", "nudity", "bare", "unclothed", 
    "topless", "bottomless", "undressed"
]


def contains_nudity(prompt: str) -> bool:
    """Check if a prompt contains nudity-related content."""
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in NUDITY_KEYWORDS)


class CacheActivationsRunner:
    def __init__(self, cfg: FineTuningCacheActivationsRunnerConfig):
        self.cfg = cfg
        self.accelerator = Accelerator()

        if self.cfg.hook_names is not None:
            from SAE.hooked_sd_noised_pipeline import (
                HookedStableDiffusionPipeline,
            )

            self.pipe = HookedStableDiffusionPipeline.from_pretrained(
                self.cfg.model_name, torch_dtype=self.cfg.dtype, safety_checker=None
            )
            if is_xformers_available():
                print("Enabling xFormers memory efficient attention")
                self.pipe.unet.enable_xformers_memory_efficient_attention()
            self.pipe.to(self.accelerator.device)
            self.pipe.vae.to("cpu")
            self.pipe.set_progress_bar_config(disable=True)

            self.scheduler = self.pipe.scheduler
            self.scheduler.set_timesteps(self.cfg.num_inference_steps, device="cpu")
            self.scheduler_timesteps = self.scheduler.timesteps

            self.features_dict = {hookpoint: None for hookpoint in self.cfg.hook_names}

            # Create prompts with nudity classification
            all_prompts = []
            all_nudity_labels = []
            
            # Read all prompts from file
            if hasattr(self.cfg, 'prompts_path'):
                with open(self.cfg.prompts_path, 'r') as f:
                    for line in f:
                        caption = line.strip()
                        if caption:  # Skip empty lines
                            all_prompts.append(caption)
                            # Classify based on nudity keywords
                            if contains_nudity(caption):
                                all_nudity_labels.append("nudity")
                            else:
                                all_nudity_labels.append("non_nudity")

            self.dataset = Dataset.from_dict({
                "caption": all_prompts,
                "nudity_label": all_nudity_labels
            })
            self.dataset = self.dataset.shuffle(self.cfg.seed)
            
            if limit := self.cfg.max_num_examples:
                self.dataset = self.dataset.select(range(limit))
            
            if self.accelerator.is_main_process:
                print("\n=== DATASET INFO ===")
                nudity_count = sum(1 for label in self.dataset["nudity_label"] if label == "nudity")
                non_nudity_count = len(self.dataset) - nudity_count
                print(f"Total samples: {len(self.dataset)}")
                print(f"Nudity samples: {nudity_count}")
                print(f"Non-nudity samples: {non_nudity_count}")
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
        """Consolidate sharded datasets into a single directory."""
        first_shard_dir_name = "shard_00000"

        assert source_dir.exists() and source_dir.is_dir()
        assert output_dir.exists() and output_dir.is_dir()
        if not (source_dir / first_shard_dir_name).exists():
            raise Exception(f"No shards in {source_dir} exist!")

        transfer_fn = shutil.copy2 if copy_files else shutil.move

        final_output_dir = output_dir
        if category_name:
            final_output_dir = output_dir / category_name
            final_output_dir.mkdir(exist_ok=True, parents=True)

        transfer_fn(
            source_dir / first_shard_dir_name / "dataset_info.json",
            final_output_dir / "dataset_info.json",
        )

        arrow_files = []
        file_count = 0

        for shard_dir in sorted(source_dir.iterdir()):
            if not shard_dir.name.startswith("shard_"):
                continue

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

        with open(final_output_dir / "state.json", "w") as f:
            json.dump(new_state, f, indent=2)

        ds = Dataset.load_from_disk(str(final_output_dir))
        fingerprint = generate_fingerprint(ds)
        del ds

        with open(final_output_dir / "state.json", "r+") as f:
            state = json.loads(f.read())
            state["_fingerprint"] = fingerprint
            f.seek(0)
            json.dump(state, f, indent=2)
            f.truncate()

        if not copy_files:
            shutil.rmtree(source_dir)

        return Dataset.load_from_disk(str(final_output_dir))

    @torch.no_grad()
    def _create_shard(
        self,
        buffer: torch.Tensor,
        hook_name: str,
        nudity_labels: list = None,
    ) -> Dataset:
        batch_size, n_steps, d_sample_size, d_in = buffer.shape

        if hook_name not in self.features_dict:
            hook_name = next(iter(self.features_dict))

        buffer = buffer[:, :: self.cfg.cache_every_n_timesteps, :, :]
        activations = buffer.reshape(-1, d_sample_size, d_in)
        timesteps = self.scheduler_timesteps[
            :: self.cfg.cache_every_n_timesteps
        ].repeat(batch_size)

        dataset_dict = {
            "activations": activations,
            "timestep": timesteps,
        }

        n_items = batch_size * (len(self.scheduler_timesteps) // self.cfg.cache_every_n_timesteps)
        dataset_dict["nudity_label"] = nudity_labels * (n_steps // self.cfg.cache_every_n_timesteps) if nudity_labels is not None else ["non_nudity"] * n_items

        shard = Dataset.from_dict(
            dataset_dict,
            features=self.features_dict[hook_name],
        )
        return shard

    def create_dataset_feature(self, hook_name, d_in, d_out):
        features = {
            "activations": Array2D(
                shape=(d_in, d_out),
                dtype=TORCH_STRING_DTYPE_MAP[self.cfg.dtype],
            ),
            "timestep": Value(dtype="uint16"),
            "nudity_label": Value(dtype="string"),
        }
        self.features_dict[hook_name] = Features(features)

    @torch.no_grad()
    def run(self) -> dict[str, Dataset]:
        """Run the caching process, organizing by nudity/non-nudity."""
        assert self.cfg.new_cached_activations_path is not None

        # Create category-specific buffers for nudity and non-nudity
        category_buffers = {}
        category_counts = {}
        final_cached_activation_paths = {}
        
        for hook_name in self.cfg.hook_names:
            for category in ["nudity", "non_nudity"]:
                key = f"{hook_name}::{category}"
                path = Path(self.cfg.new_cached_activations_path) / hook_name
                final_cached_activation_paths[key] = path
                category_buffers[key] = []
                category_counts[key] = 0

        if self.accelerator.is_main_process:
            for path in set(final_cached_activation_paths.values()):
                path.mkdir(exist_ok=True, parents=True)

            tmp_cached_activation_paths = {}
            for n, path in final_cached_activation_paths.items():
                safe_name = n.replace('::', '_').replace('.', '_')
                tmp_cached_activation_paths[n] = path / f".tmp_shards_{safe_name}"

            for path in tmp_cached_activation_paths.values():
                path.mkdir(exist_ok=True, parents=True)

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            print(f"Started caching {self.num_examples} activations")
            print("Organizing by: nudity vs non-nudity")

        for i, batch in tqdm(
            enumerate(self.dataloader),
            desc="Caching activations",
            total=self.n_buffers,
            disable=not self.accelerator.is_main_process,
        ):
            with self.accelerator.split_between_processes(batch) as prompt_batch:
                prompt = prompt_batch[self.cfg.column]
                nudity_labels = prompt_batch["nudity_label"]
                
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

            gathered_buffer = {}
            for hook_name in self.cfg.hook_names:
                if self.cfg.output_or_diff == "diff":
                    gathered_buffer[hook_name] = (
                        acts_cache["output"][hook_name] - acts_cache["input"][hook_name]
                    )
                else:
                    gathered_buffer[hook_name] = acts_cache["output"][hook_name]
            
            gathered_data = gather_object([{
                "buffer": gathered_buffer,
                "nudity_labels": nudity_labels,
            }])

            if self.accelerator.is_main_process:
                for hook_name in self.cfg.hook_names:
                    gathered_buffer_acts = torch.cat(
                        [data["buffer"][hook_name] for data in gathered_data],
                        dim=0,
                    )
                    
                    all_nudity_labels = []
                    for data in gathered_data:
                        all_nudity_labels.extend(data["nudity_labels"])
                    
                    if self.features_dict[hook_name] is None:
                        self.create_dataset_feature(
                            hook_name,
                            gathered_buffer_acts.shape[-2],
                            gathered_buffer_acts.shape[-1],
                        )

                    # Organize by nudity category
                    for idx, act in enumerate(gathered_buffer_acts):
                        category = all_nudity_labels[idx]
                        key = f"{hook_name}::{category}"

                        if key in category_buffers:
                            category_buffers[key].append(act.unsqueeze(0))

                    # Save shards when buffers reach batch size
                    for key, buffer_list in category_buffers.items():
                        if len(buffer_list) >= self.cfg.batch_size:
                            buffer_tensor = torch.cat(buffer_list[:self.cfg.batch_size], dim=0)
                            hook_name = key.split('::')[0]
                            shard = self._create_shard(buffer_tensor, hook_name)
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
        if self.accelerator.is_main_process:
            for key, buffer_list in category_buffers.items():
                if buffer_list:
                    buffer_tensor = torch.cat(buffer_list, dim=0)
                    hook_name = key.split('::')[0]
                    shard = self._create_shard(buffer_tensor, hook_name)
                    shard_idx = category_counts[key]
                    shard.save_to_disk(
                        f"{tmp_cached_activation_paths[key]}/shard_{shard_idx:05d}",
                        num_shards=1,
                    )

        # Consolidate shards
        datasets = {}
        
        if self.accelerator.is_main_process:
            print("\n=== Consolidating shards ===")
            for key, path in tmp_cached_activation_paths.items():
                if path.exists() and any(path.iterdir()):
                    print(f"Processing: {key}")
                    
                    parts = key.split('::', 1)
                    hook_name = parts[0]
                    category_name = parts[1] if len(parts) > 1 else None

                    base_hook_dir = Path(self.cfg.new_cached_activations_path) / hook_name
                    base_hook_dir.mkdir(exist_ok=True, parents=True)

                    try:
                        datasets[key] = CacheActivationsRunner._consolidate_shards(
                            self,
                            path, 
                            base_hook_dir,
                            copy_files=False,
                            category_name=category_name
                        )
                        print(f"✅ Consolidated: {key}")
                    except Exception as e:
                        print(f"❌ Error consolidating {key}: {e}")

        return datasets