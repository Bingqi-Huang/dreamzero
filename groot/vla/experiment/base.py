# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# This file is modified from https://github.com/haotian-liu/LLaVA/

from abc import ABC
import contextlib
import json
import logging
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Optional
import warnings

from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader, Dataset, Sampler
import transformers
from transformers import TrainerCallback, set_seed
from transformers.trainer import (
    # ALL_LAYERNORM_LAYERS,  # ShardedDDPOption,  # Removed deprecated import
    TRAINER_STATE_NAME,
    TrainerState,
    get_last_checkpoint,
    get_parameter_names,
    is_sagemaker_mp_enabled,
)

import groot.vla.common.utils as U
from groot.vla.data.dataset.lerobot_sharded import ShardedLeRobotMixtureDataset
from groot.vla.data.schema import EmbodimentTag
from groot.vla.data.transform import ComposedModalityTransform
from groot.vla.experiment.utils import (
    compute_grad_accum_to_match_global_bs,
    dtype_from_string,
    get_checkpoint_path,
    mprint,
    safe_save_model_for_hf_trainer,
)
from groot.vla.utils.timer import ContextTimer

# Fix resume: https://github.com/huggingface/transformers/pull/34632/files
np_core = np.core
allowlist = [np_core.multiarray._reconstruct, np.ndarray, np.dtype]
# numpy >1.25 defines numpy.dtypes.UInt32DType, but below works for
# all versions of numpy
allowlist += [type(np.dtype(np.uint32))]
torch.serialization.add_safe_globals(allowlist)

# Define LayerNorm classes locally to replace deprecated ALL_LAYERNORM_LAYERS
LAYERNORM_LAYERS = [
    torch.nn.LayerNorm,
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LocalResponseNorm,
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
]


class LossLoggerCallback(TrainerCallback):
    """Callback that writes per-step loss metrics to a JSONL file for offline analysis."""

    def __init__(self, output_path: str):
        self.output_path = output_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return
        entry = {"step": state.global_step}
        for key in ("loss", "dynamics_loss_avg", "action_loss_avg", "learning_rate"):
            if key in logs:
                entry[key] = logs[key]
        if len(entry) > 1:  # more than just "step"
            with open(self.output_path, "a") as f:
                f.write(json.dumps(entry) + "\n")


class CheckpointFormatCallback(TrainerCallback):
    """This callback format checkpoint to make them standalone. For now, it copies all config
    files to /checkpoint-{step}/experiment_cfg/:
    - conf.yaml
    - initial_actions.npz
    - metadata.json
    """

    def __init__(
        self, run_name: str, exp_cfg_dir: Path | None = None, processor_dir: Path | None = None
    ):
        """
        Args:
            run_name: Name of the experiment run
            exp_cfg_dir: Path to the directory containing all experiment metadata
        """
        self.exp_cfg_dir = exp_cfg_dir
        self.processor_dir = processor_dir

    def on_save(self, args, state, control, **kwargs):
        """Called after the trainer saves a checkpoint."""
        if state.is_world_process_zero:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"

            # Copy experiment config directory if provided
            if self.exp_cfg_dir is not None:
                exp_cfg_dst = checkpoint_dir / self.exp_cfg_dir.name
                if self.exp_cfg_dir.exists():
                    print(
                        f"Copying experiment config directory {self.exp_cfg_dir} to {exp_cfg_dst}"
                    )
                    shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)

            # Copy processor directory if provided
            if self.processor_dir is not None:
                if self.processor_dir.exists():
                    print(f"Copying processor directory {self.processor_dir} to {checkpoint_dir}")
                    shutil.copytree(self.processor_dir, checkpoint_dir, dirs_exist_ok=True)

            # Copy wandb_config.json if provided
            wandb_config_src = Path(args.output_dir) / "wandb_config.json"
            wandb_config_dst = checkpoint_dir / "wandb_config.json"
            if wandb_config_src.exists():
                print(f"Copying wandb_config.json from {wandb_config_src} to {wandb_config_dst}")
                shutil.copy2(wandb_config_src, wandb_config_dst)


class ProfCallback(transformers.TrainerCallback):
    """Callback to manage PyTorch profiler during training.

    Dynamically starts/stops the profiler within a specified session step window.
    After profiling completes, triggers optional S3 upload and removes itself.

    Args:
        profile_dir: Directory to save profile traces
        upload_callback: Optional callback to trigger S3 upload after profiling
        profile_start_step: Session step to start profiling (default: 50)
        profile_end_step: Session step to stop profiling
        warmup_steps: Number of warmup steps for profiler schedule (default: 1)
        active_steps: Number of active profiling steps (default: 5)
        trainer: Trainer instance (required for self-removal after profiling)
        record_shapes: Record tensor shapes in profiler (default: False)
        with_stack: Record Python stack traces (default: True)
        profile_memory: Record memory allocation/deallocation (default: False)
    """

    def __init__(
        self,
        profile_dir,
        upload_callback=None,
        profile_start_step=50,
        profile_end_step=55,
        warmup_steps=1,
        active_steps=5,
        trainer=None,
        record_shapes=False,
        with_stack=True,
        profile_memory=False,
    ):
        self.profile_dir = profile_dir
        self.upload_callback = upload_callback
        self.profile_start_step = profile_start_step
        self.profile_end_step = profile_end_step
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.trainer = trainer
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.profile_memory = profile_memory
        self.upload_triggered = False
        self.starting_global_step = None
        self.session_step = 0
        self.prof = None
        self.profiling_active = False
        self.profiling_complete = False
        self.removed_from_trainer = False

    def on_step_begin(self, args, state, control, **kwargs):
        # Remove callback after upload triggered to eliminate all overhead
        if self.profiling_complete and self.upload_triggered and not self.removed_from_trainer:
            if self.trainer is not None and hasattr(self.trainer, "callback_handler"):
                try:
                    self.trainer.callback_handler.callbacks.remove(self)
                    self.removed_from_trainer = True
                    logging.info(
                        f"Removed ProfCallback from trainer at global step {state.global_step}"
                    )
                except (ValueError, AttributeError) as e:
                    logging.warning(f"Failed to remove ProfCallback: {e}")
            return

        # Early return if profiling already complete
        if self.profiling_complete:
            return

        # Record starting global step on first call
        if self.starting_global_step is None:
            self.starting_global_step = state.global_step

        # Calculate session step
        self.session_step = state.global_step - self.starting_global_step

        # Start profiler when we reach the profiling window
        if self.session_step == self.profile_start_step and self.prof is None:
            logging.info(
                f"Starting profiler at global step {state.global_step} (session step {self.session_step})"
            )
            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    skip_first=0,
                    wait=0,
                    warmup=self.warmup_steps,
                    active=self.active_steps,
                    repeat=1,
                ),
                profile_memory=self.profile_memory,
                with_stack=self.with_stack,
                record_shapes=self.record_shapes,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.profile_dir)),
            )
            self.prof.__enter__()
            self.profiling_active = True

    def on_step_end(self, args, state, control, **kwargs):
        # Early return if profiling already complete
        if self.profiling_complete:
            return

        # Recalculate session_step to ensure accuracy
        if self.starting_global_step is not None:
            self.session_step = state.global_step - self.starting_global_step

        # Step profiler if active
        if self.profiling_active and self.prof is not None:
            self.prof.step()

        # Stop profiler when we reach the end of profiling window
        if self.session_step == self.profile_end_step and self.prof is not None:
            self.prof.__exit__(None, None, None)
            self.profiling_active = False

            # Explicitly release profiler resources to minimize CUPTI overhead
            # Combined with TEARDOWN_CUPTI=1 env var for full cleanup
            del self.prof
            self.prof = None

            # Force CUDA synchronization to ensure profiler cleanup completes
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            self.profiling_complete = True
            logging.info(
                f"Profiler stopped and resources released at global step {state.global_step} "
                f"(session step {self.session_step})"
            )

            # Trigger upload if callback provided
            if self.upload_callback:
                logging.info(f"Triggering upload at global step {state.global_step}...")
                self.upload_callback()

            # Mark as ready for callback removal
            self.upload_triggered = True


class BaseSampler(Sampler):
    """Sampler for dataset, which enables `set_epoch` for Dataset.
    `set_epoch` will be called by huggingface Trainer at the end of each epoch.
    `shuffle` is also supported for training set shuffling
    """

    def __init__(self, data_source: Dataset, shuffle: bool = False, seed: int = 0):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # must not add rank here, or randomization will be different for each rank
            return iter(torch.randperm(len(self.data_source), generator=g).tolist())
        return iter(range(len(self.data_source)))

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.data_source, "set_epoch"):
            # this is important for dataset
            self.data_source.set_epoch(epoch)

    def __len__(self):
        return len(self.data_source)


class BaseTrainer(transformers.Trainer):

    def __init__(self, **kwargs):
        # Increase the cache size limit for torch._dynamo to
        # accommodate videos with different numbers of frames.
        torch._dynamo.config.cache_size_limit = 1000

        self.compute_dtype = kwargs.pop("compute_dtype")
        self.output_dir = kwargs.pop("output_dir")
        self.timer = ContextTimer(self)

        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.global_rank = int(os.environ.get("RANK", "0"))
        self.node_rank = int(os.environ.get("NODE_RANK", "0"))

        # Get distributed info
        self.current_step = 0

        # Profiling (legacy per-step profiling)
        self.enable_profiling = kwargs.pop("enable_profiling", False)
        self.profiling_steps = kwargs.pop("profiling_steps", 5)
        # Pop new ProfCallback config options (handled in create_trainer, not here)
        kwargs.pop("enable_prof_callback", None)
        kwargs.pop("profile_start_step", None)
        kwargs.pop("profile_warmup_steps", None)
        kwargs.pop("profile_active_steps", None)
        kwargs.pop("profile_record_shapes", None)
        kwargs.pop("profile_with_stack", None)
        kwargs.pop("profile_memory", None)
        kwargs.pop("msc_profile_url", None)
        kwargs.pop("profile_delete_after_upload", None)
        if self.enable_profiling:
            # Setup profiling directories
            self.profile_dir = Path(self.output_dir) / "profiling"
            self.memory_profile_dir = self.profile_dir / "memory"
            self.torch_profile_dir = self.profile_dir / "torch"

            self.memory_profile_dir.mkdir(exist_ok=True, parents=True)
            self.torch_profile_dir.mkdir(exist_ok=True, parents=True)

            # Start recording the memory history.
            torch.cuda.memory._record_memory_history(max_entries=100000)

        super().__init__(**kwargs)

        self.loss_queues = {}
        self.loss_queue_size = 10

    def _get_train_sampler(self):
        return BaseSampler(self.train_dataset, shuffle=True, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset):
        return BaseSampler(eval_dataset, shuffle=False)

    def training_step(self, model, inputs, num_items_in_batch=None):
        enable_profile = self.enable_profiling and self.current_step % self.profiling_steps == 0
        if enable_profile:
            profile_context = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True,
            )
        else:
            profile_context = contextlib.nullcontext()

        start_time = time.time()

        with self.timer.with_label("training_step"), profile_context as prof:
            output = super().training_step(model, inputs)

        time_taken = time.time() - start_time
        print(
            f"Rank {self.global_rank} time taken for training_step {self.current_step}: {time_taken:.2f} seconds"
        )

        if enable_profile:
            trace_path = f"{self.torch_profile_dir}/trace_rank_{self.global_rank}_step_{self.current_step}.json.gz"
            print(f"Rank {self.global_rank} exporting torch profile to {trace_path}")
            prof.export_chrome_trace(trace_path)

            snapshot_path = f"{self.memory_profile_dir}/memory_snapshot_rank_{self.global_rank}_step_{self.current_step}.pickle"
            print(f"Rank {self.global_rank} dumping memory snapshot to {snapshot_path}")
            torch.cuda.memory._dump_snapshot(snapshot_path)

        self.current_step += 1
        return output

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        with self.timer.with_label("model_forward"):
            outputs = model(inputs)
        ### For additional losses, track and log their moving averages
        for key, value in outputs.items():
            if key.endswith("_loss") and key != "loss":
                # Initialize queue if not exists
                if key not in self.loss_queues:
                    self.loss_queues[key] = []

                # Add current loss value to queue
                current_value = value.item() if torch.is_tensor(value) else value
                self.loss_queues[key].append(current_value)

                # Keep only last N values
                if len(self.loss_queues[key]) > self.loss_queue_size:
                    self.loss_queues[key].pop(0)

                # Log average every 10 steps
                if self.current_step % self.loss_queue_size == 0:
                    avg_loss = sum(self.loss_queues[key]) / len(self.loss_queues[key])
                    self.log({f"{key}_avg": avg_loss})

        loss = outputs["loss"]

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            # DeepSpeed CPU Adam (ZeRO offload) expects 'bias_correction' in each param group.
            # HuggingFace Trainer's AdamW does not set it, causing KeyError in cpu_adam.step().
            if getattr(self.args, "deepspeed", None):
                for group in self.optimizer.param_groups:
                    group.setdefault("bias_correction", True)

        return self.optimizer

    def save_model(self, output_dir: Optional[str], _internal_call: bool):

        ## save tuned model separately
        if self.base_cfg.save_lora_only:
            # Save only the trainable (LoRA) parameters. Under DeepSpeed ZeRO-2 the
            # parameters are NOT sharded -- every rank holds a full replica -- so
            # self.model.state_dict() already contains them and no DeepSpeed gather
            # is needed. Critically, do NOT call accelerator.get_state_dict() on this
            # path: under ZeRO-2 it clones the ENTIRE 14B model to CPU host RAM on
            # every rank (~28GB x num_ranks ~= 168GB), and we would immediately throw
            # it away below. That host-RAM spike, colliding with dataloader shard
            # caching at the save step, OOM-killed a DataLoader worker (signal: Killed)
            # and crashed the whole run at the first checkpoint. The saved file is
            # identical -- this only removes the dead 168GB allocation.
            train_key = [k for k, v in self.model.named_parameters() if v.requires_grad]
            state_dict = {k: v for k, v in self.model.state_dict().items() if k in train_key}
        elif self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            state_dict = self.model.state_dict()

        if self.args.should_save:
            ret = self.model.save_pretrained(output_dir, state_dict=state_dict)

            # can separately save the VLM model for downstream evalualtion
            if self.base_cfg.save_llm:
                llm_output_dir = os.path.join(output_dir, "llm")
                self.model.backbone.model.save_pretrained(llm_output_dir)

            if self.base_cfg.save_value_model:
                assert hasattr(
                    self.model.action_head, "value_model"
                ), f"Value model not found in action head: {type(self.model.action_head)}"
                value_model_output_dir = os.path.join(output_dir, "value_model")
                self.model.action_head.value_model.save_pretrained(value_model_output_dir)

            return ret

    def train(
        self,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
        **kwargs,
    ):
        """Correctly set self.state from checkpoint so get_train_dataloader can read from it."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
        return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if not isinstance(train_dataset, (ShardedLeRobotMixtureDataset)):
            return super().get_train_dataloader()

        # During resume, don't skip the data
        self.args.ignore_data_skip = True
        curr_global_step = self.state.global_step
        print(f"Current global step: {curr_global_step}")
        if curr_global_step > 0:
            new_seed = train_dataset.seed + curr_global_step
            train_dataset.reset_seed(new_seed)
            print(
                f"Resetting seed to {new_seed}. Please note that this will make the experiment non-reproducible."
            )

        print("Creating custom train dataloader")
        # Handle the case where the dataset is an IterableDataset
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(
            data_collator, description="training"
        )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        # persistent_workers is only valid when num_workers > 0 (PyTorch raises otherwise)
        if self.args.dataloader_num_workers > 0:
            dataloader_params["persistent_workers"] = self.args.dataloader_persistent_workers

        return DataLoader(train_dataset, **dataloader_params)


class BaseExperiment(ABC):
    def __init__(self, cfg: DictConfig):
        # assert cfg.save_steps == 500, "save_steps must be 500 for standarized evaluation"
        assert cfg.max_steps > 0, "max_steps must be > 0 for standarized evaluation"
        assert cfg.save_total_limit >= 5, "save_total_limit must be >= 5 for standarized evaluation"

        if cfg.load_from_yaml is not None:
            # Override the default config with the loaded config.
            loaded_cfg = OmegaConf.load(cfg.load_from_yaml)
            cfg = loaded_cfg  # overwrite

        # Check if evaluation transforms are valid.
        assert cfg.transforms is not None, "Evaluation transforms are not provided."
        for tag, transform_cfg in cfg.transforms.items():
            try:
                # Check if the tag is a valid EmbodimentTag
                _ = EmbodimentTag(tag)
                # Check if the transform is a valid ComposedModalityTransform
                transform = instantiate(transform_cfg)
                assert isinstance(transform, ComposedModalityTransform), f"{transform=}"
            except Exception as e:
                raise ValueError(f"Evaluation transform {tag} is invalid: {e}")

        # Instantiate the training arguments.
        cfg.training_args.output_dir = cfg.training_args.output_dir.rstrip("/")
        cfg.training_args.run_name = cfg.training_args.output_dir.split("/")[-1]
        print(f"Run name: {cfg.training_args.run_name}")
        training_args = instantiate(cfg.training_args)
        set_seed(training_args.seed)

        # Set the environment variables for wandb.
        if "WANDB_PROJECT" not in os.environ:
            os.environ["WANDB_PROJECT"] = cfg.wandb_project
        if "WANDB_RUN_ID" not in os.environ:
            runtime_id = os.environ.get("RUNTIME_ID", None)
            """If a RUNTIME_ID is available in the environment, we use it as the wandb id,
            which will allow to display the evaluation results and the training results
            in the same wandb run. Otherwise, we create a new run."""
            if runtime_id:
                os.environ["WANDB_RUN_ID"] = runtime_id
        os.environ["WANDB_DIR"] = training_args.output_dir

        # Create the experiment config directory.
        output_dir = Path(training_args.output_dir)
        exp_cfg_dir = output_dir / "experiment_cfg"
        exp_cfg_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, exp_cfg_dir / "conf.yaml", resolve=True)

        wandb_config_file = output_dir / "wandb_config.json"
        with open(wandb_config_file, "w") as f:
            json.dump(
                {
                    "project": os.environ.get("WANDB_PROJECT", ""),
                    "run_id": os.environ.get("WANDB_RUN_ID", ""),
                },
                f,
            )

        # Check if we are resuming training.
        resume_path, continue_training = get_checkpoint_path(training_args.output_dir)
        if not continue_training:
            print(f"Models is ready under {training_args.output_dir}. Skip training.")
            exit(0)
        if resume_path:
            print(f"Resuming training from {resume_path}")
            resume_from_checkpoint = True
        else:
            # First time training.
            resume_from_checkpoint = False

        # Instantiate the model.
        model = self.create_model(cfg, training_args)

        if hasattr(model.action_head, "max_steps"):
            model.action_head.max_steps = cfg.max_steps

        # Make sure model_dtype and training_args dtype are compatible.
        compute_dtype = dtype_from_string(model.config.model_dtype)

        # Create the train dataset.
        # Dump the metadata; necessary for policy to normalize the input and unnormalize the output
        train_dataset = self.create_train_dataset(cfg, model)
        print("Using dataset:")
        print(train_dataset)
        assert (
            train_dataset.merged_metadata is not None
        ), "You must set metadata_config.merge=true in order to save the metadata."

        metadata_save_path = exp_cfg_dir / "metadata.json"
        U.json_dump(
            {k: v.model_dump(mode="json") for k, v in train_dataset.merged_metadata.items()},
            metadata_save_path,
            indent=4,
        )
        print("Successfully dumped metadata")

        val_dataset = self.create_val_dataset(cfg, model)
        data_collator = self.create_data_collator(cfg, model)
        trainer = self.create_trainer(
            cfg=cfg,
            exp_cfg_dir=exp_cfg_dir,
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            data_collator=data_collator,
            compute_dtype=compute_dtype,
        )
        self.cfg = cfg
        self.exp_cfg_dir = exp_cfg_dir
        self.training_args = training_args
        self.resume_from_checkpoint = resume_from_checkpoint
        self.train_dataset = train_dataset
        self.trainer = trainer

    def _rank_print(self, message: str) -> None:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            print(f"[dist-{rank}-of-{world_size}] {message}", flush=True)
        else:
            print(message, flush=True)

    def _wait_for_marker(self, marker_path: Path, timeout_sec: int, description: str) -> None:
        start_time = time.time()
        while not marker_path.exists():
            if time.time() - start_time > timeout_sec:
                raise TimeoutError(f"Timed out waiting for {description}: {marker_path}")
            time.sleep(1)

    def _stagger_model_init_begin(self, training_args):
        stagger = os.environ.get("STAGGER_MODEL_INIT", "0").lower() in ("1", "true", "yes")
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if not stagger or world_size <= 1:
            return None

        rank = int(os.environ.get("RANK", "0"))
        session_id = os.environ.get("DREAMZERO_INIT_SESSION_ID", "default")
        timeout_sec = int(os.environ.get("STAGGER_MODEL_INIT_TIMEOUT_SEC", "1800"))
        group_size = max(1, int(os.environ.get("STAGGER_MODEL_INIT_GROUP_SIZE", "1")))
        marker_dir = Path(training_args.output_dir) / ".staggered_model_init" / session_id
        marker_dir.mkdir(parents=True, exist_ok=True)

        group_start = (rank // group_size) * group_size
        if group_start > 0:
            prev_group_end_rank = group_start - 1
            prev_marker = marker_dir / f"rank_{prev_group_end_rank}.done"
            self._rank_print(
                f"Waiting for rank {prev_group_end_rank} model init marker before "
                f"initializing rank group {group_start}-{min(group_start + group_size - 1, world_size - 1)}"
            )
            self._wait_for_marker(prev_marker, timeout_sec, f"rank {prev_group_end_rank} model init")

        self._rank_print(
            f"Starting model init/load in rank group "
            f"{group_start}-{min(group_start + group_size - 1, world_size - 1)}"
        )
        return marker_dir, rank, world_size, timeout_sec

    def _stagger_model_init_finish(self, stagger_state) -> None:
        if stagger_state is None:
            return

        marker_dir, rank, world_size, timeout_sec = stagger_state
        marker_path = marker_dir / f"rank_{rank}.done"
        marker_path.write_text(str(time.time()))
        self._rank_print(f"Finished model init/load; wrote {marker_path.name}")

        final_marker = marker_dir / f"rank_{world_size - 1}.done"
        self._wait_for_marker(final_marker, timeout_sec, f"rank {world_size - 1} model init")

    def create_model(self, cfg, training_args):
        import gc

        stagger_state = self._stagger_model_init_begin(training_args)

        model_init_dtype_name = os.environ.get(
            "DREAMZERO_MODEL_INIT_DTYPE",
            str(OmegaConf.select(cfg, "model.config.model_dtype", default="float32")),
        )
        model_init_dtype = dtype_from_string(model_init_dtype_name)
        default_dtype = torch.get_default_dtype()
        if model_init_dtype != default_dtype:
            self._rank_print(f"Instantiating model with default dtype {model_init_dtype_name}")
            torch.set_default_dtype(model_init_dtype)
        try:
            model = instantiate(cfg.model)
        finally:
            if model_init_dtype != default_dtype:
                torch.set_default_dtype(default_dtype)

        if cfg.pretrained_model_path is not None:
            mprint(f"Loading pretrained weights from: {cfg.pretrained_model_path}")
            import json
            from safetensors.torch import load_file

            ckpt_dir = cfg.pretrained_model_path
            safetensors_index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
            safetensors_path = os.path.join(ckpt_dir, "model.safetensors")

            if os.path.exists(safetensors_index_path):
                with open(safetensors_index_path, 'r') as f:
                    index = json.load(f)
                for shard_file in sorted(set(index["weight_map"].values())):
                    shard_path = os.path.join(ckpt_dir, shard_file)
                    mprint(f"Loading shard: {shard_path}")
                    shard_state_dict = load_file(shard_path)
                    model.load_state_dict(shard_state_dict, strict=False)
                    del shard_state_dict
                    gc.collect()
            elif os.path.exists(safetensors_path):
                state_dict = load_file(safetensors_path)
                model.load_state_dict(state_dict, strict=False)
            else:
                raise FileNotFoundError(
                    f"No weights found at '{ckpt_dir}'. "
                    "Expected 'model.safetensors' or 'model.safetensors.index.json'."
                )

            if (hasattr(model, 'action_head')
                    and hasattr(model.action_head, 'inject_lora_after_loading')
                    and model.action_head.config.defer_lora_injection):
                model.action_head.inject_lora_after_loading()

            mprint("Successfully loaded pretrained weights")

        if os.environ.get("MOVE_MODEL_TO_CUDA_AFTER_LOAD", "0").lower() in ("1", "true", "yes"):
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                device = torch.device("cuda", local_rank)
                self._rank_print(f"Moving model to {device} with dtype {model_init_dtype_name}")
                model.to(device=device, dtype=model_init_dtype)
                gc.collect()
                torch.cuda.empty_cache()
            else:
                self._rank_print("MOVE_MODEL_TO_CUDA_AFTER_LOAD set, but CUDA is not available")

        # Optional: torch.compile (Inductor kernel fusion) on the trainable DiT
        # blocks. Gated by TORCH_COMPILE_DIT so the default path is unchanged.
        # Done AFTER weight load + LoRA injection + move-to-cuda so the compiled
        # graph includes the PEFT adapters and runs on the final device. We locate
        # blocks by type via model.modules() to stay robust to PEFT/DDP wrapping,
        # and compile in place with Module.compile() (keeps param registration,
        # so DeepSpeed/grad still work). Compilation itself is lazy: the first
        # training step pays the cost, then it is reused (shared Inductor cache).
        if os.environ.get("TORCH_COMPILE_DIT", "0").lower() in ("1", "true", "yes"):
            try:
                from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
                    CausalWanAttentionBlock,
                )
                compile_mode = os.environ.get("TORCH_COMPILE_DIT_MODE", "default")
                n_compiled = 0
                for sub in model.modules():
                    if isinstance(sub, CausalWanAttentionBlock):
                        sub.compile(mode=compile_mode)
                        n_compiled += 1
                self._rank_print(
                    f"TORCH_COMPILE_DIT enabled: compiled {n_compiled} CausalWanAttentionBlock "
                    f"module(s) with mode={compile_mode}; first step will be slow while Inductor compiles."
                )
            except Exception as e:  # never let a compile-setup failure kill training
                self._rank_print(f"TORCH_COMPILE_DIT requested but setup failed ({e}); continuing in eager mode")

        self._stagger_model_init_finish(stagger_state)

        model.config.resume_path = model.config._name_or_path = training_args.output_dir
        mprint(f"{model}\n")
        return model

    def create_train_dataset(self, cfg, model):
        assert torch.distributed.is_initialized()
        train_dataset = instantiate(cfg.train_dataset)
        return train_dataset

    def create_val_dataset(self, cfg, model):
        return None

    def create_data_collator(self, cfg, model):
        return instantiate(cfg.data_collator)

    def create_trainer(
        self,
        cfg,
        exp_cfg_dir,
        model,
        training_args,
        train_dataset,
        val_dataset,
        data_collator,
        compute_dtype,
    ):
        # Set the gradient accumulation steps.
        if cfg.global_batch_size is not None:
            global_bs = cfg.global_batch_size
            bs = training_args.per_device_train_batch_size
            grad_acc = compute_grad_accum_to_match_global_bs(global_bs, bs)
            training_args.gradient_accumulation_steps = grad_acc
            print(
                f"Set global batch size to {global_bs}, set gradient accumulation steps to {grad_acc}"
            )
        elif cfg.raise_error_if_global_batch_size_not_set:
            raise ValueError(
                "global_batch_size is not set. To ensure the scripts can be reproduced regardless of the number of nodes used, please set this."
            )
        else:
            warnings.warn(
                "global_batch_size is not set. This is fine for debugging, but please set this for real experiments."
            )

        # Instantiate the partial trainer.
        trainer_partial = instantiate(
            cfg.trainer,
            model=model,
            output_dir=training_args.output_dir,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_dtype=compute_dtype,
        )

        # Fully instantiate the trainer with dataclasses instances.
        trainer = trainer_partial(data_collator=data_collator, args=training_args)
        trainer.base_cfg = cfg
        train_dl_len = len(trainer.get_train_dataloader())
        eval_dl_len = (
            len(trainer.get_eval_dataloader()) if val_dataset is not None else "no eval dataloader"
        )

        # Save the total training steps in the config.
        with open_dict(cfg):
            cfg.total_training_steps = train_dl_len * cfg.training_args.num_train_epochs

        # Save config.
        OmegaConf.save(cfg, exp_cfg_dir / "conf.yaml", resolve=True)

        run_name = cfg.training_args.get("run_name", None)
        ckpt_format_callback = CheckpointFormatCallback(run_name=run_name, exp_cfg_dir=exp_cfg_dir)
        trainer.add_callback(ckpt_format_callback)

        loss_log_path = str(Path(training_args.output_dir) / "loss_log.jsonl")
        trainer.add_callback(LossLoggerCallback(output_path=loss_log_path))


        # Add profiling callback (local profiling only, no S3 upload)
        # Local: {output_dir}/profiling/rank_{id}/*.pt.trace.json
        if cfg.trainer.get("enable_prof_callback", False):
            output_dir = Path(training_args.output_dir)
            global_rank = int(os.environ.get("RANK", "0"))

            # Get profiling configuration from trainer config
            profile_start_step = cfg.trainer.get("profile_start_step", 50)
            profile_warmup_steps = cfg.trainer.get("profile_warmup_steps", 1)
            profile_active_steps = cfg.trainer.get("profile_active_steps", 5)
            profile_record_shapes = cfg.trainer.get("profile_record_shapes", False)
            profile_with_stack = cfg.trainer.get(
                "profile_with_stack", False
            )  # Default False to match omni (stack traces add significant file size)
            profile_memory = cfg.trainer.get("profile_memory", False)

            # Calculate end step
            profile_end_step = profile_start_step + profile_warmup_steps + profile_active_steps - 1

            # Setup profile directory with rank subdirectory: {output_dir}/profiling/rank_{id}/
            profile_dir = output_dir / "profiling" / f"rank_{global_rank}"
            profile_dir.mkdir(parents=True, exist_ok=True)

            mprint(
                f"Profiling enabled: steps {profile_start_step}-{profile_end_step}, "
                f"saving to {profile_dir}"
            )

            # Add ProfCallback
            trainer.add_callback(
                ProfCallback(
                    profile_dir=profile_dir,
                    upload_callback=None,
                    profile_start_step=profile_start_step,
                    profile_end_step=profile_end_step,
                    warmup_steps=profile_warmup_steps,
                    active_steps=profile_active_steps,
                    trainer=trainer,
                    record_shapes=profile_record_shapes,
                    with_stack=profile_with_stack,
                    profile_memory=profile_memory,
                )
            )

        mprint(
            f"train dataloader length: {train_dl_len}\n"
            f"eval dataloader length: {eval_dl_len}\n"
            f"train dataset length: {len(trainer.train_dataset)}\n"
            f"GPU memory before training: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024} GB",
            flush=True,
        )
        return trainer

    def train(self):
        # Start training.
        self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        self.trainer.save_state()
        safe_save_model_for_hf_trainer(
            trainer=self.trainer, output_dir=self.training_args.output_dir
        )
