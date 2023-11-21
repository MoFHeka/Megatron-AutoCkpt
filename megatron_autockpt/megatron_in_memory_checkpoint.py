# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""In memory checkpoint for megatron."""

import logging
import os
import shutil

import torch

from megatron.core import mpu
from megatron.utils import unwrap_model
from megatron.global_vars import get_args
from megatron.checkpointing import (
    load_checkpoint,
    get_rng_state,
    get_checkpoint_tracker_filename,
)

try:
  from megatron.checkpointing import get_checkpoint_names

  get_checkpoint_names_fun_exists = True
except:
  from megatron.checkpointing import (
      get_checkpoint_name,
      get_distributed_optimizer_checkpoint_name,
  )

  get_checkpoint_names_fun_exists = False

from megatron_autockpt.hierarchical_checkpoint import (
    _validate_checkpoint_done,
    _mark_checkpoint_done,
    _clean_checkpoint_done,
    InMemoryCheckpointManager,
)

logger = logging.getLogger(__name__)


def _get_iteration_path_for_megatron(checkpoints_path, iteration):
  directory = "iter_{:07d}".format(iteration)
  iteration_path = os.path.join(checkpoints_path, directory)
  return iteration_path


def _validate_checkpoint_done_for_megatron(checkpoints_path, iteration_prefix):
  """Validate checkpoint done."""
  if (not torch.distributed.is_initialized()
      or torch.distributed.get_rank() == 0):  # pylint: disable=too-many-nested-blocks
    if os.path.exists(checkpoints_path):
      tracker_filename = get_checkpoint_tracker_filename(checkpoints_path)
      if not os.path.exists(tracker_filename):
        folders = [
            folder for folder in os.listdir(checkpoints_path)
            if folder.startswith(iteration_prefix)
        ]
        folders.sort(key=lambda x: int(x.split("_")[1]), reverse=True)
        find_last_iteration = False
        for folder in folders:
          path_name = os.path.join(checkpoints_path, folder)
          if not find_last_iteration:
            if _validate_checkpoint_done(path_name):
              iteration = int(folder.split("_")[1])
              with open(tracker_filename, "w") as f:  # pylint: disable=unspecified-encoding
                f.write(str(iteration))
              find_last_iteration = True
          else:
            # Todo(linzheyu.lzy): need to clean each folder?
            _clean_checkpoint_done(path_name)
  # Make sure tracker_file has been written before each rank comes here.
  if torch.distributed.is_initialized():
    torch.distributed.barrier()


def load_checkpoint_for_megatron(model,
                                 optimizer,
                                 opt_param_scheduler,
                                 load_arg="load",
                                 strict=True):
  """Load checkpoint for megatron."""
  args = get_args()
  load_dir = getattr(args, load_arg)
  _validate_checkpoint_done_for_megatron(load_dir, "iter_")
  return load_checkpoint(model, optimizer, opt_param_scheduler, load_arg,
                         strict)


def _clean_up_storage_for_megatron(max_ckpt_num):
  """Delete first iter ckpts."""
  if not torch.distributed.is_initialized() or torch.distributed.get_rank(
  ) == 0:
    args = get_args()
    if os.path.exists(args.save):
      ckpt_folders = [
          folder for folder in os.listdir(args.save)
          if folder.startswith("iter_")
      ]
      if len(ckpt_folders) >= max_ckpt_num:
        ckpt_folders.sort(key=lambda x: int(x.split("_")[1]))
        first_iter_ckpts_path = os.path.join(args.save, ckpt_folders[0])
        shutil.rmtree(first_iter_ckpts_path)
  # Make sure ckpts have been deleted before each rank comes here.
  if torch.distributed.is_initialized():
    torch.distributed.barrier()


class InMemCkptMegatron(InMemoryCheckpointManager):
  """In Memory Checkpoint for Megatron."""

  def _save_to_storage_specific(self):
    """Save to storage for megatron."""
    args = get_args()

    iteration = self._share_memory["iteration"]

    # Checkpoint file names.
    if get_checkpoint_names_fun_exists is True:
      model_checkpoint_name, optim_checkpoint_name = get_checkpoint_names(
          args.save, iteration, args.no_load_optim,
          args.use_distributed_optimizer)
    else:
      model_checkpoint_name = get_checkpoint_name(args.save, iteration)
      optim_checkpoint_name = get_distributed_optimizer_checkpoint_name(
          model_checkpoint_name)

    model_state_dict = {}
    if "model_state_dict" in self._share_memory:
      model_state_dict = self._share_memory["model_state_dict"]

    optim_state_dict = {}
    if "optim_state_dict" in self._share_memory:
      optim_state_dict = self._share_memory["optim_state_dict"]

    if args.use_distributed_optimizer:
      # Save model separate from optimizer.
      self._save_to_storage(model_state_dict, model_checkpoint_name)
      self._save_to_storage(optim_state_dict, optim_checkpoint_name)
    else:
      # Save model and optimizer together.
      state_dict = {**model_state_dict, **optim_state_dict}
      self._save_to_storage(state_dict, model_checkpoint_name)

    logger.info(
        "Successfully saved checkpoints to storage at iteration {:7d} to {}".
        format(iteration, args.save))

    # The write_to_storage thread need to know whether the in_memory_ckpt of this iteration has been saved.
    self._saved_iterations.append(iteration)

    # In Megatron, when the process of each rank has saved ckpt to storage,
    # latest_checkpointed_iteration.txt will be revised at args.save(checkpoints_path). Here, we
    # write to storage asyncly, so we can not use torch.distributed.barrier() to make sure each process
    # has finished. In our way, We will revise latest_checkpointed_iteration.txt before load_checkpoint().
    iteration_path = _get_iteration_path_for_megatron(args.save, iteration)
    _mark_checkpoint_done(iteration_path)

    # Start to wait other rank.
    self._finish_save_to_storage_event.set()
    sync_result = self._finish_save_to_storage_queue.get()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized(
    ) else 0
    if sync_result and rank == 0:
      tracker_filename = get_checkpoint_tracker_filename(args.save)
      with open(tracker_filename, "w") as f:  # pylint: disable=unspecified-encoding
        f.write(str(iteration))

  def save_distributed_optimizer_parameter_state(self, optimizer):
    """Save parameter state (i.e., parameter & optimizer tensors).
        Copy from Megatron save_parameter_state function

        This method performs three steps:
        - For each DP rank, copy param & optimizer shards to contiguous CPU
          buffers. (e.g., one buffer each for main_param, exp_avg, and
          exp_avg_sq).
        - Gather contiguous buffers on DP rank 0 and concatenate to world
          buffers.
        - Save world buffers to disk (i.e., distrib_opt.pt).
        """

    # Data parallelism variables.
    data_parallel_world_size = mpu.get_data_parallel_world_size(
        with_context_parallel=True)
    data_parallel_rank = mpu.get_data_parallel_rank(with_context_parallel=True)
    data_parallel_group_gloo = mpu.get_data_parallel_group_gloo(
        with_context_parallel=True)
    data_parallel_global_ranks = list(mpu._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP)

    # Collect param states.
    state = {"per_bucket_numel": self.per_bucket_numel}
    for model_idx, gbuf_range_maps in enumerate(self.model_gbuf_ranges):
      # Iterate grad buffers (by data type).
      dtype_state = {}
      assert len(gbuf_range_maps) == 1, "single dtype supported, for now."
      for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
        world_tensors = {}
        for bucket_idx, gbuf_range_map in enumerate(
            gbuf_range_map_for_all_buckets):
          # Compute local DP contiguous shard's size.
          model = self.models[model_idx]
          gbuf_world_numel = (
              model.grad_buffers[dtype].buckets[bucket_idx].data.numel())
          assert gbuf_world_numel % data_parallel_world_size == 0
          gbuf_local_numel = gbuf_world_numel // data_parallel_world_size
          local_shards = {
              key: torch.empty((gbuf_local_numel,),
                               dtype=torch.float32,
                               device="cpu")
              for key in ("param", "exp_avg", "exp_avg_sq")
          }

          # Build contiguous DP rank shards (for param + optim states).
          for model_param, param_range_map in gbuf_range_map["param_map"].items(
          ):
            # Main param & optimizer states.
            group_index, group_order = self.model_param_group_index_map[
                model_param]
            main_param = self.optimizer.param_groups[group_index]["params"][
                group_order]
            optim_state = self.optimizer.state[main_param]

            tensors = {
                "param": main_param,
                **optim_state,
            }

            # Copy states into contiguous shard.
            gbuf_local_start = param_range_map["gbuf_local"].start
            gbuf_local_end = param_range_map["gbuf_local"].end
            for key in local_shards:
              local_shards[key][gbuf_local_start:gbuf_local_end].data.copy_(
                  tensors[key].detach().cpu())

          # Gather contiguous shards on DP rank 0.
          for key, send_tensor in local_shards.items():
            # Gather tensor list.
            if data_parallel_rank == 0:
              recv_tensors = [
                  torch.empty(
                      (gbuf_local_numel,),
                      dtype=torch.float32,
                      device="cpu",
                  ) for _ in range(data_parallel_world_size)
              ]
            else:
              recv_tensors = None

            # Gather.
            torch.distributed.gather(
                send_tensor,
                recv_tensors,
                data_parallel_global_ranks[0],
                data_parallel_group_gloo,
            )

            # Concatenate.
            if data_parallel_rank == 0:
              if key not in world_tensors:
                world_tensors[key] = []
              world_tensors[key].append(torch.cat(recv_tensors))

        # Collect world state.
        dtype_state[dtype] = world_tensors
      state[model_idx] = dtype_state

    return state

  def save_in_memory_checkpoint_for_megatron(self, iteration, model, optimizer,
                                             opt_param_scheduler):
    """Save checkpoint for megatron."""
    args = get_args()

    # Only rank zero of the data parallel writes to the storage.
    model = unwrap_model(model)

    # Collect rng state across data parallel ranks.
    rng_state = get_rng_state()

    # Collect args, model, RNG.
    model_state_dict = {}
    if not torch.distributed.is_initialized() or mpu.get_data_parallel_rank(
    ) == 0:
      # Arguments, iteration, and model.
      model_state_dict["args"] = args
      model_state_dict["checkpoint_version"] = 3.0
      model_state_dict["iteration"] = iteration
      if len(model) == 1:
        model_state_dict["model"] = model[0].state_dict_for_save_checkpoint()
      else:
        for i in range(len(model)):  # pylint: disable=consider-using-enumerate
          mpu.set_virtual_pipeline_model_parallel_rank(i)
          model_state_dict["model%d" %
                           i] = model[i].state_dict_for_save_checkpoint()

      # RNG states.
      if not args.no_save_rng:
        model_state_dict["rng_state"] = rng_state

    # Collect optimizer state. (Optimizer is saved separately from the model, due
    # to the conflicting data pattern when using the distributed optimizer.)
    optim_state_dict = {}
    if (not torch.distributed.is_initialized()
        or mpu.get_data_parallel_rank() == 0 or args.use_distributed_optimizer):
      if hasattr(optimizer, "save_parameter_state"):
        optim_state_dict = self.save_distributed_optimizer_parameter_state(
            optimizer)
        if not args.no_save_optim:
          # Optimizer stuff.
          if optimizer is not None:
            model_state_dict["optimizer"] = optimizer.state_dict()
          if opt_param_scheduler is not None:
            model_state_dict[
                "opt_param_scheduler"] = opt_param_scheduler.state_dict()
      else:
        if not args.no_save_optim:
          # Optimizer stuff.
          if optimizer is not None:
            optim_state_dict["optimizer"] = optimizer.state_dict()
          if opt_param_scheduler is not None:
            optim_state_dict[
                "opt_param_scheduler"] = opt_param_scheduler.state_dict()

    self.save_in_memory_checkpoint(
        iteration,
        model_state_dict=model_state_dict,
        optim_state_dict=optim_state_dict,
    )


in_mem_ckpt_mgr = None
save_to_mem_interval = None
save_to_storage_interval = None
max_checkpoint_num = None


def initialize_autockpt_for_megatron(save_mem_interval, save_storage_interval,
                                     max_ckpt_num):
  """Initialize AutoCkpt for Megatron."""
  global in_mem_ckpt_mgr
  if in_mem_ckpt_mgr is None:
    in_mem_ckpt_mgr = InMemCkptMegatron()
  global save_to_mem_interval
  if save_to_mem_interval is None:
    save_to_mem_interval = save_mem_interval
  global save_to_storage_interval
  if save_to_storage_interval is None:
    save_to_storage_interval = save_storage_interval
  global max_checkpoint_num
  if max_checkpoint_num is None:
    max_checkpoint_num = max_ckpt_num


def save_checkpoint_if_needed(iteration, model, optimizer, opt_param_scheduler):
  if in_mem_ckpt_mgr is None:
    logger.error(
        "You must initialize autockpt first, exit save_checkpoint_if_needed()..."
    )
    return
  if save_to_mem_interval and iteration % save_to_mem_interval == 0:
    in_mem_ckpt_mgr.save_in_memory_checkpoint_for_megatron(
        iteration, model, optimizer, opt_param_scheduler)
  if save_to_storage_interval and iteration % save_to_storage_interval == 0:
    _clean_up_storage_for_megatron(max_checkpoint_num)
    in_mem_ckpt_mgr.save_checkpoint_to_storage(iteration)


def save_checkpoint_done():
  in_mem_ckpt_mgr.save_checkpoint_done()
