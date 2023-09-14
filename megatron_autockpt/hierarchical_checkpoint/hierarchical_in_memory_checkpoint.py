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
"""In memory checkpoint manager."""

import logging
import os
import queue
import threading
import multiprocessing
from abc import ABC, abstractmethod

import torch

logger = logging.getLogger(__name__)

PREFIX = "done_"
SUFFIX = ".txt"

def _ensure_directory_exists(filename):
  dirname = os.path.dirname(filename)
  if not os.path.exists(dirname):
    os.makedirs(dirname)

def _mark_checkpoint_done(iteration_path):
  rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
  filename = PREFIX + str(rank) + SUFFIX
  filepath = os.path.join(iteration_path, filename)
  with open(filepath, 'w') as f: # pylint: disable=unused-variable, unspecified-encoding
    pass

def _validate_checkpoint_done(iteration_path):
  """Validate checkpoint done."""
  world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
  count = 0
  for filename in os.listdir(iteration_path):
    if filename.startswith(PREFIX) and filename.endswith(SUFFIX):
      count += 1
      os.remove(os.path.join(iteration_path, filename))
  return world_size == count

def _clean_checkpoint_done(iteration_path):
  for filename in os.listdir(iteration_path):
    if filename.startswith(PREFIX) and filename.endswith(SUFFIX):
      os.remove(os.path.join(iteration_path, filename))


class InMemoryCheckpointManager(ABC):
  """In memory checkpoint manager."""
  def __init__(self):
    self._pin_memory_buffer = {}
    self._manager = multiprocessing.Manager()
    self._share_memory = self._manager.dict()
    self._saved_iterations = self._manager.list()
    self._memory_buffer_lock = multiprocessing.Lock()
    self._save_to_storage_event = multiprocessing.Event()
    self._save_to_storage_queue = multiprocessing.Queue(maxsize=1)
    self._finish_save_to_storage_event = multiprocessing.Event()
    self._finish_save_to_storage_queue = multiprocessing.Queue(maxsize=1)
    self._sync_save_to_storage_group = torch.distributed.new_group(backend="gloo")
    self._sync_save_to_storage_thread = threading.Thread(target=self._sync_save_to_storage, daemon=True)
    self._sync_save_to_storage_thread.start()
    self._sub_process = multiprocessing.Process(target=self._on_demand_save_to_storage, daemon=True)
    self._sub_process.start()
    self._save_to_storage_done = torch.Tensor([0])
    if torch.distributed.is_nccl_available() and torch.cuda.is_available():
      self._save_to_storage_done = self._save_to_storage_done.cuda()

  def _make_state_dict_buffer(self, state_dict):
    """Make state dict buffer."""
    def traversal_state_dict(value):
      if isinstance(value, dict):
        temp_dict = {}
        for k, v in value.items():
          temp_dict[k] = traversal_state_dict(v)
        return temp_dict
      if isinstance(value, list):
        temp_list = []
        for item in value:
          temp_list.append(traversal_state_dict(item))
        return temp_list
      if torch.is_tensor(value):
        return torch.empty_like(value.cpu(), pin_memory=True)
      return value
    state_dict_buffer = traversal_state_dict(state_dict)
    return state_dict_buffer

  def _make_pin_memory_buffer(self, kwargs):
    """Make pin memory buffer."""
    for state_dict_name, state_dict_value in kwargs.items():
      if state_dict_value:
        self._pin_memory_buffer[state_dict_name] = self._make_state_dict_buffer(state_dict_value)

  def _sync_save_to_storage(self):
    """Sync whether each rank has finished save_to_storage."""
    while self._finish_save_to_storage_event.wait():
      self._finish_save_to_storage_event.clear()
      try:
        torch.distributed.barrier(group=self._sync_save_to_storage_group)
        self._finish_save_to_storage_queue.put(True)
      except Exception as e: # pylint: disable=broad-except
        logger.info('Exit exception when do barrier in _sync_save_to_storage: %s', e)
        self._finish_save_to_storage_queue.put(False)

  def _copy_state_to_buffer(self, source_state, target_buffer):
    """Copy state_dict to pin memory buffer."""
    def traversal_copy(state_dict, pin_memory_buffer):
      if isinstance(state_dict, dict):
        for k, v in state_dict.items():
          if isinstance(v, (dict, list)):
            traversal_copy(state_dict[k], pin_memory_buffer[k])
          elif torch.is_tensor(v):
            pin_memory_buffer[k].copy_(state_dict[k])
          else:
            pin_memory_buffer[k] = state_dict[k]
      elif isinstance(state_dict, list):
        for i in range(len(state_dict)): # pylint: disable=consider-using-enumerate
          if isinstance(state_dict[i], (dict, list)):
            traversal_copy(state_dict[i], pin_memory_buffer[i])
          elif torch.is_tensor(state_dict[i]):
            pin_memory_buffer[i].copy_(state_dict[i])
          else:
            pin_memory_buffer[i] = state_dict[i]
    traversal_copy(source_state, target_buffer)

  def _write_to_pin_memory(self, iteration, kwargs):
    """Write to pin memory."""
    # Here, we make sure the content of pin_memory is not dirty. So we only write to pin memory
    # when save_to_storage threads of each rank are not locked.
    acq = self._memory_buffer_lock.acquire(block=False)
    self._save_to_storage_done[0] = 0 if acq else 1
    if torch.distributed.is_initialized():
      try:
        torch.distributed.all_reduce(self._save_to_storage_done, op=torch.distributed.ReduceOp.SUM)
      except Exception as e: # pylint: disable=broad-except
        logger.info('Exit exception when do all_reduce in write_to_pin_memory: %s', e)
    if self._save_to_storage_done.item() > 0:
      logger.info('Still save checkpoints to storage, skip write_to_pin_memory.')
      self._save_to_storage_done[0] = 0
      if acq:
        self._memory_buffer_lock.release()
    else:
      # We assume that args.save_interval(the freqeunce of save_to_storage()) is not frequent,
      # and we know that the overhead of save_to_storage() is less than each train_step,
      # so we can write to pin_memory after at most few iterations been skipped.
      logger.info('Start to save in memory checkpoint at iteration %d .' % (iteration))
      for state_dict_name, state_dict_value in kwargs.items():
        if state_dict_value and state_dict_name in self._pin_memory_buffer:
          self._copy_state_to_buffer(state_dict_value, self._pin_memory_buffer[state_dict_name])
      self._pin_memory_buffer["iteration"] = iteration
      for key, value in self._pin_memory_buffer.items():
        self._share_memory[key] = value
      self._memory_buffer_lock.release()
      self._save_to_storage_event.set()
      logger.info('Successfully saved in memory checkpoint at iteration %d .' % (iteration))
    # Todo(linzheyu.lzy): we should guarantee that we will not skip write_to_pin_memory()
    # when the iteration is at args.save_interval.

  def save_in_memory_checkpoint(self, iteration, **kwargs):
    """Save in memory checkpoints."""
    # Make pin memory dicts.
    if not self._pin_memory_buffer:
      self._make_pin_memory_buffer(kwargs)
    # Write to pin memory.
    self._write_to_pin_memory(iteration, kwargs)

  def _save_to_storage(self, state_dict, save_path):
    """Save to storage."""
    if state_dict:
      _ensure_directory_exists(save_path)
      torch.save(state_dict, save_path)

  @abstractmethod
  def _save_to_storage_specific(self):
    """Save to storage for specific."""
    pass # pylint: disable=unnecessary-pass

  def _on_demand_save_to_storage(self):
    """On demand save to storage(background thread)."""
    while True:
      iteration = self._save_to_storage_queue.get()
      if iteration is None:
        break
      self._save_to_storage_event.wait()
      self._save_to_storage_event.clear()
      if self._share_memory and \
       "iteration" in self._share_memory and \
       self._share_memory["iteration"] not in self._saved_iterations:
        with self._memory_buffer_lock:
          logger.info('Start to save checkpoints to storage %d .' % (int(self._share_memory["iteration"])))
          self._save_to_storage_specific()

  def save_checkpoint_to_storage(self, iteration):
    """Save checkpoint to storage."""
    try:
      self._save_to_storage_queue.put(iteration, block=False)
    except queue.Full:
      logger.info('Ready to save checkpoints to storage %d .' % (iteration))

  def save_checkpoint_done(self):
    """Stop the background thread."""
    self._save_to_storage_queue.put(None)
