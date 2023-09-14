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
"""Support megatron."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .megatron_in_memory_checkpoint import (initialize_autockpt_for_megatron as initialize_autockpt,
                                                             load_checkpoint_for_megatron as load_checkpoint,
                                                             save_checkpoint_if_needed,
                                                             save_checkpoint_done)
