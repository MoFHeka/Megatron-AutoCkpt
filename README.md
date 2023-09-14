# Megatron-AutoCkpt
A Megatron checkpoint auto-saving patch at the end of each iteration, inspired by Alibaba PAI EasyCkpt for Megatron. 

# Installation
```bash
pip install --no-build-isolation git+https://github.com/MoFHeka/Megatron-AutoCkpt.git
```

# How to use

---

**1. 在import处，注释掉megatron.checkpointing⾥引⽤的load_checkpoint。**

**1. At the import section, comment out the load_checkpoint referenced in megatron.checkpointing.**

```python
# from megatron.checkpointing import load_checkpoint
```

---

**2. 在import处，加⼊megatron_autockpt的相关函数**

**2. At the import section, add the related functions of megatron_autockpt.**

```python
from megatron_autockpt import (
    load_checkpoint,
    initialize_autockpt,
    save_checkpoint_if_needed)
```

---

**3. 在“while iteration < args.train_iters”循环之前，即真的触发save_checkpoint之前，初始化AutoCkpt。其中，需要指定三个参数。**

**3. Initialize AutoCkpt before the 'while iteration < args.train_iters' loop, that is, before actually triggering save_checkpoint. Three parameters need to be specified.**

```python
initialize_autockpt(save_mem_interval=2, save_storage_interval=4, max_ckpt_num=100)
```

---

**4. 在“while iteration < args.train_iters”循环之内，“train_step“相关代码后⾯任意位置，做将 GPU Tensor 拷贝到 Host 侧和将 Tensor 保存等的相关操作。**



**4. Within the 'while iteration < args.train_iters' loop, at any place after the 'train_step' related code, perform operations such as copying GPU Tensor to the Host side and saving Tensor.**

```python
save_checkpoint_if_needed(iteration, model, optimizer, opt_param_scheduler)
```

---