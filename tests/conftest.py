"""Shared test setup.

HiPerGator login nodes report all 64 cores via ``nproc``, so torch defaults to
64 intra-op threads. For the tiny tensors in these tests that causes severe
thread-contention (a trivial training step can take tens of ms), so cap threads
to keep the suite fast. Real training runs on a GPU under SLURM, where this does
not apply.
"""

from __future__ import annotations

import torch

torch.set_num_threads(1)
