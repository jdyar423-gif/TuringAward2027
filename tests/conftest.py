import gc
import os
import sys

import pytest

# Keep CPU test runs small and deterministic: the dev container has 4 cores and a shared memory limit.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@pytest.fixture(autouse=True, scope="module")
def _release_compiled_programs():
    """Drop JAX's compilation caches after every test module. Each compiled program holds memory
    mappings of the process; one pytest process running a whole directory otherwise collects
    them until the kernel's limit on mappings (``vm.max_map_count``, 65,530 by default) and
    every later compilation fails with "LLVM compilation error: Cannot allocate memory"."""
    yield
    jax = sys.modules.get("jax")
    if jax is not None:
        jax.clear_caches()
    gc.collect()
