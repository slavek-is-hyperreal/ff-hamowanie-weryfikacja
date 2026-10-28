import os

# Intel MKL in torch 2.5.1 may dispatch AVX-512 kernels on CPUs without them (SIGILL on the author's
# i5-3450); the compatible code path is forced before torch is imported (kod/README.md).
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
