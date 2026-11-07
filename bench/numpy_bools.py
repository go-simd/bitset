"""NumPy reference for FromBools/ToBools: np.packbits / np.unpackbits, little bit order.

Prints "<op> <n> <GB/s>" for the best of several timed batches, counting one
byte per bool as the Go benchmarks do. Run on one core (taskset -c N).
"""
import sys
import time

import numpy as np

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
for n in (1 << 10, 1 << 16, 1 << 20):
    src = (np.arange(n) % 3 == 0)
    packed = np.packbits(src, bitorder="little")
    out = np.empty(n, dtype=np.uint8)
    for op, fn in (
        ("FromBools", lambda: np.packbits(src, bitorder="little")),
        ("ToBools", lambda: np.unpackbits(packed, bitorder="little")),
    ):
        iters = max(1, (1 << 24) // n)
        best = 0.0
        for _ in range(REPS):
            t = time.perf_counter()
            for _ in range(iters):
                fn()
            dt = time.perf_counter() - t
            best = max(best, n * iters / dt / 1e9)
        print(op, n, "%.2f" % best)
