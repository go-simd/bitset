<p align="center"><img src="https://raw.githubusercontent.com/go-simd/brand/main/social/go-simd.png" alt="go-simd/bitset" width="720"></p>

# bitset

[![ci](https://github.com/go-simd/bitset/actions/workflows/ci.yml/badge.svg)](https://github.com/go-simd/bitset/actions/workflows/ci.yml)
![coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
[![Go Reference](https://pkg.go.dev/badge/github.com/go-simd/bitset.svg)](https://pkg.go.dev/github.com/go-simd/bitset)

SIMD-accelerated **bulk bit-set operations** over a bit set stored as a slice of
64-bit words (`[]uint64`), in pure Go (`CGO_ENABLED=0`, stable Go, no
`GOEXPERIMENT`).

```go
import "github.com/go-simd/bitset"

bitset.And(dst, a, b)              // dst[i] = a[i] & b[i]   (in-place capable)
bitset.Or(dst, a, b)              // dst[i] = a[i] | b[i]
bitset.AndNot(dst, a, b)          // dst[i] = a[i] &^ b[i]
bitset.Xor(dst, a, b)             // dst[i] = a[i] ^ b[i]

n  := bitset.Count(a)             // total set bits (popcount)
ni := bitset.IntersectionCount(a, b) // popcount(a & b),  no allocation
nu := bitset.UnionCount(a, b)        // popcount(a | b)
nd := bitset.DifferenceCount(a, b)   // popcount(a &^ b)

bitset.FromBools(dst, bools)      // pack []bool into the words
bitset.ToBools(bools, a)          // expand the words into []bool
```

`FromBools` and `ToBools` convert between a bit set and a `[]bool` (one byte
per element, the layout a dataframe's boolean column or a validity mask has):
64 elements per iteration on amd64 (AVX2) and arm64 (NEON), eight at a time in
portable Go elsewhere (a multiply that gathers eight bool bytes into a byte, and
a 256-entry table of eight bools per byte).

- **amd64**: pack is `VPSLLW $7` + `VPMOVMSKB` (two 32-byte loads per word);
  unpack broadcasts each 32-bit half, `VPSHUFB` copies the byte, `VPAND` +
  `VPCMPEQB` test the bit, `VPAND` with 1 makes the bool.
- **arm64**: pack is `VUSHL` (shift byte *i* of each 8-byte group by *i*) and
  three rounds of `VADDP`, so four 16-byte loads reduce to the word; unpack is
  `VTBL` (replicate each byte 8 times), `VUSHL` by 0..-7, `VAND` 1.
- loong64, ppc64le, s390x, riscv64: the portable loops (not yet measured against
  NumPy; no claim is made for them).

Measured against NumPy 2.5.3 `np.packbits(..., bitorder="little")` and
`np.unpackbits(..., bitorder="little")` on a `bool` array (GB/s counting one
byte per element, single core pinned with `taskset`, 8 interleaved Go/NumPy
rounds, medians; Go 1.27.3; the Go per-element loop is the control and did not
move between rounds). Source: [`bench/`](bench) (`numpy_bools.py`, `interleave`).

| host (load at the time)                    | op          | size | Go (GB/s) | NumPy (GB/s) | Go / NumPy |
| ------------------------------------------ | ----------- | ---- | --------: | -----------: | ---------: |
| cfarm424, Neoverse N1 (16.5 on 64 cores)   | `FromBools` | 2^10 |     21.03 |         0.85 |      24.7x |
|                                            | `FromBools` | 2^16 |     24.65 |         9.49 |       2.6x |
|                                            | `FromBools` | 2^20 |     10.72 |         9.05 |       1.2x |
|                                            | `ToBools`   | 2^10 |     14.48 |         0.78 |      18.6x |
|                                            | `ToBools`   | 2^16 |     17.09 |         7.45 |       2.3x |
|                                            | `ToBools`   | 2^20 |     16.66 |         8.53 |       2.0x |
| cfarm420, Zen 3 (2.5-3.3 on 128 cores)     | `FromBools` | 2^10 |     47.84 |         1.08 |      44.3x |
|                                            | `FromBools` | 2^16 |     62.72 |        21.11 |       3.0x |
|                                            | `FromBools` | 2^20 |     56.09 |        29.47 |       1.9x |
|                                            | `ToBools`   | 2^10 |     39.95 |         1.00 |      40.0x |
|                                            | `ToBools`   | 2^16 |     52.59 |        14.45 |       3.6x |
|                                            | `ToBools`   | 2^20 |     49.42 |        18.05 |       2.7x |

The 2^10 rows are dominated by NumPy's per-call overhead (about a microsecond
from Python), so they say little about its kernel; 2^16 and 2^20 are the
throughput comparison. Both hosts are shared machines: the load averages are
those of other users' jobs on mostly idle cores, above the unloaded host the
method asks for, so treat the ratios, not the absolute figures, as the result.
Before the kernels, the portable Go was 3.3 and 3.9 GB/s on N1 (2^20) and 5.8
and 6.2 GB/s on Zen 3: below NumPy at 2^16 and 2^20 (0.2x to 0.4x on Zen 3).

Word `w` bit `b` is set bit `w*64+b`. The four logical ops write the first
`min(len(dst), len(a), len(b))` words and may alias (`dst` can be `a` and/or
`b`); the counts read the first `min(len(a), len(b))` words. Nothing is
allocated, and the read-only ops are safe for concurrent use.

## Algorithm

Two clean families — vector logical ops and vector popcount — both available on
every 64-bit SIMD target, so no integer multiply (and therefore no special-casing
of arm64) is needed. The hot loops are assembly generated by
[go-asmgen](https://github.com/go-asmgen/asmgen):

| arch        | logical ops (`And`/`Or`/`AndNot`/`Xor`)                       | counts                                                         |
| ----------- | ------------------------------------------------------------- | ------------------------------------------------------------- |
| **amd64**   | AVX2 `VPAND`/`VPOR`/`VPANDN`/`VPXOR`, 32 B/iter               | 4-way `POPCNTQ`, boolean op fused in a GPR                     |
| **arm64**   | NEON `VAND`/`VORR`/`VEOR` (`AndNot` = `a ^ (a&b)`)            | NEON `VCNT` + `VUADDLV` over the combined words                |
| **loong64** | LSX `VANDV`/`VORV`/`VXORV`/`VANDNV`                           | LSX `VPCNTV` per-64-bit lane over the combined words           |
| **ppc64le** | VSX `VAND`/`VOR`/`VXOR`/`VANDC`                               | VSX `VPOPCNTD` per-doubleword over the combined words          |
| **s390x**   | vector `VN`/`VO`/`VX`/`VNC`                                   | `VPOPCT` + `VSUMB`/`VSUMQF` reduction over the combined words  |
| **riscv64** | portable `math/bits` word loop                               | portable `math/bits.OnesCount64` word loop                    |

Five architectures carry native SIMD kernels: amd64, arm64, loong64, ppc64le and
s390x. The ppc64le path is the POWER8 VSX baseline, s390x is the z13 vector-
facility baseline, and loong64 LSX is treated as baseline — none needs a runtime
feature flag. amd64 gates on `AVX2` (logical ops) and `POPCNT` (counts) at
runtime, falling back to the scalar loop otherwise. riscv64 uses the portable
`math/bits` path (the base RVV profile gives no portable per-element word
popcount, and the bandwidth-bound logical ops gain nothing over scalar there).

The pairwise counts (`IntersectionCount`/`UnionCount`/`DifferenceCount`) **fuse**
the boolean op into the popcount kernel, so the combined words are never written
back to memory. Each kernel handles whole vector blocks; a scalar `math/bits`
tail finishes the remainder, so every result is bit-identical to the reference.

## The honest result: it's size-dependent

A scalar `math/bits` word loop already moves about one 64-bit word per cycle, so
for large, out-of-cache sets the bulk ops are **memory-bandwidth-bound** and SIMD
can only *tie* scalar — the win is real for in-cache / medium sets, where the
problem is compute-bound. The fused popcount counts win across the board because
they replace several scalar instructions per word with one vector instruction.

Representative throughput (`go test -bench .`, higher is better; ratios are the
honest story, not the absolute MB/s which depend on the host):

**arm64** (Apple-silicon class, native):

| size  | `And` SIMD vs scalar | `Count` SIMD vs scalar / bits-and-blooms | `IntersectionCount` SIMD vs bits-and-blooms |
| ----- | -------------------- | ---------------------------------------- | ------------------------------------------- |
| 1 KiB | ~1.7×                | ~1.8× / ~1.2×                            | ~1.8×                                        |
| 1 MiB | ~1.7×                | ~1.9× / ~1.2×                            | ~1.9×                                        |
| 16 MiB| ~1.9×                | ~1.9× / ~1.2×                            | ~1.9×                                        |

**amd64** (`POPCNT`+`AVX2`, native x86-64 VM):

| size  | `And` SIMD vs scalar | `Count` SIMD vs scalar / bits-and-blooms | `IntersectionCount` SIMD vs bits-and-blooms |
| ----- | -------------------- | ---------------------------------------- | ------------------------------------------- |
| 1 KiB | ~1.2×                | ~3.0× / ~2.9×                            | ~2.5×                                        |
| 1 MiB | ~1.4×                | ~3.9× / ~3.5×                            | ~2.7×                                        |
| 16 MiB| ~1.2× (ties)         | ~3.6× / ~2.2×                            | ~1.1× (ties, bandwidth-bound)               |

The `Count`/`IntersectionCount` kernels are the clearest win (hardware popcount
is far denser than a scalar `OnesCount64`); the logical ops are bandwidth-bound
and converge toward scalar as the set leaves cache.

**ppc64le is now measured on real POWER9 silicon** (GCC Compile Farm,
https://portal.cfarm.net/, VSX, Go 1.26.4, 2026-06-26): `And` over a 1 MiB set
runs at ~8869 MB/s vs ~6048 MB/s scalar — **~1.5×**. Consistent with the
size-dependent story above, this is an honest, modest margin: the logical ops are
bandwidth-bound and converge to scalar out-of-cache.

**riscv64 is now measured on a real SpacemiT X60 (RVV 1.0)** (GCC Compile Farm,
https://portal.cfarm.net/, Go 1.26.4, 2026-06-26). Note riscv64 runs the **portable
`math/bits` word loop**, not a dedicated RVV kernel (see the kernel table above),
so the comparison is that loop vs the scalar reference, and the result is
honestly **modest**: `And` is bandwidth-bound — roughly parity in-cache and
**~1.5× scalar out-of-cache** (16 MiB: ~1420 MB/s vs ~959 MB/s); `Count` is
**~1.2× scalar**. Consistent with the existing bandwidth-bound stance — no large
win is claimed on this core.

**s390x is now measured on real IBM z15 silicon** (VXE2, native execution,
2026-07-03, `-count=6`): the vector `VPOPCT` + `VSUMB`/`VSUMQF` `Count` kernel
runs **~2.8× scalar**, and `And` **~1.6×** — consistent with the size-dependent
story above (the popcount-backed `Count` is the clearest win; the bandwidth-bound
logical ops keep a modest margin).

## Existing Go bit sets

- [`github.com/bits-and-blooms/bitset`](https://github.com/bits-and-blooms/bitset)
  is the de-facto Go bit set — a rich `BitSet` type (Set/Test/Clear, iteration,
  serialization). Its bulk ops are **scalar** `math/bits` loops over its backing
  `[]uint64`. This package is its dense-`[]uint64` SIMD complement: operate on
  the words directly (it exposes them via `FromWithLength`) for the hot bulk
  kernels.
- [`github.com/RoaringBitmap/roaring`](https://github.com/RoaringBitmap/roaring)
  is a different, **compressed** structure (run/array/bitmap containers), not a
  dense `[]uint64` — a different model, not directly comparable.

This package is intentionally minimal: it accelerates the bulk word kernels over
a caller-owned `[]uint64` and composes with either of the above.

## Validation

Every architecture is validated against an independent scalar oracle by a table
test, an exhaustive size sweep across the SIMD-block/tail boundary, and the
`FuzzAnd`/`FuzzOr`/`FuzzAndNot`/`FuzzXor`/`FuzzCount`/`FuzzCounts` fuzzers. CI
runs amd64 and arm64 natively (with the fuzzers and a 100 % coverage gate),
ppc64le natively on real POWER9 silicon (GCC Compile Farm, VSX, Go 1.26.4), and
riscv64/loong64/s390x under qemu (also gated at 100 % coverage). The s390x run
additionally proves the big-endian path: `[]uint64` word ops and the byte/word-
wise popcount reductions are endian-neutral, and the table + fuzz tests are the
proof.

Beyond the six SIMD targets, the portable `math/bits` path is now build- and
test-validated on a **seventh architecture, ppc64 (big-endian)**, on real POWER9
silicon (GCC Compile Farm) — bit-exact on a big-endian target distinct from
s390x's vector kernel. ppc64 BE carries no VSX build tag, so it takes the generic
path. Framing: **six SIMD targets, validated on seven architectures.**

## Regenerating the assembly

The `.s` files are committed. To regenerate them (`//go:build ignore` generators,
one per arch):

```sh
go run bitset_amd64_gen.go
go run bitset_arm64_gen.go
go run bitset_loong64_gen.go
go run bitset_ppc64le_gen.go
go run bitset_s390x_gen.go
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
