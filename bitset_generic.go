//go:build !amd64 && !arm64 && !loong64 && !ppc64le && !s390x

package bitset

// No SIMD kernel on this architecture (this includes riscv64: the base RVV
// profile guaranteed by the toolchain does not give a portable per-element word
// popcount, and the logical ops are bandwidth-bound where scalar already
// saturates), so every operation uses the portable math/bits word loop.

func and(dst, a, b []uint64)    { andScalarRef(dst, a, b) }
func or(dst, a, b []uint64)     { orScalarRef(dst, a, b) }
func andNot(dst, a, b []uint64) { andNotScalarRef(dst, a, b) }
func xor(dst, a, b []uint64)    { xorScalarRef(dst, a, b) }

func count(a []uint64) int                 { return countScalarRef(a) }
func intersectionCount(a, b []uint64) int  { return intersectionCountScalarRef(a, b) }
func unionCount(a, b []uint64) int         { return unionCountScalarRef(a, b) }
func differenceCount(a, b []uint64) int    { return differenceCountScalarRef(a, b) }
