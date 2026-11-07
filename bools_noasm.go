//go:build !arm64 && !amd64

package bitset

// No SIMD kernel for the bool conversions on this architecture: they report
// no word done and the portable loops in bools.go do everything.
func packBools(dst []uint64, src []byte) int   { return 0 }
func unpackBools(dst []byte, src []uint64) int { return 0 }
