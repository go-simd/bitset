//go:build amd64

package bitset

import (
	"math/rand"
	"testing"
)

// TestForceKernels drives each amd64 kernel directly over whole blocks and
// finishes with the scalar tail, mirroring the dispatchers but without the size
// threshold, so every kernel is validated at every length (including ones where
// the dispatcher would have stayed scalar). The logical kernels run only when
// the CPU has AVX2 and the count kernels only when it has POPCNT (the
// instructions would #UD otherwise).
func TestForceKernels(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for n := 0; n <= 300; n++ {
		a := randWords(rng, n)
		b := randWords(rng, n)

		if hasAVX2 {
			for _, tc := range []struct {
				name string
				k    func(dst, a, b []uint64) int
				ref  func(dst, a, b []uint64) []uint64
			}{
				{"and", andKernel, refAnd},
				{"or", orKernel, refOr},
				{"xor", xorKernel, refXor},
				{"andNot", andNotKernel, refAndNot},
			} {
				dst := make([]uint64, n)
				done := tc.k(dst, a, b)
				tail := tc.ref(make([]uint64, n), a, b)
				copyTail(dst, a, b, done, tc.name)
				eq(t, tc.name+"-force", dst, tail)
			}
		}

		if hasPOPCNT {
			sum, done := countKernel(a)
			if got := sum + countScalarRef(a[done:]); got != refCount(a) {
				t.Fatalf("countKernel n=%d: %d want %d", n, got, refCount(a))
			}
			check := func(name string, k func(a, b []uint64) (int, int), op func(x, y uint64) uint64) {
				s, d := k(a, b)
				got := s + refPair(a[d:], b[d:], op)
				if want := refPair(a, b, op); got != want {
					t.Fatalf("%s n=%d: %d want %d", name, n, got, want)
				}
			}
			check("intersection", intersectionKernel, func(x, y uint64) uint64 { return x & y })
			check("union", unionKernel, func(x, y uint64) uint64 { return x | y })
			check("difference", differenceKernel, func(x, y uint64) uint64 { return x & ^y })
		}
	}
}

// copyTail finishes the logical-kernel result on [done:] with the scalar ref, so
// the forced kernel + tail equals the full operation.
func copyTail(dst, a, b []uint64, done int, op string) {
	switch op {
	case "and":
		andScalarRef(dst[done:], a[done:], b[done:])
	case "or":
		orScalarRef(dst[done:], a[done:], b[done:])
	case "xor":
		xorScalarRef(dst[done:], a[done:], b[done:])
	case "andNot":
		andNotScalarRef(dst[done:], a[done:], b[done:])
	}
}

// TestDispatch drives each public function down both its SIMD and its scalar
// branch by toggling the package feature flags, restoring them with defer. A
// branch that calls a kernel is only forced on when the CPU has that feature;
// the scalar path (flags off) is always safe. The native amd64 CI runner has
// AVX2 and POPCNT, so both branches of every function are covered there.
func TestDispatch(t *testing.T) {
	savedA, savedP := hasAVX2, hasPOPCNT
	defer func() { hasAVX2, hasPOPCNT = savedA, savedP }()

	rng := rand.New(rand.NewSource(42))
	sizes := []int{0, 1, 3, 4, 5, 8, 33, 100, 257}

	run := func() {
		for _, n := range sizes {
			a := randWords(rng, n)
			b := randWords(rng, n)
			dst := make([]uint64, n)
			And(dst, a, b)
			eq(t, "And", dst, refAnd(make([]uint64, n), a, b))
			Or(dst, a, b)
			eq(t, "Or", dst, refOr(make([]uint64, n), a, b))
			AndNot(dst, a, b)
			eq(t, "AndNot", dst, refAndNot(make([]uint64, n), a, b))
			Xor(dst, a, b)
			eq(t, "Xor", dst, refXor(make([]uint64, n), a, b))
			if Count(a) != refCount(a) {
				t.Fatalf("Count n=%d", n)
			}
			if IntersectionCount(a, b) != refPair(a, b, func(x, y uint64) uint64 { return x & y }) {
				t.Fatalf("IntersectionCount n=%d", n)
			}
			if UnionCount(a, b) != refPair(a, b, func(x, y uint64) uint64 { return x | y }) {
				t.Fatalf("UnionCount n=%d", n)
			}
			if DifferenceCount(a, b) != refPair(a, b, func(x, y uint64) uint64 { return x & ^y }) {
				t.Fatalf("DifferenceCount n=%d", n)
			}
		}
	}

	// Scalar path: features off (always safe).
	hasAVX2, hasPOPCNT = false, false
	run()

	// SIMD path: restore the real flags (kernels run only where supported).
	hasAVX2, hasPOPCNT = savedA, savedP
	run()
	if !savedA {
		t.Log("CPU lacks AVX2; logical SIMD kernels not exercised on this host")
	}
	if !savedP {
		t.Log("CPU lacks POPCNT; count SIMD kernels not exercised on this host")
	}
}
