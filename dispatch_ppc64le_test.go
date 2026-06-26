//go:build ppc64le

package bitset

import (
	"math/rand"
	"testing"

	"golang.org/x/sys/cpu"
)

// TestDispatchPPC64LE drives every public function down BOTH ppc64le paths — the
// VSX kernel and the pure-Go scalar fallback — by toggling hasVSX, restoring it
// with defer. The fallback (hasVSX=false) is always safe. The kernel branch
// emits ISA-3.0 (POWER9) instructions (e.g. MFVSRLD) that SIGILL on POWER8, so
// it is forced on only when the host is genuinely POWER9+. Under the QEMU
// power9 CI target IsPOWER9 is true, so both branches are covered there, making
// this the authoritative 100%-coverage gate for the ppc64le dispatchers.
func TestDispatchPPC64LE(t *testing.T) {
	saved := hasVSX
	defer func() { hasVSX = saved }()

	rng := rand.New(rand.NewSource(42))
	sizes := []int{0, 1, 3, 4, 5, 8, 33, 100, 257}

	run := func(label string) {
		for _, n := range sizes {
			a := randWords(rng, n)
			b := randWords(rng, n)
			dst := make([]uint64, n)
			And(dst, a, b)
			eq(t, label+" And", dst, refAnd(make([]uint64, n), a, b))
			Or(dst, a, b)
			eq(t, label+" Or", dst, refOr(make([]uint64, n), a, b))
			AndNot(dst, a, b)
			eq(t, label+" AndNot", dst, refAndNot(make([]uint64, n), a, b))
			Xor(dst, a, b)
			eq(t, label+" Xor", dst, refXor(make([]uint64, n), a, b))
			if Count(a) != refCount(a) {
				t.Fatalf("%s Count n=%d", label, n)
			}
			if IntersectionCount(a, b) != refPair(a, b, func(x, y uint64) uint64 { return x & y }) {
				t.Fatalf("%s IntersectionCount n=%d", label, n)
			}
			if UnionCount(a, b) != refPair(a, b, func(x, y uint64) uint64 { return x | y }) {
				t.Fatalf("%s UnionCount n=%d", label, n)
			}
			if DifferenceCount(a, b) != refPair(a, b, func(x, y uint64) uint64 { return x & ^y }) {
				t.Fatalf("%s DifferenceCount n=%d", label, n)
			}
		}
	}

	// Scalar fallback: always safe.
	hasVSX = false
	run("fallback")

	// Kernel: ISA-3.0 (POWER9) instructions SIGILL on POWER8, so only force the
	// VSX branch on a genuine POWER9+ host (true under QEMU power9 CI).
	if !cpu.PPC64.IsPOWER9 {
		t.Log("pre-POWER9 host; VSX kernel branch not exercised")
		return
	}
	hasVSX = true
	run("kernel")
}
