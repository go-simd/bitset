package bitset

import (
	"math/rand"
	"testing"
)

// Independent oracles, deliberately distinct from the package's *ScalarRef
// implementations, so the table tests and fuzzers check the public API against a
// second source of truth rather than against its own scalar path.

func refAnd(dst, a, b []uint64) []uint64 {
	n := min3(len(dst), len(a), len(b))
	out := append([]uint64(nil), dst...)
	for i := 0; i < n; i++ {
		out[i] = a[i] & b[i]
	}
	return out
}
func refOr(dst, a, b []uint64) []uint64 {
	n := min3(len(dst), len(a), len(b))
	out := append([]uint64(nil), dst...)
	for i := 0; i < n; i++ {
		out[i] = a[i] | b[i]
	}
	return out
}
func refAndNot(dst, a, b []uint64) []uint64 {
	n := min3(len(dst), len(a), len(b))
	out := append([]uint64(nil), dst...)
	for i := 0; i < n; i++ {
		out[i] = a[i] & ^b[i]
	}
	return out
}
func refXor(dst, a, b []uint64) []uint64 {
	n := min3(len(dst), len(a), len(b))
	out := append([]uint64(nil), dst...)
	for i := 0; i < n; i++ {
		out[i] = a[i] ^ b[i]
	}
	return out
}

func refCount(a []uint64) int {
	n := 0
	for _, w := range a {
		for w != 0 {
			n += int(w & 1)
			w >>= 1
		}
	}
	return n
}
func refPair(a, b []uint64, op func(x, y uint64) uint64) int {
	n := min2(len(a), len(b))
	total := 0
	for i := 0; i < n; i++ {
		total += refCount([]uint64{op(a[i], b[i])})
	}
	return total
}

func randWords(rng *rand.Rand, n int) []uint64 {
	w := make([]uint64, n)
	for i := range w {
		w[i] = rng.Uint64()
	}
	return w
}

func eq(t *testing.T, name string, got, want []uint64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len got %d want %d", name, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s: word %d got %#x want %#x", name, i, got[i], want[i])
		}
	}
}

func TestLogicalTable(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	ops := []struct {
		name string
		fn   func(dst, a, b []uint64)
		ref  func(dst, a, b []uint64) []uint64
	}{
		{"And", And, refAnd},
		{"Or", Or, refOr},
		{"AndNot", AndNot, refAndNot},
		{"Xor", Xor, refXor},
	}
	sizes := []int{0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 33, 64, 65, 100, 257, 1000}
	for _, op := range ops {
		for _, n := range sizes {
			a := randWords(rng, n)
			b := randWords(rng, n)
			dst := make([]uint64, n)
			want := op.ref(dst, a, b)
			op.fn(dst, a, b)
			eq(t, op.name+"-fresh", dst, want)

			// In-place dst==a.
			ina := append([]uint64(nil), a...)
			want = op.ref(ina, ina, b)
			op.fn(ina, ina, b)
			eq(t, op.name+"-inplace-a", ina, want)

			// In-place dst==b.
			inb := append([]uint64(nil), b...)
			want = op.ref(inb, a, inb)
			op.fn(inb, a, inb)
			eq(t, op.name+"-inplace-b", inb, want)
		}
	}
}

// TestLogicalMismatchedLengths exercises the min-length handling: ops must touch
// only the first min(len) words of dst and leave the rest untouched.
func TestLogicalMismatchedLengths(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	a := randWords(rng, 50)
	b := randWords(rng, 40)
	dst := randWords(rng, 60)
	want := refAnd(dst, a, b) // touches first 40
	And(dst, a, b)
	eq(t, "And-mismatch", dst, want)
}

func TestCountTable(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	sizes := []int{0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 33, 64, 65, 100, 257, 1000, 4096}
	for _, n := range sizes {
		a := randWords(rng, n)
		b := randWords(rng, n)
		if got, want := Count(a), refCount(a); got != want {
			t.Fatalf("Count n=%d: %d want %d", n, got, want)
		}
		if got, want := IntersectionCount(a, b), refPair(a, b, func(x, y uint64) uint64 { return x & y }); got != want {
			t.Fatalf("IntersectionCount n=%d: %d want %d", n, got, want)
		}
		if got, want := UnionCount(a, b), refPair(a, b, func(x, y uint64) uint64 { return x | y }); got != want {
			t.Fatalf("UnionCount n=%d: %d want %d", n, got, want)
		}
		if got, want := DifferenceCount(a, b), refPair(a, b, func(x, y uint64) uint64 { return x & ^y }); got != want {
			t.Fatalf("DifferenceCount n=%d: %d want %d", n, got, want)
		}
	}
}

func TestCountMismatchedLengths(t *testing.T) {
	rng := rand.New(rand.NewSource(10))
	a := randWords(rng, 70)
	b := randWords(rng, 33)
	if got, want := IntersectionCount(a, b), refPair(a, b, func(x, y uint64) uint64 { return x & y }); got != want {
		t.Fatalf("IntersectionCount mismatch: %d want %d", got, want)
	}
	if got, want := UnionCount(b, a), refPair(b, a, func(x, y uint64) uint64 { return x | y }); got != want {
		t.Fatalf("UnionCount mismatch: %d want %d", got, want)
	}
	if got, want := DifferenceCount(a, b), refPair(a, b, func(x, y uint64) uint64 { return x & ^y }); got != want {
		t.Fatalf("DifferenceCount mismatch: %d want %d", got, want)
	}
}

// TestSizes sweeps every length across the SIMD-block / tail boundary.
func TestSizes(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for n := 0; n <= 300; n++ {
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

		if got, want := Count(a), refCount(a); got != want {
			t.Fatalf("Count n=%d: %d want %d", n, got, want)
		}
		if got, want := IntersectionCount(a, b), refPair(a, b, func(x, y uint64) uint64 { return x & y }); got != want {
			t.Fatalf("IntersectionCount n=%d: %d want %d", n, got, want)
		}
		if got, want := UnionCount(a, b), refPair(a, b, func(x, y uint64) uint64 { return x | y }); got != want {
			t.Fatalf("UnionCount n=%d: %d want %d", n, got, want)
		}
		if got, want := DifferenceCount(a, b), refPair(a, b, func(x, y uint64) uint64 { return x & ^y }); got != want {
			t.Fatalf("DifferenceCount n=%d: %d want %d", n, got, want)
		}
	}
}

// TestScalarRefMatchesOracle checks the package scalar refs (the riscv64 path and
// the SIMD tail) against the independent oracle, so coverage of those functions
// is genuine even on a SIMD host.
func TestScalarRefMatchesOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	for n := 0; n <= 200; n++ {
		a := randWords(rng, n)
		b := randWords(rng, n)
		dst := make([]uint64, n)
		andScalarRef(dst, a, b)
		eq(t, "andScalarRef", dst, refAnd(make([]uint64, n), a, b))
		orScalarRef(dst, a, b)
		eq(t, "orScalarRef", dst, refOr(make([]uint64, n), a, b))
		andNotScalarRef(dst, a, b)
		eq(t, "andNotScalarRef", dst, refAndNot(make([]uint64, n), a, b))
		xorScalarRef(dst, a, b)
		eq(t, "xorScalarRef", dst, refXor(make([]uint64, n), a, b))
		if got := countScalarRef(a); got != refCount(a) {
			t.Fatalf("countScalarRef n=%d", n)
		}
		if intersectionCountScalarRef(a, b) != refPair(a, b, func(x, y uint64) uint64 { return x & y }) {
			t.Fatalf("intersectionCountScalarRef n=%d", n)
		}
		if unionCountScalarRef(a, b) != refPair(a, b, func(x, y uint64) uint64 { return x | y }) {
			t.Fatalf("unionCountScalarRef n=%d", n)
		}
		if differenceCountScalarRef(a, b) != refPair(a, b, func(x, y uint64) uint64 { return x & ^y }) {
			t.Fatalf("differenceCountScalarRef n=%d", n)
		}
	}
}

func FuzzAnd(f *testing.F) {
	fuzzLogic(f, And, refAnd)
}
func FuzzOr(f *testing.F) {
	fuzzLogic(f, Or, refOr)
}
func FuzzAndNot(f *testing.F) {
	fuzzLogic(f, AndNot, refAndNot)
}
func FuzzXor(f *testing.F) {
	fuzzLogic(f, Xor, refXor)
}

func fuzzLogic(f *testing.F, fn func(dst, a, b []uint64), ref func(dst, a, b []uint64) []uint64) {
	f.Add([]byte{1, 2, 3}, []byte{4, 5, 6})
	f.Add([]byte(nil), []byte(nil))
	f.Fuzz(func(t *testing.T, ab, bb []byte) {
		a := bytesToWords(ab)
		b := bytesToWords(bb)
		n := min2(len(a), len(b))
		dst := make([]uint64, n)
		want := ref(make([]uint64, n), a, b)
		fn(dst, a, b)
		eq(t, "fuzz", dst, want)
	})
}

func FuzzCount(f *testing.F) {
	f.Add([]byte{0xff, 0x00, 0x12})
	f.Fuzz(func(t *testing.T, ab []byte) {
		a := bytesToWords(ab)
		if got, want := Count(a), refCount(a); got != want {
			t.Fatalf("Count=%d want %d", got, want)
		}
	})
}

func FuzzCounts(f *testing.F) {
	f.Add([]byte{0xff, 0x00}, []byte{0x0f, 0xf0})
	f.Fuzz(func(t *testing.T, ab, bb []byte) {
		a := bytesToWords(ab)
		b := bytesToWords(bb)
		if got, want := IntersectionCount(a, b), refPair(a, b, func(x, y uint64) uint64 { return x & y }); got != want {
			t.Fatalf("Intersection=%d want %d", got, want)
		}
		if got, want := UnionCount(a, b), refPair(a, b, func(x, y uint64) uint64 { return x | y }); got != want {
			t.Fatalf("Union=%d want %d", got, want)
		}
		if got, want := DifferenceCount(a, b), refPair(a, b, func(x, y uint64) uint64 { return x & ^y }); got != want {
			t.Fatalf("Difference=%d want %d", got, want)
		}
	})
}

// bytesToWords packs a byte slice into uint64 words (little-endian), dropping a
// trailing partial word, giving the fuzzer arbitrary word slices.
func bytesToWords(b []byte) []uint64 {
	w := make([]uint64, len(b)/8)
	for i := range w {
		var v uint64
		for j := 0; j < 8; j++ {
			v |= uint64(b[i*8+j]) << (8 * j)
		}
		w[i] = v
	}
	return w
}
