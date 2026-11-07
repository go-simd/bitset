package bitset

import (
	"math/rand/v2"
	"testing"
)

func refFromBools(dst []uint64, src []bool) int {
	n := min(len(src), 64*len(dst))
	for w := 0; 64*w < n; w++ {
		dst[w] = 0
	}
	for i, v := range src[:n] {
		if v {
			dst[i>>6] |= 1 << (uint(i) & 63)
		}
	}
	return n
}

func refToBools(dst []bool, src []uint64) int {
	n := min(len(dst), 64*len(src))
	for i := range n {
		dst[i] = src[i>>6]>>(uint(i)&63)&1 == 1
	}
	return n
}

func TestBoolsMatchReference(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, n := range []int{0, 1, 7, 8, 63, 64, 65, 127, 128, 1000, 4099} {
		for _, words := range []int{0, (n + 63) / 64, (n + 63) / 64 / 2, (n+63)/64 + 3} {
			src := make([]bool, n)
			for i := range src {
				src[i] = r.IntN(3) == 0
			}
			got, want := make([]uint64, words), make([]uint64, words)
			for i := range got {
				got[i], want[i] = ^uint64(0), ^uint64(0)
			}
			if a, b := FromBools(got, src), refFromBools(want, src); a != b {
				t.Fatalf("n=%d words=%d: FromBools returned %d, want %d", n, words, a, b)
			}
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("n=%d words=%d: word %d = %x, want %x", n, words, i, got[i], want[i])
				}
			}
			bools, ref := make([]bool, n), make([]bool, n)
			if a, b := ToBools(bools, got), refToBools(ref, got); a != b {
				t.Fatalf("n=%d words=%d: ToBools returned %d, want %d", n, words, a, b)
			}
			for i := range bools {
				if bools[i] != ref[i] {
					t.Fatalf("n=%d words=%d: bool %d", n, words, i)
				}
			}
		}
	}
}

// TestPackWordsPortable runs the portable multiply kernel by itself: on
// architectures with a SIMD kernel FromBools never reaches it.
func TestPackWordsPortable(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	src := make([]bool, 5*64+9)
	for i := range src {
		src[i] = r.IntN(2) == 0
	}
	got, want := make([]uint64, 6), make([]uint64, 6)
	if w := packWordsPortable(got, boolBytes(src), 0); w != 5 {
		t.Fatalf("returned %d, want 5", w)
	}
	refFromBools(want, src[:5*64])
	for i := range 5 {
		if got[i] != want[i] {
			t.Fatalf("word %d = %x, want %x", i, got[i], want[i])
		}
	}
	if w := packWordsPortable(got, boolBytes(src), 4); w != 5 {
		t.Fatalf("resumed: returned %d, want 5", w)
	}
}

func FuzzBools(f *testing.F) {
	f.Add([]byte{0, 1, 1, 0, 1}, uint64(0xdeadbeef))
	f.Add(make([]byte, 200), uint64(0))
	f.Add(make([]byte, 200), uint64(1))
	f.Add(make([]byte, 200), uint64(2))
	f.Fuzz(func(t *testing.T, in []byte, seed uint64) {
		src := make([]bool, len(in))
		for i, b := range in {
			src[i] = b&1 == 1
		}
		need := (len(src) + 63) / 64
		// seed picks len(dst): below, at, or above the needed word count,
		// so the truncation path is fuzzed too.
		words := need
		switch seed % 3 {
		case 0:
			if need > 0 {
				words = int(seed>>2) % need
			}
		case 2:
			words = need + int(seed>>2)%4
		}
		got, want := make([]uint64, words), make([]uint64, words)
		for i := range got {
			got[i], want[i] = ^uint64(0), ^uint64(0)
		}
		if a, b := FromBools(got, src), refFromBools(want, src); a != b {
			t.Fatalf("FromBools returned %d, want %d", a, b)
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("word %d = %x, want %x", i, got[i], want[i])
			}
		}
		back, ref := make([]bool, len(src)), make([]bool, len(src))
		if a, b := ToBools(back, got), refToBools(ref, got); a != b {
			t.Fatalf("ToBools returned %d, want %d", a, b)
		}
		for i := range back {
			if back[i] != ref[i] {
				t.Fatalf("bool %d differs from the reference", i)
			}
		}
	})
}

var boolSizes = []struct {
	name string
	n    int
}{{"2^10", 1 << 10}, {"2^16", 1 << 16}, {"2^20", 1 << 20}}

func benchBools(b *testing.B, f func(n int) func()) {
	for _, s := range boolSizes {
		b.Run(s.name, func(b *testing.B) {
			run := f(s.n)
			b.SetBytes(int64(s.n)) // one byte per bool, as NumPy counts them
			for b.Loop() {
				run()
			}
		})
	}
}

func fromBoolsCase(f func([]uint64, []bool) int) func(n int) func() {
	return func(n int) func() {
		src := make([]bool, n)
		for i := range src {
			src[i] = i%3 == 0
		}
		dst := make([]uint64, n/64)
		return func() { f(dst, src) }
	}
}

func toBoolsCase(f func([]bool, []uint64) int) func(n int) func() {
	return func(n int) func() {
		src := make([]uint64, n/64)
		for i := range src {
			src[i] = uint64(i) * 0x9E3779B97F4A7C15
		}
		dst := make([]bool, n)
		return func() { f(dst, src) }
	}
}

func BenchmarkFromBools(b *testing.B)       { benchBools(b, fromBoolsCase(FromBools)) }
func BenchmarkFromBoolsScalar(b *testing.B) { benchBools(b, fromBoolsCase(refFromBools)) }
func BenchmarkToBools(b *testing.B)         { benchBools(b, toBoolsCase(ToBools)) }
func BenchmarkToBoolsScalar(b *testing.B)   { benchBools(b, toBoolsCase(refToBools)) }
