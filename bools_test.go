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

func FuzzBools(f *testing.F) {
	f.Add([]byte{0, 1, 1, 0, 1}, uint64(0xdeadbeef))
	f.Fuzz(func(t *testing.T, in []byte, seed uint64) {
		src := make([]bool, len(in))
		for i, b := range in {
			src[i] = b&1 == 1
		}
		words := (len(src) + 63) / 64
		got, want := make([]uint64, words), make([]uint64, words)
		FromBools(got, src)
		refFromBools(want, src)
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("word %d = %x, want %x", i, got[i], want[i])
			}
		}
		back := make([]bool, len(src))
		ToBools(back, got)
		for i := range src {
			if back[i] != src[i] {
				t.Fatalf("bool %d does not round-trip", i)
			}
		}
		_ = seed
	})
}

func BenchmarkFromBools(b *testing.B) {
	src := make([]bool, 1<<20)
	for i := range src {
		src[i] = i%3 == 0
	}
	dst := make([]uint64, len(src)/64)
	b.SetBytes(int64(len(src)))
	for b.Loop() {
		FromBools(dst, src)
	}
}

func BenchmarkFromBoolsScalar(b *testing.B) {
	src := make([]bool, 1<<20)
	dst := make([]uint64, len(src)/64)
	b.SetBytes(int64(len(src)))
	for b.Loop() {
		refFromBools(dst, src)
	}
}

func BenchmarkToBools(b *testing.B) {
	src := make([]uint64, 1<<14)
	for i := range src {
		src[i] = uint64(i) * 0x9E3779B97F4A7C15
	}
	dst := make([]bool, 64*len(src))
	b.SetBytes(int64(len(dst)))
	for b.Loop() {
		ToBools(dst, src)
	}
}

func BenchmarkToBoolsScalar(b *testing.B) {
	src := make([]uint64, 1<<14)
	dst := make([]bool, 64*len(src))
	b.SetBytes(int64(len(dst)))
	for b.Loop() {
		refToBools(dst, src)
	}
}
