package bitset

import (
	"math/bits"
	"math/rand"
	"testing"

	bb "github.com/bits-and-blooms/bitset"
)

// Benchmarks pit the exported ops against two honest baselines:
//   - a plain math/bits word loop (the "scalar" floor — already ~1 word/cycle,
//     so out of cache the problem is bandwidth-bound and SIMD merely ties it),
//   - github.com/bits-and-blooms/bitset, the de-facto Go bit set, whose bulk ops
//     are scalar math/bits loops over its backing []uint64,
// across set sizes spanning the cache hierarchy (in words; one word = 64 bits).
// SetBytes reports throughput in MB/s (÷1000 for GB/s). The sweep shows where
// SIMD wins (in-cache, compute-bound) and where it ties scalar (out-of-cache).

var benchSizes = []struct {
	name  string
	words int
}{
	{"1KiB", 1 << 7},   // 128 words = 1 KiB
	{"64KiB", 1 << 13}, // 8192 words = 64 KiB
	{"1MiB", 1 << 17},  // 131072 words = 1 MiB
	{"16MiB", 1 << 21}, // 16 MiB
}

func benchWords(n int) []uint64 {
	w := make([]uint64, n)
	rng := rand.New(rand.NewSource(1))
	for i := range w {
		w[i] = rng.Uint64()
	}
	return w
}

func scalarAnd(dst, a, b []uint64) {
	for i := range dst {
		dst[i] = a[i] & b[i]
	}
}

func scalarCount(a []uint64) int {
	t := 0
	for _, w := range a {
		t += bits.OnesCount64(w)
	}
	return t
}

var intSink int

func BenchmarkAnd(b *testing.B) {
	for _, s := range benchSizes {
		x := benchWords(s.words)
		y := benchWords(s.words)
		dst := make([]uint64, s.words)
		b.Run("simd/"+s.name, func(b *testing.B) {
			b.SetBytes(int64(s.words * 8))
			for i := 0; i < b.N; i++ {
				And(dst, x, y)
			}
		})
		b.Run("scalar/"+s.name, func(b *testing.B) {
			b.SetBytes(int64(s.words * 8))
			for i := 0; i < b.N; i++ {
				scalarAnd(dst, x, y)
			}
		})
	}
}

func BenchmarkCount(b *testing.B) {
	for _, s := range benchSizes {
		x := benchWords(s.words)
		set := bb.FromWithLength(uint(s.words*64), append([]uint64(nil), x...))
		b.Run("simd/"+s.name, func(b *testing.B) {
			b.SetBytes(int64(s.words * 8))
			for i := 0; i < b.N; i++ {
				intSink = Count(x)
			}
		})
		b.Run("scalar/"+s.name, func(b *testing.B) {
			b.SetBytes(int64(s.words * 8))
			for i := 0; i < b.N; i++ {
				intSink = scalarCount(x)
			}
		})
		b.Run("bitsandblooms/"+s.name, func(b *testing.B) {
			b.SetBytes(int64(s.words * 8))
			for i := 0; i < b.N; i++ {
				intSink = int(set.Count())
			}
		})
	}
}

func BenchmarkIntersectionCount(b *testing.B) {
	for _, s := range benchSizes {
		x := benchWords(s.words)
		y := benchWords(s.words)
		sx := bb.FromWithLength(uint(s.words*64), append([]uint64(nil), x...))
		sy := bb.FromWithLength(uint(s.words*64), append([]uint64(nil), y...))
		b.Run("simd/"+s.name, func(b *testing.B) {
			b.SetBytes(int64(s.words * 8))
			for i := 0; i < b.N; i++ {
				intSink = IntersectionCount(x, y)
			}
		})
		b.Run("bitsandblooms/"+s.name, func(b *testing.B) {
			b.SetBytes(int64(s.words * 8))
			for i := 0; i < b.N; i++ {
				intSink = int(sx.IntersectionCardinality(sy))
			}
		})
	}
}
