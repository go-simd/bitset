package bitset

import (
	"encoding/binary"
	"unsafe"
)

// FromBools packs bools into the bit set: bit i of dst is set when src[i]
// is true, for i in [0, n) where n = min(len(src), 64*len(dst)); the
// remaining bits of the last word written are cleared, and words of dst
// beyond it are left unchanged. It returns n.
//
// Eight bools at a time: a Go bool is one byte, 0 or 1, so eight of them
// read as one little-endian uint64 hold bit i in byte i, and one multiply
// gathers those eight bits into the top byte. This is the portable kernel;
// it already moves eight elements per few instructions, and the conversion
// is bound by the bytes it reads.
func FromBools(dst []uint64, src []bool) int {
	n := min(len(src), 64*len(dst))
	b := boolBytes(src[:n])
	w := 0
	for ; 64*(w+1) <= n; w++ {
		var word uint64
		for k := range 8 {
			x := binary.LittleEndian.Uint64(b[64*w+8*k:])
			word |= (x * 0x0102040810204080 >> 56) << (8 * k)
		}
		dst[w] = word
	}
	if rest := n - 64*w; rest > 0 {
		var word uint64
		for i, v := range src[64*w : n] {
			if v {
				word |= 1 << uint(i)
			}
		}
		dst[w] = word
	}
	return n
}

// ToBools expands the bit set into bools: dst[i] is bit i of src, for i in
// [0, n) where n = min(len(dst), 64*len(src)). It returns n.
//
// A byte at a time, through a table of the eight bools each byte value
// stands for, stored as one 8-byte word.
func ToBools(dst []bool, src []uint64) int {
	n := min(len(dst), 64*len(src))
	b := boolBytes(dst[:n])
	i := 0
	for ; i+8 <= n; i += 8 {
		binary.LittleEndian.PutUint64(b[i:], spread[byte(src[i>>6]>>(uint(i)&63))])
	}
	for ; i < n; i++ {
		dst[i] = src[i>>6]>>(uint(i)&63)&1 == 1
	}
	return n
}

// spread[v] holds, byte k, bit k of v: the eight bools v stands for.
var spread = func() (t [256]uint64) {
	for v := range t {
		for k := range 8 {
			t[v] |= uint64(v>>k&1) << (8 * k)
		}
	}
	return t
}()

// boolBytes views bools as their bytes, without copying.
func boolBytes(b []bool) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(b))), len(b))
}
