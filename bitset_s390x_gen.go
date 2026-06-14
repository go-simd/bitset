//go:build ignore

// Command gen produces bitset_s390x.s with go-asmgen: the vector-facility
// kernels for the bit-set word operations over []uint64.
//
// Logical ops (and/or/andNot/xor): a vector loop over whole 2-uint64 (16-byte)
// blocks. a[i],b[i] are loaded with VL into V0,V1 and one vector boolean
// instruction is applied:
//
//   - And -> VN, Or -> VO, Xor -> VX (3-operand, dst last: "VN Vb, Va, Vd"
//     means Vd = Va & Vb).
//   - AndNot wants a &^ b = a & ^b. The and-with-complement VNC Va, Vb, Vd =
//     Va & ^Vb (operand order verified empirically under qemu), so with V0=a,
//     V1=b: VNC V0, V1, V2 = a & ^b.
//
// Counts (count/intersection/union/difference): VPOPCT per-byte popcount, then
// VSUMB (16 bytes -> 4 uint32 lanes) and VSUMQF (4 uint32 -> 1 uint128) for the
// horizontal sum; the low doubleword (block sum <= 128) is taken with VLGVG $1.
// count VPOPCTs the loaded words; the pairwise kernels combine a[i],b[i] first.
//
// Big-endian: s390x is big-endian, but uint64 word ops and the byte/word-wise
// popcount reductions are endian-neutral — VN/VO/VX/VNC are bitwise per element,
// and VSUMB/VSUMQF are commutative horizontal sums over all lanes, so the result
// does not depend on lane numbering. VLGVG $1 takes the rightmost doubleword of
// the quadword sum, the convention the standard library's count_s390x.s uses.
// The vector facility (z13) is the s390x baseline, so no runtime feature flag.
//
// Each kernel computes its operating length in-kernel (min of the slice lens)
// and returns done as a word count (a multiple of 2); the Go wrapper finishes
// the remainder with the scalar word loop.
//
// Run: go run bitset_s390x_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/s390x"
)

func logicSig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("a"), abi.Slice("b")},
		[]abi.Arg{abi.Scalar("done", abi.Int64)},
	)
}
func countSig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("a")},
		[]abi.Arg{abi.Scalar("sum", abi.Int64), abi.Scalar("done", abi.Int64)},
	)
}
func pairSig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("a"), abi.Slice("b")},
		[]abi.Arg{abi.Scalar("sum", abi.Int64), abi.Scalar("done", abi.Int64)},
	)
}

var labelSeq int

// minInto emits Rdst = min(Rx, Ry).
func minInto(b *s390x.Builder, rx, ry, rdst string) {
	labelSeq++
	useY := fmt.Sprintf("useY%d", labelSeq)
	dn := fmt.Sprintf("doneMin%d", labelSeq)
	b.Raw("CMPBLT %s, %s, %s", ry, rx, useY). // ry < rx -> useY
							Raw("MOVD %s, %s", rx, rdst).
							Raw("BR %s", dn).
							Label(useY).
							Raw("MOVD %s, %s", ry, rdst).
							Label(dn)
}

// emitCombine combines V0=a and V1=b into V2. vinsn for the commutative ops as
// "VINSN V1, V0, V2" = V2 = V0 OP V1; for AndNot it emits VNC V1, V0, V2 = a&^b.
func emitCombine(b *s390x.Builder, vinsn string, andnot bool) {
	if andnot {
		// VNC operand order verified empirically under qemu: "VNC Va, Vb, Vd"
		// = Vd = Va & ^Vb. So with V0=a, V1=b: VNC V0, V1, V2 = a & ^b.
		b.Raw("VNC V0, V1, V2") // V2 = a & ^b
		return
	}
	b.Raw("%s V1, V0, V2", vinsn) // V2 = a OP b
}

func genLogic(f *emit.File, name, vinsn string, andnot bool) {
	b := s390x.NewFunc(name, logicSig(), 0)
	b.LoadArg("dst_base", "R1").
		LoadArg("dst_len", "R2").
		LoadArg("a_base", "R3").
		LoadArg("a_len", "R4").
		LoadArg("b_base", "R5").
		LoadArg("b_len", "R6")
	minInto(b, "R2", "R4", "R2")
	minInto(b, "R2", "R6", "R2")
	b.Raw("SRD $1, R2, R7"). // blocks = n >> 1
					Raw("MOVD $0, R8"). // word index
					Label("loop").
					Raw("CMPBEQ R7, $0, done").
					Raw("SLD $3, R8, R9").    // byte offset
					Raw("ADD R3, R9, R10").   // &a[i]
					Raw("ADD R5, R9, R11").   // &b[i]
					Raw("ADD R1, R9, R12").   // &dst[i]
					Raw("VL (R10), V0").      // a
					Raw("VL (R11), V1")       // b
	emitCombine(b, vinsn, andnot) // -> V2
	b.Raw("VST V2, (R12)").
		Raw("ADD $2, R8, R8").
		Raw("ADD $-1, R7, R7").
		Raw("BR loop").
		Label("done").
		StoreRet("R8", "done").
		Ret()
	f.Add(b.Func())
}

func genCount(f *emit.File) {
	b := s390x.NewFunc("countKernel", countSig(), 0)
	b.LoadArg("a_base", "R3").
		LoadArg("a_len", "R4").
		Raw("MOVD $0, R5").    // sum
		Raw("VZERO V5").       // zero vector for the horizontal sums
		Raw("SRD $1, R4, R7"). // blocks
		Raw("MOVD $0, R8").    // word index
		Label("loop").
		Raw("CMPBEQ R7, $0, done").
		Raw("SLD $3, R8, R9").
		Raw("ADD R3, R9, R10").
		Raw("VL (R10), V0").
		Raw("VPOPCT V0, V1").     // per-byte popcount
		Raw("VSUMB V1, V5, V1").  // 16 bytes -> 4 uint32
		Raw("VSUMQF V1, V5, V1"). // 4 uint32 -> 1 uint128
		Raw("VLGVG $1, V1, R11"). // low doubleword block sum
		Raw("ADD R11, R5, R5").
		Raw("ADD $2, R8, R8").
		Raw("ADD $-1, R7, R7").
		Raw("BR loop").
		Label("done").
		StoreRet("R5", "sum").
		StoreRet("R8", "done").
		Ret()
	f.Add(b.Func())
}

func genPair(f *emit.File, name, vinsn string, andnot bool) {
	b := s390x.NewFunc(name, pairSig(), 0)
	b.LoadArg("a_base", "R3").
		LoadArg("a_len", "R4").
		LoadArg("b_base", "R5").
		LoadArg("b_len", "R6")
	minInto(b, "R4", "R6", "R4")
	b.Raw("MOVD $0, R2").    // sum
					Raw("VZERO V5").       // zero vector for horizontal sums
					Raw("SRD $1, R4, R7"). // blocks
					Raw("MOVD $0, R8").    // word index
					Label("loop").
					Raw("CMPBEQ R7, $0, done").
					Raw("SLD $3, R8, R9").
					Raw("ADD R3, R9, R10").
					Raw("ADD R5, R9, R11").
					Raw("VL (R10), V0"). // a
					Raw("VL (R11), V1")  // b
	emitCombine(b, vinsn, andnot) // -> V2
	b.Raw("VPOPCT V2, V2").
		Raw("VSUMB V2, V5, V2").
		Raw("VSUMQF V2, V5, V2").
		Raw("VLGVG $1, V2, R12").
		Raw("ADD R12, R2, R2").
		Raw("ADD $2, R8, R8").
		Raw("ADD $-1, R7, R7").
		Raw("BR loop").
		Label("done").
		StoreRet("R2", "sum").
		StoreRet("R8", "done").
		Ret()
	f.Add(b.Func())
}

func main() {
	f := emit.NewFile("s390x")

	genLogic(f, "andKernel", "VN", false)
	genLogic(f, "orKernel", "VO", false)
	genLogic(f, "xorKernel", "VX", false)
	genLogic(f, "andNotKernel", "", true)

	genCount(f)
	genPair(f, "intersectionKernel", "VN", false)
	genPair(f, "unionKernel", "VO", false)
	genPair(f, "differenceKernel", "", true)

	if err := os.WriteFile("bitset_s390x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote bitset_s390x.s")
}
