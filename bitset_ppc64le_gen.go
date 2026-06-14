//go:build ignore

// Command gen produces bitset_ppc64le.s with go-asmgen: the VSX kernels for the
// bit-set word operations over []uint64.
//
// Logical ops (and/or/andNot/xor): a VSX loop over whole 2-uint64 (16-byte)
// blocks. a[i] and b[i] are loaded with LXVD2X into VS32 (=V0) and VS33 (=V1)
// — the VSX/VMX register-aliasing rule from go-simd/popcount: AltiVec Vn aliases
// VSX VS(32+n), so a load that the VMX boolean ops must see as V0/V1 has to
// target VS32/VS33. The boolean op writes V2 and STXVD2X stores VS34 (=V2):
//
//   - And -> VAND, Or -> VOR, Xor -> VXOR.
//   - AndNot wants a &^ b = a & ^b. PowerISA vandc vrt, vra, vrb = vra & ^vrb,
//     so VANDC V0, V1, V2 = a & ^b directly.
//
// LXVD2X uint64-element loads are byte-shuffle-free, so the ppc64le little-endian
// element order is correct without a permute (the popcount sibling relies on the
// same property).
//
// Counts (count/intersection/union/difference): VPOPCNTD — a per-64-bit-
// doubleword population count. count VPOPCNTDs the loaded words; the pairwise
// kernels first combine a[i],b[i] with the matching boolean op, then VPOPCNTD.
// The two doubleword counts are extracted to GPRs with MFVSRD (upper) and
// MFVSRLD (lower) and summed.
//
// VSX (POWER8, ISA 2.07) is the ppc64le baseline — VAND/VANDC/VPOPCNTD and the
// GPR-from-VSR moves are all POWER8 — so no runtime feature flag is needed.
//
// Each kernel computes its operating length in-kernel (min of the slice lens)
// and returns done as a word count (a multiple of 2); the Go wrapper finishes
// the remainder with the scalar word loop.
//
// Run: go run bitset_ppc64le_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/ppc64"
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
func minInto(b *ppc64.Builder, rx, ry, rdst string) {
	labelSeq++
	useY := fmt.Sprintf("useY%d", labelSeq)
	dn := fmt.Sprintf("doneMin%d", labelSeq)
	b.Raw("CMP %s, %s", ry, rx). // ry ? rx
					Raw("BLT %s", useY). // ry < rx -> useY
					Raw("MOVD %s, %s", rx, rdst).
					Raw("BR %s", dn).
					Label(useY).
					Raw("MOVD %s, %s", ry, rdst).
					Label(dn)
}

// emitCombine combines V0=a and V1=b into V2. vinsn for the commutative ops as
// "VINSN V0, V1, V2"; for AndNot it emits VANDC V0, V1, V2 = a & ^b.
func emitCombine(b *ppc64.Builder, vinsn string, andnot bool) {
	if andnot {
		b.Raw("VANDC V0, V1, V2") // a & ^b
		return
	}
	b.Raw("%s V0, V1, V2", vinsn) // a OP b
}

func genLogic(f *emit.File, name, vinsn string, andnot bool) {
	b := ppc64.NewFunc(name, logicSig(), 0)
	b.LoadArg("dst_base", "R3").
		LoadArg("dst_len", "R4").
		LoadArg("a_base", "R5").
		LoadArg("a_len", "R6").
		LoadArg("b_base", "R7").
		LoadArg("b_len", "R8")
	minInto(b, "R4", "R6", "R4")
	minInto(b, "R4", "R8", "R4")
	b.Raw("SRD $1, R4, R9"). // blocks = n >> 1
					Raw("MOVD $0, R10"). // word index
					Label("loop").
					Raw("CMP R9, $0").
					Raw("BEQ done").
					Raw("SLD $3, R10, R11"). // byte offset
					Raw("ADD R5, R11, R12"). // &a[i]
					Raw("ADD R7, R11, R14"). // &b[i]
					Raw("ADD R3, R11, R15"). // &dst[i]
					Raw("LXVD2X (R12)(R0), VS32"). // V0 = a
					Raw("LXVD2X (R14)(R0), VS33")  // V1 = b
	emitCombine(b, vinsn, andnot)
	b.Raw("STXVD2X VS34, (R15)(R0)"). // store V2
						Raw("ADD $2, R10, R10").
						Raw("ADD $-1, R9, R9").
						Raw("BR loop").
						Label("done").
						StoreRet("R10", "done").
						Ret()
	f.Add(b.Func())
}

func genCount(f *emit.File) {
	b := ppc64.NewFunc("countKernel", countSig(), 0)
	b.LoadArg("a_base", "R5").
		LoadArg("a_len", "R6").
		Raw("MOVD $0, R7").     // sum
		Raw("SRD $1, R6, R9").  // blocks
		Raw("MOVD $0, R10").    // word index
		Label("loop").
		Raw("CMP R9, $0").
		Raw("BEQ done").
		Raw("SLD $3, R10, R11").
		Raw("ADD R5, R11, R12").
		Raw("LXVD2X (R12)(R0), VS32"). // V0
		Raw("VPOPCNTD V0, V0").
		Raw("MFVSRD VS32, R13").   // upper doubleword count
		Raw("MFVSRLD VS32, R14").  // lower doubleword count
		Raw("ADD R13, R7, R7").
		Raw("ADD R14, R7, R7").
		Raw("ADD $2, R10, R10").
		Raw("ADD $-1, R9, R9").
		Raw("BR loop").
		Label("done").
		StoreRet("R7", "sum").
		StoreRet("R10", "done").
		Ret()
	f.Add(b.Func())
}

func genPair(f *emit.File, name, vinsn string, andnot bool) {
	b := ppc64.NewFunc(name, pairSig(), 0)
	b.LoadArg("a_base", "R5").
		LoadArg("a_len", "R6").
		LoadArg("b_base", "R7").
		LoadArg("b_len", "R8")
	minInto(b, "R6", "R8", "R6")
	b.Raw("MOVD $0, R4").     // sum
					Raw("SRD $1, R6, R9").
					Raw("MOVD $0, R10").
					Label("loop").
					Raw("CMP R9, $0").
					Raw("BEQ done").
					Raw("SLD $3, R10, R11").
					Raw("ADD R5, R11, R12").
					Raw("ADD R7, R11, R14").
					Raw("LXVD2X (R12)(R0), VS32"). // V0 = a
					Raw("LXVD2X (R14)(R0), VS33")  // V1 = b
	emitCombine(b, vinsn, andnot) // -> V2
	b.Raw("VPOPCNTD V2, V2").
		Raw("MFVSRD VS34, R15").
		Raw("MFVSRLD VS34, R16").
		Raw("ADD R15, R4, R4").
		Raw("ADD R16, R4, R4").
		Raw("ADD $2, R10, R10").
		Raw("ADD $-1, R9, R9").
		Raw("BR loop").
		Label("done").
		StoreRet("R4", "sum").
		StoreRet("R10", "done").
		Ret()
	f.Add(b.Func())
}

func main() {
	f := emit.NewFile("ppc64le")

	genLogic(f, "andKernel", "VAND", false)
	genLogic(f, "orKernel", "VOR", false)
	genLogic(f, "xorKernel", "VXOR", false)
	genLogic(f, "andNotKernel", "", true)

	genCount(f)
	genPair(f, "intersectionKernel", "VAND", false)
	genPair(f, "unionKernel", "VOR", false)
	genPair(f, "differenceKernel", "", true)

	if err := os.WriteFile("bitset_ppc64le.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote bitset_ppc64le.s")
}
