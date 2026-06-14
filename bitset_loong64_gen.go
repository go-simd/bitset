//go:build ignore

// Command gen produces bitset_loong64.s with go-asmgen: the LSX kernels for the
// bit-set word operations over []uint64.
//
// Logical ops (and/or/andNot/xor): an LSX loop over whole 2-uint64 (16-byte)
// blocks. Each iteration loads a[i],b[i] into 128-bit V-registers and applies
// one vector boolean instruction:
//
//   - And -> VANDV, Or -> VORV, Xor -> VXORV (commutative).
//   - AndNot wants a &^ b = a & ^b = ^b & a. LoongArch vandn.v vd, vj, vk =
//     vd = ^vj & vk, so with vj=b, vk=a we get ^b & a directly. In Plan 9
//     operand order (VANDNV vk, vj, vd) that is "VANDNV Va, Vb, Vd".
//
// Counts (count/intersection/union/difference): VPCNTV — a per-64-bit-element
// population count of the 128-bit vector. count VPCNTVs the loaded words
// directly; the pairwise kernels first combine a[i],b[i] with the matching
// vector boolean op, then VPCNTV. The two per-lane counts are extracted to GPRs
// (VMOVQ V.V[0]/V.V[1]) and summed.
//
// Each kernel computes its operating length in-kernel (min of the slice lens)
// and returns done as a word count (a multiple of 2); the Go wrapper finishes
// the remainder with the scalar word loop.
//
// Run: go run bitset_loong64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/loong64"
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

// minInto emits Rdst = min(Rx, Ry): if Ry < Rx, Rdst = Ry else Rx. Uses fresh
// label names each call so two calls in one function do not collide.
func minInto(b *loong64.Builder, rx, ry, rdst string) {
	labelSeq++
	useY := fmt.Sprintf("useY%d", labelSeq)
	doneMin := fmt.Sprintf("doneMin%d", labelSeq)
	b.Raw("BLT %s, %s, %s", ry, rx, useY). // Ry < Rx -> useY
						Raw("MOVV %s, %s", rx, rdst).
						Raw("JMP %s", doneMin).
						Label(useY).
						Raw("MOVV %s, %s", ry, rdst).
						Label(doneMin)
}

// emitCombine combines V0=a and V1=b into V0. vinsn for the commutative ops; for
// AndNot it emits VANDNV V0, V1, V0 = ^b & a = a &^ b.
func emitCombine(b *loong64.Builder, vinsn string, andnot bool) {
	if andnot {
		b.Raw("VANDNV V0, V1, V0") // vandn.v V0 = ^V1 & V0 = ^b & a
		return
	}
	b.Raw("%s V1, V0, V0", vinsn) // V0 = a OP b
}

func genLogic(f *emit.File, name, vinsn string, andnot bool) {
	b := loong64.NewFunc(name, logicSig(), 0)
	b.LoadArg("dst_base", "R4").
		LoadArg("dst_len", "R5").
		LoadArg("a_base", "R6").
		LoadArg("a_len", "R7").
		LoadArg("b_base", "R8").
		LoadArg("b_len", "R9")
	minInto(b, "R5", "R7", "R5") // n = min(dst,a)
	minInto(b, "R5", "R9", "R5") // n = min(n,b)
	b.Raw("SRLV $1, R5, R10").   // blocks = n >> 1
					Raw("MOVV $0, R11"). // word index
					Label("loop").
					Raw("BEQ R10, R0, done").
					Raw("SLLV $3, R11, R12"). // byte offset
					Raw("ADDV R6, R12, R13"). // &a[i]
					Raw("ADDV R8, R12, R14"). // &b[i]
					Raw("ADDV R4, R12, R15"). // &dst[i]
					Raw("VMOVQ (R13), V0").   // a
					Raw("VMOVQ (R14), V1")    // b
	emitCombine(b, vinsn, andnot)
	b.Raw("VMOVQ V0, (R15)").
		Raw("ADDV $2, R11, R11").
		Raw("ADDV $-1, R10, R10").
		Raw("JMP loop").
		Label("done").
		StoreRet("R11", "done").
		Ret()
	f.Add(b.Func())
}

func genCount(f *emit.File) {
	b := loong64.NewFunc("countKernel", countSig(), 0)
	b.LoadArg("a_base", "R4").
		LoadArg("a_len", "R5").
		Raw("MOVV $0, R6").       // sum
		Raw("SRLV $1, R5, R10").  // blocks
		Raw("MOVV $0, R11").      // word index
		Label("loop").
		Raw("BEQ R10, R0, done").
		Raw("SLLV $3, R11, R12").
		Raw("ADDV R4, R12, R13").
		Raw("VMOVQ (R13), V0").
		Raw("VPCNTV V0, V0").
		Raw("VMOVQ V0.V[0], R14").
		Raw("VMOVQ V0.V[1], R15").
		Raw("ADDV R14, R6, R6").
		Raw("ADDV R15, R6, R6").
		Raw("ADDV $2, R11, R11").
		Raw("ADDV $-1, R10, R10").
		Raw("JMP loop").
		Label("done").
		StoreRet("R6", "sum").
		StoreRet("R11", "done").
		Ret()
	f.Add(b.Func())
}

func genPair(f *emit.File, name, vinsn string, andnot bool) {
	b := loong64.NewFunc(name, pairSig(), 0)
	b.LoadArg("a_base", "R4").
		LoadArg("a_len", "R5").
		LoadArg("b_base", "R8").
		LoadArg("b_len", "R9")
	minInto(b, "R5", "R9", "R5") // n = min(a,b)
	b.Raw("MOVV $0, R6").        // sum
					Raw("SRLV $1, R5, R10").
					Raw("MOVV $0, R11").
					Label("loop").
					Raw("BEQ R10, R0, done").
					Raw("SLLV $3, R11, R12").
					Raw("ADDV R4, R12, R13").
					Raw("ADDV R8, R12, R14").
					Raw("VMOVQ (R13), V0").
					Raw("VMOVQ (R14), V1")
	emitCombine(b, vinsn, andnot)
	b.Raw("VPCNTV V0, V0").
		Raw("VMOVQ V0.V[0], R15").
		Raw("VMOVQ V0.V[1], R16").
		Raw("ADDV R15, R6, R6").
		Raw("ADDV R16, R6, R6").
		Raw("ADDV $2, R11, R11").
		Raw("ADDV $-1, R10, R10").
		Raw("JMP loop").
		Label("done").
		StoreRet("R6", "sum").
		StoreRet("R11", "done").
		Ret()
	f.Add(b.Func())
}

func main() {
	f := emit.NewFile("loong64")

	genLogic(f, "andKernel", "VANDV", false)
	genLogic(f, "orKernel", "VORV", false)
	genLogic(f, "xorKernel", "VXORV", false)
	genLogic(f, "andNotKernel", "", true)

	genCount(f)
	genPair(f, "intersectionKernel", "VANDV", false)
	genPair(f, "unionKernel", "VORV", false)
	genPair(f, "differenceKernel", "", true)

	if err := os.WriteFile("bitset_loong64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote bitset_loong64.s")
}
