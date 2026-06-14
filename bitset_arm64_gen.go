//go:build ignore

// Command gen produces bitset_arm64.s with go-asmgen: the arm64 NEON kernels for
// the bit-set word operations over []uint64.
//
// Logical ops (and/or/andNot/xor): a NEON loop over whole 2-uint64 (16-byte)
// blocks. Each iteration loads a[i],b[i] into Q-registers and applies one vector
// boolean instruction:
//
//   - And -> VAND, Or -> VORR, Xor -> VEOR (commutative).
//   - AndNot wants a &^ b = a & ^b. The Go arm64 assembler does not accept VBIC
//     (nor VMVN/VNOT) for the .B16 vector form, so AndNot uses the identity
//     a &^ b = a ^ (a & b): VAND a,b -> t then VEOR a,t -> dst. Two ops, no
//     bitwise-NOT instruction required.
//
// Counts (count/intersection/union/difference): the NEON popcount kernel — per
// 16-byte block VCNT (per-byte popcount) then VUADDLV (horizontal widen-sum to a
// scalar). count VCNTs the loaded words directly; the pairwise kernels first
// combine a[i],b[i] with the matching vector boolean op, then VCNT/VUADDLV the
// result. Each block sum is < 128*8, so it fits the H/D accumulator lane.
//
// Every kernel computes its operating length in-kernel (min of the slice lens)
// and returns done as a word count (a multiple of 2); the Go wrapper finishes
// the remainder with the scalar word loop.
//
// Run: go run bitset_arm64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
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

// minInto emits R_dst = min(R_x, R_y) using CMP + CSEL.
func minInto(b *arm64.Builder, rx, ry, rdst string) {
	b.Raw("CMP %s, %s", ry, rx). // compare rx ? ry
					Raw("CSEL LT, %s, %s, %s", rx, ry, rdst) // rdst = rx<ry ? rx : ry
}

// emitCombine writes the combined value of V0=a and V1=b into V0. For the
// commutative ops vinsn is the single NEON mnemonic. For AndNot (andnot=true,
// vinsn ignored) it emits a ^ (a & b) using scratch V2.
func emitCombine(b *arm64.Builder, vinsn string, andnot bool) {
	if andnot {
		b.Raw("VAND V1.B16, V0.B16, V2.B16"). // V2 = a & b
							Raw("VEOR V2.B16, V0.B16, V0.B16") // V0 = a ^ (a&b) = a &^ b
		return
	}
	b.Raw("%s V1.B16, V0.B16, V0.B16", vinsn) // V0 = a OP b
}

// genLogic emits name(dst, a, b []uint64) (done int) over 16-byte (2-word)
// blocks. vinsn applied as "VINSN <m>, <n>, <d>"; for VBIC (AndNot) n=a, m=b.
func genLogic(f *emit.File, name, vinsn string, andnot bool) {
	b := arm64.NewFunc(name, logicSig(), 0)
	b.LoadArg("dst_base", "R0").
		LoadArg("dst_len", "R1").
		LoadArg("a_base", "R2").
		LoadArg("a_len", "R3").
		LoadArg("b_base", "R4").
		LoadArg("b_len", "R5")
	// R1 = n = min(dst_len, a_len, b_len)
	minInto(b, "R1", "R3", "R1")
	minInto(b, "R1", "R5", "R1")
	b.Raw("LSR $1, R1, R6"). // blocks = n >> 1
					Raw("MOVD $0, R7"). // word index
					Label("loop").
					Raw("CBZ R6, done").
					Raw("LSL $3, R7, R8"). // byte offset = word*8
					Raw("ADD R2, R8, R9").
					Raw("ADD R4, R8, R10").
					Raw("ADD R0, R8, R11").
					Raw("VLD1 (R9), [V0.B16]"). // a[i..i+1]  (V0=a)
					Raw("VLD1 (R10), [V1.B16]")  // b[i..i+1]  (V1=b)
	emitCombine(b, vinsn, andnot)              // result in V0
	b.Raw("VST1 [V0.B16], (R11)").
		Raw("ADD $2, R7, R7").
		Raw("SUB $1, R6, R6").
		Raw("JMP loop").
		Label("done").
		StoreRet("R7", "done").
		Ret()
	f.Add(b.Func())
}

// genCount emits countKernel(a []uint64) (sum, done int): VCNT + VUADDLV per
// 16-byte block.
func genCount(f *emit.File) {
	b := arm64.NewFunc("countKernel", countSig(), 0)
	b.LoadArg("a_base", "R0").
		LoadArg("a_len", "R1").
		Raw("MOVD $0, R2"). // sum
		Raw("LSR $1, R1, R6"). // blocks
		Raw("MOVD $0, R7"). // word index
		Label("loop").
		Raw("CBZ R6, done").
		Raw("LSL $3, R7, R8").
		Raw("ADD R0, R8, R9").
		Raw("VLD1 (R9), [V0.B16]").
		Raw("VCNT V0.B16, V0.B16").
		Raw("VUADDLV V0.B16, V0").
		Raw("VMOV V0.D[0], R10").
		Raw("ADD R10, R2, R2").
		Raw("ADD $2, R7, R7").
		Raw("SUB $1, R6, R6").
		Raw("JMP loop").
		Label("done").
		StoreRet("R2", "sum").
		StoreRet("R7", "done").
		Ret()
	f.Add(b.Func())
}

// genPair emits name(a, b []uint64) (sum, done int): combine a[i],b[i] with
// vinsn, then VCNT + VUADDLV.
func genPair(f *emit.File, name, vinsn string, andnot bool) {
	b := arm64.NewFunc(name, pairSig(), 0)
	b.LoadArg("a_base", "R0").
		LoadArg("a_len", "R1").
		LoadArg("b_base", "R2").
		LoadArg("b_len", "R3")
	minInto(b, "R1", "R3", "R1") // n = min(a_len, b_len)
	b.Raw("MOVD $0, R4").       // sum
					Raw("LSR $1, R1, R6"). // blocks
					Raw("MOVD $0, R7").    // word index
					Label("loop").
					Raw("CBZ R6, done").
					Raw("LSL $3, R7, R8").
					Raw("ADD R0, R8, R9").
					Raw("ADD R2, R8, R10").
					Raw("VLD1 (R9), [V0.B16]"). // V0=a
					Raw("VLD1 (R10), [V1.B16]")  // V1=b
	emitCombine(b, vinsn, andnot)              // result in V0
	b.Raw("VCNT V0.B16, V0.B16").
		Raw("VUADDLV V0.B16, V0").
		Raw("VMOV V0.D[0], R11").
		Raw("ADD R11, R4, R4").
		Raw("ADD $2, R7, R7").
		Raw("SUB $1, R6, R6").
		Raw("JMP loop").
		Label("done").
		StoreRet("R4", "sum").
		StoreRet("R7", "done").
		Ret()
	f.Add(b.Func())
}

func main() {
	f := emit.NewFile("arm64")

	genLogic(f, "andKernel", "VAND", false)
	genLogic(f, "orKernel", "VORR", false)
	genLogic(f, "xorKernel", "VEOR", false)
	genLogic(f, "andNotKernel", "", true)

	genCount(f)
	genPair(f, "intersectionKernel", "VAND", false)
	genPair(f, "unionKernel", "VORR", false)
	genPair(f, "differenceKernel", "", true)

	if err := os.WriteFile("bitset_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote bitset_arm64.s")
}
