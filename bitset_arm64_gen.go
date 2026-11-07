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
					Raw("VLD1 (R10), [V1.B16]") // b[i..i+1]  (V1=b)
	emitCombine(b, vinsn, andnot) // result in V0
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
		Raw("MOVD $0, R2").    // sum
		Raw("LSR $1, R1, R6"). // blocks
		Raw("MOVD $0, R7").    // word index
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
	b.Raw("MOVD $0, R4").        // sum
					Raw("LSR $1, R1, R6"). // blocks
					Raw("MOVD $0, R7").    // word index
					Label("loop").
					Raw("CBZ R6, done").
					Raw("LSL $3, R7, R8").
					Raw("ADD R0, R8, R9").
					Raw("ADD R2, R8, R10").
					Raw("VLD1 (R9), [V0.B16]"). // V0=a
					Raw("VLD1 (R10), [V1.B16]") // V1=b
	emitCombine(b, vinsn, andnot) // result in V0
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

func boolsSig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("words"), abi.Slice("bools")},
		[]abi.Arg{abi.Scalar("done", abi.Int64)},
	)
}

// genPackBools emits packBoolsKernel(words []uint64, bools []byte) (done int):
// 64 bools (bytes 0 or 1) become one word per iteration. Four 16-byte loads;
// USHL shifts byte i of every 8-byte group left by i (V4 = 0..7, 0..7); three
// rounds of ADDP then add the eight shifted bytes of each group together, in
// order (no carries: the bits are distinct), leaving the 8 result bytes in the
// low half of V0, which is the word.
func genPackBools(f *emit.File) {
	b := arm64.NewFunc("packBoolsKernel", boolsSig(), 0)
	b.LoadArg("words_base", "R0").
		LoadArg("words_len", "R1").
		LoadArg("bools_base", "R2").
		LoadArg("bools_len", "R3").
		Raw("LSR $6, R3, R3"). // full 64-byte groups available
		Raw("CMP R3, R1").
		Raw("CSEL LT, R1, R3, R1"). // n = min(words_len, groups)
		Raw("MOVD $0x0706050403020100, R4").
		Raw("VMOV R4, V4.D[0]").
		Raw("VMOV R4, V4.D[1]").
		Raw("MOVD $0, R7").
		Label("loop").
		Raw("CMP R1, R7").
		Raw("BEQ done").
		Raw("VLD1.P 64(R2), [V0.B16, V1.B16, V2.B16, V3.B16]").
		Raw("VUSHL V4.B16, V0.B16, V0.B16").
		Raw("VUSHL V4.B16, V1.B16, V1.B16").
		Raw("VUSHL V4.B16, V2.B16, V2.B16").
		Raw("VUSHL V4.B16, V3.B16, V3.B16").
		Raw("VADDP V1.B16, V0.B16, V0.B16").
		Raw("VADDP V3.B16, V2.B16, V2.B16").
		Raw("VADDP V2.B16, V0.B16, V0.B16").
		Raw("VADDP V0.B16, V0.B16, V0.B16").
		Raw("VMOV V0.D[0], R10").
		Raw("MOVD.P R10, 8(R0)").
		Raw("ADD $1, R7, R7").
		Raw("JMP loop").
		Label("done").
		StoreRet("R7", "done").
		Ret()
	f.Add(b.Func())
}

// genUnpackBools emits unpackBoolsKernel(words []uint64, bools []byte) (done
// int): one word becomes 64 bool bytes per iteration. TBL against the word
// replicates each of its bytes 8 times (V5..V8 hold the indices), USHL by 0,
// -1 .. -7 brings bit k of each byte to the bottom of lane k, AND with 1 keeps it.
func genUnpackBools(f *emit.File) {
	b := arm64.NewFunc("unpackBoolsKernel", boolsSig(), 0)
	b.LoadArg("words_base", "R0").
		LoadArg("words_len", "R1").
		LoadArg("bools_base", "R2").
		LoadArg("bools_len", "R3").
		Raw("LSR $6, R3, R3").
		Raw("CMP R3, R1").
		Raw("CSEL LT, R1, R3, R1")
	// index vectors: lane 8h+k of V(5+j) selects byte 2j+h.
	for j := range 4 {
		lo := uint64(2*j) * 0x0101010101010101
		hi := uint64(2*j+1) * 0x0101010101010101
		b.Raw("MOVD $0x%016x, R4", lo).
			Raw("VMOV R4, V%d.D[0]", 5+j).
			Raw("MOVD $0x%016x, R4", hi).
			Raw("VMOV R4, V%d.D[1]", 5+j)
	}
	b.Raw("MOVD $0xf9fafbfcfdfeff00, R4"). // shifts 0,-1,..,-7
						Raw("VMOV R4, V9.D[0]").
						Raw("VMOV R4, V9.D[1]").
						Raw("MOVD $1, R4").
						Raw("VDUP R4, V10.B16").
						Raw("MOVD $0, R7").
						Label("loop").
						Raw("CMP R1, R7").
						Raw("BEQ done").
						Raw("MOVD.P 8(R0), R10").
						Raw("VMOV R10, V4.D[0]")
	for j := range 4 {
		b.Raw("VTBL V%d.B16, [V4.B16], V%d.B16", 5+j, j).
			Raw("VUSHL V9.B16, V%d.B16, V%d.B16", j, j).
			Raw("VAND V10.B16, V%d.B16, V%d.B16", j, j)
	}
	b.Raw("VST1.P [V0.B16, V1.B16, V2.B16, V3.B16], 64(R2)").
		Raw("ADD $1, R7, R7").
		Raw("JMP loop").
		Label("done").
		StoreRet("R7", "done").
		Ret()
	f.Add(b.Func())
}

func main() {
	f := emit.NewFile("arm64")

	genPackBools(f)
	genUnpackBools(f)

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
