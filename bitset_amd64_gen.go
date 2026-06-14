//go:build ignore

// Command gen produces bitset_amd64.s with go-asmgen: the amd64 SIMD kernels for
// the bit-set word operations over []uint64.
//
// Logical ops (and/or/andNot/xor): an AVX2 loop over whole 4-uint64 (32-byte)
// blocks. Each iteration loads a[i..i+3] and b[i..i+3] into YMM registers,
// applies one vector boolean instruction and stores to dst[i..i+3].
//
//   - And -> VPAND, Or -> VPOR, Xor -> VPXOR are symmetric.
//   - AndNot wants dst = a &^ b = a & ^b. Plan 9 amd64 VPANDN src1, src2, dst
//     computes dst = ^src1 & src2, so to get a & ^b we pass src1=b, src2=a:
//     "VPANDN Yb, Ya, Ydst".
//
// Counts (count/intersection/union/difference): a 4-way-unrolled hardware
// POPCNTQ loop over whole 4-word blocks. count POPCNTQs each word directly; the
// pairwise kernels load a[i] and b[i], combine them in a GPR with the matching
// integer boolean op (ANDQ / ORQ / ANDNQ for a&^b) and POPCNTQ the result. Four
// independent POPCNTQ destination registers break POPCNT's documented false
// output dependency on several Intel parts. Note ANDNQ src1, src2, dst computes
// dst = ^src1 & src2, so difference (a&^b) loads a into the second operand.
//
// All kernels compute their operating length in-kernel (min of the slice lens)
// and return done as a word count (a multiple of 4); the Go wrapper finishes the
// remainder with the scalar word loop.
//
// Run: go run bitset_amd64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/amd64"
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

// genLogic emits name(dst, a, b []uint64) (done int) — an AVX2 boolean loop.
// vinsn is the vector mnemonic; if andnot is true the AndNot operand order
// (VPANDN b, a, dst) is used so dst = a &^ b.
func genLogic(f *emit.File, name, vinsn string, andnot bool) {
	b := amd64.NewFunc(name, logicSig(), 0)
	b.LoadArg("dst_base", "DI").
		LoadArg("dst_len", "CX").
		LoadArg("a_base", "SI").
		LoadArg("a_len", "AX").
		LoadArg("b_base", "BX").
		LoadArg("b_len", "DX").
		Raw("CMPQ AX, CX"). // CX = min(dst_len, a_len)
		Raw("CMOVQLT AX, CX").
		Raw("CMPQ DX, CX"). // CX = min(CX, b_len) = n
		Raw("CMOVQLT DX, CX").
		Raw("MOVQ CX, R8"). // blocks = n >> 2
		Raw("SHRQ $2, R8").
		Raw("XORQ R9, R9"). // R9 = byte offset
		Raw("TESTQ R8, R8").
		Raw("JZ done").
		Label("loop").
		Raw("VMOVDQU (SI)(R9*1), Y0"). // a
		Raw("VMOVDQU (BX)(R9*1), Y1")  // b
	if andnot {
		// Plan 9 VPANDN A, B, dst = dst = ^B & A. We want a &^ b = ^b & a, so
		// A=a (Y0), B=b (Y1): VPANDN Y0, Y1, Y0.
		b.Raw("%s Y0, Y1, Y0", vinsn)
	} else {
		b.Raw("%s Y1, Y0, Y0", vinsn) // dst = a OP b
	}
	b.Raw("VMOVDQU Y0, (DI)(R9*1)").
		Raw("ADDQ $32, R9").
		Raw("DECQ R8").
		Raw("JNZ loop").
		Label("done").
		Raw("VZEROUPPER").
		Raw("SHRQ $3, R9"). // bytes -> words (32 bytes = 4 words: /8)
		StoreRet("R9", "done").
		Ret()
	f.Add(b.Func())
}

// genCount emits countKernel(a []uint64) (sum, done int): 4-way POPCNTQ.
func genCount(f *emit.File) {
	b := amd64.NewFunc("countKernel", countSig(), 0)
	b.LoadArg("a_base", "SI").
		LoadArg("a_len", "DX").
		Raw("XORQ AX, AX"). // sum
		Raw("XORQ DI, DI"). // byte offset
		Raw("MOVQ DX, CX"). // blocks = len >> 2
		Raw("SHRQ $2, CX").
		Raw("TESTQ CX, CX").
		Raw("JZ done").
		Label("loop").
		Raw("POPCNTQ 0(SI)(DI*1), R8").
		Raw("POPCNTQ 8(SI)(DI*1), R9").
		Raw("POPCNTQ 16(SI)(DI*1), R10").
		Raw("POPCNTQ 24(SI)(DI*1), R11").
		Raw("ADDQ R8, AX").
		Raw("ADDQ R9, AX").
		Raw("ADDQ R10, AX").
		Raw("ADDQ R11, AX").
		Raw("ADDQ $32, DI").
		Raw("DECQ CX").
		Raw("JNZ loop").
		Label("done").
		Raw("SHRQ $3, DI"). // bytes -> words
		StoreRet("AX", "sum").
		StoreRet("DI", "done").
		Ret()
	f.Add(b.Func())
}

// genPair emits name(a, b []uint64) (sum, done int): load a[i],b[i], combine
// with op (ANDQ/ORQ/ANDNQ) into a GPR, POPCNTQ. op2nd selects which source
// holds a for the ANDNQ (a&^b) case where ANDNQ src1,src2,dst = ^src1 & src2.
func genPair(f *emit.File, name, op string, andnot bool) {
	b := amd64.NewFunc(name, pairSig(), 0)
	b.LoadArg("a_base", "SI").
		LoadArg("a_len", "AX").
		LoadArg("b_base", "BX").
		LoadArg("b_len", "DX").
		Raw("CMPQ AX, DX"). // DX = n = min(a_len, b_len)
		Raw("CMOVQLT AX, DX").
		Raw("XORQ AX, AX"). // sum
		Raw("XORQ DI, DI"). // byte offset
		Raw("MOVQ DX, CX"). // blocks = n >> 2
		Raw("SHRQ $2, CX").
		Raw("TESTQ CX, CX").
		Raw("JZ done").
		Label("loop")
	// Process 4 words, two combine+popcount pairs to keep two POPCNTQ in flight.
	for _, off := range []int{0, 8, 16, 24} {
		ai := fmt.Sprintf("%d(SI)(DI*1)", off)
		bi := fmt.Sprintf("%d(BX)(DI*1)", off)
		dst := map[int]string{0: "R8", 8: "R9", 16: "R10", 24: "R11"}[off]
		if andnot {
			// a &^ b = a & ^b. Use NOTQ + ANDQ (no BMI1 dependency, so the count
			// path stays gated on POPCNT alone).
			b.Raw("MOVQ %s, R12", bi). // R12 = b
							Raw("NOTQ R12").          // R12 = ^b
							Raw("ANDQ %s, R12", ai)   // R12 = a & ^b
		} else {
			b.Raw("MOVQ %s, R12", ai).
				Raw("%s %s, R12", op, bi) // R12 = a OP b
		}
		b.Raw("POPCNTQ R12, %s", dst)
	}
	b.Raw("ADDQ R8, AX").
		Raw("ADDQ R9, AX").
		Raw("ADDQ R10, AX").
		Raw("ADDQ R11, AX").
		Raw("ADDQ $32, DI").
		Raw("DECQ CX").
		Raw("JNZ loop").
		Label("done").
		Raw("SHRQ $3, DI").
		StoreRet("AX", "sum").
		StoreRet("DI", "done").
		Ret()
	f.Add(b.Func())
}

func main() {
	f := emit.NewFile("amd64")

	genLogic(f, "andKernel", "VPAND", false)
	genLogic(f, "orKernel", "VPOR", false)
	genLogic(f, "xorKernel", "VPXOR", false)
	genLogic(f, "andNotKernel", "VPANDN", true)

	genCount(f)
	genPair(f, "intersectionKernel", "ANDQ", false)
	genPair(f, "unionKernel", "ORQ", false)
	genPair(f, "differenceKernel", "", true) // a &^ b via NOTQ + ANDQ

	if err := os.WriteFile("bitset_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote bitset_amd64.s")
}
