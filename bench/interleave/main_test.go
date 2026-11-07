package main

import "testing"

func TestParse(t *testing.T) {
	g := parseGo("BenchmarkFromBools/2^10-64   \t 1000\t 100.0 ns/op\t 10240.00 MB/s\nnoise\n")
	if g["FromBools/2^10"] != 10.24 || len(g) != 1 {
		t.Fatalf("go: %v", g)
	}
	p := parsePy("ToBools 65536 3.50\nFromBools 7 1\n")
	if p["ToBools/2^16"] != 3.5 || len(p) != 1 {
		t.Fatalf("py: %v", p)
	}
	if median([]float64{3, 1, 2}) != 2 || median(nil) != 0 {
		t.Fatal("median")
	}
}
