// Command interleave runs the Go FromBools/ToBools benchmarks and the NumPy
// reference (bench/numpy_bools.py) in alternating rounds on one core, and
// prints the median GB/s of each cell with the 1-minute load seen around the
// rounds that were kept. Rounds whose load before or after exceeds maxload
// are dropped. The Scalar benchmarks are the control pass: the per-element Go
// loop, which must not move between rounds.
package main

import (
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

var goLine = regexp.MustCompile(`^Benchmark(\w+)/(2\^\d+)(?:-\d+)?\s+\d+\s+[\d.]+ ns/op\s+([\d.]+) MB/s`)

// parseGo returns "Op/size" -> GB/s from `go test -bench` output.
func parseGo(out string) map[string]float64 {
	m := map[string]float64{}
	for _, l := range strings.Split(out, "\n") {
		if g := goLine.FindStringSubmatch(l); g != nil {
			v, _ := strconv.ParseFloat(g[3], 64)
			m[g[1]+"/"+g[2]] = v / 1000
		}
	}
	return m
}

var sizeName = map[string]string{"1024": "2^10", "65536": "2^16", "1048576": "2^20"}

// parsePy returns "Op/size" -> GB/s from numpy_bools.py output.
func parsePy(out string) map[string]float64 {
	m := map[string]float64{}
	for _, l := range strings.Split(out, "\n") {
		f := strings.Fields(l)
		if len(f) == 3 && sizeName[f[1]] != "" {
			v, _ := strconv.ParseFloat(f[2], 64)
			m[f[0]+"/"+sizeName[f[1]]] = v
		}
	}
	return m
}

func load() float64 {
	b, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		return -1
	}
	v, _ := strconv.ParseFloat(strings.Fields(string(b))[0], 64)
	return v
}

func median(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	sort.Float64s(v)
	return v[len(v)/2]
}

func run(name string, args ...string) string {
	out, err := exec.Command(name, args...).CombinedOutput()
	if err != nil {
		fmt.Fprintln(os.Stderr, name, err, string(out))
		os.Exit(1)
	}
	return string(out)
}

func main() {
	if len(os.Args) != 7 {
		fmt.Fprintln(os.Stderr, "usage: interleave testbin python script core rounds maxload")
		os.Exit(2)
	}
	bin, py, script, core := os.Args[1], os.Args[2], os.Args[3], os.Args[4]
	rounds, _ := strconv.Atoi(os.Args[5])
	maxload, _ := strconv.ParseFloat(os.Args[6], 64)
	goV, pyV := map[string][]float64{}, map[string][]float64{}
	var loads []float64
	kept := 0
	for range rounds {
		l0 := load()
		g := parseGo(run("taskset", "-c", core, bin, "-test.run", "^$", "-test.bench", "Bools", "-test.benchtime", "300ms"))
		p := parsePy(run("taskset", "-c", core, py, script, "3"))
		l1 := load()
		if l0 > maxload || l1 > maxload {
			fmt.Printf("round dropped: load %.2f -> %.2f\n", l0, l1)
			continue
		}
		kept++
		loads = append(loads, l0, l1)
		for k, v := range g {
			goV[k] = append(goV[k], v)
		}
		for k, v := range p {
			pyV[k] = append(pyV[k], v)
		}
	}
	if kept == 0 {
		fmt.Println("no round kept")
		os.Exit(1)
	}
	sort.Float64s(loads)
	fmt.Printf("rounds kept %d/%d, load %.2f..%.2f\n", kept, rounds, loads[0], loads[len(loads)-1])
	for _, op := range []string{"FromBools", "ToBools"} {
		for _, n := range []string{"2^10", "2^16", "2^20"} {
			k := op + "/" + n
			g, s, np := median(goV[k]), median(goV[op+"Scalar/"+n]), median(pyV[k])
			fmt.Printf("%-9s %-5s go %7.2f  scalar(control) %6.2f  numpy %7.2f  go/numpy %.2fx\n", op, n, g, s, np, g/np)
		}
	}
}
