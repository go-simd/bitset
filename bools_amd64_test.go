//go:build amd64

package bitset

import "testing"

// TestBoolsNoAVX2 runs the bool conversions on the portable path, as a CPU
// without AVX2 would.
func TestBoolsNoAVX2(t *testing.T) {
	saved := hasAVX2
	defer func() { hasAVX2 = saved }()
	hasAVX2 = false
	TestBoolsMatchReference(t)
}
