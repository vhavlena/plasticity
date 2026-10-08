package plasticity

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/vhavlena/plasticity/internal/model"
)

var t0 = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

func approx(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

// fire simulates n solo firings of the edge's source endpoint.
func fire(e *model.Edge, n int64) { e.SrcFires += n }

func assoc(w, s float64) *model.Edge {
	return &model.Edge{Src: "a", Dst: "b", Kind: model.KindAssociative, Weight: w, Stability: s}
}

func TestAgeCountsSoloFiringsOfBothEndpoints(t *testing.T) {
	e := assoc(0.5, 10)
	e.SrcFires, e.DstFires = 3, 4
	if e.Age() != 7 {
		t.Fatalf("age = %d, want 7", e.Age())
	}
	e.SrcMark, e.DstMark = 3, 2
	if e.Age() != 2 {
		t.Fatalf("age = %d, want 2", e.Age())
	}
}

func TestDecayIsExponentialInInterference(t *testing.T) {
	p := model.DefaultParams()
	e := assoc(0.8, 10)
	if got := EffectiveWeight(e, p); !approx(got, 0.8) {
		t.Fatalf("w(0) = %g", got)
	}
	fire(e, 10)
	if got, want := EffectiveWeight(e, p), 0.8*math.Exp(-1); !approx(got, want) {
		t.Fatalf("w(S) = %g, want %g", got, want)
	}
	prev := 1.0
	for i := 0; i < 100; i++ {
		fire(e, 1)
		w := EffectiveWeight(e, p)
		if w > prev || w < 0 {
			t.Fatalf("decay not monotone at %d: %g", i, w)
		}
		prev = w
	}
}

func TestNoInterferenceNoForgetting(t *testing.T) {
	p := model.DefaultParams()
	e := assoc(0.6, 5)
	// Wall-clock time is irrelevant: only firings age a synapse.
	e.LastUpdate = t0.Add(-10 * 365 * 24 * time.Hour)
	if got := EffectiveWeight(e, p); !approx(got, 0.6) {
		t.Fatalf("weight changed without interference: %g", got)
	}
}

func TestTypedEdgesDecayToFloor(t *testing.T) {
	p := model.DefaultParams()
	e := &model.Edge{Kind: model.KindSupports, Weight: 0.9, Stability: 5}
	fire(e, 100000)
	if w := EffectiveWeight(e, p); !approx(w, p.TypedWeightFloor) {
		t.Fatalf("typed edge decayed to %g, want floor %g", w, p.TypedWeightFloor)
	}
	if ShouldPrune(e, p, false) {
		t.Fatal("typed edge must not be pruned by default")
	}
	low := &model.Edge{Kind: model.KindSupports, Weight: 0.1, Stability: 5}
	fire(low, 1000)
	if w := EffectiveWeight(low, p); !approx(w, 0.1) {
		t.Fatalf("typed edge below floor changed: %g", w)
	}
}

func TestMaterializePreservesWeightAndResetsAge(t *testing.T) {
	p := model.DefaultParams()
	e := assoc(0.7, 7)
	fire(e, 5)
	e.DstFires = 2
	before := EffectiveWeight(e, p)
	Materialize(e, p)
	if e.Age() != 0 || !approx(EffectiveWeight(e, p), before) {
		t.Fatalf("age %d, weight %g vs %g", e.Age(), EffectiveWeight(e, p), before)
	}
	fire(e, 3)
	f := assoc(0.7, 7)
	fire(f, 8)
	f.DstFires = 2
	if !approx(EffectiveWeight(e, p), EffectiveWeight(f, p)) {
		t.Fatal("materializing must not change future weights")
	}
}

func TestNewAssociativeIsCanonical(t *testing.T) {
	p := model.DefaultParams()
	e := NewAssociative("b", "a", 5, 9, t0, p)
	if e.Src != "a" || e.SrcMark != 9 || e.DstMark != 5 || e.Age() != 0 {
		t.Fatalf("not canonical: %+v", e)
	}
}

func TestPotentiateSaturatesBelowOne(t *testing.T) {
	p := model.DefaultParams()
	e := NewAssociative("a", "b", 0, 0, t0, p)
	prev := 0.0
	for i := 0; i < 50; i++ {
		Potentiate(e, 1, 1, t0, p)
		if e.Weight <= prev || e.Weight >= 1 {
			t.Fatalf("step %d: weight %g not strictly increasing below 1", i, e.Weight)
		}
		prev = e.Weight
	}
	f := NewAssociative("a", "b", 0, 0, t0, p)
	Potentiate(f, 1, 1, t0, p)
	if !approx(f.Weight, p.LearningRate) {
		t.Fatalf("first step = %g, want η", f.Weight)
	}
}

func TestPotentiateScalesWithActivation(t *testing.T) {
	p := model.DefaultParams()
	strong := NewAssociative("a", "b", 0, 0, t0, p)
	weak := NewAssociative("a", "c", 0, 0, t0, p)
	Potentiate(strong, 1, 0.9, t0, p)
	Potentiate(weak, 1, 0.2, t0, p)
	if !(strong.Weight > weak.Weight) {
		t.Fatalf("activation-weighted LTP violated")
	}
}

func TestSpacingEffect(t *testing.T) {
	p := model.DefaultParams()
	massed := NewAssociative("a", "b", 0, 0, t0, p)
	spaced := NewAssociative("a", "b", 0, 0, t0, p)
	for i := 0; i < 4; i++ {
		Potentiate(massed, 1, 1, t0, p)
		Potentiate(spaced, 1, 1, t0, p)
		fire(spaced, 15) // endpoints used elsewhere in between
	}
	if !(spaced.Stability > massed.Stability) {
		t.Fatalf("spaced stability %g should exceed massed %g", spaced.Stability, massed.Stability)
	}
	if massed.Stability <= p.InitialStability {
		t.Fatalf("massed repetition should still grow stability a little")
	}
	big := NewAssociative("a", "b", 0, 0, t0, p)
	for i := 0; i < 200; i++ {
		Potentiate(big, 1, 1, t0, p)
		fire(big, 100000)
	}
	if big.Stability > p.MaxStability {
		t.Fatalf("stability %g exceeds cap", big.Stability)
	}
}

func TestDepress(t *testing.T) {
	p := model.DefaultParams()
	e := assoc(0.5, 7)
	Depress(e, 1, 1, t0, p)
	if want := 0.5 * (1 - p.DepressionRate); !approx(e.Weight, want) {
		t.Fatalf("LTD weight %g, want %g", e.Weight, want)
	}
}

func TestPruneAfterInterference(t *testing.T) {
	p := model.DefaultParams()
	e := NewAssociative("a", "b", 0, 0, t0, p)
	Potentiate(e, 1, 1, t0, p)
	fire(e, 5)
	if ShouldPrune(e, p, false) {
		t.Fatal("lightly interfered edge pruned")
	}
	fire(e, 50)
	if !ShouldPrune(e, p, false) {
		t.Fatalf("edge survived 55 solo firings: w=%g", EffectiveWeight(e, p))
	}
}

func TestScaleFactor(t *testing.T) {
	if f := ScaleFactor(3, 5); f != 1 {
		t.Fatalf("no scaling expected, got %g", f)
	}
	if f := ScaleFactor(10, 5); !approx(f*10, 5) {
		t.Fatalf("scaled sum %g", f*10)
	}
}

func TestBaseLevel(t *testing.T) {
	if !math.IsInf(BaseLevel(nil, 100, 0.5), -1) || Sigmoid(math.Inf(-1)) != 0 {
		t.Fatal("no accesses must give -Inf → 0")
	}
	recent := BaseLevel([]int64{999}, 1000, 0.5)
	old := BaseLevel([]int64{900}, 1000, 0.5)
	freq := BaseLevel([]int64{900, 910, 920}, 1000, 0.5)
	if !(recent > old) || !(freq > old) {
		t.Fatalf("recency/frequency: %g %g %g", recent, old, freq)
	}
	if !approx(BaseLevel([]int64{1000}, 1000, 0.5), 0) {
		t.Fatal("access at the current tick should give B=0")
	}
}

// Property: random sequences of LTP/LTD/interference keep weights in [0,1]
// and stability in (0, max].
func TestRandomDynamicsStayBounded(t *testing.T) {
	p := model.DefaultParams()
	rng := rand.New(rand.NewSource(1))
	for trial := 0; trial < 200; trial++ {
		e := NewAssociative("a", "b", 0, 0, t0, p)
		for step := 0; step < 100; step++ {
			e.SrcFires += int64(rng.Intn(30))
			e.DstFires += int64(rng.Intn(30))
			switch rng.Intn(3) {
			case 0:
				Potentiate(e, rng.Float64(), rng.Float64(), t0, p)
			case 1:
				Depress(e, rng.Float64(), rng.Float64(), t0, p)
			case 2:
				Materialize(e, p)
			}
			w := EffectiveWeight(e, p)
			if w < 0 || w > 1 || e.Stability <= 0 || e.Stability > p.MaxStability {
				t.Fatalf("trial %d step %d: w=%g S=%g", trial, step, w, e.Stability)
			}
		}
	}
}
