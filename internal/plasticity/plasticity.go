// Package plasticity implements the synaptic dynamics of the knowledge graph
// as pure functions over edges:
//
//   - interference-based forgetting: a synapse decays exponentially in the
//     number of times its endpoints fire without each other (its Age), with
//     a spacing-effect stability (FSRS-style),
//   - soft-bounded Hebbian potentiation (LTP) on co-activation,
//   - anti-Hebbian depression (LTD) on negative feedback,
//   - homeostatic synaptic scaling,
//   - ACT-R base-level activation of neurons over a global recall clock.
//
// Weakening (decay, depression, scaling) never removes a synapse: its weight
// stays at or above a per-kind floor (see Floor).
//
// Wall-clock time plays no role: idle periods do not erode memory.
package plasticity

import (
	"math"
	"time"

	"github.com/vhavlena/plasticity/internal/model"
)

// Retrievability is R = exp(−age/S): the fraction of the stored weight that
// survives the interference since the last update.
func Retrievability(e *model.Edge) float64 {
	if e.Stability <= 0 {
		return 0
	}
	return math.Exp(-float64(e.Age()) / e.Stability)
}

// Floor is the weight below which weakening never pushes e:
// min(w, AssocWeightFloor) for associative edges and min(w, TypedWeightFloor)
// for typed ones, so an edge created weaker than its floor is not raised.
func Floor(e *model.Edge, p model.Params) float64 {
	floor := p.AssocWeightFloor
	if e.Kind.Typed() {
		floor = p.TypedWeightFloor
	}
	return math.Min(e.Weight, floor)
}

// EffectiveWeight is the current (decayed) weight of e: it decays from w0
// towards Floor.
func EffectiveWeight(e *model.Edge, p model.Params) float64 {
	floor := Floor(e, p)
	return clamp01(floor + (e.Weight-floor)*Retrievability(e))
}

// Excess is the part of e's effective weight above its floor: the share that
// homeostatic scaling budgets and can take away.
func Excess(e *model.Edge, p model.Params) float64 {
	return EffectiveWeight(e, p) - Floor(e, p)
}

// Materialize folds the decay into Weight and resets the age to 0 by moving
// the marks to the endpoints' current fire counts. The effective weight is
// unchanged.
func Materialize(e *model.Edge, p model.Params) {
	if e.Age() > 0 {
		e.Weight = EffectiveWeight(e, p)
	}
	e.SrcMark, e.DstMark = e.SrcFires, e.DstFires
}

// NewAssociative returns an empty learned edge between a and b (with current
// fire counts fa and fb), stored in canonical order.
func NewAssociative(a, b string, fa, fb int64, now time.Time, p model.Params) *model.Edge {
	if b < a {
		a, b, fa, fb = b, a, fb, fa
	}
	return &model.Edge{
		Src: a, Dst: b, Kind: model.KindAssociative,
		Weight: 0, Stability: p.InitialStability, LastUpdate: now,
		SrcMark: fa, DstMark: fb, SrcFires: fa, DstFires: fb,
	}
}

// NewTyped returns an explicit directed edge with the given weight
// (DefaultTypedWeight if weight ≤ 0) between endpoints with fire counts fs, fd.
func NewTyped(src, dst string, kind model.EdgeKind, weight float64, fs, fd int64, now time.Time, p model.Params) *model.Edge {
	if weight <= 0 {
		weight = p.DefaultTypedWeight
	}
	return &model.Edge{
		Src: src, Dst: dst, Kind: kind,
		Weight: clamp01(weight), Stability: p.TypedStability, LastUpdate: now,
		SrcMark: fs, DstMark: fd, SrcFires: fs, DstFires: fd,
	}
}

// Potentiate applies one Hebbian co-activation event with activations
// ai, aj ∈ [0,1]:
//
//	w ← w + η·a_i·a_j·(1 − w)               (soft bound: w stays < 1)
//	w ← max(w, AssocWeightFloor)            (associative edges only)
//	S ← S·(1 + α·(g + 1 − R))               (spacing effect)
//
// where R is the retrievability just before the event: reinforcing a link
// that interference had nearly erased (low R) increases its stability much
// more than massed repetition (R ≈ 1, gain only g).
func Potentiate(e *model.Edge, ai, aj float64, now time.Time, p model.Params) {
	r := Retrievability(e)
	if e.Coactivations == 0 && e.Weight == 0 {
		r = 1 // a brand-new synapse has nothing to be "spaced" from
	}
	Materialize(e, p)
	e.Weight = clamp01(e.Weight + p.LearningRate*clamp01(ai)*clamp01(aj)*(1-e.Weight))
	if !e.Kind.Typed() {
		e.Weight = math.Max(e.Weight, p.AssocWeightFloor)
	}
	e.Stability = math.Min(p.MaxStability, e.Stability*(1+p.StabilityGrowth*(p.MinStabilityGain+1-r)))
	e.Coactivations++
	e.LastUpdate = now
}

// Depress applies anti-Hebbian long-term depression:
//
//	w ← max(Floor, w − η_d·a_i·a_j·w)
//
// Stability is not changed.
func Depress(e *model.Edge, ai, aj float64, now time.Time, p model.Params) {
	Materialize(e, p)
	floor := Floor(e, p)
	e.Weight = math.Max(floor, clamp01(e.Weight-p.DepressionRate*clamp01(ai)*clamp01(aj)*e.Weight))
	e.LastUpdate = now
}

// Scale multiplies e's Excess by the homeostatic factor f:
//
//	w ← Floor + (w − Floor)·f
//
// so scaling never takes an edge below its floor, and an edge at the floor
// is left unchanged.
func Scale(e *model.Edge, f float64, p model.Params) {
	Materialize(e, p)
	floor := Floor(e, p)
	e.Weight = floor + (e.Weight-floor)*f
}

// ScaleFactor returns the multiplicative homeostatic factor for a node whose
// summed associative Excess is total: min(1, maxOut/total). Floor weight is
// left out of the budget, so links that faded to their floor neither trigger
// scaling nor force it onto the node's strong links, and one Scale pass with
// this factor brings the node to maxOut. A relative tolerance absorbs the
// rounding left by that pass, so a node at the budget is not rescaled again.
func ScaleFactor(total, maxOut float64) float64 {
	if total <= maxOut*(1+1e-9) || total <= 0 {
		return 1
	}
	return maxOut / total
}

// BaseLevel is the weighted ACT-R base-level activation B = ln Σ_j w_j·t_j^−d,
// where t_j is the number of global recall ticks since the j-th access (at
// least 1) and w_j its weight. It is −Inf if there are no accesses.
func BaseLevel(accesses []model.Access, now int64, d float64) float64 {
	if len(accesses) == 0 {
		return math.Inf(-1)
	}
	sum := 0.0
	for _, a := range accesses {
		sum += a.Weight * math.Pow(float64(max(1, now-a.Tick)), -d)
	}
	if sum <= 0 {
		return math.Inf(-1)
	}
	return math.Log(sum)
}

// Sigmoid maps B to (0,1); −Inf maps to 0.
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func clamp01(x float64) float64 {
	switch {
	case x < 0 || math.IsNaN(x):
		return 0
	case x > 1:
		return 1
	}
	return x
}
