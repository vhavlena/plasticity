// Package plasticity implements the synaptic dynamics of the knowledge graph
// as pure functions over edges:
//
//   - interference-based forgetting: a synapse decays exponentially in the
//     number of times its endpoints fire without each other (its Age), with
//     a spacing-effect stability (FSRS-style),
//   - soft-bounded Hebbian potentiation (LTP) on co-activation,
//   - anti-Hebbian depression (LTD) on negative feedback,
//   - homeostatic synaptic scaling,
//   - pruning of weak synapses,
//   - ACT-R base-level activation of neurons over a global recall clock.
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

// EffectiveWeight is the current (decayed) weight of e. Associative edges
// decay towards 0; typed edges towards min(w0, TypedWeightFloor).
func EffectiveWeight(e *model.Edge, p model.Params) float64 {
	r := Retrievability(e)
	if !e.Kind.Typed() {
		return clamp01(e.Weight * r)
	}
	floor := math.Min(e.Weight, p.TypedWeightFloor)
	return clamp01(floor + (e.Weight-floor)*r)
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
	e.Stability = math.Min(p.MaxStability, e.Stability*(1+p.StabilityGrowth*(p.MinStabilityGain+1-r)))
	e.Coactivations++
	e.LastUpdate = now
}

// Depress applies anti-Hebbian long-term depression:
//
//	w ← w − η_d·a_i·a_j·w
//
// Stability is not changed.
func Depress(e *model.Edge, ai, aj float64, now time.Time, p model.Params) {
	Materialize(e, p)
	e.Weight = clamp01(e.Weight - p.DepressionRate*clamp01(ai)*clamp01(aj)*e.Weight)
	e.LastUpdate = now
}

// ScaleFactor returns the multiplicative homeostatic factor for a node whose
// summed associative weight is total: min(1, maxOut/total).
func ScaleFactor(total, maxOut float64) float64 {
	if total <= maxOut || total <= 0 {
		return 1
	}
	return maxOut / total
}

// ShouldPrune reports whether e is too weak to keep. Typed edges are only
// pruned when pruneTyped is set.
func ShouldPrune(e *model.Edge, p model.Params, pruneTyped bool) bool {
	if e.Kind.Typed() && !pruneTyped {
		return false
	}
	return EffectiveWeight(e, p) < p.PruneThreshold
}

// BaseLevel is the ACT-R base-level activation B = ln Σ_j t_j^−d, where t_j is
// the number of global recall ticks since the j-th access (at least 1). It is
// −Inf if there are no accesses.
func BaseLevel(accessTicks []int64, now int64, d float64) float64 {
	if len(accessTicks) == 0 {
		return math.Inf(-1)
	}
	sum := 0.0
	for _, t := range accessTicks {
		sum += math.Pow(float64(max(1, now-t)), -d)
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
