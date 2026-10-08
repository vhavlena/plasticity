package model

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
)

// Params holds every tunable constant of the plasticity and retrieval
// algorithms. Field JSON names are the keys used by `plasticity params`.
type Params struct {
	// --- Hebbian learning (LTP / LTD) ---

	// LearningRate is η in Δw = η·a_i·a_j·(1−w).
	LearningRate float64 `json:"learning_rate"`
	// DepressionRate is η_d in Δw = −η_d·a_i·a_j·w (anti-Hebbian feedback).
	DepressionRate float64 `json:"depression_rate"`
	// LearnMinConfidence: a recall stores a trace for Hebbian learning only
	// if its query confidence is at least this.
	LearnMinConfidence float64 `json:"learn_min_confidence"`
	// LearnMinActivation: only recalled nodes with at least this activation
	// enter the trace.
	LearnMinActivation float64 `json:"learn_min_activation"`
	// TraceRetention: pending traces older than this many recalls are
	// discarded by consolidation without being learned from.
	TraceRetention int `json:"trace_retention"`
	// ReinforceTopK limits how many trace nodes take part in pairwise
	// co-activation learning per recall (cost is quadratic).
	ReinforceTopK int `json:"reinforce_top_k"`

	// --- Forgetting (lazy exponential decay with spacing effect) ---

	// InitialStability is S (in solo endpoint firings) for a newly learned
	// associative edge: after S firings of its endpoints without each other,
	// the weight drops to 1/e.
	InitialStability float64 `json:"initial_stability"`
	// TypedStability is S for explicit (typed) edges.
	TypedStability float64 `json:"typed_stability"`
	// StabilityGrowth is α in S ← S·(1 + α·(MinStabilityGain + 1 − R)).
	StabilityGrowth float64 `json:"stability_growth"`
	// MinStabilityGain is the stability gain of massed (non-spaced) repetition.
	MinStabilityGain float64 `json:"min_stability_gain"`
	// MaxStability caps S.
	MaxStability float64 `json:"max_stability"`
	// TypedWeightFloor is the weight typed edges decay towards (never below).
	TypedWeightFloor float64 `json:"typed_weight_floor"`
	// DefaultTypedWeight is the weight of a newly linked typed edge.
	DefaultTypedWeight float64 `json:"default_typed_weight"`

	// --- Homeostasis and pruning ---

	// MaxOutWeight caps the summed associative weight around one node
	// (homeostatic synaptic scaling).
	MaxOutWeight float64 `json:"max_out_weight"`
	// PruneThreshold: associative edges whose effective weight falls below it
	// are removed during consolidation.
	PruneThreshold float64 `json:"prune_threshold"`

	// --- ACT-R base-level activation ---

	// ActrDecay is d in B = ln Σ t_j^−d (t = recalls since the access).
	ActrDecay float64 `json:"actr_decay"`
	// MaxAccessRecords is the number of access timestamps kept per node.
	MaxAccessRecords int `json:"max_access_records"`
	// DormantBelow marks isolated nodes dormant when B drops below it
	// (−4 ≈ a single access about 3000 recalls ago).
	DormantBelow float64 `json:"dormant_below"`

	// --- Retrieval (weight-gated spreading activation) ---

	// SeedLimit is the number of BM25 seeds.
	SeedLimit int `json:"seed_limit"`
	// AbsentTermWeight scales the IDF of query terms that occur nowhere in
	// the graph when computing query confidence (0 = ignore unknown words,
	// 1 = treat them as fully unexplained query content).
	AbsentTermWeight float64 `json:"absent_term_weight"`
	// MinSeedStrength drops seeds whose strength (relative BM25 × query
	// confidence) is below it; a query without a good match recalls nothing.
	MinSeedStrength float64 `json:"min_seed_strength"`
	// HopDecay is δ, the per-hop attenuation.
	HopDecay float64 `json:"hop_decay"`
	// FiringThreshold is θ: a node is reached only if its activation ≥ θ.
	FiringThreshold float64 `json:"firing_threshold"`
	// FanK0 and FanBeta implement the fan effect: a node of degree k passes
	// activation divided by max(1, k/K0)^β.
	FanK0   float64 `json:"fan_k0"`
	FanBeta float64 `json:"fan_beta"`
	// MaxNodes caps the number of distinct nodes expanded per recall.
	MaxNodes int `json:"max_nodes"`
	// MaxHops is a safety cap on path length.
	MaxHops int `json:"max_hops"`
	// EdgesPerNode caps edges fetched per expanded node (strongest first).
	EdgesPerNode int `json:"edges_per_node"`
	// ResultLimit is the default number of recalled nodes.
	ResultLimit int `json:"result_limit"`

	// --- Final ranking: λ1·graph + λ2·σ(B) + λ3·importance ---

	ScoreGraph      float64 `json:"score_graph"`
	ScoreBaseLevel  float64 `json:"score_base_level"`
	ScoreImportance float64 `json:"score_importance"`

	// --- BM25 column weights (title, content, rationale, tags) ---

	BM25Title     float64 `json:"bm25_title"`
	BM25Content   float64 `json:"bm25_content"`
	BM25Rationale float64 `json:"bm25_rationale"`
	BM25Tags      float64 `json:"bm25_tags"`

	// --- PageRank rerank ---

	PPRRestart float64 `json:"ppr_restart"`
}

// DefaultParams returns the default configuration.
func DefaultParams() Params {
	return Params{
		LearningRate:   0.2,
		DepressionRate: 0.15,
		ReinforceTopK:  12,
		TraceRetention: 1000,

		LearnMinConfidence: 0.4,
		LearnMinActivation: 0.2,

		InitialStability:   20,
		TypedStability:     200,
		StabilityGrowth:    1.0,
		MinStabilityGain:   0.2,
		MaxStability:       10000,
		TypedWeightFloor:   0.3,
		DefaultTypedWeight: 0.8,

		MaxOutWeight:   5,
		PruneThreshold: 0.05,

		ActrDecay:        0.5,
		MaxAccessRecords: 50,
		DormantBelow:     -4,

		SeedLimit:        10,
		AbsentTermWeight: 0.5,
		MinSeedStrength:  0.2,
		HopDecay:         0.8,
		FiringThreshold:  0.1,
		FanK0:            8,
		FanBeta:          0.5,
		MaxNodes:         200,
		MaxHops:          10,
		EdgesPerNode:     64,
		ResultLimit:      10,

		ScoreGraph:      0.7,
		ScoreBaseLevel:  0.2,
		ScoreImportance: 0.1,

		BM25Title:     4,
		BM25Content:   1,
		BM25Rationale: 1,
		BM25Tags:      3,

		PPRRestart: 0.15,
	}
}

// Validate checks parameter ranges.
func (p Params) Validate() error {
	in01 := map[string]float64{
		"learning_rate": p.LearningRate, "depression_rate": p.DepressionRate,
		"typed_weight_floor": p.TypedWeightFloor, "default_typed_weight": p.DefaultTypedWeight,
		"prune_threshold": p.PruneThreshold, "hop_decay": p.HopDecay,
		"firing_threshold": p.FiringThreshold, "ppr_restart": p.PPRRestart,
		"absent_term_weight": p.AbsentTermWeight, "min_seed_strength": p.MinSeedStrength,
		"learn_min_confidence": p.LearnMinConfidence, "learn_min_activation": p.LearnMinActivation,
	}
	for k, v := range in01 {
		if v < 0 || v > 1 {
			return fmt.Errorf("%s must be in [0,1], got %g", k, v)
		}
	}
	pos := map[string]float64{
		"initial_stability": p.InitialStability, "typed_stability": p.TypedStability,
		"max_stability": p.MaxStability, "max_out_weight": p.MaxOutWeight,
		"fan_k0": p.FanK0, "actr_decay": p.ActrDecay,
	}
	for k, v := range pos {
		if v <= 0 {
			return fmt.Errorf("%s must be > 0, got %g", k, v)
		}
	}
	if p.FiringThreshold == 0 {
		return fmt.Errorf("firing_threshold must be > 0")
	}
	if p.ReinforceTopK < 2 || p.SeedLimit < 1 || p.MaxNodes < 1 || p.MaxHops < 1 ||
		p.EdgesPerNode < 1 || p.ResultLimit < 1 || p.MaxAccessRecords < 1 || p.TraceRetention < 1 {
		return fmt.Errorf("integer limits must be positive (reinforce_top_k ≥ 2)")
	}
	return nil
}

// ToMap renders params as key → string value.
func (p Params) ToMap() map[string]string {
	raw, _ := json.Marshal(p)
	var m map[string]json.RawMessage
	_ = json.Unmarshal(raw, &m)
	out := make(map[string]string, len(m))
	for k, v := range m {
		out[k] = string(v)
	}
	return out
}

// Keys returns all parameter names, sorted.
func (p Params) Keys() []string {
	m := p.ToMap()
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// With returns a copy of p with the overrides in m applied. Unknown keys and
// malformed numbers are errors.
func (p Params) With(m map[string]string) (Params, error) {
	known := p.ToMap()
	obj := make(map[string]json.RawMessage, len(m))
	for k, v := range m {
		if _, ok := known[k]; !ok {
			return p, fmt.Errorf("unknown parameter %q", k)
		}
		if _, err := strconv.ParseFloat(v, 64); err != nil {
			return p, fmt.Errorf("parameter %q: %q is not a number", k, v)
		}
		obj[k] = json.RawMessage(v)
	}
	raw, _ := json.Marshal(obj)
	q := p
	if err := json.Unmarshal(raw, &q); err != nil {
		return p, fmt.Errorf("parameter override: %w", err)
	}
	return q, q.Validate()
}
