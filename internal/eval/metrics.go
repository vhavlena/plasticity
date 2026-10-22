package eval

import (
	"math"

	"github.com/vhavlena/plasticity/internal/engine"
	"regexp"
	"sort"
	"strings"
	"time"
)

// QueryMetrics scores one recall against its relevant set.
type QueryMetrics struct {
	Recall    float64 `json:"recall"`    // |hits ∩ R| / |R|
	Precision float64 `json:"precision"` // |hits ∩ R| / |hits| (0 if no hits)
	RR        float64 `json:"rr"`        // 1 / rank of the first relevant hit
	NDCG      float64 `json:"ndcg"`      // binary-relevance nDCG over the returned list
	// Associative relevant nodes share no word with the query, so only the
	// graph (links, learning) or embeddings can retrieve them.
	AssocTotal int  `json:"assoc_total"`
	AssocFound int  `json:"assoc_found"`
	Empty      bool `json:"empty"`
	// Unlabeled queries (empty relevant set) should recall nothing.
	Unlabeled bool `json:"unlabeled,omitempty"`
	// Irrelevant hits are those not in the relevant set.
	Irrelevant int `json:"irrelevant"`
	// CrossProject hits are irrelevant hits from another project than the
	// session's (both known).
	CrossProject int `json:"cross_project"`
	// StaleHits are hits labeled stale; StaleFirst: a stale hit ranks above
	// every relevant hit (the superseded answer is the one read first).
	StaleLabels int  `json:"stale_labels,omitempty"`
	StaleHits   int  `json:"stale_hits"`
	StaleFirst  bool `json:"stale_first,omitempty"`
}

// noise fills the noise metrics of a recall in a session of the given
// project.
func (m *QueryMetrics) noise(hits []string, q Query, sessProject string, project map[string]string) {
	m.Unlabeled, m.StaleLabels = len(q.Relevant) == 0, len(q.Stale)
	rel, stale := set(q.Relevant), set(q.Stale)
	sawRelevant := false
	for _, h := range hits {
		if rel[h] {
			sawRelevant = true
			continue
		}
		m.Irrelevant++
		if p := project[h]; p != "" && sessProject != "" && p != sessProject {
			m.CrossProject++
		}
		if stale[h] {
			m.StaleHits++
			if !sawRelevant && m.StaleHits == 1 {
				m.StaleFirst = true
			}
		}
	}
}

func set(keys []string) map[string]bool {
	out := make(map[string]bool, len(keys))
	for _, k := range keys {
		out[k] = true
	}
	return out
}

// score computes metrics for a ranked list of hit keys.
func score(hits []string, relevant []string, assoc map[string]bool) QueryMetrics {
	rel := make(map[string]bool, len(relevant))
	for _, r := range relevant {
		rel[r] = true
	}
	var m QueryMetrics
	m.Empty = len(hits) == 0
	found := 0
	dcg := 0.0
	for i, h := range hits {
		if !rel[h] {
			continue
		}
		found++
		if m.RR == 0 {
			m.RR = 1 / float64(i+1)
		}
		dcg += 1 / math.Log2(float64(i+2))
		if assoc[h] {
			m.AssocFound++
		}
	}
	ideal := 0.0
	for i := 0; i < min(len(rel), max(len(hits), 1)); i++ {
		ideal += 1 / math.Log2(float64(i+2))
	}
	if len(rel) > 0 {
		m.Recall = float64(found) / float64(len(rel))
	}
	if len(hits) > 0 {
		m.Precision = float64(found) / float64(len(hits))
	}
	if ideal > 0 {
		m.NDCG = dcg / ideal
	}
	for r := range rel {
		if assoc[r] {
			m.AssocTotal++
		}
	}
	return m
}

var wordRE = regexp.MustCompile(`[\p{L}\p{N}]+`)

// stem is a crude normalization (lowercase, common English suffixes
// stripped) used only to decide lexical overlap; the engine's FTS5 uses the
// Porter stemmer, so this is an approximation.
func stem(w string) string {
	w = strings.ToLower(w)
	for _, suf := range []string{"ing", "ed", "es", "ly", "s"} {
		if len(w) > len(suf)+3 && strings.HasSuffix(w, suf) {
			return w[:len(w)-len(suf)]
		}
	}
	return w
}

func terms(text string) map[string]bool {
	out := map[string]bool{}
	for _, w := range wordRE.FindAllString(strings.ToLower(text), -1) {
		if len(w) >= 2 && !engine.IsStopword(w) {
			out[stem(w)] = true
		}
	}
	return out
}

// associative reports which relevant nodes share no term with the query.
func associative(query string, relevant []string, nodeTerms map[string]map[string]bool) map[string]bool {
	q := terms(query)
	out := map[string]bool{}
	for _, r := range relevant {
		shared := false
		for t := range nodeTerms[r] {
			if q[t] {
				shared = true
				break
			}
		}
		if !shared {
			out[r] = true
		}
	}
	return out
}

// Aggregate is the mean of query metrics over a set of queries.
type Aggregate struct {
	Queries     int     `json:"queries"`
	Recall      float64 `json:"recall"`
	Precision   float64 `json:"precision"`
	MRR         float64 `json:"mrr"`
	NDCG        float64 `json:"ndcg"`
	AssocRecall float64 `json:"assoc_recall"` // found / total over associative relevant nodes
	EmptyRate   float64 `json:"empty_rate"`
	TraceRate   float64 `json:"trace_rate"` // share of recalls that stored a learning trace
	// Labeled queries have relevant nodes; recall, precision, MRR, nDCG,
	// assoc-recall and the empty rate are means over them only.
	Labeled int `json:"labeled"`
	// NoiseRate: share of unlabeled queries that recalled anything.
	NoiseRate float64 `json:"noise_rate"`
	// HitsPerRecall: mean hits per query (skipped queries count as 0).
	HitsPerRecall float64 `json:"hits_per_recall"`
	// IrrelevantShare and CrossProjectShare: shares of all hits.
	IrrelevantShare   float64 `json:"irrelevant_share"`
	CrossProjectShare float64 `json:"cross_project_share"`
	// StaleRate and StaleFirstRate: shares of queries with stale labels that
	// recalled a stale node / ranked it above every relevant one.
	StaleQueries   int     `json:"stale_queries"`
	StaleRate      float64 `json:"stale_rate"`
	StaleFirstRate float64 `json:"stale_first_rate"`
	SkippedRate    float64 `json:"skipped_rate"` // share filtered by the prompt hook
}

func aggregate(rs []QueryResult) Aggregate {
	a := Aggregate{Queries: len(rs)}
	if len(rs) == 0 {
		return a
	}
	assocT, assocF := 0, 0
	unlabeled, noisy, hits, irrelevant, cross := 0, 0, 0, 0, 0
	for _, r := range rs {
		hits += len(r.Hits)
		irrelevant += r.Metrics.Irrelevant
		cross += r.Metrics.CrossProject
		if r.Skipped {
			a.SkippedRate++
		}
		if r.Traced {
			a.TraceRate++
		}
		if r.Metrics.StaleLabels > 0 {
			a.StaleQueries++
			if r.Metrics.StaleHits > 0 {
				a.StaleRate++
			}
			if r.Metrics.StaleFirst {
				a.StaleFirstRate++
			}
		}
		if r.Metrics.Unlabeled {
			unlabeled++
			if len(r.Hits) > 0 {
				noisy++
			}
			continue
		}
		a.Labeled++
		a.Recall += r.Metrics.Recall
		a.Precision += r.Metrics.Precision
		a.MRR += r.Metrics.RR
		a.NDCG += r.Metrics.NDCG
		assocT += r.Metrics.AssocTotal
		assocF += r.Metrics.AssocFound
		if r.Metrics.Empty {
			a.EmptyRate++
		}
	}
	n := float64(len(rs))
	a.TraceRate /= n
	a.SkippedRate /= n
	a.HitsPerRecall = float64(hits) / n
	if l := float64(a.Labeled); l > 0 {
		a.Recall /= l
		a.Precision /= l
		a.MRR /= l
		a.NDCG /= l
		a.EmptyRate /= l
	}
	if unlabeled > 0 {
		a.NoiseRate = float64(noisy) / float64(unlabeled)
	}
	if hits > 0 {
		a.IrrelevantShare = float64(irrelevant) / float64(hits)
		a.CrossProjectShare = float64(cross) / float64(hits)
	}
	if a.StaleQueries > 0 {
		a.StaleRate /= float64(a.StaleQueries)
		a.StaleFirstRate /= float64(a.StaleQueries)
	}
	if assocT > 0 {
		a.AssocRecall = float64(assocF) / float64(assocT)
	}
	return a
}

// Latency summarizes durations in milliseconds.
type Latency struct {
	Count int     `json:"count"`
	P50   float64 `json:"p50_ms"`
	P95   float64 `json:"p95_ms"`
	Max   float64 `json:"max_ms"`
	Mean  float64 `json:"mean_ms"`
	Total float64 `json:"total_ms"`
}

func latency(ds []time.Duration) Latency {
	l := Latency{Count: len(ds)}
	if len(ds) == 0 {
		return l
	}
	ms := make([]float64, len(ds))
	for i, d := range ds {
		ms[i] = float64(d) / float64(time.Millisecond)
		l.Total += ms[i]
	}
	sort.Float64s(ms)
	pct := func(p float64) float64 { return ms[min(len(ms)-1, int(math.Ceil(p*float64(len(ms))))-1)] }
	l.P50, l.P95, l.Max = pct(0.5), pct(0.95), ms[len(ms)-1]
	l.Mean = l.Total / float64(len(ms))
	return l
}
