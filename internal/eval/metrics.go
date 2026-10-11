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
	m.Recall = float64(found) / float64(len(rel))
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
}

func aggregate(rs []QueryResult) Aggregate {
	a := Aggregate{Queries: len(rs)}
	if len(rs) == 0 {
		return a
	}
	assocT, assocF := 0, 0
	for _, r := range rs {
		a.Recall += r.Metrics.Recall
		a.Precision += r.Metrics.Precision
		a.MRR += r.Metrics.RR
		a.NDCG += r.Metrics.NDCG
		assocT += r.Metrics.AssocTotal
		assocF += r.Metrics.AssocFound
		if r.Metrics.Empty {
			a.EmptyRate++
		}
		if r.Traced {
			a.TraceRate++
		}
	}
	n := float64(len(rs))
	a.Recall /= n
	a.Precision /= n
	a.MRR /= n
	a.NDCG /= n
	a.EmptyRate /= n
	a.TraceRate /= n
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
