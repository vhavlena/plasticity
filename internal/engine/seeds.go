package engine

import (
	"math"
	"regexp"
	"strings"

	"github.com/vhavlena/plasticity/internal/store"
)

var wordRE = regexp.MustCompile(`[\p{L}\p{N}]+`)

var stopwords = map[string]bool{}

func init() {
	for _, w := range strings.Fields(`a also an and are as at be but by can could do does for from has have how
		i if in into is it its just me my need not of on or our please should so that the their them
		then there these this to us want was we what when where which who why will with would you your`) {
		stopwords[w] = true
	}
}

// queryTerms extracts the distinct searchable words of a query: lowercase,
// without stopwords and one-letter words.
func queryTerms(query string) []string {
	seen := map[string]bool{}
	var terms []string
	for _, w := range wordRE.FindAllString(strings.ToLower(query), -1) {
		if len(w) < 2 || stopwords[w] || seen[w] {
			continue
		}
		seen[w] = true
		terms = append(terms, w)
	}
	return terms
}

func quote(term string) string { return `"` + term + `"` }

// MatchExpr turns free text into a safe FTS5 expression: an OR of quoted
// terms without stopwords. It returns "" if nothing searchable remains.
func MatchExpr(query string) string {
	terms := queryTerms(query)
	for i, t := range terms {
		terms[i] = quote(t)
	}
	return strings.Join(terms, " OR ")
}

// TermInfo explains how one query term contributed to query confidence.
type TermInfo struct {
	Term    string  `json:"term"`
	DocFreq int     `json:"doc_freq"`
	IDF     float64 `json:"idf"`
	Matched bool    `json:"matched"` // matched by at least one seed candidate
}

// idf is the BM25 inverse document frequency, always positive.
func idf(n, df int) float64 {
	return math.Log((float64(n-df)+0.5)/(float64(df)+0.5) + 1)
}

// seeding is the outcome of turning BM25 hits into seed activations.
type seeding struct {
	seeds      map[string]float64
	confidence float64
	terms      []TermInfo
}

// computeSeeds calibrates BM25 hits into seed strengths:
//
//	seed_i     = (bm25_i / bm25_best) · C
//	C          = Σ_{t matched by some hit} idf_t / (Σ_{t known} idf_t + w_abs · Σ_{t unknown} idf_t)
//
// C (query confidence) is the IDF-weighted share of the query that the hits
// explain. Words that occur nowhere in the graph count against it with weight
// w_abs. The relative BM25 factor ranks seeds within one query; C makes the
// strength comparable across queries, so a query whose best match covers only
// an unimportant fraction of it produces weak seeds (or none, below
// MinSeedStrength) instead of a full-strength one.
func (e *Engine) computeSeeds(tx *store.Tx, terms []string, hits []store.SearchHit) (*seeding, error) {
	res := &seeding{seeds: map[string]float64{}, terms: []TermInfo{}}
	if len(terms) == 0 {
		return res, nil
	}
	n, err := tx.NodeCount()
	if err != nil {
		return nil, err
	}
	ids := make([]string, len(hits))
	for i, h := range hits {
		ids[i] = h.Node.ID
	}
	matched, known, total := 0.0, 0.0, 0.0
	for _, t := range terms {
		df, err := tx.DocFreq(quote(t))
		if err != nil {
			return nil, err
		}
		ti := TermInfo{Term: t, DocFreq: df, IDF: idf(n, df)}
		if df == 0 {
			total += e.P.AbsentTermWeight * ti.IDF
		} else {
			total += ti.IDF
			known += ti.IDF
			if len(ids) > 0 {
				among, err := tx.MatchingAmong(quote(t), ids)
				if err != nil {
					return nil, err
				}
				if ti.Matched = len(among) > 0; ti.Matched {
					matched += ti.IDF
				}
			}
		}
		res.terms = append(res.terms, ti)
	}
	if total > 0 {
		res.confidence = matched / total
	}
	for _, h := range hits {
		if s := h.Score / hits[0].Score * res.confidence; s >= e.P.MinSeedStrength {
			res.seeds[h.Node.ID] = s
		}
	}
	return res, nil
}
