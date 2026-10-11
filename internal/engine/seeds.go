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

// seeding is the outcome of turning lexical and semantic matches into seed
// activations.
type seeding struct {
	seeds map[string]float64
	// lexical is the BM25 query confidence C; semantic is the strongest
	// calibrated similarity S among the vector matches.
	lexical, semantic float64
	similarity        map[string]float64 // raw cosine of semantic candidates
	terms             []TermInfo
}

// confidence is the evidence that the query matched something: the stronger
// of the lexical and semantic signals.
func (s *seeding) confidence() float64 { return math.Max(s.lexical, s.semantic) }

// lexicalStrengths calibrates BM25 hits:
//
//	L_i = (bm25_i / bm25_best) · C
//	C   = Σ_{t matched by some hit} idf_t / (Σ_{t known} idf_t + w_abs · Σ_{t unknown} idf_t)
//
// C (query confidence) is the IDF-weighted share of the query that the hits
// explain. Words that occur nowhere in the graph count against it with weight
// w_abs. The relative BM25 factor ranks hits within one query; C makes the
// strength comparable across queries, so a query whose best match covers only
// an unimportant fraction of it produces weak seeds instead of full-strength
// ones.
func (e *Engine) lexicalStrengths(tx *store.Tx, terms []string, hits []store.SearchHit, sd *seeding) (map[string]float64, error) {
	strengths := map[string]float64{}
	if len(terms) == 0 {
		return strengths, nil
	}
	n, err := tx.NodeCount()
	if err != nil {
		return nil, err
	}
	ids := make([]string, len(hits))
	for i, h := range hits {
		ids[i] = h.Node.ID
	}
	matched, total := 0.0, 0.0
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
		sd.terms = append(sd.terms, ti)
	}
	if total > 0 {
		sd.lexical = matched / total
	}
	for _, h := range hits {
		strengths[h.Node.ID] = h.Score / hits[0].Score * sd.lexical
	}
	return strengths, nil
}

// semanticStrength calibrates cosine similarity: S = clamp((cos − low)/(high − low)).
func (e *Engine) semanticStrength(cos float64) float64 {
	return math.Max(0, math.Min(1, (cos-e.P.VectorCosLow)/(e.P.VectorCosHigh-e.P.VectorCosLow)))
}

// computeSeeds fuses calibrated lexical strengths L and semantic strengths S
// by noisy-OR, seed = 1 − (1 − L)(1 − S): either signal alone can seed, and
// agreement of both makes a seed stronger. Because both scores are already
// calibrated to [0,1], fusing the scores keeps more information than
// rank-based fusion (RRF) would. Seeds below MinSeedStrength are dropped.
func (e *Engine) computeSeeds(tx *store.Tx, terms []string, hits []store.SearchHit, sem []vecHit) (*seeding, error) {
	sd := &seeding{seeds: map[string]float64{}, similarity: map[string]float64{}, terms: []TermInfo{}}
	lex, err := e.lexicalStrengths(tx, terms, hits, sd)
	if err != nil {
		return nil, err
	}
	semS := map[string]float64{}
	for _, h := range sem {
		sd.similarity[h.id] = h.cos
		semS[h.id] = e.semanticStrength(h.cos)
		sd.semantic = math.Max(sd.semantic, semS[h.id])
	}
	for _, src := range []map[string]float64{lex, semS} {
		for id := range src {
			if s := 1 - (1-lex[id])*(1-semS[id]); s >= e.P.MinSeedStrength {
				sd.seeds[id] = s
			}
		}
	}
	return sd, nil
}
