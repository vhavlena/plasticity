package engine

import (
	"math"
	"regexp"
	"strings"

	"github.com/vhavlena/plasticity/internal/store"
)

var wordRE = regexp.MustCompile(`[\p{L}\p{N}]+`)

var stopwords = map[string]bool{}

// The NLTK English stopword list plus conversational filler common in
// prompts: acknowledgements and the follow-ups of an ongoing task ("how is
// it going?", "go with item 2", "show me the example", "ok, it makes
// sense"). Their words are generic, but in a small graph they are rare
// enough to look like strong evidence (and Porter stemming turns "going"
// into the Go language). Words that carry a topic ("build", "run", "test",
// "fix") are kept.
func init() {
	for _, w := range strings.Fields(`i me my myself we our ours ourselves you your yours yourself yourselves
		he him his himself she her hers herself it its itself they them their theirs themselves what which
		who whom this that these those am is are was were be been being have has had having do does did
		doing a an the and but if or because as until while of at by for with about against between into
		through during before after above below to from up down in out on off over under again further
		then once here there when where why how all any both each few more most other some such no nor not
		only own same so than too very s t can will just don should now d ll m o re ve y ain aren couldn
		didn doesn hadn hasn haven isn ma mightn mustn needn shan shouldn wasn weren won wouldn
		also could would might must shall may please want need let lets get got ok okay yes thanks thank
		go goes going gone went ahead item items option options first second third next last
		show example examples done makes sense keep start continue proceed explain briefly
		simple terms suggest mean think guess sure fine great good right`) {
		stopwords[w] = true
	}
}

// IsStopword reports whether w (lower-case) is ignored by search.
func IsStopword(w string) bool { return stopwords[w] }

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
//	C   = Σ_{t matched by some hit} idf_t / (Σ_{t known} idf_t + w_abs · Σ_{t unknown} idf(N, 1))
//
// C (query confidence) is the IDF-weighted share of the query that the hits
// explain. Words that occur nowhere in the graph count against it with weight
// w_abs, each weighted like a word occurring once: an unseen word is not more
// informative than the rarest known one. (Using idf(N, 0) would let a few
// filler words of a conversational prompt swamp the matched terms, especially
// in small graphs.) The relative BM25 factor ranks hits within one query; C makes the
// strength comparable across queries, so a query whose best match covers only
// an unimportant fraction of it produces weak seeds instead of full-strength
// ones.
//
// In graphs of at least MinEvidenceNodes neurons, a hit seeds only if the
// rarest term it matched has IDF ≥ MinMatchedIDFShare · idf(N, N/10): a match
// on generic words alone ("use", "project", "build") is no evidence, however
// well it covers a short query.
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
	rarest := map[string]float64{} // hit → highest IDF among the terms it matched
	matched, total := 0.0, 0.0
	for _, t := range terms {
		df, err := tx.DocFreq(quote(t))
		if err != nil {
			return nil, err
		}
		ti := TermInfo{Term: t, DocFreq: df, IDF: idf(n, max(df, 1))}
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
				for id := range among {
					rarest[id] = math.Max(rarest[id], ti.IDF)
				}
			}
		}
		sd.terms = append(sd.terms, ti)
	}
	if total > 0 {
		sd.lexical = matched / total
	}
	minIDF := 0.0
	if n >= e.P.MinEvidenceNodes {
		minIDF = e.P.MinMatchedIDFShare * idf(n, n/10)
	}
	for _, h := range hits {
		if rarest[h.Node.ID] >= minIDF {
			strengths[h.Node.ID] = h.Score / hits[0].Score * sd.lexical
		}
	}
	return strengths, nil
}

// corroboration is how far the embedding cosine of a lexical hit confirms
// the keyword match: clamp((cos − low)/(high − low)) with LexicalCosLow and
// LexicalCosHigh. The bounds sit below the semantic seed calibration
// (VectorCosLow): a hit need not be semantically close enough to seed on its
// own, only not unrelated. Measured on the real-use scenarios, lexical seeds
// of prompts that should recall nothing have a median cosine of 0.08, those
// of relevant neurons 0.41.
func (e *Engine) corroboration(cos float64) float64 {
	return math.Max(0, math.Min(1, (cos-e.P.LexicalCosLow)/(e.P.LexicalCosHigh-e.P.LexicalCosLow)))
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
//
// With embeddings, lexCos holds the cosine of each lexical hit to the query
// (NaN: no vector), and a lexical strength first counts only as far as the
// embedding corroborates it (see corroboration). The lexical confidence is
// then the strongest corroborated lexical strength: a query whose words all
// matched, but only by coincidence, is not a confident one.
func (e *Engine) computeSeeds(tx *store.Tx, terms []string, hits []store.SearchHit, sem []vecHit, lexCos map[string]float64) (*seeding, error) {
	sd := &seeding{seeds: map[string]float64{}, similarity: map[string]float64{}, terms: []TermInfo{}}
	lex, err := e.lexicalStrengths(tx, terms, hits, sd)
	if err != nil {
		return nil, err
	}
	if lexCos != nil {
		best := 0.0
		for id, l := range lex {
			if c := lexCos[id]; !math.IsNaN(c) {
				lex[id] = l * e.corroboration(c)
			}
			best = math.Max(best, lex[id])
		}
		sd.lexical = math.Min(sd.lexical, best)
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
