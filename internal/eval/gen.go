package eval

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/vhavlena/plasticity/internal/model"
)

// GenConfig parameterizes the synthetic scenario generator.
type GenConfig struct {
	Nodes             int     `json:"nodes"`
	Topics            int     `json:"topics"`
	GroupSize         int     `json:"group_size"` // nodes used together (hidden ground truth)
	Sessions          int     `json:"sessions"`
	QueriesPerSession int     `json:"queries_per_session"`
	TypedLinkRate     float64 `json:"typed_link_rate"` // share of groups with authored links
	NoiseLinks        int     `json:"noise_links"`     // random learned edges per node (scale/distractors)
	ZipfS             float64 `json:"zipf_s"`          // query popularity skew over groups
	Seed              int64   `json:"seed"`
}

// DefaultGenConfig returns a small but non-trivial configuration.
func DefaultGenConfig() GenConfig {
	return GenConfig{Nodes: 1000, Topics: 50, GroupSize: 4, Sessions: 20, QueriesPerSession: 10,
		TypedLinkRate: 0.2, NoiseLinks: 0, ZipfS: 1.1, Seed: 1}
}

// Generate builds a synthetic scenario.
//
// Nodes belong to topics (shared topic vocabulary) and, within a topic, to
// hidden groups of GroupSize nodes that are "used together". Every node also
// has two unique words. A query names the unique words of one group member
// (the anchor) plus a topic word, and all group members are relevant: the
// anchor is found lexically, the rest are associative targets that only
// links, learning (or embeddings) can bring in. Query popularity over groups
// follows a Zipf law, so some groups recur and learning has a chance to pay
// off, as in real use.
func Generate(c GenConfig) (*Scenario, error) {
	if c.Nodes < c.GroupSize || c.GroupSize < 2 || c.Topics < 1 || c.Sessions < 1 || c.QueriesPerSession < 1 {
		return nil, fmt.Errorf("invalid generator config %+v", c)
	}
	rng := rand.New(rand.NewSource(c.Seed))
	words := newWordFactory(rng)
	global := words.n(300)
	topicVocab := make([][]string, c.Topics)
	for i := range topicVocab {
		topicVocab[i] = words.n(25)
	}
	pick := func(vocab []string, k int) []string {
		out := make([]string, k)
		for i := range out {
			out[i] = vocab[rng.Intn(len(vocab))]
		}
		return out
	}
	types := []model.NodeType{model.TypeFact, model.TypeDecision, model.TypeConcept, model.TypeProcedure, model.TypeRationale}

	s := &Scenario{
		Name:        fmt.Sprintf("synthetic-%dn-%dt-s%d", c.Nodes, c.Topics, c.Seed),
		Description: fmt.Sprintf("generated: %+v", c),
	}
	unique := make([][2]string, c.Nodes)
	var groups [][]string
	perTopic := c.Nodes / c.Topics
	for i := 0; i < c.Nodes; i++ {
		topic := min(i/max(perTopic, 1), c.Topics-1)
		u := words.n(2)
		unique[i] = [2]string{u[0], u[1]}
		key := fmt.Sprintf("n%d", i)
		title := strings.Join(append(u, pick(topicVocab[topic], 2)...), " ")
		content := strings.Join(append(pick(topicVocab[topic], 5), pick(global, 4)...), " ")
		s.Nodes = append(s.Nodes, ScenarioNode{Key: key, Type: types[rng.Intn(len(types))], Title: title, Content: content})
		if i%c.GroupSize == 0 {
			groups = append(groups, nil)
		}
		groups[len(groups)-1] = append(groups[len(groups)-1], key)
	}
	if last := groups[len(groups)-1]; len(last) < 2 { // fold a singleton tail into the previous group
		groups[len(groups)-2] = append(groups[len(groups)-2], last...)
		groups = groups[:len(groups)-1]
	}

	for _, g := range groups {
		if rng.Float64() < c.TypedLinkRate {
			s.Links = append(s.Links, ScenarioLink{Src: g[0], Dst: g[1], Kind: model.KindSupports, Weight: 0.8})
		}
	}
	if c.NoiseLinks > 0 {
		seen := map[[2]int]bool{}
		for i := 0; i < c.Nodes; i++ {
			for k := 0; k < c.NoiseLinks; k++ {
				j := rng.Intn(c.Nodes)
				a, b := min(i, j), max(i, j)
				if a == b || seen[[2]int{a, b}] {
					continue
				}
				seen[[2]int{a, b}] = true
				s.Links = append(s.Links, ScenarioLink{Src: fmt.Sprintf("n%d", a), Dst: fmt.Sprintf("n%d", b),
					Kind: model.KindAssociative, Weight: 0.05 + 0.4*rng.Float64()})
			}
		}
	}

	// Zipf popularity over a random permutation of groups.
	perm := rng.Perm(len(groups))
	cdf := make([]float64, len(groups))
	total := 0.0
	for r := range cdf {
		total += 1 / math.Pow(float64(r+1), c.ZipfS)
		cdf[r] = total
	}
	drawGroup := func() []string {
		x := rng.Float64() * total
		lo, hi := 0, len(cdf)-1
		for lo < hi {
			mid := (lo + hi) / 2
			if cdf[mid] < x {
				lo = mid + 1
			} else {
				hi = mid
			}
		}
		return groups[perm[lo]]
	}
	for si := 0; si < c.Sessions; si++ {
		sess := Session{Name: fmt.Sprintf("session-%d", si)}
		for qi := 0; qi < c.QueriesPerSession; qi++ {
			g := drawGroup()
			anchor := g[rng.Intn(len(g))]
			var idx int
			fmt.Sscanf(anchor, "n%d", &idx)
			topic := min(idx/max(perTopic, 1), c.Topics-1)
			q := unique[idx][0] + " " + unique[idx][1] + " " + pick(topicVocab[topic], 1)[0]
			sess.Queries = append(sess.Queries, Query{Q: q, Relevant: append([]string(nil), g...)})
		}
		s.Sessions = append(s.Sessions, sess)
	}
	return s, s.Validate()
}

// wordFactory makes distinct pronounceable pseudo-words.
type wordFactory struct {
	rng  *rand.Rand
	used map[string]bool
}

func newWordFactory(rng *rand.Rand) *wordFactory {
	return &wordFactory{rng: rng, used: map[string]bool{}}
}

var (
	consonants = []string{"b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z"}
	vowels     = []string{"a", "e", "i", "o", "u"}
)

func (w *wordFactory) n(k int) []string {
	out := make([]string, 0, k)
	for len(out) < k {
		var b strings.Builder
		for s := 0; s < 3; s++ {
			b.WriteString(consonants[w.rng.Intn(len(consonants))])
			b.WriteString(vowels[w.rng.Intn(len(vowels))])
		}
		if word := b.String(); !w.used[word] {
			w.used[word] = true
			out = append(out, word)
		}
	}
	return out
}
