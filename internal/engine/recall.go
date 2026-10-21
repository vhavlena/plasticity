package engine

import (
	"context"
	"fmt"
	"math"
	"slices"
	"sort"

	"github.com/oklog/ulid/v2"

	"github.com/vhavlena/plasticity/internal/model"
	"github.com/vhavlena/plasticity/internal/plasticity"
	"github.com/vhavlena/plasticity/internal/retrieval"
	"github.com/vhavlena/plasticity/internal/store"
)

func (e *Engine) bm25Weights() [4]float64 {
	return [4]float64{e.P.BM25Title, e.P.BM25Content, e.P.BM25Rationale, e.P.BM25Tags}
}

// Search runs lexical BM25 search only (no spreading, no learning).
func (e *Engine) Search(query string, limit int) ([]store.SearchHit, error) {
	var hits []store.SearchHit
	err := e.st.Read(func(tx *store.Tx) error {
		var err error
		if hits, err = tx.Search(MatchExpr(query), e.bm25Weights(), limit); err != nil {
			return err
		}
		nodes := make([]*model.Node, len(hits))
		for i, h := range hits {
			nodes[i] = h.Node
		}
		return fillSuccessorTitles(tx, nodes...)
	})
	if hits == nil {
		hits = []store.SearchHit{}
	}
	return hits, err
}

// txGraph exposes the stored graph, with decayed weights, to retrieval.
type txGraph struct {
	tx *store.Tx
	p  model.Params
}

// Neighbors returns the node's strongest EdgesPerNode edges and its fan load:
// their summed effective weight plus, if the list was truncated, the stored
// weight of the remaining edges (an upper bound of their effective weight).
func (g txGraph) Neighbors(id string) ([]retrieval.Neighbor, float64, error) {
	edges, err := g.tx.EdgesOf(id, g.p.EdgesPerNode)
	if err != nil {
		return nil, 0, err
	}
	load, stored := 0.0, 0.0
	out := make([]retrieval.Neighbor, 0, len(edges))
	for _, ed := range edges {
		stored += ed.Weight
		if w := plasticity.EffectiveWeight(ed, g.p); w > 0 {
			load += w
			out = append(out, retrieval.Neighbor{ID: ed.Other(id), Weight: w, Kind: string(ed.Kind)})
		}
	}
	if len(edges) == g.p.EdgesPerNode {
		total, err := g.tx.WeightSum(id)
		if err != nil {
			return nil, 0, err
		}
		load += max(0, total-stored)
	}
	return out, load, nil
}

// RecallRequest parameterizes a recall. Zero values select the defaults.
type RecallRequest struct {
	Query     string
	Session   string // if set, a trace is stored for later reinforcement
	Limit     int
	Threshold float64
	HopDecay  float64
	Rerank    string // "" or "ppr"
	NoRecord  bool   // do not record accesses or traces
	// Project is the project the recall is made in; neurons of other
	// projects only seed weaker (see ForeignProjectFactor). If empty, the
	// session's project is used.
	Project string
	// Auto marks a recall injected automatically by the prompt hook, as
	// opposed to one Claude issued itself (explicit).
	Auto bool
	// MinConfidence: a recall whose query confidence stays below it returns
	// no hits and records nothing (no access, firing or trace).
	MinConfidence float64
	// ExcludeTypes leaves neurons of these types out of the hits; they
	// still carry activation to their neighbors.
	ExcludeTypes []model.NodeType
	// ExcludeSuperseded leaves superseded neurons out of the hits (they still
	// carry activation to their neighbors): each one's activation goes to its
	// current successor instead, so a query worded like the old neuron finds
	// the new one. Without it, superseded neurons stay visible with their
	// score scaled by superseded_factor.
	ExcludeSuperseded bool
}

// RecallHit is one recalled neuron.
type RecallHit struct {
	Node       *model.Node `json:"node"`
	Score      float64     `json:"score"`
	Activation float64     `json:"activation"`
	BaseLevel  *float64    `json:"base_level,omitempty"`
	IsSeed     bool        `json:"is_seed"`
	// Similarity is the embedding cosine to the query, for semantic
	// candidates.
	Similarity *float64 `json:"similarity,omitempty"`
	Hops       int      `json:"hops"`
	Path       []string `json:"path"`
	PathTitles []string `json:"path_titles"`
	// Supersedes are the titles of superseded neurons whose activation was
	// redirected to this one (with ExcludeSuperseded).
	Supersedes []string `json:"supersedes,omitempty"`
}

// RecallResult is the answer to a recall.
type RecallResult struct {
	// RecallID is set when a learning trace was stored.
	RecallID string `json:"recall_id,omitempty"`
	// TraceSkipped explains why no trace was stored despite a session.
	TraceSkipped string `json:"trace_skipped,omitempty"`
	// Suppressed explains why a recall with seeds returned no hits.
	Suppressed string `json:"suppressed,omitempty"`
	Query      string `json:"query"`
	// Confidence is max(LexicalConfidence, SemanticConfidence); it gates
	// learning.
	Confidence float64 `json:"confidence"`
	// LexicalConfidence is the IDF-weighted share of the query explained by
	// the BM25 matches; SemanticConfidence is the strongest calibrated
	// embedding similarity (0 without embeddings).
	LexicalConfidence  float64     `json:"lexical_confidence"`
	SemanticConfidence float64     `json:"semantic_confidence"`
	Terms              []TermInfo  `json:"terms"`
	Seeds              int         `json:"seeds"`
	Reached            int         `json:"reached"`
	Expanded           int         `json:"expanded"`
	Hits               []RecallHit `json:"hits"`
}

// Recall finds calibrated lexical (BM25) and semantic (embedding) seeds,
// spreads activation through the
// weighted graph, ranks by λ1·graph + λ2·σ(B) + λ3·importance, and (unless
// NoRecord) records accesses and, if the query confidence is high enough, a
// trace for Hebbian learning.
func (e *Engine) Recall(req RecallRequest) (*RecallResult, error) {
	p := e.P
	if req.Limit <= 0 {
		req.Limit = p.ResultLimit
	}
	cfg := retrieval.SpreadConfig{
		HopDecay: p.HopDecay, Threshold: p.FiringThreshold,
		FanK0: p.FanK0, FanBeta: p.FanBeta, MaxNodes: p.MaxNodes, MaxHops: p.MaxHops,
	}
	if req.Threshold > 0 {
		cfg.Threshold = req.Threshold
	}
	if req.HopDecay > 0 {
		cfg.HopDecay = math.Min(1, req.HopDecay)
	}
	now := timestamp()
	res := &RecallResult{Query: req.Query, Terms: []TermInfo{}, Hits: []RecallHit{}}
	qvec := e.embedQuery(context.Background(), req.Query) // nil: lexical only

	err := e.st.Read(func(tx *store.Tx) error {
		terms := queryTerms(req.Query)
		hits, err := tx.Search(MatchExpr(req.Query), e.bm25Weights(), p.SeedLimit)
		if err != nil {
			return err
		}
		tick, err := tx.Tick()
		if err != nil {
			return err
		}
		var lexCos map[string]float64 // cosines of the lexical hits, to corroborate them
		if qvec != nil && p.LexicalCosHigh > 0 {
			lexCos = make(map[string]float64, len(hits))
			for _, h := range hits {
				lexCos[h.Node.ID] = math.NaN() // NaN: the neuron has no vector
			}
		}
		sem, err := e.semanticSearch(tx, qvec, p.SeedLimit, lexCos)
		if err != nil {
			return err
		}
		sd, err := e.computeSeeds(tx, terms, hits, sem, lexCos)
		if err != nil {
			return err
		}
		if err := e.applyProjectAffinity(tx, &req, sd.seeds); err != nil {
			return err
		}
		seeds := sd.seeds
		res.Confidence, res.Terms, res.Seeds = sd.confidence(), sd.terms, len(seeds)
		res.LexicalConfidence, res.SemanticConfidence = sd.lexical, sd.semantic
		if len(seeds) == 0 {
			return nil
		}
		if res.Confidence < req.MinConfidence {
			res.Suppressed = fmt.Sprintf("query confidence %.2f below %.2f", res.Confidence, req.MinConfidence)
			return nil
		}

		spread, err := retrieval.Spread(txGraph{tx, p}, seeds, cfg)
		if err != nil {
			return err
		}
		res.Reached, res.Expanded = len(spread.Activations), spread.Expanded

		graphScore := make(map[string]float64, len(spread.Activations))
		ids := make([]string, 0, len(spread.Activations))
		for id, a := range spread.Activations {
			ids = append(ids, id)
			graphScore[id] = a.Score
		}
		sort.Strings(ids)
		if req.Rerank == "ppr" {
			pr := retrieval.PersonalizedPageRank(ids, spread.Edges, seeds, p.PPRRestart, 1e-8, 200)
			top := 0.0
			for _, v := range pr {
				top = math.Max(top, v)
			}
			for id, v := range pr {
				graphScore[id] = v / top
			}
		}

		nodes, err := tx.GetNodes(ids)
		if err != nil {
			return err
		}
		act := spread.Activations
		redirected := map[string][]string{} // successor → superseded neurons it took activation from
		if req.ExcludeSuperseded {
			if act, ids, err = redirectSuperseded(tx, act, ids, graphScore, nodes, redirected); err != nil {
				return err
			}
		}
		acc, err := tx.Accesses(ids)
		if err != nil {
			return err
		}
		for _, id := range ids {
			n, a := nodes[id], act[id]
			if n == nil || slices.Contains(req.ExcludeTypes, n.Type) || req.ExcludeSuperseded && n.SupersededBy != "" {
				continue
			}
			b := plasticity.BaseLevel(acc[id], tick, p.ActrDecay)
			h := RecallHit{
				Node: n, Activation: a.Score, IsSeed: a.IsSeed, Hops: a.Hops, Path: a.Path,
				Score: p.ScoreGraph*graphScore[id] + p.ScoreBaseLevel*plasticity.Sigmoid(b) + p.ScoreImportance*n.Importance,
			}
			if n.SupersededBy != "" {
				h.Score *= p.SupersededFactor
			}
			for _, old := range redirected[id] {
				h.Supersedes = append(h.Supersedes, nodes[old].Title)
			}
			if len(acc[id]) > 0 {
				h.BaseLevel = &b
			}
			if c, ok := sd.similarity[id]; ok {
				h.Similarity = &c
			}
			for _, pid := range a.Path {
				if pn := nodes[pid]; pn != nil {
					h.PathTitles = append(h.PathTitles, pn.Title)
				}
			}
			res.Hits = append(res.Hits, h)
		}
		hitNodes := make([]*model.Node, len(res.Hits))
		for i, h := range res.Hits {
			hitNodes[i] = h.Node
		}
		if err := fillSuccessorTitles(tx, hitNodes...); err != nil {
			return err
		}
		sort.Slice(res.Hits, func(i, j int) bool {
			if res.Hits[i].Score != res.Hits[j].Score {
				return res.Hits[i].Score > res.Hits[j].Score
			}
			return res.Hits[i].Node.ID < res.Hits[j].Node.ID
		})
		if len(res.Hits) > req.Limit {
			res.Hits = res.Hits[:req.Limit]
		}
		return nil
	})
	if err != nil || req.NoRecord || len(res.Hits) == 0 {
		return res, err
	}

	// Learning gate: only confident recalls, and only strongly activated
	// nodes, may later rewire the graph.
	tr := &store.Trace{Session: req.Session}
	if req.Session != "" {
		if res.Confidence < p.LearnMinConfidence {
			res.TraceSkipped = "low query confidence"
		} else {
			for _, h := range res.Hits {
				if h.Activation >= p.LearnMinActivation {
					tr.Entries = append(tr.Entries, store.TraceEntry{NodeID: h.Node.ID, Activation: h.Activation, Path: h.Path})
				}
			}
			if len(tr.Entries) == 0 {
				res.TraceSkipped = "no strongly activated nodes"
			} else {
				tr.RecallID = ulid.MustNew(ulid.Timestamp(now), ulid.DefaultEntropy()).String()
				res.RecallID = tr.RecallID
			}
		}
	}
	// A recorded recall advances the global clock. Only a confident recall
	// is evidence about its hits: it fires them (which ages their synapses
	// to neurons that did not fire) and records a light access for each hit
	// that is not foreign to the recall's project. A low-confidence recall
	// changes nothing else, so noisy recalls neither age true links nor
	// feed the ACT-R base level.
	confident := res.Confidence >= p.LearnMinConfidence
	err = e.st.Write(func(tx *store.Tx) error {
		tick, err := tx.AdvanceTick()
		if err != nil {
			return err
		}
		tr.Tick = tick
		if !confident {
			return nil
		}
		ids := make([]string, len(res.Hits))
		for i, h := range res.Hits {
			ids[i] = h.Node.ID
		}
		foreign, err := e.foreignSet(tx, req.Project, ids)
		if err != nil {
			return err
		}
		ids = slices.DeleteFunc(ids, func(id string) bool { return foreign[id] })
		if err := tx.Fire(ids); err != nil {
			return err
		}
		for _, h := range res.Hits {
			if foreign[h.Node.ID] {
				continue
			}
			h.Node.Fires++
			if err := tx.RecordAccess(h.Node.ID, tick, p.RecallAccessWeight, p.MaxAccessRecords); err != nil {
				return err
			}
			if h.Node.Dormant { // recalling a dormant memory reactivates it
				if err := tx.SetDormant(h.Node.ID, false); err != nil {
					return err
				}
				h.Node.Dormant = false
			}
		}
		if tr.RecallID == "" {
			return nil
		}
		return tx.InsertTrace(tr)
	})
	return res, err
}

// redirectSuperseded moves the activation of every reached superseded neuron
// to its current successor: the successor's activation (and graph score)
// becomes the maximum of its own and the old neuron's, and if that came from
// the old neuron its path is the old one's path followed by the successor.
// Successors not reached by spreading are added to ids (and nodes).
// redirected collects, per successor, the superseded neurons it absorbed.
func redirectSuperseded(tx *store.Tx, act map[string]*retrieval.Activation, ids []string, graphScore map[string]float64,
	nodes map[string]*model.Node, redirected map[string][]string) (map[string]*retrieval.Activation, []string, error) {
	out := make(map[string]*retrieval.Activation, len(act))
	for id, a := range act {
		out[id] = a
	}
	for _, id := range ids {
		n := nodes[id]
		if n == nil || n.SupersededBy == "" {
			continue
		}
		s, err := currentSuccessor(tx, id, nodes)
		if err != nil {
			return nil, nil, err
		}
		if s == "" {
			continue // broken or cyclic chain: nothing to redirect to
		}
		redirected[s] = append(redirected[s], id)
		old := act[id]
		if cur, ok := out[s]; !ok || cur.Score < old.Score {
			if !ok {
				ids = append(ids, s)
			}
			out[s] = &retrieval.Activation{ID: s, Score: old.Score, Seed: old.Seed, Hops: old.Hops + 1,
				Path: append(slices.Clone(old.Path), s), IsSeed: ok && cur.IsSeed}
		}
		graphScore[s] = math.Max(graphScore[s], graphScore[id])
	}
	sort.Strings(ids)
	return out, ids, nil
}

// GlobalTag marks a neuron as knowledge shared by all projects (preferences,
// tool tips, shared infrastructure): it is never foreign.
const GlobalTag = "global"

// applyProjectAffinity scales the seeds that are foreign to the recall's
// project by ForeignProjectFactor (ForeignProjectFactorExplicit for recalls
// Claude issued) and drops those that fall below MinSeedStrength. A neuron is foreign if it has project records, none of
// them for this project, it is used in fewer than SharedMinProjects
// projects, and it is not tagged global. Neurons without project records
// (legacy, imported) are never foreign. The project defaults to the
// session's.
func (e *Engine) applyProjectAffinity(tx *store.Tx, req *RecallRequest, seeds map[string]float64) error {
	if req.Project == "" && req.Session != "" { // also used when recording the recall
		s, err := tx.GetSession(req.Session)
		if err != nil {
			return err
		}
		if s != nil {
			req.Project = s.Project
		}
	}
	factor := e.P.ForeignProjectFactor
	if !req.Auto {
		factor = e.P.ForeignProjectFactorExplicit
	}
	if len(seeds) == 0 || factor >= 1 || req.Project == "" {
		return nil
	}
	ids := make([]string, 0, len(seeds))
	for id := range seeds {
		ids = append(ids, id)
	}
	foreign, err := e.foreignSet(tx, req.Project, ids)
	if err != nil {
		return err
	}
	for id := range foreign {
		// The scaled seed faces MinSeedStrength again: foreign knowledge
		// needs 1/factor times the evidence to seed at all.
		if seeds[id] *= factor; seeds[id] < e.P.MinSeedStrength {
			delete(seeds, id)
		}
	}
	return nil
}

// foreignSet returns which of ids are foreign to project (none if project
// is ""): neurons with project records, none for project, used in fewer
// than SharedMinProjects projects, and not tagged global.
func (e *Engine) foreignSet(tx *store.Tx, project string, ids []string) (map[string]bool, error) {
	foreign := map[string]bool{}
	if project == "" || len(ids) == 0 {
		return foreign, nil
	}
	projects, err := tx.NodeProjects(ids)
	if err != nil {
		return nil, err
	}
	var candidates []string
	for _, id := range ids {
		if foreignByProjects(projects[id], project, e.P.SharedMinProjects) {
			candidates = append(candidates, id)
		}
	}
	if len(candidates) == 0 {
		return foreign, nil
	}
	nodes, err := tx.GetNodes(candidates)
	if err != nil {
		return nil, err
	}
	for _, id := range candidates {
		if n := nodes[id]; n != nil && !slices.Contains(n.Tags, GlobalTag) {
			foreign[id] = true
		}
	}
	return foreign, nil
}

// foreignByProjects reports whether a neuron with these project records is
// foreign to project: it has records, none for project, and is used in
// fewer than sharedMin projects.
func foreignByProjects(ps []store.NodeProject, project string, sharedMin int) bool {
	if len(ps) == 0 {
		return false
	}
	used := 0
	for _, p := range ps {
		if p.Project == project {
			return false
		}
		if p.Uses > 0 {
			used++
		}
	}
	return used < sharedMin
}
