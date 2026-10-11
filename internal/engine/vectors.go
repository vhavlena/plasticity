package engine

import (
	"container/heap"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/vhavlena/plasticity/internal/embed"
	"github.com/vhavlena/plasticity/internal/model"
	"github.com/vhavlena/plasticity/internal/paths"
	"github.com/vhavlena/plasticity/internal/store"
)

// Config keys: the embedder configuration (JSON) and the ID its embedder
// reports, under which its vectors are stored. The ID is recorded when the
// embedder is enabled, so the active model is known without loading it.
const (
	embedderKey   = "embedder"
	embedderIDKey = "embedder_id"
)

// loadEmbedConfig reads the embedder configuration (nil if embeddings are
// disabled). The model itself is loaded lazily by embedder().
func (e *Engine) loadEmbedConfig(tx *store.Tx) error {
	raw, err := tx.GetConfig(embedderKey)
	if err != nil || raw == "" {
		return err
	}
	var c embed.Config
	if err := json.Unmarshal([]byte(raw), &c); err != nil {
		return fmt.Errorf("embedder config: %w", err)
	}
	if e.embID, err = tx.GetConfig(embedderIDKey); err != nil {
		return err
	}
	if e.embID == "" {
		e.embID = c.ID()
	}
	e.embCfg = &c
	return nil
}

// activeModel is the ID of the embedder this engine uses ("" if none),
// known without loading the model.
func (e *Engine) activeModel() string {
	switch {
	case e.noEmbed:
		return ""
	case e.injected:
		return e.emb.ID()
	case e.embCfg != nil:
		return e.embID
	}
	return ""
}

// embedder returns the active embedder, loading the configured one on first
// use, or nil if embeddings are off.
func (e *Engine) embedder(ctx context.Context) (embed.Embedder, error) {
	if e.noEmbed {
		return nil, nil
	}
	if e.emb != nil || e.embCfg == nil {
		return e.emb, nil
	}
	emb, err := embed.New(ctx, *e.embCfg)
	if err != nil {
		return nil, err
	}
	e.emb, e.embOwned = emb, true
	return emb, nil
}

// releaseEmbedder closes the active embedder if the engine created it.
func (e *Engine) releaseEmbedder() {
	if e.emb != nil && e.embOwned {
		e.emb.Close()
	}
	if !e.injected {
		e.emb, e.embOwned = nil, false
	}
}

// warn records a non-fatal problem for the caller to report.
func (e *Engine) warn(format string, args ...any) {
	e.Warnings = append(e.Warnings, fmt.Sprintf(format, args...))
}

// nodeText is the text embedded for a node.
func nodeText(n *model.Node) string {
	parts := []string{n.Title, n.Content, n.Rationale}
	if len(n.Tags) > 0 {
		parts = append(parts, strings.Join(n.Tags, " "))
	}
	return strings.Join(parts, "\n")
}

func textHash(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:8])
}

// nodeVector is a computed embedding ready to be stored.
type nodeVector struct {
	model, hash string
	blob        []byte
}

// embedNode computes a node's embedding, or returns nil (with a warning) if
// embeddings are disabled or the embedder fails: a node without a vector is
// still found lexically, and `embed reindex` fills the gap later.
func (e *Engine) embedNode(ctx context.Context, n *model.Node) *nodeVector {
	emb, err := e.embedder(ctx)
	if err != nil {
		e.warn("embedding skipped for %s: %v", n.ID, err)
		return nil
	}
	if emb == nil {
		return nil
	}
	text := embed.Truncate(nodeText(n), e.P.EmbedMaxWords)
	v, err := emb.Embed(ctx, text, embed.Document)
	if err != nil {
		e.warn("embedding skipped for %s: %v", n.ID, err)
		return nil
	}
	return &nodeVector{model: emb.ID(), hash: textHash(text), blob: embed.Encode(v)}
}

// vecHit is a semantic match.
type vecHit struct {
	id  string
	cos float64
}

type vecHeap []vecHit // min-heap on cos: the root is the weakest kept hit

func (h vecHeap) Len() int           { return len(h) }
func (h vecHeap) Less(i, j int) bool { return h[i].cos < h[j].cos }
func (h vecHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *vecHeap) Push(x any)        { *h = append(*h, x.(vecHit)) }
func (h *vecHeap) Pop() any {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
}

// embedQuery embeds a recall query, or returns nil (with a warning) if
// embeddings are disabled or fail, in which case recall is lexical only.
func (e *Engine) embedQuery(ctx context.Context, query string) []float32 {
	emb, err := e.embedder(ctx)
	if err != nil {
		e.warn("semantic search disabled for this recall: %v", err)
		return nil
	}
	if emb == nil {
		return nil
	}
	v, err := emb.Embed(ctx, embed.Truncate(query, e.P.EmbedMaxWords), embed.Query)
	if err != nil {
		e.warn("semantic search disabled for this recall: %v", err)
		return nil
	}
	return v
}

// semanticSearch returns the k nodes most similar to q (brute force over
// all vectors of the current model), best first.
func (e *Engine) semanticSearch(tx *store.Tx, q []float32, k int) ([]vecHit, error) {
	if q == nil {
		return nil, nil
	}
	h := &vecHeap{}
	err := tx.ScanVectors(e.emb.ID(), func(id string, blob []byte) error {
		v, err := embed.Decode(blob)
		if err != nil {
			return fmt.Errorf("vector of %s: %w", id, err)
		}
		c := embed.Dot(q, v)
		if h.Len() < k {
			heap.Push(h, vecHit{id, c})
		} else if c > (*h)[0].cos {
			(*h)[0] = vecHit{id, c}
			heap.Fix(h, 0)
		}
		return nil
	})
	out := make([]vecHit, h.Len())
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = heap.Pop(h).(vecHit)
	}
	return out, err
}

// ---------------------------------------------------------------- management

// EmbedStatus describes the embedding index.
type EmbedStatus struct {
	// Config is the embedder stored in the database (nil if none).
	Config *embed.Config `json:"config,omitempty"`
	// Active is the embedder this engine uses ("" if semantic search is
	// off, e.g. with --no-embed).
	Active  string `json:"active"`
	Indexed int    `json:"indexed"`
	Missing int    `json:"missing"`
	// Models counts stored vectors per embedder; inactive ones can be
	// switched back to without re-indexing, or removed with PruneVectors.
	Models map[string]int `json:"models"`
	// Providers lists the registered embedder backends.
	Providers []string `json:"providers"`
}

// EmbedStatus reports the embedder configuration and index coverage.
func (e *Engine) EmbedStatus() (*EmbedStatus, error) {
	st := &EmbedStatus{Config: e.embCfg, Active: e.activeModel(), Providers: embed.Providers()}
	err := e.st.Read(func(tx *store.Tx) error {
		var err error
		if st.Models, err = tx.VectorModels(); err != nil {
			return err
		}
		if st.Active == "" {
			return nil
		}
		st.Indexed, st.Missing, err = tx.VectorCounts(st.Active)
		return err
	})
	return st, err
}

// ModelsDir is where downloaded models are kept (shared by all graphs).
func (e *Engine) ModelsDir() string { return paths.ModelsDir() }

// EnableEmbedding configures a registered embedder: it runs the provider's
// setup (e.g. model download), verifies the embedder works, stores the
// configuration and indexes the nodes missing a vector from it. Vectors of
// other embedders are kept, so switching back is instant.
func (e *Engine) EnableEmbedding(ctx context.Context, c embed.Config) (*ReindexReport, error) {
	c, err := embed.Prepare(ctx, c, e.ModelsDir())
	if err != nil {
		return nil, err
	}
	emb, err := embed.New(ctx, c)
	if err != nil {
		return nil, err
	}
	if _, err := emb.Embed(ctx, "probe", embed.Query); err != nil {
		emb.Close()
		return nil, fmt.Errorf("embedder does not work: %w", err)
	}
	raw, _ := json.Marshal(c)
	if err := e.st.Write(func(tx *store.Tx) error {
		if err := tx.SetConfig(embedderKey, string(raw)); err != nil {
			return err
		}
		return tx.SetConfig(embedderIDKey, emb.ID())
	}); err != nil {
		emb.Close()
		return nil, err
	}
	e.releaseEmbedder()
	e.embCfg, e.embID, e.emb, e.embOwned, e.injected, e.noEmbed = &c, emb.ID(), emb, true, false, false
	return e.Reindex(ctx, false)
}

// DisableEmbedding removes the stored embedder configuration. Vectors are
// kept (see PruneVectors).
func (e *Engine) DisableEmbedding() error {
	if err := e.st.Write(func(tx *store.Tx) error {
		if err := tx.SetConfig(embedderKey, ""); err != nil {
			return err
		}
		return tx.SetConfig(embedderIDKey, "")
	}); err != nil {
		return err
	}
	e.releaseEmbedder()
	e.embCfg, e.embID = nil, ""
	return nil
}

// PruneVectors deletes the vectors of every embedder except the active one
// (all vectors if none is active).
func (e *Engine) PruneVectors() (int64, error) {
	var n int64
	err := e.st.Write(func(tx *store.Tx) error {
		var err error
		n, err = tx.DeleteVectorsNotOf(e.activeModel())
		return err
	})
	return n, err
}

// ReindexReport summarizes a reindex.
type ReindexReport struct {
	Embedded  int `json:"embedded"`
	UpToDate  int `json:"up_to_date"`
	Failed    int `json:"failed"`
	TotalSeen int `json:"total"`
}

// Reindex embeds every node that has no current vector (or whose text
// changed since); with all=true it re-embeds every node.
func (e *Engine) Reindex(ctx context.Context, all bool) (*ReindexReport, error) {
	emb, err := e.embedder(ctx)
	if err != nil {
		return nil, err
	}
	if emb == nil {
		return nil, fmt.Errorf("embeddings are disabled (run `plasticity embed enable`)")
	}
	rep := &ReindexReport{}
	after := ""
	for {
		var ids []string
		var nodes map[string]*model.Node
		hashes := map[string]string{}
		if err := e.st.Read(func(tx *store.Tx) error {
			var err error
			if ids, err = tx.NodeIDsAfter(after, 100); err != nil {
				return err
			}
			if nodes, err = tx.GetNodes(ids); err != nil {
				return err
			}
			for _, id := range ids {
				if hashes[id], err = tx.VectorHash(id, emb.ID()); err != nil {
					return err
				}
			}
			return nil
		}); err != nil {
			return rep, err
		}
		for _, id := range ids {
			rep.TotalSeen++
			n := nodes[id]
			if !all && hashes[id] == textHash(embed.Truncate(nodeText(n), e.P.EmbedMaxWords)) {
				rep.UpToDate++
				continue
			}
			vec := e.embedNode(ctx, n)
			if vec == nil {
				rep.Failed++
				continue
			}
			if err := e.st.Write(func(tx *store.Tx) error {
				return tx.PutVector(id, vec.model, vec.hash, vec.blob)
			}); err != nil {
				return rep, err
			}
			rep.Embedded++
		}
		if len(ids) < 100 {
			return rep, nil
		}
		after = ids[len(ids)-1]
	}
}
