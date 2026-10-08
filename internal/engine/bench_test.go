package engine

import (
	"fmt"
	"math/rand"
	"path/filepath"
	"testing"
	"time"

	"github.com/vhavlena/plasticity/internal/model"
	"github.com/vhavlena/plasticity/internal/plasticity"
	"github.com/vhavlena/plasticity/internal/store"
)

// BenchmarkRecall measures recall latency on a synthetic graph of 10k
// neurons with ~50k learned synapses (random small-world-ish wiring).
func BenchmarkRecall(b *testing.B) {
	const nodes, edgesPerNode = 10000, 5
	t0 := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	e, err := Open(filepath.Join(b.TempDir(), "bench.db"))
	if err != nil {
		b.Fatal(err)
	}
	defer e.Close()
	rng := rand.New(rand.NewSource(1))
	vocab := make([]string, 2000)
	for i := range vocab {
		vocab[i] = fmt.Sprintf("term%d", i)
	}
	word := func() string { return vocab[rng.Intn(len(vocab))] }
	ids := make([]string, nodes)
	if err := e.st.Write(func(tx *store.Tx) error {
		for i := range ids {
			n := &model.Node{ID: fmt.Sprintf("n%05d", i), Type: model.TypeFact,
				Title: word() + " " + word(), Content: word() + " " + word() + " " + word(),
				CreatedAt: t0, UpdatedAt: t0}
			ids[i] = n.ID
			if err := tx.InsertNode(n); err != nil {
				return err
			}
		}
		for i := range ids {
			for k := 0; k < edgesPerNode; k++ {
				j := (i + 1 + rng.Intn(50)) % nodes // mostly local wiring
				if rng.Intn(10) == 0 {
					j = rng.Intn(nodes) // plus some long-range shortcuts
				}
				if i == j {
					continue
				}
				ed := plasticity.NewAssociative(ids[i], ids[j], 0, 0, t0, e.P)
				ed.Weight = 0.2 + 0.8*rng.Float64()
				if err := tx.UpsertEdge(ed); err != nil {
					return err
				}
			}
		}
		return nil
	}); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r, err := e.Recall(RecallRequest{Query: word() + " " + word(), NoRecord: true})
		if err != nil {
			b.Fatal(err)
		}
		if i == 0 {
			b.ReportMetric(float64(r.Expanded), "expanded/op")
		}
	}
}
