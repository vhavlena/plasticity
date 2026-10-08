package store

import (
	"errors"
	"fmt"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/vhavlena/plasticity/internal/model"
)

var t0 = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

func openTest(t *testing.T) (*Store, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "graph.db")
	s, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s, path
}

func node(id, title, content string) *model.Node {
	return &model.Node{ID: id, Type: model.TypeFact, Title: title, Content: content,
		Tags: []string{"tag-" + id}, CreatedAt: t0, UpdatedAt: t0}
}

func mustWrite(t *testing.T, s *Store, fn func(*Tx) error) {
	t.Helper()
	if err := s.Write(fn); err != nil {
		t.Fatal(err)
	}
}

var w = [4]float64{4, 1, 1, 3}

func TestNodeCRUDAndFTSSync(t *testing.T) {
	s, _ := openTest(t)
	mustWrite(t, s, func(tx *Tx) error {
		if err := tx.InsertNode(node("a", "SQLite WAL mode", "concurrent readers")); err != nil {
			return err
		}
		return tx.InsertNode(node("b", "Badger", "single process lock"))
	})

	search := func(q string) []string {
		var ids []string
		_ = s.Read(func(tx *Tx) error {
			hits, err := tx.Search(q, w, 10)
			for _, h := range hits {
				ids = append(ids, h.Node.ID)
			}
			return err
		})
		return ids
	}
	if got := search("reader"); len(got) != 1 || got[0] != "a" { // porter stemming
		t.Fatalf("search reader: %v", got)
	}

	mustWrite(t, s, func(tx *Tx) error {
		n, err := tx.GetNode("a")
		if err != nil {
			return err
		}
		if n.Tags[0] != "tag-a" || !n.CreatedAt.Equal(t0) {
			t.Errorf("round trip lost data: %+v", n)
		}
		n.Content = "exclusive writer"
		return tx.UpdateNode(n)
	})
	if got := search("reader"); len(got) != 0 {
		t.Fatalf("FTS not updated: %v", got)
	}
	if got := search("writer"); len(got) != 1 {
		t.Fatalf("FTS missing update: %v", got)
	}
	mustWrite(t, s, func(tx *Tx) error { return tx.DeleteNode("a") })
	if got := search("writer"); len(got) != 0 {
		t.Fatalf("FTS not cleaned on delete: %v", got)
	}
	err := s.Read(func(tx *Tx) error { _, err := tx.GetNode("a"); return err })
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("want ErrNotFound, got %v", err)
	}
}

func TestSearchRanksTitleHigher(t *testing.T) {
	s, _ := openTest(t)
	mustWrite(t, s, func(tx *Tx) error {
		if err := tx.InsertNode(node("body", "unrelated", "mentions hebbian once among many other words here")); err != nil {
			return err
		}
		return tx.InsertNode(node("title", "Hebbian learning", "rule"))
	})
	_ = s.Read(func(tx *Tx) error {
		hits, err := tx.Search(`"hebbian"`, w, 10)
		if err != nil || len(hits) != 2 || hits[0].Node.ID != "title" || hits[0].Score <= hits[1].Score {
			t.Fatalf("ranking: %v %+v", err, hits)
		}
		return nil
	})
}

func TestEdgesCanonicalAndCascade(t *testing.T) {
	s, _ := openTest(t)
	mustWrite(t, s, func(tx *Tx) error {
		for _, id := range []string{"a", "b", "c"} {
			if err := tx.InsertNode(node(id, id, "")); err != nil {
				return err
			}
		}
		if err := tx.UpsertEdge(&model.Edge{Src: "b", Dst: "a", Kind: model.KindAssociative, Weight: 0.5, Stability: 7, LastUpdate: t0}); err != nil {
			return err
		}
		return tx.UpsertEdge(&model.Edge{Src: "c", Dst: "a", Kind: model.KindSupports, Weight: 0.9, Stability: 90, LastUpdate: t0})
	})
	_ = s.Read(func(tx *Tx) error {
		e, err := tx.GetEdge("a", "b", model.KindAssociative)
		if err != nil || e == nil || e.Src != "a" || e.Weight != 0.5 {
			t.Fatalf("canonical lookup: %v %+v", err, e)
		}
		if e, _ := tx.GetEdge("a", "c", model.KindSupports); e != nil {
			t.Fatal("typed edges are directed")
		}
		edges, _ := tx.EdgesOf("a", 0)
		if len(edges) != 2 || edges[0].Kind != model.KindSupports {
			t.Fatalf("EdgesOf(a) = %+v", edges)
		}
		if d, _ := tx.Degree("a"); d != 2 {
			t.Fatalf("degree(a) = %d", d)
		}
		if d, _ := tx.Degree("b"); d != 1 {
			t.Fatalf("degree(b) = %d", d)
		}
		page, _ := tx.EdgesAfter(EdgeKey{}, 1)
		page2, _ := tx.EdgesAfter(EdgeKey{page[0].Src, page[0].Dst, page[0].Kind}, 10)
		if len(page) != 1 || len(page2) != 1 {
			t.Fatalf("paging: %v %v", page, page2)
		}
		return nil
	})
	mustWrite(t, s, func(tx *Tx) error { return tx.DeleteNode("a") })
	_ = s.Read(func(tx *Tx) error {
		if st, _ := tx.Stats(); st.Edges != 0 {
			t.Fatalf("edges not cascaded: %+v", st)
		}
		return nil
	})
}

func TestAccessTrimAndTraces(t *testing.T) {
	s, _ := openTest(t)
	mustWrite(t, s, func(tx *Tx) error {
		if err := tx.InsertNode(node("a", "a", "")); err != nil {
			return err
		}
		for i := 0; i < 10; i++ {
			if err := tx.RecordAccess("a", int64(i), 3); err != nil {
				return err
			}
		}
		for i, rid := range []string{"r1", "r2"} {
			if err := tx.InsertTrace(&Trace{RecallID: rid, Session: "s", Tick: int64(i + 1),
				Entries: []TraceEntry{{NodeID: "a", Activation: 0.5, Path: []string{"seed", "a"}}}}); err != nil {
				return err
			}
		}
		return nil
	})
	_ = s.Read(func(tx *Tx) error {
		acc, _ := tx.Accesses([]string{"a"})
		if len(acc["a"]) != 3 {
			t.Fatalf("accesses not trimmed: %v", acc)
		}
		for _, a := range acc["a"] {
			if a < 7 {
				t.Fatalf("kept an old access: %v", a)
			}
		}
		trs, _ := tx.PendingTraces("s")
		if len(trs) != 2 || trs[0].RecallID != "r1" {
			t.Fatalf("traces: %+v", trs)
		}
		if p := trs[0].Entries[0].Path; len(p) != 2 || p[0] != "seed" || p[1] != "a" {
			t.Fatalf("path not round-tripped: %v", p)
		}
		return nil
	})
	mustWrite(t, s, func(tx *Tx) error { return tx.DeleteTraces([]string{"r1"}) })
	mustWrite(t, s, func(tx *Tx) error {
		trs, _ := tx.PendingTraces("s")
		if len(trs) != 1 || trs[0].RecallID != "r2" {
			t.Fatalf("delete: %+v", trs)
		}
		if n, err := tx.PurgeTraces(2); n != 0 || err != nil { // r2 is at tick 2
			t.Fatalf("purged %d %v", n, err)
		}
		n, err := tx.PurgeTraces(3)
		if n != 1 {
			t.Fatalf("purged %d", n)
		}
		return err
	})
}

func TestParams(t *testing.T) {
	s, _ := openTest(t)
	mustWrite(t, s, func(tx *Tx) error { return tx.SetParam("hop_decay", "0.5") })
	_ = s.Read(func(tx *Tx) error {
		p, err := tx.LoadParams()
		if err != nil || p.HopDecay != 0.5 || p.LearningRate != model.DefaultParams().LearningRate {
			t.Fatalf("params: %v %+v", err, p)
		}
		return nil
	})
}

// Several independent Store handles (as separate processes would have)
// writing concurrently must all succeed thanks to WAL + busy timeout.
func TestConcurrentHandles(t *testing.T) {
	_, path := openTest(t)
	const workers, perWorker = 8, 25
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			s, err := Open(path)
			if err != nil {
				errs <- err
				return
			}
			defer s.Close()
			for i := 0; i < perWorker; i++ {
				id := fmt.Sprintf("w%d-%d", w, i)
				if err := s.Write(func(tx *Tx) error { return tx.InsertNode(node(id, "concurrent", "")) }); err != nil {
					errs <- err
					return
				}
				if err := s.Read(func(tx *Tx) error { _, err := tx.Search("concurrent", [4]float64{1, 1, 1, 1}, 5); return err }); err != nil {
					errs <- err
					return
				}
			}
		}(w)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	s, _ := Open(path)
	defer s.Close()
	_ = s.Read(func(tx *Tx) error {
		st, _ := tx.Stats()
		if st.Nodes != workers*perWorker {
			t.Fatalf("nodes = %d", st.Nodes)
		}
		return nil
	})
}

func TestFireAgesEdgesExceptBetweenCoFiringNodes(t *testing.T) {
	s, _ := openTest(t)
	mustWrite(t, s, func(tx *Tx) error {
		for _, id := range []string{"a", "b", "c"} {
			if err := tx.InsertNode(node(id, id, "")); err != nil {
				return err
			}
		}
		for _, p := range [][2]string{{"a", "b"}, {"a", "c"}} {
			if err := tx.UpsertEdge(&model.Edge{Src: p[0], Dst: p[1], Kind: model.KindAssociative, Weight: 0.5, Stability: 10, LastUpdate: t0}); err != nil {
				return err
			}
		}
		if err := tx.Fire([]string{"a", "b"}); err != nil { // a and b together
			return err
		}
		return tx.Fire([]string{"a"}) // a alone
	})
	_ = s.Read(func(tx *Tx) error {
		ab, _ := tx.GetEdge("a", "b", model.KindAssociative)
		ac, _ := tx.GetEdge("a", "c", model.KindAssociative)
		if ab.SrcFires != 2 || ab.DstFires != 1 {
			t.Fatalf("fires not joined: %+v", ab)
		}
		if ab.Age() != 1 {
			t.Fatalf("a–b age = %d, want 1 (only a's solo firing)", ab.Age())
		}
		if ac.Age() != 2 {
			t.Fatalf("a–c age = %d, want 2", ac.Age())
		}
		edges, _ := tx.EdgesOf("c", 0)
		if len(edges) != 1 || edges[0].Age() != 2 {
			t.Fatalf("EdgesOf must carry fires: %+v", edges)
		}
		return nil
	})
	mustWrite(t, s, func(tx *Tx) error {
		if v, err := tx.Tick(); err != nil || v != 0 {
			t.Fatalf("tick = %d %v", v, err)
		}
		if v, err := tx.AdvanceTick(); err != nil || v != 1 {
			t.Fatalf("advance = %d %v", v, err)
		}
		return nil
	})
}
