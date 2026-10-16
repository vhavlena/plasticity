package eval

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/vhavlena/plasticity/internal/engine"
	"github.com/vhavlena/plasticity/internal/model"
	"github.com/vhavlena/plasticity/internal/store"
)

// eventLog builds an event log; IDs and timestamps increase.
type eventLog struct {
	evs []store.Event
	at  time.Time
}

func (l *eventLog) add(session, kind string, payload any) {
	raw, _ := json.Marshal(payload)
	if l.at.IsZero() {
		l.at = time.Date(2026, 10, 1, 9, 0, 0, 0, time.Local)
	}
	l.at = l.at.Add(time.Minute)
	l.evs = append(l.evs, store.Event{ID: int64(len(l.evs) + 1), At: l.at, Session: session, Kind: kind, Payload: raw})
}

func (l *eventLog) prompt(session, prompt string, conf float64, hits ...string) {
	l.add(session, engine.EventPrompt, map[string]string{"prompt": prompt})
	hs := make([]map[string]any, len(hits))
	for i, h := range hits {
		hs[i] = map[string]any{"id": h, "seed": i == 0}
	}
	l.add(session, engine.EventRecall, map[string]any{"prompt": prompt, "hits": hs, "confidence": conf,
		"latency_ms": 2.0 + float64(len(l.evs)), "injected_chars": 100 * len(hits)})
}

// claude logs a recall Claude ran itself (no prompt event).
func (l *eventLog) claude(session, query string, conf float64, hits ...string) {
	hs := make([]map[string]any, len(hits))
	for i, h := range hits {
		hs[i] = map[string]any{"id": h}
	}
	l.add(session, engine.EventRecall, map[string]any{"source": engine.RecallSourceClaude, "query": query,
		"hits": hs, "confidence": conf, "latency_ms": 3.0})
}

func (l *eventLog) used(session string, created int, ids ...string) {
	l.add(session, engine.EventUsed, map[string]any{"ids": ids, "report": engine.LearnReport{Created: created}})
}

// twoSessions: S1 recalls A,B and uses A, creates N; S2 recalls N, then
// Claude recalls C itself, and both are used; a duplicate is refused; a nudge
// in S2 is answered.
func twoSessions() *eventLog {
	l := &eventLog{}
	l.prompt("S1", "how do we store the graph", 0.9, "A", "B")
	l.used("S1", 0, "A")
	l.add("S1", engine.EventAdd, map[string]any{"id": "N", "type": "decision", "title": "new"})
	l.prompt("S1", "and what about vectors", 0.1) // empty recall
	l.add("S1", engine.EventEpisode, map[string]any{"id": "E1"})
	l.at = l.at.Add(24 * time.Hour)
	l.prompt("S2", "the new decision please", 0.5, "N", "B")
	l.claude("S2", "cache invalidation rule", 0.7, "C")
	l.add("S2", engine.EventNudge, map[string]int{"tool_calls": 7})
	l.used("S2", 1, "N", "C")
	l.add("S2", engine.EventAdd, map[string]any{"title": "new", "duplicate_of": "N"})
	l.add("S2", engine.EventConsolidate, engine.ConsolidateReport{})
	l.add("", engine.EventAdd, map[string]any{"id": "M"})
	l.add("", engine.EventError, map[string]string{"op": "setup"})
	return l
}

func TestAnalyzeUsage(t *testing.T) {
	u := AnalyzeUsage(twoSessions().evs)
	check := func(name string, got, want float64) {
		t.Helper()
		if math.Abs(got-want) > 1e-9 {
			t.Errorf("%s = %v, want %v", name, got, want)
		}
	}
	check("sessions", float64(u.Sessions), 2)
	check("prompts", float64(u.Prompts), 3)
	check("recalls", float64(u.Recalls), 4)
	check("empty rate", u.EmptyRate, 1.0/4)
	check("mean hits", u.MeanHits, 5.0/4)
	check("mean confidence", u.MeanConf, 0.55)
	check("bucket 0.6-0.8", float64(u.Confidence["0.6-0.8"]), 1)
	check("bucket <0.25", float64(u.Confidence["<0.25"]), 1)
	check("bucket 0.4-0.6", float64(u.Confidence["0.4-0.6"]), 1)
	check("bucket >=0.8", float64(u.Confidence[">=0.8"]), 1)
	// Injected pairs: S1/A, S1/B, S2/N, S2/B by prompt recalls, S2/C by
	// Claude's recall; used: S1/A, S2/N, S2/C.
	check("injected", float64(u.Injected), 5)
	check("used of injected", float64(u.UsedOfInj), 3)
	check("usage rate", u.UsageRate, 0.6)
	check("used other", float64(u.UsedOther), 0)
	pr, cl := u.BySource[engine.RecallSourcePrompt], u.BySource[engine.RecallSourceClaude]
	if pr == nil || cl == nil {
		t.Fatalf("by source: %+v", u.BySource)
	}
	check("prompt recalls", float64(pr.Recalls), 3)
	check("prompt usage rate", pr.UsageRate, 0.5)
	check("claude recalls", float64(cl.Recalls), 1)
	check("claude usage rate", cl.UsageRate, 1)
	check("useful recalls", u.RecallsUseful, 1)
	check("created", float64(u.Created), 1)
	check("created outside", float64(u.CreatedOutside), 1)
	check("duplicates", float64(u.Duplicates), 1)
	check("duplicate rate", u.DuplicateRate, 0.5)
	check("created per session", u.CreatedPerSession, 0.5)
	check("created recalled later", float64(u.CreatedRecalled), 1)
	check("created used later", float64(u.CreatedUsed), 1)
	check("nudge response", u.NudgeResponse, 1)
	check("links created", float64(u.LinksCreated), 1)
	check("consolidations", float64(u.Consolidated), 1)
	check("episodes", float64(u.Episodes), 1)
	if u.Errors["setup"] != 1 {
		t.Errorf("errors: %v", u.Errors)
	}
	if len(u.Days) != 2 || u.Days[0].Recalls != 2 || u.Days[1].Created != 0 {
		t.Errorf("days: %+v", u.Days)
	}
	var b strings.Builder
	if err := u.WriteText(&b); err != nil || !strings.Contains(b.String(), "usage rate") {
		t.Errorf("text report: %v\n%s", err, b.String())
	}
	b.Reset()
	AnalyzeUsage(nil).WriteText(&b)
	if !strings.Contains(b.String(), "no logged activity") {
		t.Errorf("empty report: %q", b.String())
	}
}

func TestExportSessions(t *testing.T) {
	l := twoSessions()
	// A turn whose used report names a neuron created later in the turn.
	l.prompt("S3", "remember this new thing", 0.3, "A")
	l.add("S3", engine.EventAdd, map[string]any{"id": "P"})
	l.used("S3", 0, "P", "gone")

	nodes := []*model.Node{}
	for _, id := range []string{"A", "B", "C", "N", "P"} {
		nodes = append(nodes, &model.Node{ID: id, Type: model.TypeFact, Title: "node " + id})
	}
	nodes = append(nodes,
		&model.Node{ID: "E1", Type: model.TypeEpisode, Title: "Session: how do we store the graph", Source: "session:S1"},
		&model.Node{ID: "E9", Type: model.TypeEpisode, Title: "Session: older", Source: "session:S9"})
	edges := []engine.ExportEdge{
		{Edge: model.Edge{Src: "A", Dst: "B", Kind: model.KindSupports, Weight: 0.8}, EffectiveWeight: 0.7},
		{Edge: model.Edge{Src: "A", Dst: "N", Kind: model.KindAssociative, Weight: 0.5}, EffectiveWeight: 0.4},
		{Edge: model.Edge{Src: "A", Dst: "E1", Kind: model.KindAssociative, Weight: 0.3}, EffectiveWeight: 0.3},
	}

	s, st, err := ExportSessions(l.evs, nodes, edges, ExportOptions{Name: "real"})
	if err != nil {
		t.Fatal(err)
	}
	// The episode of exported session S1 quotes its prompts and is left out;
	// the episode of an unexported session stays.
	if len(s.Nodes) != 6 || st.Episodes != 1 || len(s.Links) != 1 || s.Links[0].Kind != model.KindSupports || s.Links[0].Weight != 0.7 {
		t.Fatalf("knowledge base: %d nodes, links %+v", len(s.Nodes), s.Links)
	}
	// S1: first recall labeled A, the empty one unlabeled; S2: N and C;
	// S3: P was created after the recall and "gone" no longer exists.
	// Claude's recall in S2 is a query of its own, labeled by the same turn.
	if st.Recalls != 5 || st.Queries != 3 || st.Unlabeled != 2 || st.Dropped != 2 || st.Sessions != 2 {
		t.Fatalf("stats: %+v", st)
	}
	if q := s.Sessions[0].Queries[0]; q.Q != "how do we store the graph" || strings.Join(q.Relevant, ",") != "A" {
		t.Errorf("S1 query: %+v", q)
	}
	if q := s.Sessions[1].Queries[0]; strings.Join(q.Relevant, ",") != "N,C" {
		t.Errorf("S2 query: %+v", q)
	}
	if q := s.Sessions[1].Queries[1]; q.Q != "cache invalidation rule" || strings.Join(q.Relevant, ",") != "N,C" {
		t.Errorf("S2 Claude query: %+v", q)
	}

	// Session labels give every recall of the session all its used neurons.
	s, st, err = ExportSessions(l.evs, nodes, edges, ExportOptions{Label: LabelSession, Learned: true})
	if err != nil {
		t.Fatal(err)
	}
	if st.Queries != 4 || len(s.Links) != 2 || strings.Join(s.Sessions[0].Queries[1].Relevant, ",") != "A" {
		t.Errorf("session labels: %+v, %+v", st, s.Sessions)
	}

	if _, _, err := ExportSessions(nil, nodes, edges, ExportOptions{}); err == nil {
		t.Error("export without labeled recalls must fail")
	}
	if _, _, err := ExportSessions(l.evs, nodes, edges, ExportOptions{Label: "bogus"}); err == nil {
		t.Error("unknown label strategy accepted")
	}
}

func TestNotesAndUsageByChange(t *testing.T) {
	l := &eventLog{}
	l.prompt("S1", "how do we store the graph", 0.9, "A")
	l.add("S1", engine.EventNote, map[string]any{"kind": "noise", "text": "A was irrelevant", "ids": []string{"A"}})
	// Two changes in a row (one tuning step) start one period.
	l.add("", engine.EventParam, map[string]string{"key": "min_seed_strength", "old": "0.25", "new": "0.3"})
	l.add("", engine.EventParam, map[string]string{"key": "result_limit", "old": "10", "new": "6"})
	l.prompt("S2", "the schema pin", 0.2)
	l.add("S2", engine.EventNote, map[string]any{"kind": "miss", "text": "the schema pin memory was not recalled"})
	l.prompt("S2", "and the loader", 0.7, "B")
	l.used("S2", 0, "B")
	l.add("", engine.EventParam, map[string]string{"key": "embedder", "old": "", "new": "hugot:x"})

	u := AnalyzeUsage(l.evs)
	if u.Notes["noise"] != 1 || u.Notes["miss"] != 1 || len(u.NoteLog) != 2 || u.NoteLog[0].IDs[0] != "A" || len(u.Changes) != 3 {
		t.Fatalf("notes/changes: %+v %+v %+v", u.Notes, u.NoteLog, u.Changes)
	}
	var b strings.Builder
	u.WriteText(&b)
	for _, want := range []string{"1 miss", "noise: A was irrelevant", "min_seed_strength 0.25→0.3", "embedder -→hugot:x"} {
		if !strings.Contains(b.String(), want) {
			t.Errorf("text report lacks %q:\n%s", want, b.String())
		}
	}

	ps := UsageByChange(l.evs)
	if len(ps) != 3 {
		t.Fatalf("periods: %d", len(ps))
	}
	if ps[0].Change != nil || ps[0].Usage.Recalls != 1 || ps[0].Usage.Notes["noise"] != 1 {
		t.Errorf("first period: %+v", ps[0].Usage)
	}
	if c := ps[1].Change; c == nil || !strings.Contains(c.String(), "min_seed_strength") || !strings.Contains(c.String(), "result_limit 10→6") ||
		ps[1].Usage.Recalls != 2 || ps[1].Usage.Notes["miss"] != 1 || ps[1].Usage.BySource["prompt"].UsageRate != 1 {
		t.Errorf("second period: %v %+v", c, ps[1].Usage)
	}
	if c := ps[2].Change; c == nil || c.Key != "embedder" || ps[2].Usage.Recalls != 0 {
		t.Errorf("last period: %+v", ps[2])
	}
	b.Reset()
	if err := WritePeriods(&b, ps); err != nil || !strings.Contains(b.String(), "(initial settings)") || !strings.Contains(b.String(), "100%") {
		t.Errorf("periods table: %v\n%s", err, b.String())
	}
}
