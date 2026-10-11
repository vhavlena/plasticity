package engine

import (
	"errors"
	"strings"
	"testing"

	"github.com/vhavlena/plasticity/internal/model"
	"github.com/vhavlena/plasticity/internal/store"
)

func TestAddDetectsDuplicates(t *testing.T) {
	f := newFixture(t)
	n := &model.Node{Type: model.TypeFact, Title: "  hebbian   LEARNING ", Content: "different wording"}
	res, err := f.e.AddNodeChecked(n, AddOptions{})
	if err != nil || res.Node != nil || res.Duplicate == nil || res.Duplicate.ID != f.ids["hebb"] || res.Reason != "same title" {
		t.Fatalf("same-title duplicate not caught: %v %+v", err, res)
	}
	if st, _ := f.e.Stats(); st.Nodes != len(f.ids) {
		t.Fatal("duplicate was inserted")
	}
	res, err = f.e.AddNodeChecked(&model.Node{Type: model.TypeFact, Title: "Hebbian learning"}, AddOptions{Force: true})
	if err != nil || res.Node == nil || res.Duplicate != nil {
		t.Fatalf("forced add: %v %+v", err, res)
	}
}

func TestAddDetectsSemanticDuplicates(t *testing.T) {
	f := newFixture(t, WithEmbedder(conceptEmbedder{}))
	// Same concept vector as "Cooking pasta / boil water, add salt", different title.
	res, err := f.e.AddNodeChecked(&model.Node{Type: model.TypeProcedure, Title: "Kitchen meal", Content: "food"}, AddOptions{})
	if err != nil || res.Duplicate == nil || res.Duplicate.ID != f.ids["pasta"] || res.Similarity < 0.99 {
		t.Fatalf("semantic duplicate not caught: %v %+v", err, res)
	}
	res, err = f.e.AddNodeChecked(&model.Node{Type: model.TypeConcept, Title: "Totally new idea"}, AddOptions{})
	if err != nil || res.Node == nil {
		t.Fatalf("distinct node rejected: %v %+v", err, res)
	}
}

func TestSessionContextLinksAndUsed(t *testing.T) {
	f := newFixture(t)
	if err := f.e.StartSession("S", "/repo", "startup"); err != nil {
		t.Fatal(err)
	}
	if _, err := f.e.Used("S", []string{f.ids["sqlite"], f.ids["badger"]}); err != nil {
		t.Fatal(err)
	}
	if f.weight("sqlite", "badger") == 0 {
		t.Fatal("used nodes should be associated")
	}
	res, err := f.e.AddNodeChecked(&model.Node{Type: model.TypeDecision, Title: "Never open the DB twice",
		Rationale: "lock conflicts"}, AddOptions{Session: "S"})
	if err != nil || res.ContextLinks != 2 {
		t.Fatalf("context links: %v %+v", err, res)
	}
	f.ids["new"] = res.Node.ID
	if w := f.weight("new", "badger"); w != f.e.P.ContextLinkWeight {
		t.Fatalf("context link weight %g", w)
	}
	// A second neuron in the session also links to the first new one.
	res2, _ := f.e.AddNodeChecked(&model.Node{Type: model.TypeFact, Title: "Another fact"}, AddOptions{Session: "S"})
	f.ids["new2"] = res2.Node.ID
	if res2.ContextLinks != 3 || f.weight("new2", "new") == 0 {
		t.Fatalf("created nodes join the session context: %+v", res2)
	}
	if _, err := f.e.Used("", []string{f.ids["spread"]}); err == nil {
		t.Fatal("sessionless used with one node must fail")
	}
}

func TestNudgePolicy(t *testing.T) {
	f := newFixture(t)
	f.e.StartSession("S", "", "startup")
	if ok, _ := f.e.ShouldNudge("S", 3); ok {
		t.Fatal("nudged below the tool-call threshold")
	}
	if ok, _ := f.e.ShouldNudge("S", 6); !ok {
		t.Fatal("expected a nudge after substantive work")
	}
	if ok, _ := f.e.ShouldNudge("S", 50); ok {
		t.Fatal("nudged twice in one session")
	}
	f.e.StartSession("T", "", "startup")
	f.e.Used("T", []string{f.ids["hebb"]})
	if ok, _ := f.e.ShouldNudge("T", 50); ok {
		t.Fatal("a session that reported memories must not be nudged")
	}
	if ok, _ := f.e.ShouldNudge("unknown", 50); ok {
		t.Fatal("unknown session nudged")
	}
}

func TestEndSessionCreatesEpisode(t *testing.T) {
	f := newFixture(t)
	f.e.SetParam("consolidate_every_sessions", "1")
	f.e.StartSession("S", "/repo", "startup")
	f.e.NotePrompt("S", "/repo", "Why did we choose SQLite over Badger?")
	f.e.NotePrompt("S", "/repo", "And how do hooks talk to it?")
	f.e.Used("S", []string{f.ids["sqlite"], f.ids["hooks"]})
	rep, err := f.e.EndSession("S", []string{"internal/store/store.go"})
	if err != nil || rep.Episode == nil || rep.Links != 2 || rep.Consolidated == nil {
		t.Fatalf("end session: %v %+v", err, rep)
	}
	ep := rep.Episode
	if ep.Type != model.TypeEpisode || !strings.Contains(ep.Title, "SQLite over Badger") ||
		!strings.Contains(ep.Content, "hooks talk") || !strings.Contains(ep.Content, "store.go") {
		t.Fatalf("episode content: %+v", ep)
	}
	again, _ := f.e.EndSession("S", nil)
	if again.Episode != nil {
		t.Fatal("ending twice must not create a second episode")
	}
	s, _ := f.e.Session("S")
	if s.EpisodeID != ep.ID || s.EndedAt.IsZero() {
		t.Fatalf("session after end: %+v", s)
	}
	// The episode is recallable and leads to the session's knowledge.
	r := f.recall("why sqlite badger hooks session", "")
	if _, ok := hitIDs(r)[ep.ID]; !ok {
		t.Fatalf("episode not recallable: %+v", r.Hits)
	}
	kinds := map[string]int{}
	evs, _ := f.e.Events(store.EventFilter{Session: "S"})
	for _, ev := range evs {
		kinds[ev.Kind]++
	}
	for _, k := range []string{EventSessionStart, EventPrompt, EventUsed, EventEpisode, EventConsolidate} {
		if kinds[k] == 0 {
			t.Errorf("no %s event logged: %v", k, kinds)
		}
	}
	// Sessions without prompts get no episode.
	f.e.StartSession("empty", "", "startup")
	if rep, _ := f.e.EndSession("empty", nil); rep.Episode != nil {
		t.Fatal("episode for an empty session")
	}
}

func TestEndSessionKilledAfterEpisode(t *testing.T) {
	f := newFixture(t)
	f.e.StartSession("S", "/repo", "startup")
	f.e.NotePrompt("S", "/repo", "Why did we choose SQLite over Badger?")
	f.e.Used("S", []string{f.ids["sqlite"]})
	f.e.afterEpisode = func() error { return errors.New("killed") }
	if _, err := f.e.EndSession("S", nil); err == nil {
		t.Fatal("expected the simulated kill")
	}
	s, _ := f.e.Session("S")
	if s.EpisodeID == "" || !s.EndedAt.IsZero() {
		t.Fatalf("after kill: %+v", s)
	}
	// The abandoned session is closed later: same episode, no duplicate.
	f.e.afterEpisode = nil
	rep, err := f.e.EndSession("S", nil)
	if err != nil || rep.Episode == nil || rep.Episode.ID != s.EpisodeID {
		t.Fatalf("closing again: %v %+v", err, rep)
	}
	eps := 0
	for _, h := range f.recall("why sqlite badger session", "").Hits {
		if h.Node.Type == model.TypeEpisode {
			eps++
		}
	}
	if st, _ := f.e.Stats(); eps != 1 || st.Nodes != len(f.ids)+1 {
		t.Fatalf("episodes: %d recalled, %d nodes for %d fixture nodes", eps, st.Nodes, len(f.ids))
	}
}

func TestWithoutLog(t *testing.T) {
	f := newFixture(t, WithoutLog())
	f.e.StartSession("S", "", "startup")
	if evs, _ := f.e.Events(store.EventFilter{}); len(evs) != 0 {
		t.Fatalf("events logged despite WithoutLog: %v", evs)
	}
}

func TestNotesAndParamChangesAreLogged(t *testing.T) {
	f := newFixture(t)
	if err := f.e.Note("S", "miss", "forgot the schema pin", []string{f.ids["hebb"]}); err != nil {
		t.Fatal(err)
	}
	if err := f.e.Note("S", "bogus", "x", nil); err == nil {
		t.Error("unknown note kind accepted")
	}
	if err := f.e.Note("S", "noise", "  ", nil); err == nil {
		t.Error("empty note accepted")
	}
	if err := f.e.SetParam("min_seed_strength", "0.3"); err != nil {
		t.Fatal(err)
	}
	if err := f.e.SetParam("min_seed_strength", "0.3"); err != nil { // no change: not logged
		t.Fatal(err)
	}
	if err := f.e.ResetParam("min_seed_strength"); err != nil {
		t.Fatal(err)
	}
	notes, _ := f.e.Events(store.EventFilter{Kind: EventNote})
	if len(notes) != 1 || notes[0].Session != "S" || !strings.Contains(string(notes[0].Payload), `"kind":"miss"`) {
		t.Fatalf("notes: %+v", notes)
	}
	changes, _ := f.e.Events(store.EventFilter{Kind: EventParam})
	if len(changes) != 2 ||
		string(changes[0].Payload) != `{"key":"min_seed_strength","new":"0.3","old":"0.25"}` ||
		string(changes[1].Payload) != `{"key":"min_seed_strength","new":"0.25","old":"0.3"}` {
		for _, c := range changes {
			t.Log(string(c.Payload))
		}
		t.Fatal("param changes not logged as expected")
	}
}
