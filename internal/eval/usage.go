package eval

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/vhavlena/plasticity/internal/engine"
	"github.com/vhavlena/plasticity/internal/hooks"
	"github.com/vhavlena/plasticity/internal/store"
)

// Event payloads written by the engine and the hooks (only the fields read
// here).
type (
	recallPayload struct {
		Source string `json:"source"`
		Query  string `json:"query"`
		Prompt string `json:"prompt"` // query of events logged before "query" existed
		Hits   []struct {
			ID   string `json:"id"`
			Seed bool   `json:"seed"`
		} `json:"hits"`
		Confidence    float64 `json:"confidence"`
		LatencyMS     float64 `json:"latency_ms"`
		InjectedChars int     `json:"injected_chars"`
	}
	usedPayload struct {
		IDs    []string            `json:"ids"`
		Report *engine.LearnReport `json:"report"`
	}
	addPayload struct {
		ID          string   `json:"id"`
		DuplicateOf string   `json:"duplicate_of"`
		Supersedes  []string `json:"supersedes"`
		Similar     int      `json:"similar"`
	}
	errorPayload struct {
		Op string `json:"op"`
	}
)

// source returns the recall's source. Events before sources existed were
// all prompt recalls, and before notifications were told apart, task
// notifications were recalled as prompts.
func (p recallPayload) source() string {
	switch {
	case p.Source != "" && p.Source != engine.RecallSourcePrompt:
		return p.Source
	case hooks.IsNotification(p.query()):
		return engine.RecallSourceNotification
	}
	return engine.RecallSourcePrompt
}

func (p recallPayload) query() string {
	if p.Query != "" {
		return p.Query
	}
	return p.Prompt
}

func decode[T any](ev store.Event) (T, bool) {
	var v T
	return v, json.Unmarshal(ev.Payload, &v) == nil
}

// Usage summarizes how the memory behaved in real sessions, computed from
// the event log.
type Usage struct {
	From     time.Time `json:"from,omitzero"`
	To       time.Time `json:"to,omitzero"`
	Sessions int       `json:"sessions"` // sessions with at least one prompt
	Prompts  int       `json:"prompts"`
	// Notifications are task notifications delivered as prompts (not
	// counted in Prompts).
	Notifications int `json:"notifications"`

	// Recall (UserPromptSubmit). Trivial prompts are not recalled.
	Recalls       int            `json:"recalls"`
	EmptyRate     float64        `json:"empty_rate"` // share of recalls injecting nothing
	MeanHits      float64        `json:"mean_hits"`
	MeanInjected  float64        `json:"mean_injected_chars"`
	Confidence    map[string]int `json:"confidence"` // histogram of query confidence
	MeanConf      float64        `json:"mean_confidence"`
	Latency       Latency        `json:"latency"`
	RecallsUseful float64        `json:"recalls_useful"` // share of non-empty recalls with ≥1 hit later reported used in the session

	// Use (plasticity used).
	Injected   int     `json:"injected"` // distinct (session, neuron) pairs injected
	UsedOfInj  int     `json:"used_of_injected"`
	UsageRate  float64 `json:"usage_rate"`  // used_of_injected / injected
	UsedOther  int     `json:"used_other"`  // used neurons that were not injected (found via search/recall)
	UsedEvents int     `json:"used_events"` // `used` calls

	// Creation (plasticity add). Created counts neurons added with a
	// session; CreatedOutside those added without one (e.g. by hand).
	Created        int     `json:"created"`
	CreatedOutside int     `json:"created_outside_sessions"`
	Duplicates     int     `json:"duplicates"`     // adds refused as near-duplicates
	DuplicateRate  float64 `json:"duplicate_rate"` // duplicates / (created + duplicates)
	// Supersession at add time: adds that listed similar neurons, and adds
	// that superseded some (with the number of neurons superseded).
	AddsWithSimilar   int     `json:"adds_with_similar"`
	AddsSuperseding   int     `json:"adds_superseding"`
	SupersededAtAdd   int     `json:"superseded_at_add"`
	CreatedPerSession float64 `json:"created_per_session"`
	CreatedRecalled   int     `json:"created_recalled"` // created neurons injected in a later session
	CreatedUsed       int     `json:"created_used"`     // created neurons reported used in a later session
	Episodes          int     `json:"episodes"`

	// Reflection nudge (Stop hook).
	Nudges        int     `json:"nudges"`
	NudgeResponse float64 `json:"nudge_response"` // share of nudges followed by add/used in the session

	// Learning.
	LinksCreated int `json:"links_created"`
	Potentiated  int `json:"links_potentiated"`
	Depressed    int `json:"links_depressed"`
	Consolidated int `json:"consolidations"`

	// BySource splits recall and use by recall source: "prompt" (automatic,
	// per user prompt) and "claude" (recalls Claude ran itself).
	BySource map[string]*SourceUsage `json:"by_source"`

	// The user's observations (plasticity note) and the parameter or
	// embedder changes in the period.
	Notes   map[string]int `json:"notes"` // by kind
	NoteLog []UsageNote    `json:"note_log,omitempty"`
	Changes []UsageChange  `json:"changes,omitempty"`

	Errors map[string]int `json:"errors,omitempty"` // by operation
	Days   []UsageDay     `json:"days"`
}

// UsageNote is a logged observation of the user.
type UsageNote struct {
	At      time.Time `json:"ts"`
	Session string    `json:"session,omitempty"`
	Kind    string    `json:"kind"`
	Text    string    `json:"text"`
	IDs     []string  `json:"ids,omitempty"`
}

// UsageChange is a logged parameter or embedder change.
type UsageChange struct {
	At  time.Time `json:"ts"`
	Key string    `json:"key"`
	Old string    `json:"old"`
	New string    `json:"new"`
}

func (c UsageChange) String() string {
	old, new := c.Old, c.New
	if old == "" {
		old = "-"
	}
	if new == "" {
		new = "-"
	}
	return fmt.Sprintf("%s %s→%s", c.Key, old, new)
}

// SourceUsage is recall and use of one recall source. A neuron used in a
// session counts for every source that surfaced it in that session.
type SourceUsage struct {
	Recalls   int     `json:"recalls"`
	Empty     int     `json:"empty"`
	Injected  int     `json:"injected"` // distinct (session, neuron) pairs surfaced
	UsedOfInj int     `json:"used_of_injected"`
	UsageRate float64 `json:"usage_rate"`
}

// UsageDay is the activity of one calendar day.
type UsageDay struct {
	Day          string `json:"day"`
	Sessions     int    `json:"sessions"`
	Recalls      int    `json:"recalls"`
	Empty        int    `json:"empty"`
	Used         int    `json:"used"` // neurons reported used
	Created      int    `json:"created"`
	LinksCreated int    `json:"links_created"`
}

// ConfidenceBuckets are the histogram bins of Usage.Confidence.
var ConfidenceBuckets = []struct {
	Name string
	Max  float64
}{{"<0.25", 0.25}, {"0.25-0.4", 0.4}, {"0.4-0.6", 0.6}, {"0.6-0.8", 0.8}, {">=0.8", 2}}

func confidenceBucket(c float64) string {
	for _, b := range ConfidenceBuckets {
		if c < b.Max {
			return b.Name
		}
	}
	return ConfidenceBuckets[len(ConfidenceBuckets)-1].Name
}

// AnalyzeUsage computes the usage report from events (oldest first).
func AnalyzeUsage(events []store.Event) *Usage {
	u := &Usage{Confidence: map[string]int{}, Errors: map[string]int{}, BySource: map[string]*SourceUsage{}, Notes: map[string]int{}}
	for _, b := range ConfidenceBuckets {
		u.Confidence[b.Name] = 0
	}
	days := map[string]*UsageDay{}
	day := func(t time.Time) *UsageDay {
		k := t.Local().Format(time.DateOnly)
		if days[k] == nil {
			days[k] = &UsageDay{Day: k}
		}
		return days[k]
	}

	type pair struct{ session, id string }
	injected := map[pair]bool{}
	injectedBy := map[string]map[pair]bool{} // source -> pairs it surfaced
	used := map[pair]bool{}
	sessions := map[string]bool{}
	daySessions := map[string]map[string]bool{}
	createdIn := map[string]string{}             // neuron -> creating session
	createdRecalled := map[string]bool{}         // neuron -> injected in another session
	createdUsed := map[string]bool{}             // neuron -> used in another session
	recallHits := map[string][]map[string]bool{} // session -> hit sets of its non-empty recalls
	nudgedAt := map[string]int64{}               // session -> nudge event ID
	responded := map[string]bool{}
	var latencies []time.Duration
	var hits, chars int
	var conf float64

	for _, ev := range events {
		if u.From.IsZero() {
			u.From = ev.At
		}
		u.To = ev.At
		d := day(ev.At)
		switch ev.Kind {
		case engine.EventNotification:
			u.Notifications++
		case engine.EventPrompt:
			if p, ok := decode[struct{ Prompt string }](ev); ok && hooks.IsNotification(p.Prompt) {
				u.Notifications++ // logged as a prompt before notifications were told apart
				continue
			}
			u.Prompts++
			sessions[ev.Session] = true
			if daySessions[d.Day] == nil {
				daySessions[d.Day] = map[string]bool{}
			}
			if !daySessions[d.Day][ev.Session] {
				daySessions[d.Day][ev.Session] = true
				d.Sessions++
			}
		case engine.EventRecall:
			p, ok := decode[recallPayload](ev)
			if !ok {
				continue
			}
			src := p.source()
			su := u.BySource[src]
			if su == nil {
				su = &SourceUsage{}
				u.BySource[src] = su
				injectedBy[src] = map[pair]bool{}
			}
			su.Recalls++
			u.Recalls++
			d.Recalls++
			conf += p.Confidence
			u.Confidence[confidenceBucket(p.Confidence)]++
			latencies = append(latencies, time.Duration(p.LatencyMS*float64(time.Millisecond)))
			hits += len(p.Hits)
			chars += p.InjectedChars
			if len(p.Hits) == 0 {
				u.EmptyRate++
				d.Empty++
				su.Empty++
				continue
			}
			set := map[string]bool{}
			for _, h := range p.Hits {
				set[h.ID] = true
				injected[pair{ev.Session, h.ID}] = true
				injectedBy[src][pair{ev.Session, h.ID}] = true
				if s, ok := createdIn[h.ID]; ok && s != ev.Session {
					createdRecalled[h.ID] = true
				}
			}
			recallHits[ev.Session] = append(recallHits[ev.Session], set)
		case engine.EventUsed:
			p, ok := decode[usedPayload](ev)
			if !ok {
				continue
			}
			u.UsedEvents++
			d.Used += len(p.IDs)
			for _, id := range p.IDs {
				used[pair{ev.Session, id}] = true
				if s, ok := createdIn[id]; ok && s != ev.Session {
					createdUsed[id] = true
				}
			}
			if r := p.Report; r != nil {
				u.LinksCreated += r.Created
				u.Potentiated += r.Potentiated
				u.Depressed += r.Depressed
				d.LinksCreated += r.Created
			}
			if _, ok := nudgedAt[ev.Session]; ok {
				responded[ev.Session] = true
			}
		case engine.EventAdd:
			p, ok := decode[addPayload](ev)
			if !ok {
				continue
			}
			if p.DuplicateOf != "" {
				u.Duplicates++
				continue
			}
			if p.Similar > 0 {
				u.AddsWithSimilar++
			}
			if len(p.Supersedes) > 0 {
				u.AddsSuperseding++
				u.SupersededAtAdd += len(p.Supersedes)
			}
			if ev.Session == "" {
				u.CreatedOutside++
				continue
			}
			u.Created++
			d.Created++
			if p.ID != "" {
				createdIn[p.ID] = ev.Session
			}
			if _, ok := nudgedAt[ev.Session]; ok {
				responded[ev.Session] = true
			}
		case engine.EventNudge:
			u.Nudges++
			nudgedAt[ev.Session] = ev.ID
		case engine.EventEpisode:
			if p, ok := decode[struct{ Updated bool }](ev); ok && !p.Updated {
				u.Episodes++ // refreshes of resumed sessions' episodes are not new episodes
			}
		case engine.EventConsolidate:
			u.Consolidated++
		case engine.EventNote:
			if n, ok := decode[UsageNote](ev); ok {
				n.At, n.Session = ev.At, ev.Session
				u.Notes[n.Kind]++
				u.NoteLog = append(u.NoteLog, n)
			}
		case engine.EventParam:
			if c, ok := decode[UsageChange](ev); ok {
				c.At = ev.At
				u.Changes = append(u.Changes, c)
			}
		case engine.EventError:
			p, _ := decode[errorPayload](ev)
			if p.Op == "" {
				p.Op = "unknown"
			}
			u.Errors[p.Op]++
		}
	}

	u.Sessions = len(sessions)
	if u.Recalls > 0 {
		n := float64(u.Recalls)
		u.EmptyRate /= n
		u.MeanHits = float64(hits) / n
		u.MeanInjected = float64(chars) / n
		u.MeanConf = conf / n
	}
	u.Latency = latency(latencies)

	u.Injected = len(injected)
	for p := range used {
		if injected[p] {
			u.UsedOfInj++
		} else {
			u.UsedOther++
		}
	}
	if u.Injected > 0 {
		u.UsageRate = float64(u.UsedOfInj) / float64(u.Injected)
	}
	for src, su := range u.BySource {
		su.Injected = len(injectedBy[src])
		for p := range injectedBy[src] {
			if used[p] {
				su.UsedOfInj++
			}
		}
		if su.Injected > 0 {
			su.UsageRate = float64(su.UsedOfInj) / float64(su.Injected)
		}
	}
	nonEmpty, useful := 0, 0
	for s, sets := range recallHits {
		for _, set := range sets {
			nonEmpty++
			for id := range set {
				if used[pair{s, id}] {
					useful++
					break
				}
			}
		}
	}
	if nonEmpty > 0 {
		u.RecallsUseful = float64(useful) / float64(nonEmpty)
	}

	if total := u.Created + u.Duplicates; total > 0 {
		u.DuplicateRate = float64(u.Duplicates) / float64(total)
	}
	if u.Sessions > 0 {
		u.CreatedPerSession = float64(u.Created) / float64(u.Sessions)
	}
	u.CreatedRecalled, u.CreatedUsed = len(createdRecalled), len(createdUsed)
	if u.Nudges > 0 {
		u.NudgeResponse = float64(len(responded)) / float64(u.Nudges)
	}

	for _, d := range days {
		u.Days = append(u.Days, *d)
	}
	sort.Slice(u.Days, func(i, j int) bool { return u.Days[i].Day < u.Days[j].Day })
	if u.Days == nil {
		u.Days = []UsageDay{}
	}
	if len(u.Errors) == 0 {
		u.Errors = nil
	}
	return u
}

// WriteText prints the usage report.
func (u *Usage) WriteText(w io.Writer) error {
	if u.Prompts == 0 && u.Recalls == 0 && u.Created == 0 && len(u.NoteLog) == 0 {
		_, err := fmt.Fprintln(w, "no logged activity in the selected range")
		return err
	}
	fmt.Fprintf(w, "activity %s .. %s: %d sessions, %d prompts, %d task notifications\n\n",
		u.From.Local().Format("2006-01-02 15:04"), u.To.Local().Format("2006-01-02 15:04"), u.Sessions, u.Prompts, u.Notifications)

	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	row := func(k, f string, a ...any) { fmt.Fprintf(tw, "  %s\t"+f+"\n", append([]any{k}, a...)...) }
	fmt.Fprintln(tw, "recall")
	row("recalls", "%d", u.Recalls)
	for _, src := range []string{engine.RecallSourcePrompt, engine.RecallSourceNotification, engine.RecallSourceClaude} {
		su := u.BySource[src]
		if su == nil {
			su = &SourceUsage{}
		}
		switch src {
		case engine.RecallSourcePrompt:
			row("  per prompt", "%d (%s of prompts; trivial prompts are skipped), %d empty", su.Recalls, ratio(su.Recalls, u.Prompts), su.Empty)
		case engine.RecallSourceNotification:
			row("  per notification", "%d (%s of task notifications), %d empty", su.Recalls, ratio(su.Recalls, u.Notifications), su.Empty)
		case engine.RecallSourceClaude:
			row("  by Claude", "%d (%.1f per session), %d empty", su.Recalls, perSession(su.Recalls, u.Sessions), su.Empty)
		}
	}
	row("empty", "%s", pct(u.EmptyRate))
	row("hits / recall", "%.1f, %.0f chars injected", u.MeanHits, u.MeanInjected)
	row("confidence", "mean %.2f  %s", u.MeanConf, u.confidenceText())
	row("latency", "p50 %.1f ms, p95 %.1f ms, max %.1f ms", u.Latency.P50, u.Latency.P95, u.Latency.Max)
	row("useful recalls", "%s of non-empty recalls had a hit reported used", pct(u.RecallsUseful))
	fmt.Fprintln(tw, "use")
	row("usage rate", "%s (%d of %d injected neurons reported used)", pct(u.UsageRate), u.UsedOfInj, u.Injected)
	for _, src := range []string{engine.RecallSourcePrompt, engine.RecallSourceNotification, engine.RecallSourceClaude} {
		if su := u.BySource[src]; su != nil {
			row("  from "+src+" recalls", "%s (%d of %d)", pct(su.UsageRate), su.UsedOfInj, su.Injected)
		}
	}
	row("used, not injected", "%d (found via search/recall)", u.UsedOther)
	fmt.Fprintln(tw, "creation")
	row("created", "%d (%.1f per session), %d episodes, %d outside sessions", u.Created, u.CreatedPerSession, u.Episodes, u.CreatedOutside)
	row("duplicates caught", "%d (%s of adds)", u.Duplicates, pct(u.DuplicateRate))
	row("supersession at add", "%d of %d adds with similar neurons; %d adds superseded %d neurons", u.AddsWithSimilar, u.Created+u.CreatedOutside, u.AddsSuperseding, u.SupersededAtAdd)
	row("reused later", "%d recalled, %d used in a later session (of %d created)", u.CreatedRecalled, u.CreatedUsed, u.Created)
	row("nudges", "%d, %s answered with add/used", u.Nudges, pct(u.NudgeResponse))
	fmt.Fprintln(tw, "learning")
	row("links", "%d created, %d potentiated, %d depressed (%d consolidations)",
		u.LinksCreated, u.Potentiated, u.Depressed, u.Consolidated)
	if len(u.NoteLog) > 0 {
		fmt.Fprintln(tw, "your notes")
		var kinds []string
		for _, k := range engine.NoteKinds {
			if u.Notes[k] > 0 {
				kinds = append(kinds, fmt.Sprintf("%d %s", u.Notes[k], k))
			}
		}
		row("notes", "%s", strings.Join(kinds, ", "))
		from := max(0, len(u.NoteLog)-maxNotesShown)
		for _, n := range u.NoteLog[from:] {
			row("  "+n.At.Local().Format("01-02 15:04"), "%s: %s", n.Kind, truncateText(n.Text, 100))
		}
	}
	if len(u.Changes) > 0 {
		fmt.Fprintln(tw, "changes")
		for _, c := range u.Changes {
			row("  "+c.At.Local().Format("01-02 15:04"), "%s", c)
		}
	}
	if len(u.Errors) > 0 {
		row("errors", "%v", u.Errors)
	}
	if err := tw.Flush(); err != nil {
		return err
	}

	fmt.Fprintln(w, "\nper day")
	tw = tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "day\tsessions\trecalls\tempty\tused\tcreated\tlinks+\t")
	for _, d := range u.Days {
		fmt.Fprintf(tw, "%s\t%d\t%d\t%d\t%d\t%d\t%d\t\n", d.Day, d.Sessions, d.Recalls, d.Empty, d.Used, d.Created, d.LinksCreated)
	}
	return tw.Flush()
}

func (u *Usage) confidenceText() string {
	s := ""
	for _, b := range ConfidenceBuckets {
		s += fmt.Sprintf(" %s:%d", b.Name, u.Confidence[b.Name])
	}
	return "[" + s[1:] + "]"
}

// maxNotesShown bounds the notes listed in the text report.
const maxNotesShown = 10

func truncateText(s string, n int) string {
	s = strings.Join(strings.Fields(s), " ")
	if r := []rune(s); len(r) > n {
		return string(r[:n-1]) + "…"
	}
	return s
}

// UsagePeriod is the usage between two consecutive changes.
type UsagePeriod struct {
	Change *UsageChange `json:"change,omitempty"` // the change that starts the period (nil: the first)
	Usage  *Usage       `json:"usage"`
}

// UsageByChange splits the events at each parameter or embedder change, so
// periods with different settings can be compared. Consecutive changes
// without prompts between them (one tuning step) start a single period.
func UsageByChange(events []store.Event) []UsagePeriod {
	var out []UsagePeriod
	start := 0
	var change *UsageChange
	flush := func(end int) {
		if end > start || change != nil {
			out = append(out, UsagePeriod{Change: change, Usage: AnalyzeUsage(events[start:end])})
		}
	}
	for i, ev := range events {
		if ev.Kind != engine.EventParam {
			continue
		}
		c, ok := decode[UsageChange](ev)
		if !ok {
			continue
		}
		c.At = ev.At
		if change != nil && !hasActivity(events[start:i]) {
			merged := *change
			merged.Key, merged.Old, merged.New = change.String()+", "+c.Key, c.Old, c.New
			change = &merged
			continue
		}
		flush(i)
		start, change = i, &c
	}
	flush(len(events))
	return out
}

func hasActivity(evs []store.Event) bool {
	for _, ev := range evs {
		if ev.Kind == engine.EventPrompt || ev.Kind == engine.EventNotification || ev.Kind == engine.EventRecall {
			return true
		}
	}
	return false
}

// WritePeriods prints the key metrics of each period side by side.
func WritePeriods(w io.Writer, ps []UsagePeriod) error {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "from\tchange\tsessions\trecalls\tempty\tusage(prompt)\tclaude/session\tcreated/session\treused\tmiss\tnoise\t")
	for _, p := range ps {
		u := p.Usage
		from, change := "start", "(initial settings)"
		if p.Change != nil {
			from, change = p.Change.At.Local().Format("01-02 15:04"), p.Change.String()
		} else if !u.From.IsZero() {
			from = u.From.Local().Format("01-02 15:04")
		}
		prompt, claude := u.BySource[engine.RecallSourcePrompt], u.BySource[engine.RecallSourceClaude]
		usage, own := "-", 0
		if prompt != nil && prompt.Injected > 0 {
			usage = pct(prompt.UsageRate)
		}
		if claude != nil {
			own = claude.Recalls
		}
		fmt.Fprintf(tw, "%s\t%s\t%d\t%d\t%s\t%s\t%.1f\t%.1f\t%d\t%d\t%d\t\n", from, change, u.Sessions, u.Recalls,
			pct(u.EmptyRate), usage, perSession(own, u.Sessions), u.CreatedPerSession, u.CreatedRecalled+u.CreatedUsed,
			u.Notes["miss"], u.Notes["noise"])
	}
	return tw.Flush()
}

func perSession(n, sessions int) float64 {
	if sessions == 0 {
		return 0
	}
	return float64(n) / float64(sessions)
}

func ratio(a, b int) string {
	if b == 0 {
		return "-"
	}
	return pct(float64(a) / float64(b))
}
