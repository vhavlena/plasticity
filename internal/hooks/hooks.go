// Package hooks implements the Claude Code hook handlers of the plasticity
// plugin. Each handler reads the hook's JSON input and returns the JSON
// output Claude Code expects. Handlers never fail the user's turn: problems
// are logged and reported on stderr by the caller, and the hook exits 0.
package hooks

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/vhavlena/plasticity/internal/engine"
	"github.com/vhavlena/plasticity/internal/model"
)

// Input is the subset of the hook input JSON used by the handlers.
type Input struct {
	SessionID      string `json:"session_id"`
	TranscriptPath string `json:"transcript_path"`
	Cwd            string `json:"cwd"`
	HookEventName  string `json:"hook_event_name"`
	Prompt         string `json:"prompt"`           // UserPromptSubmit
	Source         string `json:"source"`           // SessionStart
	StopHookActive bool   `json:"stop_hook_active"` // Stop
	Reason         string `json:"reason"`           // SessionEnd
}

// Output is the hook output JSON.
type Output struct {
	Decision string          `json:"decision,omitempty"` // "block" (Stop)
	Reason   string          `json:"reason,omitempty"`
	Specific *SpecificOutput `json:"hookSpecificOutput,omitempty"`
}

// SpecificOutput carries injected context.
type SpecificOutput struct {
	HookEventName     string `json:"hookEventName"`
	AdditionalContext string `json:"additionalContext"`
}

func contextOutput(event, text string) *Output {
	return &Output{Specific: &SpecificOutput{HookEventName: event, AdditionalContext: text}}
}

// Config holds the plugin options (userConfig), read from the environment.
type Config struct {
	Scope       string // global | project
	RecallLimit int    // memories injected per prompt
	Reflect     bool   // Stop-hook nudge to store/report memories
	Log         bool   // event log
	Embeddings  string // auto | off
	// Episodes: create an episode neuron for each session when it ends.
	// Session history is kept in the database either way (plasticity sessions).
	Episodes bool
	// SessionIdle: a session idle this long without a SessionEnd hook is
	// closed (its episode created, if enabled) at the next session start.
	// Resuming it later reopens it, and its episode is refreshed when it ends
	// again.
	SessionIdle time.Duration
	// ContextBudget caps the injected characters per prompt (Claude Code
	// truncates additional context above 10,000 characters).
	ContextBudget int
	// Notifications selects which task notifications are recalled (see
	// PlanRecall): NotifyFailed, NotifyAll (as plain prompts) or NotifyOff.
	Notifications string
	// Task notifications (see PlanRecall): only failed tasks are recalled,
	// with at most NotificationLimit hits (0: RecallLimit), only when the query confidence
	// reaches NotificationMinConfidence, on at most NotificationMaxChars
	// characters of description and error output.
	NotificationLimit         int
	NotificationMinConfidence float64
	NotificationMaxChars      int
	// DB is the graph the hook opened. Commands Claude runs through Bash see
	// neither the plugin options nor the session's cwd, so the commands the
	// hooks suggest name this graph (and the log option) explicitly (see CLI).
	DB string
}

// CLI returns the command prefix for the commands the hooks suggest to
// Claude: plasticity --db '<DB>' (without a DB: plain plasticity), plus
// --no-log when the event log is off.
func (c Config) CLI() string {
	cli := "plasticity"
	if c.DB != "" {
		cli += " --db " + shellQuote(c.DB)
	}
	if !c.Log {
		cli += " --no-log"
	}
	return cli
}

// shellQuote quotes s as one POSIX shell word.
func shellQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

// DefaultConfig returns the plugin options' defaults.
func DefaultConfig() Config {
	return Config{Scope: "global", RecallLimit: 6, Reflect: true, Log: true, Embeddings: "auto", ContextBudget: 3000,
		SessionIdle: 48 * time.Hour, Notifications: NotifyFailed, NotificationLimit: 0, NotificationMinConfidence: 0.6, NotificationMaxChars: 600}
}

// ConfigFromEnv reads CLAUDE_PLUGIN_OPTION_<KEY> (as declared or upper-case),
// with PLASTICITY_<KEY> overrides for use outside the plugin.
func ConfigFromEnv() Config {
	c := DefaultConfig()
	get := func(key string) (string, bool) {
		for _, k := range []string{"PLASTICITY_" + strings.ToUpper(key), "CLAUDE_PLUGIN_OPTION_" + strings.ToUpper(key), "CLAUDE_PLUGIN_OPTION_" + key} {
			if v, ok := os.LookupEnv(k); ok && strings.TrimSpace(v) != "" {
				return strings.TrimSpace(v), true
			}
		}
		return "", false
	}
	if v, ok := get("scope"); ok {
		c.Scope = strings.ToLower(v)
	}
	if v, ok := get("recall_limit"); ok {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			c.RecallLimit = n
		}
	}
	if v, ok := get("session_idle_hours"); ok {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			c.SessionIdle = time.Duration(n) * time.Hour
		}
	}
	if v, ok := get("reflect"); ok {
		c.Reflect = truthy(v)
	}
	if v, ok := get("log"); ok {
		c.Log = truthy(v)
	}
	if v, ok := get("episodes"); ok {
		c.Episodes = truthy(v)
	}
	if v, ok := get("embeddings"); ok {
		c.Embeddings = strings.ToLower(v)
	}
	if v, ok := get("notifications"); ok {
		switch v = strings.ToLower(v); v {
		case NotifyFailed, NotifyAll, NotifyOff:
			c.Notifications = v
		}
	}
	if v, ok := get("notification_limit"); ok {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			c.NotificationLimit = n
		}
	}
	if v, ok := get("notification_min_confidence"); ok {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f >= 0 && f <= 1 {
			c.NotificationMinConfidence = f
		}
	}
	if v, ok := get("notification_max_chars"); ok {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			c.NotificationMaxChars = n
		}
	}
	return c
}

func truthy(v string) bool {
	switch strings.ToLower(v) {
	case "0", "false", "no", "off":
		return false
	}
	return true
}

// SessionStart registers the session and tells Claude that memory is active,
// its session ID (needed by `plasticity add/used`, since Bash commands do not
// see it) and how to use the memory.
func SessionStart(e *engine.Engine, in Input, project string, cfg Config) (*Output, error) {
	if err := e.StartSession(in.SessionID, project, in.Source, in.TranscriptPath); err != nil {
		return nil, err
	}
	st, err := e.Stats()
	if err != nil {
		return nil, err
	}
	emb, _ := e.EmbedStatus()
	search := "keyword search"
	if emb != nil && emb.Active != "" {
		search = "keyword + semantic search"
	} else if cfg.Embeddings != "off" {
		search = "keyword search (semantic search is being set up in the background)"
	}
	id, cli := in.SessionID, cfg.CLI()
	text := fmt.Sprintf(`Plasticity long-term memory is active: %d neurons, %s scope, %s.
This session's ID is %s.
Run every plasticity command as %s ... (it selects this session's memory).
- Memories relevant to each prompt are injected automatically.
- Recall more yourself when that context is missing: before a design decision, when starting
  work in an unfamiliar area, when stuck on an error, before re-investigating a problem:
  %s recall "<specific question or error>" --session %s
- Store durable knowledge (decisions with rationale, root causes, working procedures, preferences):
  %s add -t decision|fact|procedure|concept|rationale|question --title "..." -c "..." -r "why" --importance 0..1 --session %s
  (importance: 0.8-1 user preferences/corrections and costly decisions or root causes, 0.4-0.7 useful facts and procedures, 0.1-0.3 minor details)
- Report memories you actually used (also ones you found via plasticity search/recall):
  %s used --session %s ID [ID...]
Use the plasticity:memory skill for guidance.`, st.Nodes, cfg.Scope, search, id, cli, cli, id, cli, id, cli, id)
	return contextOutput("SessionStart", text), nil
}

// CloseAbandoned closes up to limit sessions that ended without a SessionEnd
// hook (crash, kill, hook timeout), so their episodes (if enabled) are not
// lost. It runs
// in the async setup hook: closing embeds and may consolidate, which must
// not delay or block the session-start context. A session that fails to
// close is logged and skipped (it is retried at the next session start).
func CloseAbandoned(e *engine.Engine, cfg Config, limit int) (int, error) {
	stale, err := e.AbandonedSessions(cfg.SessionIdle, limit)
	if err != nil {
		return 0, err
	}
	closed := 0
	for _, s := range stale {
		files, _ := EditedFiles(s.Transcript, s.Project)
		if _, err := e.EndSession(s.ID, files, cfg.Episodes); err != nil {
			e.Log(s.ID, engine.EventError, map[string]string{"op": "close-session", "error": err.Error()})
			continue
		}
		closed++
	}
	return closed, nil
}

// Prompt recalls memories for the user's prompt and injects them (see
// PlanRecall for which prompts are recalled and how).
func Prompt(e *engine.Engine, in Input, project string, cfg Config) (*Output, error) {
	plan := cfg.PlanRecall(in.Prompt, ReadOutputTail)
	source := engine.RecallSourcePrompt
	if plan.Kind == KindNotification {
		source = engine.RecallSourceNotification
		if err := e.NoteNotification(in.SessionID, project, in.Prompt); err != nil {
			return nil, err
		}
	} else if err := e.NotePrompt(in.SessionID, project, in.Prompt); err != nil {
		return nil, err
	}
	if plan.Skip {
		return nil, nil
	}
	limit := cfg.RecallLimit
	if plan.Limit > 0 {
		limit = min(limit, plan.Limit)
	}
	start := time.Now()
	// Episodes quote raw prompts: they match many queries lexically and would
	// crowd knowledge out of the few injected slots. They stay reachable
	// through explicit recall and search.
	r, err := e.Recall(engine.RecallRequest{Query: plan.Query, Session: in.SessionID, Project: project, Limit: limit,
		MinConfidence: plan.MinConfidence, ExcludeTypes: []model.NodeType{model.TypeEpisode}})
	if err != nil {
		return nil, err
	}
	latency := time.Since(start)
	text := FormatMemories(r, in.SessionID, cfg.CLI(), cfg.ContextBudget)
	e.LogRecall(in.SessionID, source, plan.Query, r, latency, len(text))
	if text == "" {
		return nil, nil
	}
	return contextOutput("UserPromptSubmit", text), nil
}

// FormatMemories renders recalled memories as compact context within budget
// characters ("" if there are none); cli is the command prefix (Config.CLI).
func FormatMemories(r *engine.RecallResult, session, cli string, budget int) string {
	if len(r.Hits) == 0 {
		return ""
	}
	var b strings.Builder
	fmt.Fprintf(&b, "<plasticity-memories session=%q>\nPossibly relevant memories (verify before relying on them):\n", session)
	footer := fmt.Sprintf("If any helped, report them: %s used --session %s ID...\n</plasticity-memories>", cli, session)
	for _, h := range r.Hits {
		n := h.Node
		line := fmt.Sprintf("- [%s] %s: %s", n.ID, n.Type, truncate(n.Title, 120))
		if n.Content != "" {
			line += " — " + truncate(n.Content, 220)
		}
		if n.Rationale != "" {
			line += " Why: " + truncate(n.Rationale, 140)
		}
		if !h.IsSeed && len(h.PathTitles) > 1 {
			line += fmt.Sprintf(" (associated via %q)", truncate(h.PathTitles[len(h.PathTitles)-2], 60))
		}
		line += "\n"
		if b.Len()+len(line)+len(footer) > budget {
			break
		}
		b.WriteString(line)
	}
	b.WriteString(footer)
	return b.String()
}

// nudgeReason is the Stop-hook message asking Claude to consolidate the
// session into memory.
func nudgeReason(session, cli string) string {
	return fmt.Sprintf(`Before finishing, update long-term memory (plasticity) if this session produced anything worth keeping:
- durable knowledge (decisions with rationale, root causes, working procedures, user preferences):
  %s add -t <type> --title "..." -c "..." -r "why" --importance 0..1 --session %s
- memories that helped: %s used --session %s ID [ID...]
If nothing is worth keeping, simply finish; you will not be asked again in this session.`, cli, session, cli, session)
}

// Stop asks Claude, at most once per session and only after substantive
// work, to store or report memories.
func Stop(e *engine.Engine, in Input, cfg Config) (*Output, error) {
	if !cfg.Reflect || in.StopHookActive {
		return nil, nil
	}
	calls, err := CountToolCalls(in.TranscriptPath)
	if err != nil {
		return nil, err
	}
	ok, err := e.ShouldNudge(in.SessionID, calls)
	if err != nil || !ok {
		return nil, err
	}
	return &Output{Decision: "block", Reason: nudgeReason(in.SessionID, cfg.CLI())}, nil
}

// SessionEnd closes the session and, if enabled, creates its episode neuron
// from the transcript.
func SessionEnd(e *engine.Engine, in Input, project string, cfg Config) (*engine.EpisodeReport, error) {
	var files []string
	if cfg.Episodes {
		var err error
		if files, err = EditedFiles(in.TranscriptPath, project); err != nil {
			files = nil // a missing transcript must not lose the episode
		}
	}
	return e.EndSession(in.SessionID, files, cfg.Episodes)
}

func truncate(s string, n int) string {
	s = strings.Join(strings.Fields(s), " ")
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n-1]) + "…"
}
