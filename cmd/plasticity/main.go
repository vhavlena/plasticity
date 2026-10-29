// Command plasticity is the CLI of the neural knowledge graph. Every command
// prints JSON on stdout so it can be driven by Claude Code hooks and skills.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/vhavlena/plasticity/internal/embed"
	"github.com/vhavlena/plasticity/internal/engine"
	"github.com/vhavlena/plasticity/internal/hooks"
	"github.com/vhavlena/plasticity/internal/model"
	"github.com/vhavlena/plasticity/internal/paths"
	"github.com/vhavlena/plasticity/internal/store"
	"github.com/vhavlena/plasticity/internal/version"
)

type app struct {
	dbPath  string
	pretty  bool
	noEmbed bool
	noLog   bool
	out     io.Writer
	in      io.Reader
}

func main() {
	if err := newRoot(os.Stdout, os.Stdin).Execute(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

// defaultDB resolves the database from PLASTICITY_DB or the scope (global
// graph, or the graph of the current project).
func defaultDB() string {
	wd, err := os.Getwd()
	if err != nil {
		wd = "."
	}
	return paths.DB(wd)
}

func (a *app) open() (*engine.Engine, error) {
	var opts []engine.Option
	if a.noEmbed {
		opts = append(opts, engine.WithoutEmbeddings())
	}
	if a.noLog || !hooks.ConfigFromEnv().Log {
		opts = append(opts, engine.WithoutLog())
	}
	return engine.Open(a.dbPath, opts...)
}

// with opens the engine, runs fn and prints its result as JSON on stdout.
// Engine warnings go to stderr so stdout stays machine-readable.
func (a *app) with(fn func(e *engine.Engine) (any, error)) error {
	e, err := a.open()
	if err != nil {
		return err
	}
	defer e.Close()
	v, err := fn(e)
	for _, w := range e.Warnings {
		fmt.Fprintln(os.Stderr, "warning:", w)
	}
	if err != nil {
		return err
	}
	return a.print(v)
}

func (a *app) print(v any) error {
	enc := json.NewEncoder(a.out)
	if a.pretty {
		enc.SetIndent("", "  ")
	}
	return enc.Encode(v)
}

func newRoot(out io.Writer, in io.Reader) *cobra.Command {
	a := &app{out: out, in: in}
	root := &cobra.Command{
		Use:           "plasticity",
		Short:         "Neural knowledge graph with Hebbian plasticity",
		Version:       version.String(),
		SilenceUsage:  true,
		SilenceErrors: true,
	}
	root.SetOut(out)
	root.PersistentFlags().StringVar(&a.dbPath, "db", defaultDB(), "graph database file (env PLASTICITY_DB; default by scope: $PLASTICITY_HOME/graph.db, or per project with PLASTICITY_SCOPE=project)")
	root.PersistentFlags().BoolVar(&a.pretty, "pretty", false, "indent JSON output")
	root.PersistentFlags().BoolVar(&a.noEmbed, "no-embed", false, "ignore embeddings: lexical seeding only (for comparisons)")
	root.PersistentFlags().BoolVar(&a.noLog, "no-log", false, "do not write the event log (env PLASTICITY_LOG=false)")

	root.AddCommand(
		a.initCmd(), a.addCmd(), a.usedCmd(), a.noteCmd(), a.logCmd(), a.hookCmd(), a.getCmd(), a.updateCmd(), a.deleteCmd(),
		a.linkCmd(), a.unlinkCmd(), a.searchCmd(), a.recallCmd(),
		a.reinforceCmd(), a.feedbackCmd(), a.consolidateCmd(), a.maintainCmd(),
		a.statsCmd(), a.sessionsCmd(), a.exportCmd(), a.vizCmd(), a.paramsCmd(), a.embedCmd(), a.evalCmd(),
	)
	return root
}

func (a *app) initCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "init",
		Short: "Create (or migrate) the graph database",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			return a.with(func(e *engine.Engine) (any, error) {
				return map[string]string{"db": a.dbPath}, nil
			})
		},
	}
}

// nodeFlags are shared by add and update.
type nodeFlags struct {
	typ, title, content, rationale, source string
	tags                                   []string
	importance                             float64
}

func (f *nodeFlags) register(c *cobra.Command) {
	c.Flags().StringVarP(&f.typ, "type", "t", string(model.TypeFact), "node type: "+joinTypes())
	c.Flags().StringVar(&f.title, "title", "", "short title")
	c.Flags().StringVarP(&f.content, "content", "c", "", "the information")
	c.Flags().StringVarP(&f.rationale, "rationale", "r", "", "why it holds / why it was decided")
	c.Flags().StringSliceVar(&f.tags, "tags", nil, "comma-separated tags")
	c.Flags().StringVar(&f.source, "source", "", "provenance (file, URL, session)")
	c.Flags().Float64Var(&f.importance, "importance", 0, "importance in [0,1]")
}

func joinTypes() string {
	s := make([]string, len(model.NodeTypes))
	for i, t := range model.NodeTypes {
		s[i] = string(t)
	}
	return strings.Join(s, "|")
}

func joinKinds() string {
	s := make([]string, len(model.EdgeKinds))
	for i, k := range model.EdgeKinds {
		s[i] = string(k)
	}
	return strings.Join(s, "|")
}

func (a *app) addCmd() *cobra.Command {
	var f nodeFlags
	var stdin bool
	var links []string
	var opt engine.AddOptions
	c := &cobra.Command{
		Use:   "add",
		Short: "Add a neuron (flags, or a JSON node on stdin with --stdin)",
		Long: `Adds a neuron after a duplicate check: if a neuron with the same title or a
near-identical embedding exists, nothing is inserted and the result names it in
"duplicate_of" (update or link it instead, or pass --force).

With --session, the neuron is linked to the neurons the session used or created
before it, so new knowledge is encoded in its context.

With --supersedes, the listed neurons are marked as replaced by the new one: the
prompt hook no longer injects them and recall redirects to the new neuron. The
result lists "similar" existing neurons; if one of them is outdated, mark it with
'update ID --superseded-by NEW' (or update its content if it is partly true).`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			n := &model.Node{Type: model.NodeType(f.typ), Title: f.title, Content: f.content,
				Rationale: f.rationale, Tags: f.tags, Source: f.source, Importance: f.importance}
			importanceSet := cmd.Flags().Changed("importance")
			if stdin {
				var raw map[string]json.RawMessage
				if err := json.NewDecoder(a.in).Decode(&raw); err != nil {
					return fmt.Errorf("decode node from stdin: %w", err)
				}
				_, importanceSet = raw["importance"]
				b, _ := json.Marshal(raw)
				n = &model.Node{Type: model.TypeFact}
				if err := json.Unmarshal(b, n); err != nil {
					return fmt.Errorf("decode node from stdin: %w", err)
				}
			}
			specs, err := parseLinks(links)
			if err != nil {
				return err
			}
			return a.with(func(e *engine.Engine) (any, error) {
				for _, l := range specs { // validate targets before inserting anything
					if _, err := e.GetNode(l.dst); err != nil {
						return nil, fmt.Errorf("--link: %w", err)
					}
				}
				res, err := e.AddNodeChecked(n, opt)
				if err != nil || res.Node == nil {
					return res, err
				}
				// Importance stays 0 by default (stored data keeps its meaning),
				// but a forgotten one is the most common way a neuron is mis-weighted.
				if !importanceSet {
					res.Warning = importanceWarning
					e.Warnings = append(e.Warnings, importanceWarning)
				}
				for _, l := range specs {
					if _, err := e.Link(res.Node.ID, l.dst, l.kind, l.weight); err != nil {
						return nil, fmt.Errorf("node %s added, but link to %s failed: %w", res.Node.ID, l.dst, err)
					}
				}
				return res, nil
			})
		},
	}
	f.register(c)
	c.Flags().BoolVar(&stdin, "stdin", false, "read the node as JSON from stdin")
	c.Flags().StringArrayVar(&links, "link", nil, "link new node → ID, as ID[:kind[:weight]] (repeatable)")
	c.Flags().StringVarP(&opt.Session, "session", "s", "", "session ID: link to the session's context and record as created by it")
	c.Flags().BoolVar(&opt.Force, "force", false, "insert even if a near-duplicate exists")
	c.Flags().StringSliceVar(&opt.Supersedes, "supersedes", nil, "IDs of neurons the new one replaces (they are no longer injected)")
	return c
}

const importanceWarning = "importance not set: pass --importance 0..1"

func (a *app) usedCmd() *cobra.Command {
	var session string
	c := &cobra.Command{
		Use:   "used ID [ID...]",
		Short: "Report neurons that were actually used (strengthens them together)",
		Long: `Report the memories that helped, including ones found via search or recall.
With --session this is feedback: the used neurons are associated with each other
and links to neurons recalled in the session but not used are weakened. Without
a session, at least two neurons are reinforced as used together.`,
		Args: cobra.MinimumNArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			return a.with(func(e *engine.Engine) (any, error) { return e.Used(session, args) })
		},
	}
	c.Flags().StringVarP(&session, "session", "s", "", "session ID")
	return c
}

func (a *app) noteCmd() *cobra.Command {
	var session, kind string
	var ids []string
	c := &cobra.Command{
		Use:   "note TEXT...",
		Short: "Log an observation about the memory (a miss, noise, a memory that helped)",
		Long: `Notes are your observations during evaluation, shown by 'eval report' next
to the measured numbers:
  miss    something stored was not recalled when it should have been
  noise   an injected memory was irrelevant
  helped  a memory clearly helped
  other   anything else`,
		Args: cobra.MinimumNArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			text := strings.Join(args, " ")
			return a.with(func(e *engine.Engine) (any, error) {
				if err := e.Note(session, kind, text, ids); err != nil {
					return nil, err
				}
				return map[string]string{"kind": kind, "text": text}, nil
			})
		},
	}
	c.Flags().StringVarP(&session, "session", "s", "", "session ID")
	c.Flags().StringVarP(&kind, "kind", "k", "other", "note kind: "+strings.Join(engine.NoteKinds, "|"))
	c.Flags().StringSliceVar(&ids, "id", nil, "neurons concerned (repeatable)")
	return c
}

func (a *app) logCmd() *cobra.Command {
	var f store.EventFilter
	var since time.Duration
	c := &cobra.Command{
		Use:   "log",
		Short: "Show the event log (recalls, used, added neurons, nudges, episodes)",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			if since > 0 {
				f.Since = time.Now().Add(-since)
			}
			return a.with(func(e *engine.Engine) (any, error) { return e.Events(f) })
		},
	}
	c.Flags().StringVarP(&f.Session, "session", "s", "", "only this session")
	c.Flags().StringVar(&f.Kind, "kind", "", "only this event kind")
	c.Flags().DurationVar(&since, "since", 0, "only events newer than this (e.g. 24h)")
	c.Flags().IntVarP(&f.Limit, "limit", "n", 0, "at most this many events")
	return c
}

type linkSpec struct {
	dst    string
	kind   model.EdgeKind
	weight float64
}

func parseLinks(specs []string) ([]linkSpec, error) {
	var out []linkSpec
	for _, s := range specs {
		parts := strings.Split(s, ":")
		l := linkSpec{dst: parts[0], kind: model.KindExplicit}
		if len(parts) > 1 {
			l.kind = model.EdgeKind(parts[1])
		}
		if len(parts) > 2 {
			w, err := strconv.ParseFloat(parts[2], 64)
			if err != nil {
				return nil, fmt.Errorf("--link %q: bad weight", s)
			}
			l.weight = w
		}
		if len(parts) > 3 || l.dst == "" || !l.kind.Valid() {
			return nil, fmt.Errorf("--link %q: want ID[:kind[:weight]] with kind %s", s, joinKinds())
		}
		out = append(out, l)
	}
	return out, nil
}

func (a *app) getCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "get ID",
		Short: "Show a neuron with its synapses",
		Args:  cobra.ExactArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			return a.with(func(e *engine.Engine) (any, error) { return e.GetNode(args[0]) })
		},
	}
}

func (a *app) updateCmd() *cobra.Command {
	var f nodeFlags
	var supersededBy string
	var clearSuperseded bool
	c := &cobra.Command{
		Use:   "update ID",
		Short: "Update fields of a neuron (only the given flags change)",
		Args:  cobra.ExactArgs(1),
		RunE: func(c *cobra.Command, args []string) error {
			var p engine.NodePatch
			ch := c.Flags().Changed
			if ch("type") {
				t := model.NodeType(f.typ)
				p.Type = &t
			}
			if ch("title") {
				p.Title = &f.title
			}
			if ch("content") {
				p.Content = &f.content
			}
			if ch("rationale") {
				p.Rationale = &f.rationale
			}
			if ch("tags") {
				p.Tags = &f.tags
			}
			if ch("source") {
				p.Source = &f.source
			}
			if ch("importance") {
				p.Importance = &f.importance
			}
			if ch("superseded-by") && clearSuperseded {
				return fmt.Errorf("--superseded-by and --clear-superseded exclude each other")
			}
			if ch("superseded-by") {
				p.SupersededBy = &supersededBy
			}
			if clearSuperseded {
				none := ""
				p.SupersededBy = &none
			}
			return a.with(func(e *engine.Engine) (any, error) { return e.UpdateNode(args[0], p) })
		},
	}
	f.register(c)
	c.Flags().StringVar(&supersededBy, "superseded-by", "", "mark the neuron as replaced by this ID")
	c.Flags().BoolVar(&clearSuperseded, "clear-superseded", false, "mark the neuron as current again")
	return c
}

func (a *app) deleteCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "delete ID",
		Short: "Delete a neuron and its synapses",
		Args:  cobra.ExactArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			return a.with(func(e *engine.Engine) (any, error) {
				return map[string]any{"deleted": args[0]}, e.DeleteNode(args[0])
			})
		},
	}
}

func (a *app) linkCmd() *cobra.Command {
	var kind string
	var weight float64
	c := &cobra.Command{
		Use:   "link SRC DST",
		Short: "Create or overwrite an explicit synapse SRC → DST",
		Args:  cobra.ExactArgs(2),
		RunE: func(_ *cobra.Command, args []string) error {
			return a.with(func(e *engine.Engine) (any, error) {
				return e.Link(args[0], args[1], model.EdgeKind(kind), weight)
			})
		},
	}
	c.Flags().StringVarP(&kind, "kind", "k", string(model.KindExplicit), "edge kind: "+joinKinds())
	c.Flags().Float64VarP(&weight, "weight", "w", 0, "weight in (0,1] (default: default_typed_weight)")
	return c
}

func (a *app) unlinkCmd() *cobra.Command {
	var kind string
	c := &cobra.Command{
		Use:   "unlink SRC DST",
		Short: "Remove a synapse",
		Args:  cobra.ExactArgs(2),
		RunE: func(_ *cobra.Command, args []string) error {
			return a.with(func(e *engine.Engine) (any, error) {
				ok, err := e.Unlink(args[0], args[1], model.EdgeKind(kind))
				return map[string]bool{"removed": ok}, err
			})
		},
	}
	c.Flags().StringVarP(&kind, "kind", "k", string(model.KindExplicit), "edge kind: "+joinKinds())
	return c
}

func (a *app) searchCmd() *cobra.Command {
	var limit int
	c := &cobra.Command{
		Use:   "search QUERY...",
		Short: "Lexical BM25 search (no spreading, no learning)",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			return a.with(func(e *engine.Engine) (any, error) { return e.Search(strings.Join(args, " "), limit) })
		},
	}
	c.Flags().IntVarP(&limit, "limit", "n", 10, "max results")
	return c
}

func (a *app) recallCmd() *cobra.Command {
	var req engine.RecallRequest
	c := &cobra.Command{
		Use:   "recall QUERY...",
		Short: "Associative recall: BM25 seeds + weight-gated spreading activation",
		Long: `Finds lexical seeds and spreads activation along weighted synapses; strong
connections carry activation further than weak ones. Accesses are recorded
(ACT-R) and, with --session, a trace is stored for 'reinforce --session'.`,
		Args: cobra.MinimumNArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			req.Query = strings.Join(args, " ")
			if req.Project != "" {
				req.Project = paths.Project(req.Project)
			}
			if req.Rerank != "" && req.Rerank != "ppr" {
				return fmt.Errorf("--rerank: only 'ppr' is supported")
			}
			return a.with(func(e *engine.Engine) (any, error) {
				start := time.Now()
				r, err := e.Recall(req)
				if err == nil && req.Session != "" && !req.NoRecord {
					e.LogRecall(req.Session, engine.RecallSourceClaude, req.Query, r, time.Since(start), 0)
				}
				return r, err
			})
		},
	}
	c.Flags().StringVarP(&req.Session, "session", "s", "", "session ID; stores a trace for Hebbian learning and logs the recall")
	c.Flags().IntVarP(&req.Limit, "limit", "n", 0, "max results (default: result_limit)")
	c.Flags().Float64Var(&req.Threshold, "threshold", 0, "firing threshold θ (default: firing_threshold)")
	c.Flags().Float64Var(&req.HopDecay, "hop-decay", 0, "per-hop decay δ (default: hop_decay)")
	c.Flags().StringVar(&req.Rerank, "rerank", "", "rerank activated subgraph: ppr")
	c.Flags().BoolVar(&req.NoRecord, "no-record", false, "do not record accesses or traces")
	c.Flags().StringVar(&req.Project, "project", "", "project directory of the recall (default: the session's); neurons of other projects seed weaker")
	return c
}

func (a *app) reinforceCmd() *cobra.Command {
	var session string
	c := &cobra.Command{
		Use:   "reinforce [ID...]",
		Short: "Hebbian learning: strengthen synapses between co-used neurons",
		Long: `With IDs: the given neurons were used together; every pair is potentiated.
With --session: consume the session's pending recall traces and potentiate
co-activated neurons proportionally to their activations.`,
		RunE: func(_ *cobra.Command, args []string) error {
			if (session == "") == (len(args) == 0) {
				return fmt.Errorf("give either node IDs or --session")
			}
			return a.with(func(e *engine.Engine) (any, error) {
				if session != "" {
					return e.ReinforceSession(session)
				}
				return e.Reinforce(args)
			})
		},
	}
	c.Flags().StringVarP(&session, "session", "s", "", "session whose recall traces to learn from")
	return c
}

func (a *app) feedbackCmd() *cobra.Command {
	var session string
	var used []string
	c := &cobra.Command{
		Use:   "feedback --session S --used ID,ID",
		Short: "Report which recalled neurons were useful (LTP for used, LTD for unused)",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			return a.with(func(e *engine.Engine) (any, error) { return e.Feedback(session, used) })
		},
	}
	c.Flags().StringVarP(&session, "session", "s", "", "session ID")
	c.Flags().StringSliceVarP(&used, "used", "u", nil, "IDs of neurons that were actually used")
	_ = c.MarkFlagRequired("session")
	_ = c.MarkFlagRequired("used")
	return c
}

func (a *app) consolidateCmd() *cobra.Command {
	c := &cobra.Command{
		Use:   "consolidate",
		Short: "Sleep phase: homeostatic scaling, dormancy, trace purge",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			return a.with(func(e *engine.Engine) (any, error) { return e.Consolidate() })
		},
	}
	return c
}

func (a *app) maintainCmd() *cobra.Command {
	c := &cobra.Command{
		Use:   "maintain",
		Short: "One-off repairs of an existing graph",
	}
	var from, to float64
	var dryRun bool
	demote := &cobra.Command{
		Use:   "demote-unconfirmed",
		Short: "Lower context links that were never co-used to the tentative weight",
		Long: "Context links made before they became tentative (weight 0.3) spread activation\n" +
			"although nothing confirmed them. This lowers associative links with no\n" +
			"co-activation whose weight is still --from to --to (default: context_link_weight).",
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			return a.with(func(e *engine.Engine) (any, error) {
				if !cmd.Flags().Changed("to") {
					to = e.P.ContextLinkWeight
				}
				return e.DemoteUnconfirmed(from, to, dryRun)
			})
		},
	}
	demote.Flags().Float64Var(&from, "from", 0.3, "weight of the links to demote (the old context_link_weight)")
	demote.Flags().Float64Var(&to, "to", 0, "new weight (default: context_link_weight)")
	demote.Flags().BoolVar(&dryRun, "dry-run", false, "only list the links that would change")
	c.AddCommand(demote)
	return c
}

func (a *app) statsCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "stats",
		Short: "Graph statistics",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			return a.with(func(e *engine.Engine) (any, error) { return e.Stats() })
		},
	}
}

func (a *app) sessionsCmd() *cobra.Command {
	var project string
	var limit int
	c := &cobra.Command{
		Use:   "sessions",
		Short: "List past Claude Code sessions with the neurons each used and created",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			return a.with(func(e *engine.Engine) (any, error) { return e.Sessions(project, limit) })
		},
	}
	c.Flags().StringVar(&project, "project", "", "only sessions of this project (git repository root)")
	c.Flags().IntVarP(&limit, "limit", "n", 20, "maximum number of sessions, most recent first")
	return c
}

func (a *app) exportCmd() *cobra.Command {
	var format string
	c := &cobra.Command{
		Use:   "export",
		Short: "Export the whole graph (json or Graphviz dot)",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			e, err := a.open()
			if err != nil {
				return err
			}
			defer e.Close()
			return e.Export(a.out, format)
		},
	}
	c.Flags().StringVarP(&format, "format", "f", "json", "json|dot|html")
	return c
}

func (a *app) vizCmd() *cobra.Command {
	var file string
	var noOpen, threeD bool
	c := &cobra.Command{
		Use:   "viz",
		Short: "Write an interactive HTML visualization of the graph and open it",
		Long: "Write the whole graph as a self-contained HTML page (no network access needed)\n" +
			"and open it in the browser. The page has a 2D and a 3D (WebGL) view.\n" +
			"The default file sits next to the database.",
		Args: cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			if file == "" {
				file = strings.TrimSuffix(a.dbPath, filepath.Ext(a.dbPath)) + ".html"
			}
			return a.with(func(e *engine.Engine) (any, error) {
				var buf bytes.Buffer
				view := ""
				if threeD {
					view = "3d"
				}
				if err := e.ExportHTML(&buf, view); err != nil {
					return nil, err
				}
				if err := os.WriteFile(file, buf.Bytes(), 0o600); err != nil {
					return nil, err
				}
				abs, err := filepath.Abs(file)
				if err != nil {
					return nil, err
				}
				res := map[string]any{"file": abs, "opened": false}
				if !noOpen {
					if err := openBrowser(abs); err != nil {
						fmt.Fprintln(os.Stderr, "warning: could not open a browser:", err)
					} else {
						res["opened"] = true
					}
				}
				return res, nil
			})
		},
	}
	c.Flags().StringVarP(&file, "output", "o", "", "HTML file to write (default: the database path with .html)")
	c.Flags().BoolVar(&noOpen, "no-open", false, "only write the file, do not open a browser")
	c.Flags().BoolVar(&threeD, "3d", false, "open in the 3D view (otherwise the view last used)")
	return c
}

// openBrowser opens path with the platform's default handler.
func openBrowser(path string) error {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", path)
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", path)
	default:
		cmd = exec.Command("xdg-open", path)
	}
	return cmd.Start()
}

func (a *app) paramsCmd() *cobra.Command {
	c := &cobra.Command{
		Use:   "params",
		Short: "Show or tune algorithm parameters",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			return a.with(func(e *engine.Engine) (any, error) { return e.P, nil })
		},
	}
	c.AddCommand(&cobra.Command{
		Use:   "set KEY VALUE",
		Short: "Override a parameter",
		Args:  cobra.ExactArgs(2),
		RunE: func(_ *cobra.Command, args []string) error {
			return a.with(func(e *engine.Engine) (any, error) {
				err := e.SetParam(args[0], args[1])
				for _, w := range e.P.Warnings() {
					fmt.Fprintln(os.Stderr, "warning:", w)
				}
				return e.P, err
			})
		},
	}, &cobra.Command{
		Use:   "reset KEY",
		Short: "Restore a parameter's default",
		Args:  cobra.ExactArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			return a.with(func(e *engine.Engine) (any, error) {
				err := e.ResetParam(args[0])
				return e.P, err
			})
		},
	})
	return c
}

func (a *app) embedCmd() *cobra.Command {
	c := &cobra.Command{
		Use:   "embed",
		Short: "Manage embeddings for semantic seed search",
	}
	status := &cobra.Command{
		Use:   "status",
		Short: "Show the embedder and how many neurons are indexed",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			return a.with(func(e *engine.Engine) (any, error) { return e.EmbedStatus() })
		},
	}
	var cfg embed.Config
	var opts []string
	enable := &cobra.Command{
		Use:   "enable PROVIDER",
		Short: "Select an embedder and index the neurons it has no vectors for",
		Long: `Registered providers: ` + strings.Join(embed.Providers(), ", ") + `

hugot   runs a sentence-transformer in-process (pure Go, no server). The model
        (default ` + embed.DefaultHugotModel + `, ~90 MB) is downloaded
        from Hugging Face next to the database unless --path points to a local copy.
ollama  calls a local Ollama server (default model ` + embed.DefaultOllamaModel + `,
        host ` + embed.DefaultOllamaHost + `); run 'ollama pull <model>' first.

Vectors are stored per embedder: switching back to an embedder used before
needs no re-indexing. Remove other embedders' vectors with 'embed prune'.`,
		Args:      cobra.ExactArgs(1),
		ValidArgs: embed.Providers(),
		RunE: func(_ *cobra.Command, args []string) error {
			cfg.Provider = args[0]
			for _, o := range opts {
				k, v, ok := strings.Cut(o, "=")
				if !ok {
					return fmt.Errorf("--opt %q: want key=value", o)
				}
				if cfg.Options == nil {
					cfg.Options = map[string]string{}
				}
				cfg.Options[k] = v
			}
			return a.with(func(e *engine.Engine) (any, error) {
				return e.EnableEmbedding(context.Background(), cfg)
			})
		},
	}
	enable.Flags().StringVar(&cfg.Model, "model", "", "model name (Hugging Face repo for hugot, Ollama model for ollama)")
	enable.Flags().StringVar(&cfg.Path, "path", "", "hugot: local model directory (skips the download)")
	enable.Flags().StringVar(&cfg.Host, "host", "", "ollama: server URL")
	enable.Flags().StringVar(&cfg.QueryPrefix, "query-prefix", "", "text prepended to queries (asymmetric models)")
	enable.Flags().StringVar(&cfg.DocPrefix, "doc-prefix", "", "text prepended to neurons (asymmetric models)")
	enable.Flags().StringArrayVar(&opts, "opt", nil, "provider-specific option key=value (repeatable)")
	var all bool
	reindex := &cobra.Command{
		Use:   "reindex",
		Short: "Embed neurons that are missing a vector or were edited (--all: every neuron)",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			return a.with(func(e *engine.Engine) (any, error) { return e.Reindex(context.Background(), all) })
		},
	}
	reindex.Flags().BoolVar(&all, "all", false, "re-embed every neuron")
	disable := &cobra.Command{
		Use:   "disable",
		Short: "Stop using embeddings (vectors are kept; see prune)",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			return a.with(func(e *engine.Engine) (any, error) {
				return map[string]bool{"enabled": false}, e.DisableEmbedding()
			})
		},
	}
	prune := &cobra.Command{
		Use:   "prune",
		Short: "Delete the vectors of all embedders except the active one",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			return a.with(func(e *engine.Engine) (any, error) {
				n, err := e.PruneVectors()
				return map[string]int64{"deleted": n}, err
			})
		},
	}
	c.AddCommand(status, enable, reindex, disable, prune)
	return c
}
