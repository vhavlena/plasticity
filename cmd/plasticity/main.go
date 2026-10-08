// Command plasticity is the CLI of the neural knowledge graph. Every command
// prints JSON on stdout so it can be driven by Claude Code hooks and skills.
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/spf13/cobra"

	"github.com/vhavlena/plasticity/internal/engine"
	"github.com/vhavlena/plasticity/internal/model"
)

type app struct {
	dbPath string
	pretty bool
	out    io.Writer
	in     io.Reader
}

func main() {
	if err := newRoot(os.Stdout, os.Stdin).Execute(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func defaultDB() string {
	if p := os.Getenv("PLASTICITY_DB"); p != "" {
		return p
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "graph.db"
	}
	return filepath.Join(home, ".plasticity", "graph.db")
}

func (a *app) open() (*engine.Engine, error) { return engine.Open(a.dbPath) }

// with opens the engine, runs fn and prints its result as JSON.
func (a *app) with(fn func(e *engine.Engine) (any, error)) error {
	e, err := a.open()
	if err != nil {
		return err
	}
	defer e.Close()
	v, err := fn(e)
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
		SilenceUsage:  true,
		SilenceErrors: true,
	}
	root.SetOut(out)
	root.PersistentFlags().StringVar(&a.dbPath, "db", defaultDB(), "graph database file (env PLASTICITY_DB)")
	root.PersistentFlags().BoolVar(&a.pretty, "pretty", false, "indent JSON output")

	root.AddCommand(
		a.initCmd(), a.addCmd(), a.getCmd(), a.updateCmd(), a.deleteCmd(),
		a.linkCmd(), a.unlinkCmd(), a.searchCmd(), a.recallCmd(),
		a.reinforceCmd(), a.feedbackCmd(), a.consolidateCmd(),
		a.statsCmd(), a.exportCmd(), a.paramsCmd(),
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
	c := &cobra.Command{
		Use:   "add",
		Short: "Add a neuron (flags, or a JSON node on stdin with --stdin)",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			n := &model.Node{Type: model.NodeType(f.typ), Title: f.title, Content: f.content,
				Rationale: f.rationale, Tags: f.tags, Source: f.source, Importance: f.importance}
			if stdin {
				n = &model.Node{Type: model.TypeFact}
				if err := json.NewDecoder(a.in).Decode(n); err != nil {
					return fmt.Errorf("decode node from stdin: %w", err)
				}
			}
			specs, err := parseLinks(links)
			if err != nil {
				return err
			}
			return a.with(func(e *engine.Engine) (any, error) {
				added, err := e.AddNode(n)
				if err != nil {
					return nil, err
				}
				for _, l := range specs {
					if _, err := e.Link(added.ID, l.dst, l.kind, l.weight); err != nil {
						return nil, fmt.Errorf("node %s added, but link to %s failed: %w", added.ID, l.dst, err)
					}
				}
				return added, nil
			})
		},
	}
	f.register(c)
	c.Flags().BoolVar(&stdin, "stdin", false, "read the node as JSON from stdin")
	c.Flags().StringArrayVar(&links, "link", nil, "link new node → ID, as ID[:kind[:weight]] (repeatable)")
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
			return a.with(func(e *engine.Engine) (any, error) { return e.UpdateNode(args[0], p) })
		},
	}
	f.register(c)
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
			if req.Rerank != "" && req.Rerank != "ppr" {
				return fmt.Errorf("--rerank: only 'ppr' is supported")
			}
			return a.with(func(e *engine.Engine) (any, error) { return e.Recall(req) })
		},
	}
	c.Flags().StringVarP(&req.Session, "session", "s", "", "session ID; stores a trace for Hebbian learning")
	c.Flags().IntVarP(&req.Limit, "limit", "n", 0, "max results (default: result_limit)")
	c.Flags().Float64Var(&req.Threshold, "threshold", 0, "firing threshold θ (default: firing_threshold)")
	c.Flags().Float64Var(&req.HopDecay, "hop-decay", 0, "per-hop decay δ (default: hop_decay)")
	c.Flags().StringVar(&req.Rerank, "rerank", "", "rerank activated subgraph: ppr")
	c.Flags().BoolVar(&req.NoRecord, "no-record", false, "do not record accesses or traces")
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
	var pruneTyped bool
	c := &cobra.Command{
		Use:   "consolidate",
		Short: "Sleep phase: apply decay, prune weak synapses, homeostatic scaling, dormancy",
		Args:  cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			return a.with(func(e *engine.Engine) (any, error) { return e.Consolidate(pruneTyped) })
		},
	}
	c.Flags().BoolVar(&pruneTyped, "prune-typed", false, "also prune weak typed (explicit) edges")
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
	c.Flags().StringVarP(&format, "format", "f", "json", "json|dot")
	return c
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
