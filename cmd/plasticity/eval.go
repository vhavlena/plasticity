package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/vhavlena/plasticity/internal/embed"
	"github.com/vhavlena/plasticity/internal/eval"
	"github.com/vhavlena/plasticity/internal/paths"
)

func (a *app) evalCmd() *cobra.Command {
	c := &cobra.Command{
		Use:   "eval",
		Short: "Evaluate recall and learning offline on scenarios",
		Long: `Scenarios are JSON files with nodes, explicit links and sessions of queries
labeled with their relevant nodes. 'eval run' replays them through the engine in
a temporary database (your graph is never touched) and compares configurations.`,
	}
	c.AddCommand(a.evalRunCmd(), a.evalGenCmd())
	return c
}

func (a *app) evalRunCmd() *cobra.Command {
	var (
		learn, embeds, sets        []string
		passes, k, consEvery       int
		jsonOut                    string
		withQueries                bool
		embModel, embPath, embHost string
	)
	c := &cobra.Command{
		Use:   "run SCENARIO.json",
		Short: "Replay a scenario under several configurations and compare them",
		Long: `Configurations are the cross product of --learn and --embed.

learning modes:
  none       recall only (baseline)
  reinforce  'reinforce --session' at the end of each session (unsupervised)
  feedback   after each recall, 'feedback' with the relevant nodes that were recalled
  explicit   after each recall, 'reinforce' with all relevant nodes (upper bound)

Metrics: recall/precision/MRR/nDCG of the top-k against the labels, and
assoc-recall: the share of relevant nodes sharing no word with the query that
were retrieved (only links, learning or embeddings can find those).`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			s, err := eval.Load(args[0])
			if err != nil {
				return err
			}
			params := map[string]string{}
			for _, kv := range sets {
				k, v, ok := strings.Cut(kv, "=")
				if !ok {
					return fmt.Errorf("--set %q: want key=value", kv)
				}
				params[k] = v
			}
			embedders, err := a.evalEmbedders(embeds, embed.Config{Model: embModel, Path: embPath, Host: embHost})
			if err != nil {
				return err
			}
			defer func() {
				for _, e := range embedders {
					if e != nil {
						e.Close()
					}
				}
			}()
			rep := &eval.Report{Scenario: s.Name, Nodes: len(s.Nodes), Links: len(s.Links),
				Sessions: len(s.Sessions), Queries: s.QueryCount(), K: k}
			for _, emb := range embedders {
				for _, mode := range learn {
					id := ""
					if emb != nil {
						id = emb.ID()
					}
					cfg := eval.Config{Name: eval.ConfigName(mode, id, params), Learn: mode, Embedder: emb,
						Params: params, Passes: passes, K: k, ConsolidateEvery: consEvery}
					fmt.Fprintf(os.Stderr, "running %s ...\n", cfg.Name)
					dir, err := os.MkdirTemp("", "plasticity-eval-")
					if err != nil {
						return err
					}
					res, err := eval.Run(s, cfg, dir)
					os.RemoveAll(dir)
					if err != nil {
						return fmt.Errorf("%s: %w", cfg.Name, err)
					}
					if !withQueries {
						res.Queries = nil
					}
					rep.Results = append(rep.Results, res)
				}
			}
			if jsonOut != "" {
				raw, _ := json.MarshalIndent(rep, "", "  ")
				if err := os.WriteFile(jsonOut, append(raw, '\n'), 0o644); err != nil {
					return err
				}
			}
			return rep.WriteText(a.out)
		},
	}
	c.Flags().StringSliceVar(&learn, "learn", []string{eval.LearnNone, eval.LearnReinforce, eval.LearnFeedback},
		"learning modes: "+strings.Join(eval.LearnModes, ","))
	c.Flags().StringSliceVar(&embeds, "embed", []string{"none"}, "embedders: none or registered providers ("+strings.Join(embed.Providers(), ",")+")")
	c.Flags().StringVar(&embModel, "embed-model", "", "model for the embedders (provider default if empty)")
	c.Flags().StringVar(&embPath, "embed-path", "", "hugot: local model directory")
	c.Flags().StringVar(&embHost, "embed-host", "", "ollama: server URL")
	c.Flags().StringArrayVar(&sets, "set", nil, "parameter override key=value for all configurations (repeatable)")
	c.Flags().IntVar(&passes, "passes", 2, "how many times the sessions are replayed (learning curve)")
	c.Flags().IntVarP(&k, "k", "k", 10, "results per recall")
	c.Flags().IntVar(&consEvery, "consolidate-every", 0, "consolidate after every n sessions (0: after each pass)")
	c.Flags().StringVar(&jsonOut, "json", "", "also write the full report as JSON to this file")
	c.Flags().BoolVar(&withQueries, "queries", false, "include per-query results in the JSON report")
	return c
}

// evalEmbedders builds one embedder per --embed entry (nil for "none").
func (a *app) evalEmbedders(names []string, base embed.Config) ([]embed.Embedder, error) {
	modelsDir := paths.ModelsDir()
	var out []embed.Embedder
	for _, n := range names {
		if n == "none" {
			out = append(out, nil)
			continue
		}
		cfg := base
		cfg.Provider = n
		cfg, err := embed.Prepare(context.Background(), cfg, modelsDir)
		if err != nil {
			return nil, err
		}
		e, err := embed.New(context.Background(), cfg)
		if err != nil {
			return nil, err
		}
		out = append(out, e)
	}
	return out, nil
}

func (a *app) evalGenCmd() *cobra.Command {
	c := eval.DefaultGenConfig()
	var out string
	cmd := &cobra.Command{
		Use:   "gen -o SCENARIO.json",
		Short: "Generate a synthetic scenario (topic clusters with hidden co-use groups)",
		Long: `Nodes belong to topics and, within a topic, to hidden groups that are "used
together". Each query names the unique words of one group member; all members
are relevant, so the others can only be found through links, learning or
embeddings. Query popularity over groups is Zipf-distributed. Use --noise-links
to add random learned edges for scale and performance testing.`,
		Args: cobra.NoArgs,
		RunE: func(*cobra.Command, []string) error {
			if out == "" {
				return fmt.Errorf("-o is required")
			}
			s, err := eval.Generate(c)
			if err != nil {
				return err
			}
			if err := s.Save(out); err != nil {
				return err
			}
			return a.print(map[string]any{"file": out, "nodes": len(s.Nodes), "links": len(s.Links),
				"sessions": len(s.Sessions), "queries": s.QueryCount()})
		},
	}
	f := cmd.Flags()
	f.StringVarP(&out, "out", "o", "", "output scenario file")
	f.IntVar(&c.Nodes, "nodes", c.Nodes, "number of nodes")
	f.IntVar(&c.Topics, "topics", c.Topics, "number of topics")
	f.IntVar(&c.GroupSize, "group-size", c.GroupSize, "nodes per hidden co-use group")
	f.IntVar(&c.Sessions, "sessions", c.Sessions, "number of sessions")
	f.IntVar(&c.QueriesPerSession, "queries", c.QueriesPerSession, "queries per session")
	f.Float64Var(&c.TypedLinkRate, "typed-link-rate", c.TypedLinkRate, "share of groups with an authored link")
	f.IntVar(&c.NoiseLinks, "noise-links", c.NoiseLinks, "random learned edges per node")
	f.Float64Var(&c.ZipfS, "zipf", c.ZipfS, "query popularity skew (Zipf exponent)")
	f.Int64Var(&c.Seed, "seed", c.Seed, "random seed")
	return cmd
}
