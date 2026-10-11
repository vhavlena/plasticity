package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/vhavlena/plasticity/internal/engine"
	"github.com/vhavlena/plasticity/internal/hooks"
	"github.com/vhavlena/plasticity/internal/paths"
)

// hookCmd groups the Claude Code hook handlers. They read the hook input
// JSON on stdin and write the hook output JSON on stdout. They never fail the
// user's turn: errors are reported on stderr and the process exits 0.
func (a *app) hookCmd() *cobra.Command {
	c := &cobra.Command{
		Use:    "hook",
		Short:  "Claude Code hook handlers (used by the plugin)",
		Hidden: true,
	}
	handler := func(name string, run func(e *engine.Engine, in hooks.Input, project string, cfg hooks.Config) (any, error)) *cobra.Command {
		return &cobra.Command{
			Use:  name,
			Args: cobra.NoArgs,
			RunE: func(cmd *cobra.Command, _ []string) error {
				if err := a.runHook(cmd, run); err != nil {
					fmt.Fprintf(os.Stderr, "plasticity hook %s: %v\n", name, err)
				}
				return nil
			},
		}
	}
	c.AddCommand(
		handler("session-start", func(e *engine.Engine, in hooks.Input, project string, cfg hooks.Config) (any, error) {
			return hooks.SessionStart(e, in, project, cfg)
		}),
		handler("prompt", func(e *engine.Engine, in hooks.Input, project string, cfg hooks.Config) (any, error) {
			return hooks.Prompt(e, in, project, cfg)
		}),
		handler("stop", func(e *engine.Engine, in hooks.Input, _ string, cfg hooks.Config) (any, error) {
			return hooks.Stop(e, in, cfg)
		}),
		handler("session-end", func(e *engine.Engine, in hooks.Input, project string, _ hooks.Config) (any, error) {
			_, err := hooks.SessionEnd(e, in, project)
			return nil, err // SessionEnd output is not shown to Claude
		}),
		handler("setup", func(e *engine.Engine, _ hooks.Input, _ string, cfg hooks.Config) (any, error) {
			status, err := hooks.Setup(context.Background(), e, cfg, paths.Home())
			if err == nil {
				fmt.Fprintln(os.Stderr, "plasticity setup:", status)
			}
			return nil, err
		}),
	)
	return c
}

func (a *app) runHook(cmd *cobra.Command, run func(*engine.Engine, hooks.Input, string, hooks.Config) (any, error)) error {
	var in hooks.Input
	if err := json.NewDecoder(a.in).Decode(&in); err != nil {
		return fmt.Errorf("decode hook input: %w", err)
	}
	if in.Cwd == "" {
		in.Cwd, _ = os.Getwd()
	}
	project := paths.Project(in.Cwd)
	if !cmd.Flags().Changed("db") && os.Getenv("PLASTICITY_DB") == "" {
		a.dbPath = paths.DB(in.Cwd) // the session's project, not the hook's cwd
	}
	cfg := hooks.ConfigFromEnv()
	cfg.DB = a.dbPath
	var opts []engine.Option
	if !cfg.Log {
		opts = append(opts, engine.WithoutLog())
	}
	e, err := engine.Open(a.dbPath, opts...)
	if err != nil {
		return err
	}
	defer e.Close()
	out, err := run(e, in, project, cfg)
	for _, w := range e.Warnings {
		fmt.Fprintln(os.Stderr, "warning:", w)
	}
	if err != nil {
		e.Log(in.SessionID, engine.EventError, map[string]string{"hook": cmd.Name(), "error": err.Error()})
		return err
	}
	if out == nil || isNilOutput(out) {
		return nil
	}
	return json.NewEncoder(a.out).Encode(out)
}

// isNilOutput reports a typed nil (*hooks.Output)(nil) wrapped in any.
func isNilOutput(v any) bool {
	o, ok := v.(*hooks.Output)
	return ok && o == nil
}
