package hooks

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/vhavlena/plasticity/internal/embed"
	"github.com/vhavlena/plasticity/internal/engine"
)

const (
	staleLock  = time.Hour      // a setup lock older than this is abandoned
	retryAfter = 24 * time.Hour // after a failed setup, retry at most daily
)

// Setup runs in the background (async SessionStart hook): it enables
// embeddings on first use (downloading the model) and indexes neurons that
// lack vectors. A lock file in home prevents concurrent setups; after a
// failure it waits retryAfter before trying again. It returns a short status.
func Setup(ctx context.Context, e *engine.Engine, cfg Config, home string) (string, error) {
	if cfg.Embeddings == "off" {
		return "embeddings disabled", nil
	}
	st, err := e.EmbedStatus()
	if err != nil {
		return "", err
	}
	if st.Config != nil { // already enabled: only fill gaps
		if st.Missing == 0 {
			return "up to date", nil
		}
		rep, err := e.Reindex(ctx, false)
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("indexed %d neurons", rep.Embedded), nil
	}

	if err := os.MkdirAll(home, 0o755); err != nil {
		return "", err
	}
	failed := filepath.Join(home, "setup.failed")
	if fi, err := os.Stat(failed); err == nil && time.Since(fi.ModTime()) < retryAfter {
		return "previous setup failed; retrying later", nil
	}
	lock := filepath.Join(home, "setup.lock")
	release, ok, err := acquire(lock)
	if err != nil || !ok {
		return "setup already running", err
	}
	defer release()

	provider := cfg.Embeddings
	if provider == "auto" || provider == "" {
		provider = embed.ProviderHugot
	}
	rep, err := e.EnableEmbedding(ctx, embed.Config{Provider: provider})
	if err != nil {
		_ = os.WriteFile(failed, []byte(err.Error()+"\n"), 0o644)
		e.Log("", engine.EventError, map[string]string{"op": "setup", "error": err.Error()})
		return "", err
	}
	_ = os.Remove(failed)
	e.Log("", "setup", map[string]any{"provider": provider, "embedded": rep.Embedded})
	return fmt.Sprintf("embeddings enabled (%s), indexed %d neurons", provider, rep.Embedded), nil
}

// acquire creates the lock file exclusively, taking over a stale one.
func acquire(path string) (release func(), ok bool, err error) {
	for attempt := 0; attempt < 2; attempt++ {
		f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
		if err == nil {
			fmt.Fprintf(f, "%d\n", os.Getpid())
			f.Close()
			return func() { os.Remove(path) }, true, nil
		}
		if !errors.Is(err, os.ErrExist) {
			return nil, false, err
		}
		fi, statErr := os.Stat(path)
		if statErr != nil || time.Since(fi.ModTime()) < staleLock {
			return nil, false, nil
		}
		os.Remove(path) // stale: previous setup died
	}
	return nil, false, nil
}
