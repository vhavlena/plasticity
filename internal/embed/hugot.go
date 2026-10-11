package embed

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/pipelines"
)

func init() {
	Register(ProviderHugot, Provider{
		Factory: func(ctx context.Context, c Config) (Embedder, error) { return NewHugot(ctx, c) },
		Defaults: func(c Config) Config {
			if c.Model == "" {
				c.Model = DefaultHugotModel
			}
			return c
		},
		Prepare: func(ctx context.Context, c Config, modelsDir string) (Config, error) {
			if c.Path != "" {
				return c, nil
			}
			dir, err := DownloadHugotModel(ctx, modelsDir, c.Model)
			c.Path = dir
			return c, err
		},
	})
}

// Hugot runs a sentence-transformer ONNX model in-process with hugot's
// pure-Go backend (no cgo, no server).
type Hugot struct {
	cfg      Config
	session  *hugot.Session
	pipeline *pipelines.FeatureExtractionPipeline
}

// NewHugot loads the model from c.Path.
func NewHugot(ctx context.Context, c Config) (*Hugot, error) {
	if c.Path == "" {
		return nil, fmt.Errorf("hugot: no model path configured (run `plasticity embed enable hugot`)")
	}
	if _, err := os.Stat(filepath.Join(c.Path, "tokenizer.json")); err != nil {
		return nil, fmt.Errorf("hugot: model directory %s is missing or incomplete: %w", c.Path, err)
	}
	// hugot's Go backend compiles and caches one graph per distinct input
	// token length, with no limit and no padding (sequence buckets only pad
	// on the XLA backend; on Go they merely cap the cache, making further
	// lengths fail). Short-lived CLI calls see one or two lengths; a
	// long-running process grows by one graph per new length, bounded by the
	// truncation in Truncate (embed_max_words).
	s, err := hugot.NewGoSession(ctx)
	if err != nil {
		return nil, fmt.Errorf("hugot session: %w", err)
	}
	p, err := s.NewPipeline(hugot.FeatureExtractionConfig{
		ModelPath: c.Path,
		Name:      "plasticity",
		Options:   []hugot.FeatureExtractionOption{pipelines.WithNormalization()},
	})
	if err != nil {
		_ = s.Destroy()
		return nil, fmt.Errorf("hugot: load %s: %w", c.Path, err)
	}
	return &Hugot{cfg: c, session: s, pipeline: p}, nil
}

// ID implements Embedder.
func (h *Hugot) ID() string { return h.cfg.ID() }

// Embed implements Embedder. Texts are embedded one at a time: hugot v0.8.2's
// Go backend pads batches to size buckets and then indexes past the real
// inputs, so only bucket-sized batches (e.g. 1) are safe.
func (h *Hugot) Embed(ctx context.Context, text string, kind Kind) ([]float32, error) {
	out, err := h.pipeline.RunPipeline(ctx, []string{h.cfg.Prefix(kind) + text})
	if err != nil {
		return nil, fmt.Errorf("hugot embed: %w", err)
	}
	if len(out.Embeddings) != 1 {
		return nil, fmt.Errorf("hugot embed: got %d embeddings for 1 input", len(out.Embeddings))
	}
	return Normalize(append([]float32(nil), out.Embeddings[0]...)), nil
}

// Close implements Embedder.
func (h *Hugot) Close() error { return h.session.Destroy() }

// ModelDir is where DownloadHugotModel stores a Hugging Face model.
func ModelDir(modelsDir, model string) string {
	return filepath.Join(modelsDir, strings.ReplaceAll(model, "/", "_"))
}

// DownloadHugotModel fetches an ONNX sentence-transformer from Hugging Face
// into modelsDir (if not present) and returns its directory.
func DownloadHugotModel(ctx context.Context, modelsDir, model string) (string, error) {
	dir := ModelDir(modelsDir, model)
	if _, err := os.Stat(filepath.Join(dir, "tokenizer.json")); err == nil {
		return dir, nil
	}
	if err := os.MkdirAll(modelsDir, 0o755); err != nil {
		return "", err
	}
	opts := hugot.NewDownloadOptions()
	opts.OnnxFilePath = "onnx/model.onnx" // sentence-transformers repos keep the ONNX export there
	path, err := hugot.DownloadModel(ctx, model, modelsDir, opts)
	if err != nil {
		return "", fmt.Errorf("download %s: %w", model, err)
	}
	return path, nil
}
