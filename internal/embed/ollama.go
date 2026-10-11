package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

func init() {
	Register(ProviderOllama, Provider{
		Factory: func(_ context.Context, c Config) (Embedder, error) { return NewOllama(c), nil },
		Defaults: func(c Config) Config {
			if c.Model == "" {
				c.Model = DefaultOllamaModel
			}
			if c.Host == "" {
				c.Host = DefaultOllamaHost
			}
			if strings.HasPrefix(c.Model, "nomic-embed-text") && c.QueryPrefix == "" && c.DocPrefix == "" {
				c.QueryPrefix, c.DocPrefix = "search_query: ", "search_document: "
			}
			return c
		},
	})
}

// Ollama calls a local Ollama server's /api/embed endpoint.
type Ollama struct {
	cfg    Config
	client *http.Client
}

// NewOllama returns an Ollama embedder; the server is contacted lazily.
func NewOllama(c Config) *Ollama {
	return &Ollama{cfg: c, client: &http.Client{Timeout: 10 * time.Second}}
}

// ID implements Embedder.
func (o *Ollama) ID() string { return o.cfg.ID() }

// Embed implements Embedder.
func (o *Ollama) Embed(ctx context.Context, text string, kind Kind) ([]float32, error) {
	body, _ := json.Marshal(map[string]any{"model": o.cfg.Model, "input": o.cfg.Prefix(kind) + text})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(o.cfg.Host, "/")+"/api/embed", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := o.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama at %s: %w", o.cfg.Host, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("ollama: %s: %s", resp.Status, strings.TrimSpace(string(msg)))
	}
	var out struct {
		Embeddings [][]float32 `json:"embeddings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("ollama: decode response: %w", err)
	}
	if len(out.Embeddings) != 1 || len(out.Embeddings[0]) == 0 {
		return nil, fmt.Errorf("ollama: expected one embedding, got %d", len(out.Embeddings))
	}
	return Normalize(out.Embeddings[0]), nil
}

// Close implements Embedder.
func (o *Ollama) Close() error { return nil }
