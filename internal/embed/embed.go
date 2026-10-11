// Package embed turns text into dense vectors for semantic seed search.
//
// Embedders are pluggable: anything implementing Embedder can be used, either
// passed to the engine directly (engine.WithEmbedder) or registered under a
// provider name (Register) so it can be selected by configuration and from
// the CLI (`plasticity embed enable <provider>`).
//
// Two free, local backends are built in: hugot (pure-Go ONNX inference,
// in-process, default model all-MiniLM-L6-v2) and Ollama (HTTP to a local
// Ollama server). Vectors are L2-normalized, so cosine similarity is a dot
// product.
package embed

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"
)

// Kind tells asymmetric models whether a text is a query or a document.
type Kind int

const (
	Query Kind = iota
	Document
)

// Embedder computes embeddings. Implementations must return vectors of a
// fixed dimension for a given ID.
type Embedder interface {
	// ID identifies the model (e.g. "hugot:sentence-transformers/all-MiniLM-L6-v2").
	// Vectors are stored per ID, so several embedders can be indexed side by
	// side and compared; vectors of different IDs are never mixed.
	ID() string
	// Embed returns the L2-normalized embedding of text. kind lets
	// asymmetric models treat queries and documents differently.
	Embed(ctx context.Context, text string, kind Kind) ([]float32, error)
	// Close releases resources (models, sessions).
	Close() error
}

// Factory builds an embedder from its configuration.
type Factory func(ctx context.Context, c Config) (Embedder, error)

// Provider is a registered embedder backend.
type Provider struct {
	Factory Factory
	// Defaults fills in unset configuration fields (may be nil).
	Defaults func(Config) Config
	// Prepare runs once when the provider is enabled, before the factory,
	// e.g. to download a model into modelsDir (may be nil).
	Prepare func(ctx context.Context, c Config, modelsDir string) (Config, error)
}

var (
	registryMu sync.RWMutex
	registry   = map[string]Provider{}
)

// Register makes a provider available under name. Registering a name twice
// replaces the earlier provider.
func Register(name string, p Provider) {
	registryMu.Lock()
	defer registryMu.Unlock()
	registry[name] = p
}

// Providers lists the registered provider names, sorted.
func Providers() []string {
	registryMu.RLock()
	defer registryMu.RUnlock()
	return providersLocked()
}

func lookup(name string) (Provider, error) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	p, ok := registry[name]
	if !ok {
		return Provider{}, fmt.Errorf("unknown embedding provider %q (registered: %s)", name, strings.Join(providersLocked(), ", "))
	}
	return p, nil
}

func providersLocked() []string {
	names := make([]string, 0, len(registry))
	for n := range registry {
		names = append(names, n)
	}
	slices.Sort(names)
	return names
}

// Providers.
const (
	ProviderHugot  = "hugot"
	ProviderOllama = "ollama"
)

// Defaults.
const (
	DefaultHugotModel  = "sentence-transformers/all-MiniLM-L6-v2"
	DefaultOllamaModel = "nomic-embed-text"
	DefaultOllamaHost  = "http://localhost:11434"
)

// Config selects and parameterizes an embedder. It is stored in the graph
// database, because all vectors of a graph must come from the same model.
type Config struct {
	Provider string `json:"provider"`
	Model    string `json:"model"`
	// Path is the local model directory (hugot).
	Path string `json:"path,omitempty"`
	// Host is the Ollama base URL.
	Host string `json:"host,omitempty"`
	// QueryPrefix and DocPrefix are prepended to texts for asymmetric
	// models (e.g. nomic-embed-text: "search_query: ", "search_document: ").
	QueryPrefix string `json:"query_prefix,omitempty"`
	DocPrefix   string `json:"doc_prefix,omitempty"`
	// Options carries provider-specific settings for custom providers.
	Options map[string]string `json:"options,omitempty"`
}

// ID returns provider:model.
func (c Config) ID() string { return c.Provider + ":" + c.Model }

// WithDefaults fills in unset fields using the provider's defaults.
func (c Config) WithDefaults() Config {
	if p, err := lookup(c.Provider); err == nil && p.Defaults != nil {
		return p.Defaults(c)
	}
	return c
}

// Prepare runs the provider's one-time setup (e.g. a model download) and
// returns the completed configuration.
func Prepare(ctx context.Context, c Config, modelsDir string) (Config, error) {
	p, err := lookup(c.Provider)
	if err != nil {
		return c, err
	}
	c = c.WithDefaults()
	if p.Prepare == nil {
		return c, nil
	}
	return p.Prepare(ctx, c, modelsDir)
}

// New creates the embedder described by c via its registered provider.
func New(ctx context.Context, c Config) (Embedder, error) {
	p, err := lookup(c.Provider)
	if err != nil {
		return nil, err
	}
	return p.Factory(ctx, c.WithDefaults())
}

// Prefix returns the configured prefix for queries or documents.
func (c Config) Prefix(kind Kind) string {
	if kind == Query {
		return c.QueryPrefix
	}
	return c.DocPrefix
}

// Truncate keeps the first maxWords whitespace-separated words of text.
// Small sentence models only attend to a few hundred tokens, and inference
// cost grows quadratically with length.
func Truncate(text string, maxWords int) string {
	words := strings.Fields(text)
	if len(words) > maxWords {
		words = words[:maxWords]
	}
	return strings.Join(words, " ")
}

// Normalize scales v to unit length in place (zero vectors stay zero).
func Normalize(v []float32) []float32 {
	var n float64
	for _, x := range v {
		n += float64(x) * float64(x)
	}
	if n == 0 {
		return v
	}
	inv := float32(1 / math.Sqrt(n))
	for i := range v {
		v[i] *= inv
	}
	return v
}

// Dot is the cosine similarity of two normalized vectors.
func Dot(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return float64(s)
}

// Encode serializes a vector as little-endian float32s.
func Encode(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return b
}

// Decode deserializes a vector written by Encode.
func Decode(b []byte) ([]float32, error) {
	if len(b)%4 != 0 {
		return nil, fmt.Errorf("vector blob of %d bytes is not a float32 array", len(b))
	}
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v, nil
}
