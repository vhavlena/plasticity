// Package viz renders the knowledge graph as a self-contained, interactive
// HTML page (force-directed layout, filters, search, neuron details). The page
// has no external dependencies, so it works offline and the graph never
// leaves the machine.
package viz

import (
	_ "embed"
	"encoding/json"
	"io"
	"strings"
)

//go:embed viz.html
var page string

// placeholder is replaced by the graph data in the page template.
const placeholder = "/*GRAPH_DATA*/null"

// Write renders the page with data (any JSON-marshalable value holding
// "nodes" and "edges") embedded in it.
func Write(w io.Writer, data any) error {
	js, err := json.Marshal(data)
	if err != nil {
		return err
	}
	// The JSON sits inside a <script> element: escape "</" so neuron text
	// cannot close it. json.Marshal already escapes <, > and & as < etc.,
	// this is a second line of defense.
	s := strings.ReplaceAll(string(js), "</", `<\/`)
	before, after, _ := strings.Cut(page, placeholder)
	_, err = io.WriteString(w, before+s+after)
	return err
}
