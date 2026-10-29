package viz

import (
	"bytes"
	"strings"
	"testing"
)

func TestWriteEmbedsDataSafely(t *testing.T) {
	var buf bytes.Buffer
	data := map[string]any{"nodes": []map[string]string{{"id": "a", "title": "</script><script>alert(1)</script>"}}}
	if err := Write(&buf, data); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	if strings.Contains(out, placeholder) || strings.Contains(out, "/*VIEW3D*/") || !strings.Contains(out, "const view3d") {
		t.Fatal("placeholder not replaced")
	}
	if n := strings.Count(out, "</script>"); n != 1 {
		t.Fatalf("neuron text closes the script element: %d </script> tags", n)
	}
	if !strings.Contains(out, `"id":"a"`) {
		t.Fatal("graph data missing")
	}
}
