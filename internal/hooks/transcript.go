package hooks

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// transcriptEntry is the subset of a Claude Code transcript line (JSONL)
// needed here: assistant messages carry tool_use content blocks.
type transcriptEntry struct {
	Type    string `json:"type"`
	Message struct {
		Content json.RawMessage `json:"content"`
	} `json:"message"`
}

type contentBlock struct {
	Type  string `json:"type"`
	Name  string `json:"name"`
	Input struct {
		FilePath     string `json:"file_path"`
		NotebookPath string `json:"notebook_path"`
	} `json:"input"`
}

// toolUses streams the tool_use blocks of a transcript. Unparseable lines
// are skipped: the format is not a stable API.
func toolUses(path string, fn func(contentBlock)) error {
	if path == "" {
		return nil
	}
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 64<<20)
	for sc.Scan() {
		var e transcriptEntry
		if json.Unmarshal(sc.Bytes(), &e) != nil || e.Type != "assistant" || len(e.Message.Content) == 0 {
			continue
		}
		var blocks []contentBlock
		if json.Unmarshal(e.Message.Content, &blocks) != nil {
			continue // content may be a plain string
		}
		for _, b := range blocks {
			if b.Type == "tool_use" {
				fn(b)
			}
		}
	}
	return sc.Err()
}

// CountToolCalls returns the number of tool calls in a transcript.
func CountToolCalls(path string) (int, error) {
	n := 0
	err := toolUses(path, func(contentBlock) { n++ })
	return n, err
}

// EditedFiles returns the files written or edited in a transcript, sorted,
// relative to project when inside it.
func EditedFiles(path, project string) ([]string, error) {
	set := map[string]bool{}
	err := toolUses(path, func(b contentBlock) {
		switch b.Name {
		case "Edit", "Write", "MultiEdit", "NotebookEdit":
			p := b.Input.FilePath
			if p == "" {
				p = b.Input.NotebookPath
			}
			if p == "" {
				return
			}
			if project != "" {
				if rel, err := filepath.Rel(project, p); err == nil && !strings.HasPrefix(rel, "..") {
					p = rel
				}
			}
			set[p] = true
		}
	})
	files := make([]string, 0, len(set))
	for f := range set {
		files = append(files, f)
	}
	sort.Strings(files)
	return files, err
}
