package eval

import (
	"fmt"
	"io"
	"strings"
	"text/tabwriter"
)

// Report is the outcome of evaluating several configurations on a scenario.
type Report struct {
	Scenario string    `json:"scenario"`
	Nodes    int       `json:"nodes"`
	Links    int       `json:"links"`
	Sessions int       `json:"sessions"`
	Queries  int       `json:"queries_per_pass"`
	K        int       `json:"k"`
	Results  []*Result `json:"results"`
}

// WriteText prints a human-readable comparison.
func (r *Report) WriteText(w io.Writer) error {
	fmt.Fprintf(w, "scenario %s: %d nodes, %d links, %d sessions, %d queries/pass, k=%d\n\n",
		r.Scenario, r.Nodes, r.Links, r.Sessions, r.Queries, r.K)

	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "config\tpass\trecall\tassoc-recall\tprecision\tMRR\tnDCG\tempty\ttraced\t")
	for _, res := range r.Results {
		for i, a := range res.Passes {
			name := res.Config
			if i > 0 {
				name = ""
			}
			fmt.Fprintf(tw, "%s\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s\t%s\t\n",
				name, i+1, a.Recall, a.AssocRecall, a.Precision, a.MRR, a.NDCG, pct(a.EmptyRate), pct(a.TraceRate))
		}
	}
	if err := tw.Flush(); err != nil {
		return err
	}

	fmt.Fprintln(w, "\nnoise (noise: unlabeled queries that recalled anything; stale: queries recalling a superseded node, first: above the current one)")
	tw = tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "config\tpass\thits/q\tirrelevant\tcross-project\tnoise\tstale\tstale-first\tskipped\t")
	for _, res := range r.Results {
		for i, a := range res.Passes {
			name := res.Config
			if i > 0 {
				name = ""
			}
			noise, stale, staleFirst := "-", "-", "-"
			if a.Queries > a.Labeled {
				noise = pct(a.NoiseRate)
			}
			if a.StaleQueries > 0 {
				stale, staleFirst = pct(a.StaleRate), pct(a.StaleFirstRate)
			}
			fmt.Fprintf(tw, "%s\t%d\t%.1f\t%s\t%s\t%s\t%s\t%s\t%s\t\n", name, i+1, a.HitsPerRecall,
				pct(a.IrrelevantShare), pct(a.CrossProjectShare), noise, stale, staleFirst, pct(a.SkippedRate))
		}
	}
	if err := tw.Flush(); err != nil {
		return err
	}

	fmt.Fprintln(w, "\ngraph after run")
	tw = tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "config\tlearned\tspurious\tcross-project\tmean-w\tmax-deg\ttyped\tpreloaded\tpreloaded-w\tdormant\tpending\t")
	for _, res := range r.Results {
		g := res.Graph
		fmt.Fprintf(tw, "%s\t%d\t%d (%s)\t%d\t%.3f\t%d\t%d\t%d\t%.3f\t%d\t%d\t\n", res.Config, g.Learned, g.Spurious,
			pct(g.SpuriousRate), g.CrossProject, g.MeanWeight, g.MaxDegree, g.Typed, g.Preloaded, g.PreloadedWeight,
			g.Dormant, g.PendingTraces)
	}
	if err := tw.Flush(); err != nil {
		return err
	}

	for _, res := range r.Results {
		for i, warn := range res.Warnings {
			if i == 3 {
				fmt.Fprintf(w, "  ... %d more warnings for %s\n", len(res.Warnings)-3, res.Config)
				break
			}
			fmt.Fprintf(w, "warning [%s]: %s\n", res.Config, warn)
		}
	}

	fmt.Fprintln(w, "\nperformance (ms)")
	tw = tabwriter.NewWriter(w, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "config\tload\trecall p50\tp95\tmax\tlearn mean\tconsolidate mean\theap MB\tdb MB\t")
	for _, res := range r.Results {
		p := res.Perf
		fmt.Fprintf(tw, "%s\t%.0f\t%.2f\t%.2f\t%.2f\t%.2f\t%.1f\t%.1f\t%.1f\t\n", res.Config, p.LoadMS,
			p.Recall.P50, p.Recall.P95, p.Recall.Max, p.Learn.Mean, p.Consolidate.Mean, p.HeapInUseMB, p.DBSizeMB)
	}
	return tw.Flush()
}

func pct(x float64) string { return fmt.Sprintf("%.0f%%", 100*x) }

// ConfigName builds a label such as "feedback/hugot:all-MiniLM-L6-v2".
func ConfigName(learn, embedID string, params map[string]string) string {
	name := learn + "/" + shortEmbed(embedID)
	if len(params) > 0 {
		var kv []string
		for k, v := range params {
			kv = append(kv, k+"="+v)
		}
		name += "[" + strings.Join(kv, ",") + "]"
	}
	return name
}

func shortEmbed(id string) string {
	if id == "" {
		return "lexical"
	}
	if i := strings.LastIndex(id, "/"); i >= 0 {
		return id[:strings.Index(id, ":")+1] + id[i+1:]
	}
	return id
}
