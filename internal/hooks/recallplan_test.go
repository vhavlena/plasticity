package hooks

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vhavlena/plasticity/internal/engine"
	"github.com/vhavlena/plasticity/internal/store"
)

// Shapes of real task notifications (paths and IDs shortened).
const (
	notifDone = `<task-notification>
<task-id>beeaj2ueu</task-id>
<tool-use-id>toolu_01GMZCVLizzwrSzrb4uVtco2</tool-use-id>
<output-file>/private/tmp/claude-501/x/0dd3b131/tasks/beeaj2ueu.output</output-file>
<status>completed</status>
<summary>Background command "Wait for full build" completed (exit code 0)</summary>
</task-notification>`
	notifAgent = `<task-notification>
<task-id>a70bf941f5aacfb56</task-id>
<output-file>/private/tmp/claude-501/x/tasks/a70bf941f5aacfb56.output</output-file>
<status>completed</status>
<summary>Agent "Build knowledge graph explainer" finished</summary>
<result>Saved the page to /Users/me/plasticity/.explain/page.html and added it to the index.</result>
</task-notification>`
	notifMonitor = `<task-notification>
<task-id>b88aau0ei</task-id>
<summary>Monitor event: "strict-bj results: norm-only vs full fix-sat-handling on verifit3"</summary>
<event>norm-only 7: timeout  :final-checks 928</event>
If this event is something the user would act on now, send a PushNotification.
</task-notification>`
	notifFailed = `<task-notification>
<task-id>bsuv0wncx</task-id>
<tool-use-id>toolu_016Wsr6Emu8CtJ4ZTsdGdR2B</tool-use-id>
<output-file>%OUT%</output-file>
<status>failed</status>
<summary>Background command "cd ~/verirego-bench/bench &amp;&amp; ./run_analysis.py ../input/weave-examples.input --config verirego" failed with exit code 1</summary>
</task-notification>`
)

func TestPlanRecall(t *testing.T) {
	c := DefaultConfig()
	if p := c.PlanRecall("why did we pick sqlite storage?", nil); p.Skip || p.Kind != KindPrompt || p.Query != "why did we pick sqlite storage?" || p.Limit != 0 || p.MinConfidence != 0 {
		t.Fatalf("plain prompt: %+v", p)
	}
	for _, s := range []string{"/compact", "ok go"} {
		if p := c.PlanRecall(s, nil); !p.Skip || p.Kind != KindTrivial {
			t.Fatalf("trivial %q: %+v", s, p)
		}
	}
	for _, s := range []string{notifDone, notifAgent, notifMonitor, "  \n" + notifDone} {
		if p := c.PlanRecall(s, nil); !p.Skip || p.Kind != KindNotification {
			t.Fatalf("successful notification must be skipped: %+v\n%s", p, s)
		}
	}

	out := filepath.Join(t.TempDir(), "bsuv0wncx.output")
	if err := os.WriteFile(out, []byte(strings.Repeat("progress line\n", 500)+"error: cannot locate VeriRego tool(s): analyses\n[exited with code 1]\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	failed := strings.ReplaceAll(notifFailed, "%OUT%", out)
	p := c.PlanRecall(failed, ReadOutputTail)
	if p.Skip || p.Kind != KindNotification || p.Limit != 0 || p.MinConfidence != 0.6 {
		t.Fatalf("failed notification: %+v", p)
	}
	for _, want := range []string{"cannot locate VeriRego tool(s): analyses", "run_analysis.py", "--config verirego"} {
		if !strings.Contains(p.Query, want) {
			t.Fatalf("query lacks %q: %q", want, p.Query)
		}
	}
	for _, noise := range []string{"/private/tmp", "toolu_", "<status>", "&amp;", "exit code", "Background command", "~/verirego-bench"} {
		if strings.Contains(p.Query, noise) {
			t.Fatalf("query keeps noise %q: %q", noise, p.Query)
		}
	}
	if n := len([]rune(p.Query)); n > c.NotificationMaxChars {
		t.Fatalf("query of %d chars exceeds the budget", n)
	}
	// Without the output file only the description is recalled.
	if p := c.PlanRecall(failed, nil); p.Skip || !strings.Contains(p.Query, "run_analysis.py") || strings.Contains(p.Query, "cannot locate") {
		t.Fatalf("failed notification without output: %+v", p)
	}
	killed := `<task-notification>
<status>killed</status>
<summary>Task "Rebuild release and reproduce issue 538" was stopped by the user</summary>
</task-notification>`
	if p := c.PlanRecall(killed, nil); !p.Skip {
		t.Fatalf("a task the user stopped is no failure: %+v", p)
	}
	// A non-zero exit code counts as a failure whatever the status says.
	if p := c.PlanRecall(strings.Replace(notifDone, "exit code 0", "exit code 255", 1), nil); p.Skip {
		t.Fatal("exit code 255 must be recalled")
	}
	all, off := c, c
	all.Notifications, off.Notifications = NotifyAll, NotifyOff
	if p := all.PlanRecall(notifDone, nil); p.Skip || p.Query != notifDone || p.MinConfidence != 0 {
		t.Fatalf("notifications=all recalls like a prompt: %+v", p)
	}
	if p := off.PlanRecall(failed, nil); !p.Skip {
		t.Fatalf("notifications=off: %+v", p)
	}
	// An unparsable notification is recalled strictly, not skipped.
	if p := c.PlanRecall("<task-notification>\nsomething about the lean build cache\n</task-notification>", nil); p.Skip || p.MinConfidence != 0.6 || p.Query != "something about the lean build cache" {
		t.Fatalf("unparsable notification: %+v", p)
	}
}

func TestReadOutputTail(t *testing.T) {
	dir := t.TempDir()
	out := filepath.Join(dir, "x.output")
	os.WriteFile(out, []byte(strings.Repeat("a", 10000)+"END"), 0o644)
	if s := ReadOutputTail(out); len(s) != maxTailBytes || !strings.HasSuffix(s, "END") {
		t.Fatalf("tail: %d bytes", len(s))
	}
	other := filepath.Join(dir, "secret.txt")
	os.WriteFile(other, []byte("secret"), 0o644)
	for _, p := range []string{other, "x.output", filepath.Join(dir, "missing.output"), dir} {
		if ReadOutputTail(p) != "" {
			t.Fatalf("%s must not be read", p)
		}
	}
}

func TestPromptHookNotifications(t *testing.T) {
	e := newEngine(t)
	cfg := DefaultConfig()
	cfg.Embeddings = "off"
	add(t, e, "verifit3 VeriRego experiment pitfalls",
		"Without source ~/.profile inside the screen job, run_analysis.py fails with cannot locate VeriRego tool(s): analyses")
	add(t, e, "Graph visualization: self-contained HTML", "plasticity viz renders the build output page")
	if _, err := SessionStart(e, Input{SessionID: "S", Source: "startup"}, "/repo", cfg); err != nil {
		t.Fatal(err)
	}
	if out, err := Prompt(e, Input{SessionID: "S", Prompt: notifDone}, "/repo", cfg); err != nil || out != nil {
		t.Fatalf("successful notification injected: %v %+v", err, out)
	}
	out := filepath.Join(t.TempDir(), "b.output")
	os.WriteFile(out, []byte("Traceback ...\nerror: cannot locate VeriRego tool(s): analyses\n"), 0o644)
	res, err := Prompt(e, Input{SessionID: "S", Prompt: strings.ReplaceAll(notifFailed, "%OUT%", out)}, "/repo", cfg)
	if err != nil || res == nil || !strings.Contains(res.Specific.AdditionalContext, "experiment pitfalls") {
		t.Fatalf("failed notification must recall the pitfall: %v %+v", err, res)
	}
	cfg.NotificationLimit = 1
	res, err = Prompt(e, Input{SessionID: "S", Prompt: strings.ReplaceAll(notifFailed, "%OUT%", out)}, "/repo", cfg)
	if err != nil || res == nil || strings.Count(res.Specific.AdditionalContext, "\n- [") != 1 {
		t.Fatalf("notification limit not applied: %v %+v", err, res)
	}
	// Notifications are not prompts of the session.
	s, _ := e.Session("S")
	if s.Prompts != 0 || s.FirstPrompt != "" {
		t.Fatalf("notifications counted as prompts: %+v", s)
	}
	notes, _ := e.Events(store.EventFilter{Session: "S", Kind: engine.EventNotification})
	recalls, _ := e.Events(store.EventFilter{Session: "S", Kind: engine.EventRecall})
	if len(notes) != 3 || len(recalls) != 2 || !strings.Contains(string(recalls[0].Payload), `"source":"notification"`) {
		t.Fatalf("events: %d notifications, recalls %+v", len(notes), recalls)
	}
}
