package hooks

import (
	"html"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// Prompt kinds, as classified by PlanRecall.
const (
	KindPrompt       = "prompt"       // a user prompt
	KindTrivial      = "trivial"      // slash command or acknowledgement
	KindNotification = "notification" // a <task-notification> delivered as a prompt
)

// RecallPlan is the prompt hook's decision how to recall for a prompt. The
// eval runner replays scenario prompts through the same function.
type RecallPlan struct {
	Kind string `json:"kind"`
	Skip bool   `json:"skip"`
	// Query is the text recalled: the prompt itself, or for a failed
	// background task its cleaned description and error output.
	Query string `json:"query,omitempty"`
	// Limit caps the hits (0: the configured recall limit).
	Limit int `json:"limit,omitempty"`
	// MinConfidence suppresses recalls with weaker query confidence.
	MinConfidence float64 `json:"min_confidence,omitempty"`
}

// Config.Notifications values.
const (
	NotifyFailed = "failed" // recall only for failed tasks (default)
	NotifyAll    = "all"    // recall every notification like a prompt
	NotifyOff    = "off"    // never recall for notifications
)

const notificationTag = "<task-notification>"

// IsNotification reports whether a prompt is a task notification of Claude
// Code (a finished background command, agent or monitor event).
func IsNotification(prompt string) bool {
	return strings.HasPrefix(strings.TrimSpace(prompt), notificationTag)
}

// trivial reports prompts that need no recall: slash commands and very
// short inputs (acknowledgements).
func trivial(prompt string) bool {
	p := strings.TrimSpace(prompt)
	return strings.HasPrefix(p, "/") || len(strings.Fields(p)) < 3
}

func tagRE(name string) *regexp.Regexp {
	return regexp.MustCompile(`(?s)<` + name + `>(.*?)</` + name + `>`)
}

var (
	statusRE     = tagRE("status")
	summaryRE    = tagRE("summary")
	resultRE     = tagRE("result")
	eventRE      = tagRE("event")
	outputFileRE = tagRE("output-file")
	exitCodeRE   = regexp.MustCompile(`exit(?:ed with)? code (\d+)`)
)

func tag(re *regexp.Regexp, s string) string {
	if m := re.FindStringSubmatch(s); m != nil {
		return strings.TrimSpace(m[1])
	}
	return ""
}

// PlanRecall decides how the prompt hook recalls for a prompt:
//
//   - slash commands and acknowledgements are skipped;
//   - task notifications of tasks that finished normally are skipped: they
//     are machine output, mostly paths and log words, and matched memories
//     by chance;
//   - a notification of a failed task (status failed or a non-zero exit
//     code; a task the user stopped is killed, not failed) is recalled on its cleaned description and the end of the
//     task's output (read by tail, nil: none), with a smaller limit and a
//     confidence threshold: a pitfall memory may explain the failure;
//   - an unparsable notification is recalled like a failed one, so a format
//     change of Claude Code loses no recall but cannot flood the context;
//   - any other prompt is recalled as it is.
//
// Config.Notifications can instead recall every notification like a prompt
// (the behavior before notifications were told apart) or none.
func (c Config) PlanRecall(prompt string, tail func(path string) string) RecallPlan {
	if !IsNotification(prompt) {
		if trivial(prompt) {
			return RecallPlan{Kind: KindTrivial, Skip: true}
		}
		return RecallPlan{Kind: KindPrompt, Query: prompt}
	}
	switch c.Notifications {
	case NotifyAll:
		return RecallPlan{Kind: KindNotification, Query: prompt}
	case NotifyOff:
		return RecallPlan{Kind: KindNotification, Skip: true}
	}
	strict := RecallPlan{Kind: KindNotification, Limit: c.NotificationLimit, MinConfidence: c.NotificationMinConfidence}
	status, summary := tag(statusRE, prompt), tag(summaryRE, prompt)
	if status == "" && summary == "" {
		strict.Query = truncateRunes(cleanNotification(prompt), c.NotificationMaxChars)
		strict.Skip = strict.Query == ""
		return strict
	}
	failed := status == "failed"
	if m := exitCodeRE.FindStringSubmatch(summary); m != nil {
		if code, err := strconv.Atoi(m[1]); err == nil && code != 0 {
			failed = true
		}
	}
	if !failed {
		return RecallPlan{Kind: KindNotification, Skip: true}
	}
	// The error output says most about the failure. The description is
	// often a long script whose words (hosts, flags, tools) match every
	// procedure using them, so with an error output it only gets a short
	// share of the budget.
	var out string
	if f := tag(outputFileRE, prompt); f != "" && tail != nil {
		out = cleanNotification(tail(html.UnescapeString(f)))
		out = lastRunes(out, c.NotificationMaxChars*3/4)
	}
	descBudget := c.NotificationMaxChars
	if out != "" {
		descBudget = c.NotificationMaxChars / 4
	}
	desc := cleanNotification(summary + "\n" + tag(resultRE, prompt) + "\n" + tag(eventRE, prompt))
	desc = truncateRunes(desc, descBudget)
	strict.Query = strings.TrimSpace(out + " " + desc)
	strict.Skip = strict.Query == ""
	return strict
}

var (
	xmlTagRE    = regexp.MustCompile(`</?[A-Za-z][\w-]*>`)
	absPathRE   = regexp.MustCompile(`(^|[\s"'=(:])(?:~|/)[^\s"'<>;|&)]*`)
	idRE        = regexp.MustCompile(`\btoolu_\w+|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b|\b[0-9a-f]{12,}\b`)
	boilerRE    = regexp.MustCompile(`(?i)background command|failed with exit code \d+|completed \(exit code \d+\)|\[exited with code \d+\]`)
	whitespaceR = regexp.MustCompile(`\s+`)
)

// cleanNotification strips what only adds noise to a recall query: XML tags,
// HTML entities, absolute paths, tool-use and session IDs and the
// notification boilerplate.
func cleanNotification(s string) string {
	s = html.UnescapeString(s)
	s = xmlTagRE.ReplaceAllString(s, " ")
	s = idRE.ReplaceAllString(s, " ")
	s = absPathRE.ReplaceAllString(s, "$1 ")
	s = boilerRE.ReplaceAllString(s, " ")
	return strings.TrimSpace(whitespaceR.ReplaceAllString(s, " "))
}

func truncateRunes(s string, n int) string {
	if n <= 0 {
		return ""
	}
	if r := []rune(s); len(r) > n {
		return string(r[:n])
	}
	return s
}

func lastRunes(s string, n int) string {
	if n <= 0 {
		return ""
	}
	if r := []rune(s); len(r) > n {
		return string(r[len(r)-n:])
	}
	return s
}

// maxTailBytes bounds what is read from a task's output file.
const maxTailBytes = 4096

// ReadOutputTail returns the end of a background task's output file. Only
// absolute paths of Claude Code task outputs (*.output) are read, so a
// prompt that merely looks like a notification cannot make the hook read
// other files; any problem yields "".
func ReadOutputTail(path string) string {
	if !filepath.IsAbs(path) || filepath.Ext(path) != ".output" {
		return ""
	}
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	if fi, err := f.Stat(); err != nil || !fi.Mode().IsRegular() {
		return ""
	} else if fi.Size() > maxTailBytes {
		if _, err := f.Seek(-maxTailBytes, io.SeekEnd); err != nil {
			return ""
		}
	}
	b, err := io.ReadAll(io.LimitReader(f, maxTailBytes))
	if err != nil {
		return ""
	}
	return string(b)
}
