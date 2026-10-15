// Package version reports which plasticity build is running, so exports (and
// the HTML viewer) show the version that produced them.
package version

import "runtime/debug"

// Version is the plugin version, set at build time by the launcher with
// -ldflags "-X github.com/vhavlena/plasticity/internal/version.Version=...".
var Version = "dev"

// String returns Version, followed by the short commit of the source tree when
// it is known (development builds from a git checkout).
func String() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return Version
	}
	var rev string
	dirty := false
	for _, s := range info.Settings {
		switch s.Key {
		case "vcs.revision":
			rev = s.Value
		case "vcs.modified":
			dirty = s.Value == "true"
		}
	}
	if rev == "" {
		return Version
	}
	if len(rev) > 7 {
		rev = rev[:7]
	}
	if dirty {
		rev += "-dirty"
	}
	return Version + " (" + rev + ")"
}
