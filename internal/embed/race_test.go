//go:build race

package embed

// raceEnabled: the race detector also enables checkptr, which rejects the
// unsafe pointer arithmetic in gomlx's pure-Go matmul packing used by hugot.
// That is not a data race, but it aborts the process, so tests that run the
// real model skip themselves under -race.
const raceEnabled = true
