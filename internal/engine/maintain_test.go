package engine

import (
	"testing"

	"github.com/vhavlena/plasticity/internal/model"
)

// A hand-made associative link of the old context weight between neurons no
// session paired is not a context link and is left alone.
func TestDemoteUnconfirmedSparesManualLinks(t *testing.T) {
	f := newFixture(t)
	if err := f.e.AddLinks([]LinkSpec{{Src: f.ids["pasta"], Dst: f.ids["hebb"], Kind: model.KindAssociative, Weight: 0.3}}); err != nil {
		t.Fatal(err)
	}
	if rep, err := f.e.DemoteUnconfirmed(0.3, 0.1, false); err != nil || rep.Demoted != 0 || f.edge("pasta", "hebb").Weight != 0.3 {
		t.Fatalf("manual link demoted: %v %+v", err, rep)
	}
}
