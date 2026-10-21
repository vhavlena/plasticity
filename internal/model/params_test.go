package model

import "testing"

// Context links strong enough to spread activation are valid but warned
// about: they would act before co-use confirms them.
func TestContextLinkWeightWarning(t *testing.T) {
	p := DefaultParams()
	if err := p.Validate(); err != nil || len(p.Warnings()) != 0 {
		t.Fatalf("defaults: %v %v", err, p.Warnings())
	}
	p.ContextLinkWeight = 0.3
	if err := p.Validate(); err != nil || len(p.Warnings()) != 1 {
		t.Fatalf("0.3: %v %v", err, p.Warnings())
	}
}
