package engine

import (
	"fmt"

	"github.com/vhavlena/plasticity/internal/model"
	"github.com/vhavlena/plasticity/internal/store"
)

// currentSuccessor follows the superseded_by chain from id to the neuron
// that is current (not superseded itself). It returns "" if the chain is
// broken (a successor was deleted) or loops. nodes caches the neurons read;
// missing ones are loaded and added.
func currentSuccessor(tx *store.Tx, id string, nodes map[string]*model.Node) (string, error) {
	seen := map[string]bool{}
	cur := id
	for {
		n := nodes[cur]
		if n == nil {
			m, err := tx.GetNodes([]string{cur})
			if err != nil {
				return "", err
			}
			if n = m[cur]; n == nil {
				return "", nil
			}
			nodes[cur] = n
		}
		if n.SupersededBy == "" {
			return cur, nil
		}
		if seen[cur] {
			return "", nil
		}
		seen[cur] = true
		cur = n.SupersededBy
	}
}

// supersede marks old as replaced by successor ("" clears the mark). It
// refuses self-supersession and chains that would loop back to old.
func supersede(tx *store.Tx, old, successor string) error {
	if _, err := tx.GetNode(old); err != nil {
		return err
	}
	if successor != "" {
		if successor == old {
			return fmt.Errorf("node %s cannot supersede itself", old)
		}
		if _, err := tx.GetNode(successor); err != nil {
			return err
		}
		// Existing chains are acyclic, so walking from successor ends.
		for cur := successor; cur != ""; {
			if cur == old {
				return fmt.Errorf("%s is already superseded (directly or not) by %s: a cycle", successor, old)
			}
			m, err := tx.GetNodes([]string{cur})
			if err != nil {
				return err
			}
			if m[cur] == nil {
				break // broken chain: cannot loop back
			}
			cur = m[cur].SupersededBy
		}
	}
	return tx.SetSupersededBy(old, successor)
}

// fillSuccessorTitles sets SuccessorTitle on superseded nodes.
func fillSuccessorTitles(tx *store.Tx, nodes ...*model.Node) error {
	var ids []string
	for _, n := range nodes {
		if n != nil && n.SupersededBy != "" {
			ids = append(ids, n.SupersededBy)
		}
	}
	if len(ids) == 0 {
		return nil
	}
	m, err := tx.GetNodes(ids)
	if err != nil {
		return err
	}
	for _, n := range nodes {
		if n != nil && n.SupersededBy != "" {
			if s := m[n.SupersededBy]; s != nil {
				n.SuccessorTitle = s.Title
			}
		}
	}
	return nil
}
