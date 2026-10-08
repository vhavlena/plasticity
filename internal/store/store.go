// Package store persists the knowledge graph in SQLite (WAL mode). Several
// processes (CLI invocations from hooks, concurrent sessions) may use the same
// database file: readers never block, writers are serialized with BEGIN
// IMMEDIATE and a busy timeout.
package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	_ "modernc.org/sqlite" // registers the "sqlite" driver

	"github.com/vhavlena/plasticity/internal/model"
)

// ErrNotFound is returned when a node does not exist.
var ErrNotFound = errors.New("not found")

// Store is a handle to a graph database.
type Store struct {
	rdb *sql.DB // deferred transactions, many connections
	wdb *sql.DB // immediate transactions, single connection
}

const pragmas = "_pragma=busy_timeout(10000)&_pragma=journal_mode(WAL)&_pragma=synchronous(NORMAL)&_pragma=foreign_keys(1)"

// Open opens (creating if needed) the database at path and migrates it.
func Open(path string) (*Store, error) {
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("create db dir: %w", err)
		}
	}
	rdb, err := sql.Open("sqlite", path+"?"+pragmas)
	if err != nil {
		return nil, err
	}
	wdb, err := sql.Open("sqlite", path+"?"+pragmas+"&_txlock=immediate")
	if err != nil {
		rdb.Close()
		return nil, err
	}
	wdb.SetMaxOpenConns(1)
	s := &Store{rdb: rdb, wdb: wdb}
	if err := s.migrate(); err != nil {
		s.Close()
		return nil, fmt.Errorf("migrate %s: %w", path, err)
	}
	return s, nil
}

// Close releases the database.
func (s *Store) Close() error {
	return errors.Join(s.rdb.Close(), s.wdb.Close())
}

func (s *Store) migrate() error {
	return s.Write(func(tx *Tx) error {
		var v int
		if err := tx.tx.QueryRow(`PRAGMA user_version`).Scan(&v); err != nil {
			return err
		}
		for ; v < len(migrations); v++ {
			if _, err := tx.tx.Exec(migrations[v]); err != nil {
				return fmt.Errorf("migration %d: %w", v+1, err)
			}
		}
		_, err := tx.tx.Exec(fmt.Sprintf(`PRAGMA user_version = %d`, v))
		return err
	})
}

// Tx is a transaction over the graph.
type Tx struct{ tx *sql.Tx }

// Read runs fn in a read transaction (a consistent snapshot).
func (s *Store) Read(fn func(*Tx) error) error { return run(s.rdb, fn) }

// Write runs fn in a write transaction (BEGIN IMMEDIATE).
func (s *Store) Write(fn func(*Tx) error) error { return run(s.wdb, fn) }

func run(db *sql.DB, fn func(*Tx) error) error {
	tx, err := db.BeginTx(context.Background(), nil)
	if err != nil {
		return err
	}
	if err := fn(&Tx{tx}); err != nil {
		_ = tx.Rollback()
		return err
	}
	return tx.Commit()
}

func ts(t time.Time) int64     { return t.UnixNano() }
func fromTS(v int64) time.Time { return time.Unix(0, v).UTC() }
func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}
func tagsJSON(tags []string) string {
	if tags == nil {
		tags = []string{}
	}
	b, _ := json.Marshal(tags)
	return string(b)
}

// ---------------------------------------------------------------- nodes

const nodeCols = `id, type, title, content, rationale, tags, source, importance, dormant, fires, created_tick, created_at, updated_at`

type scanner interface{ Scan(dest ...any) error }

// scanNode scans the nodeCols columns followed by any extra columns.
func scanNode(r scanner, extra ...any) (*model.Node, error) {
	var n model.Node
	var tags string
	var dormant int
	var created, updated int64
	dest := append([]any{&n.ID, &n.Type, &n.Title, &n.Content, &n.Rationale, &tags, &n.Source,
		&n.Importance, &dormant, &n.Fires, &n.CreatedTick, &created, &updated}, extra...)
	if err := r.Scan(dest...); err != nil {
		return nil, err
	}
	if err := json.Unmarshal([]byte(tags), &n.Tags); err != nil {
		return nil, fmt.Errorf("node %s: bad tags: %w", n.ID, err)
	}
	n.Dormant = dormant != 0
	n.CreatedAt, n.UpdatedAt = fromTS(created), fromTS(updated)
	return &n, nil
}

// InsertNode adds a new node.
func (t *Tx) InsertNode(n *model.Node) error {
	_, err := t.tx.Exec(`INSERT INTO nodes (`+nodeCols+`) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)`,
		n.ID, n.Type, n.Title, n.Content, n.Rationale, tagsJSON(n.Tags), n.Source,
		n.Importance, b2i(n.Dormant), n.Fires, n.CreatedTick, ts(n.CreatedAt), ts(n.UpdatedAt))
	return err
}

// UpdateNode overwrites all mutable fields of an existing node.
func (t *Tx) UpdateNode(n *model.Node) error {
	res, err := t.tx.Exec(`UPDATE nodes SET type=?, title=?, content=?, rationale=?, tags=?, source=?,
		importance=?, dormant=?, updated_at=? WHERE id=?`,
		n.Type, n.Title, n.Content, n.Rationale, tagsJSON(n.Tags), n.Source,
		n.Importance, b2i(n.Dormant), ts(n.UpdatedAt), n.ID)
	if err != nil {
		return err
	}
	if k, _ := res.RowsAffected(); k == 0 {
		return fmt.Errorf("node %s: %w", n.ID, ErrNotFound)
	}
	return nil
}

// SetDormant updates the dormancy flag.
func (t *Tx) SetDormant(id string, dormant bool) error {
	_, err := t.tx.Exec(`UPDATE nodes SET dormant=? WHERE id=?`, b2i(dormant), id)
	return err
}

// GetNode returns a node by ID.
func (t *Tx) GetNode(id string) (*model.Node, error) {
	n, err := scanNode(t.tx.QueryRow(`SELECT `+nodeCols+` FROM nodes WHERE id=?`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, fmt.Errorf("node %s: %w", id, ErrNotFound)
	}
	return n, err
}

// GetNodes returns the existing nodes among ids.
func (t *Tx) GetNodes(ids []string) (map[string]*model.Node, error) {
	out := make(map[string]*model.Node, len(ids))
	for _, chunk := range chunks(ids, 500) {
		rows, err := t.tx.Query(`SELECT `+nodeCols+` FROM nodes WHERE id IN (`+placeholders(len(chunk))+`)`, anys(chunk)...)
		if err != nil {
			return nil, err
		}
		for rows.Next() {
			n, err := scanNode(rows)
			if err != nil {
				rows.Close()
				return nil, err
			}
			out[n.ID] = n
		}
		if err := rows.Close(); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// DeleteNode removes a node and (by cascade) its edges, accesses and traces.
func (t *Tx) DeleteNode(id string) error {
	res, err := t.tx.Exec(`DELETE FROM nodes WHERE id=?`, id)
	if err != nil {
		return err
	}
	if k, _ := res.RowsAffected(); k == 0 {
		return fmt.Errorf("node %s: %w", id, ErrNotFound)
	}
	return nil
}

// NodeIDsAfter pages through node IDs in order.
func (t *Tx) NodeIDsAfter(after string, limit int) ([]string, error) {
	rows, err := t.tx.Query(`SELECT id FROM nodes WHERE id > ? ORDER BY id LIMIT ?`, after, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}

// ---------------------------------------------------------------- search

// SearchHit is a full-text match. Score is −bm25, so higher is better.
type SearchHit struct {
	Node  *model.Node `json:"node"`
	Score float64     `json:"score"`
}

// Search runs an FTS5 MATCH expression ranked by weighted BM25 with column
// weights (title, content, rationale, tags).
func (t *Tx) Search(match string, weights [4]float64, limit int) ([]SearchHit, error) {
	if strings.TrimSpace(match) == "" {
		return nil, nil
	}
	cols := "n." + strings.ReplaceAll(nodeCols, ", ", ", n.")
	rows, err := t.tx.Query(`SELECT `+cols+`, bm25(nodes_fts, ?, ?, ?, ?) AS rank
		FROM nodes_fts JOIN nodes n ON n.rowid = nodes_fts.rowid
		WHERE nodes_fts MATCH ? ORDER BY rank LIMIT ?`,
		weights[0], weights[1], weights[2], weights[3], match, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var hits []SearchHit
	for rows.Next() {
		var rank float64
		n, err := scanNode(rows, &rank)
		if err != nil {
			return nil, err
		}
		hits = append(hits, SearchHit{Node: n, Score: -rank})
	}
	return hits, rows.Err()
}

// NodeCount returns the number of nodes.
func (t *Tx) NodeCount() (int, error) {
	var n int
	err := t.tx.QueryRow(`SELECT count(*) FROM nodes`).Scan(&n)
	return n, err
}

// DocFreq returns the number of nodes matching the FTS5 expression (normally
// a single quoted term, so stemming is applied as in search).
func (t *Tx) DocFreq(match string) (int, error) {
	var n int
	err := t.tx.QueryRow(`SELECT count(*) FROM nodes_fts WHERE nodes_fts MATCH ?`, match).Scan(&n)
	return n, err
}

// MatchingAmong returns which of ids match the FTS5 expression.
func (t *Tx) MatchingAmong(match string, ids []string) (map[string]bool, error) {
	out := map[string]bool{}
	for _, chunk := range chunks(ids, 500) {
		rows, err := t.tx.Query(`SELECT n.id FROM nodes_fts JOIN nodes n ON n.rowid = nodes_fts.rowid
			WHERE nodes_fts MATCH ? AND n.id IN (`+placeholders(len(chunk))+`)`,
			append([]any{match}, anys(chunk)...)...)
		if err != nil {
			return nil, err
		}
		for rows.Next() {
			var id string
			if err := rows.Scan(&id); err != nil {
				rows.Close()
				return nil, err
			}
			out[id] = true
		}
		if err := rows.Close(); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// ---------------------------------------------------------------- edges

// edgeCols are the persisted edge columns.
const edgeCols = `src, dst, kind, weight, stability, src_mark, dst_mark, last_update, coactivations`

// edgeSelect reads edges (aliased e) together with the endpoints' current
// fire counts, which define the edges' ages.
const edgeSelect = `SELECT e.src, e.dst, e.kind, e.weight, e.stability, e.src_mark, e.dst_mark,
	e.last_update, e.coactivations, s.fires, d.fires
	FROM edges e JOIN nodes s ON s.id = e.src JOIN nodes d ON d.id = e.dst `

func scanEdge(r scanner) (*model.Edge, error) {
	var e model.Edge
	var lu int64
	if err := r.Scan(&e.Src, &e.Dst, &e.Kind, &e.Weight, &e.Stability, &e.SrcMark, &e.DstMark,
		&lu, &e.Coactivations, &e.SrcFires, &e.DstFires); err != nil {
		return nil, err
	}
	e.LastUpdate = fromTS(lu)
	return &e, nil
}

func scanEdges(rows *sql.Rows, err error) ([]*model.Edge, error) {
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []*model.Edge
	for rows.Next() {
		e, err := scanEdge(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, e)
	}
	return out, rows.Err()
}

// GetEdge returns the edge (src, dst, kind) or nil if it does not exist.
// Associative edges are looked up in canonical order.
func (t *Tx) GetEdge(src, dst string, kind model.EdgeKind) (*model.Edge, error) {
	if kind == model.KindAssociative {
		src, dst = model.CanonicalPair(src, dst)
	}
	e, err := scanEdge(t.tx.QueryRow(edgeSelect+`WHERE e.src=? AND e.dst=? AND e.kind=?`, src, dst, kind))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	return e, err
}

// UpsertEdge inserts or replaces an edge.
func (t *Tx) UpsertEdge(e *model.Edge) error {
	if e.Kind == model.KindAssociative && e.Dst < e.Src {
		e.Src, e.Dst = e.Dst, e.Src
		e.SrcMark, e.DstMark = e.DstMark, e.SrcMark
		e.SrcFires, e.DstFires = e.DstFires, e.SrcFires
	}
	_, err := t.tx.Exec(`INSERT INTO edges (`+edgeCols+`) VALUES (?,?,?,?,?,?,?,?,?)
		ON CONFLICT (src, dst, kind) DO UPDATE SET weight=excluded.weight, stability=excluded.stability,
		src_mark=excluded.src_mark, dst_mark=excluded.dst_mark,
		last_update=excluded.last_update, coactivations=excluded.coactivations`,
		e.Src, e.Dst, e.Kind, e.Weight, e.Stability, e.SrcMark, e.DstMark, ts(e.LastUpdate), e.Coactivations)
	return err
}

// DeleteEdge removes an edge.
func (t *Tx) DeleteEdge(src, dst string, kind model.EdgeKind) (bool, error) {
	if kind == model.KindAssociative {
		src, dst = model.CanonicalPair(src, dst)
	}
	res, err := t.tx.Exec(`DELETE FROM edges WHERE src=? AND dst=? AND kind=?`, src, dst, kind)
	if err != nil {
		return false, err
	}
	k, _ := res.RowsAffected()
	return k > 0, nil
}

// EdgesOf returns up to limit edges incident to id (either direction),
// strongest stored weight first. limit ≤ 0 means no limit.
func (t *Tx) EdgesOf(id string, limit int) ([]*model.Edge, error) {
	if limit <= 0 {
		limit = -1
	}
	return scanEdges(t.tx.Query(`SELECT * FROM (
			`+edgeSelect+`WHERE e.src = ?
			UNION ALL
			`+edgeSelect+`WHERE e.dst = ? AND e.src <> ?
		) ORDER BY 4 DESC, 1, 2, 3 LIMIT ?`, id, id, id, limit))
}

// Degree returns the number of edges incident to id.
func (t *Tx) Degree(id string) (int, error) {
	var n int
	err := t.tx.QueryRow(`SELECT (SELECT count(*) FROM edges WHERE src=?) + (SELECT count(*) FROM edges WHERE dst=? AND src<>?)`,
		id, id, id).Scan(&n)
	return n, err
}

// EdgeKey identifies an edge for paging.
type EdgeKey struct {
	Src, Dst string
	Kind     model.EdgeKind
}

// EdgesAfter pages through all edges in primary-key order.
func (t *Tx) EdgesAfter(after EdgeKey, limit int) ([]*model.Edge, error) {
	return scanEdges(t.tx.Query(edgeSelect+`WHERE (e.src, e.dst, e.kind) > (?, ?, ?)
		ORDER BY e.src, e.dst, e.kind LIMIT ?`, after.Src, after.Dst, after.Kind, limit))
}

// ---------------------------------------------------------------- firing & clock

// Tick returns the global recall clock.
func (t *Tx) Tick() (int64, error) {
	var v int64
	err := t.tx.QueryRow(`SELECT value FROM counters WHERE name = 'tick'`).Scan(&v)
	return v, err
}

// AdvanceTick increments the global recall clock and returns the new value.
func (t *Tx) AdvanceTick() (int64, error) {
	var v int64
	err := t.tx.QueryRow(`UPDATE counters SET value = value + 1 WHERE name = 'tick' RETURNING value`).Scan(&v)
	return v, err
}

// Fire records that the given neurons fired together in one recall: each
// fire count is incremented (ageing their synapses), except that synapses
// between two co-firing neurons have their marks advanced too, so being
// recalled together never ages a synapse.
func (t *Tx) Fire(ids []string) error {
	if len(ids) == 0 {
		return nil
	}
	ph, args := placeholders(len(ids)), anys(ids)
	if _, err := t.tx.Exec(`UPDATE nodes SET fires = fires + 1 WHERE id IN (`+ph+`)`, args...); err != nil {
		return err
	}
	_, err := t.tx.Exec(`UPDATE edges SET src_mark = src_mark + 1, dst_mark = dst_mark + 1
		WHERE src IN (`+ph+`) AND dst IN (`+ph+`)`, append(args, args...)...)
	return err
}

// ---------------------------------------------------------------- accesses

// RecordAccess appends an access at the given recall tick and keeps only the
// newest max accesses.
func (t *Tx) RecordAccess(id string, tick int64, max int) error {
	if _, err := t.tx.Exec(`INSERT INTO node_access (node_id, tick) VALUES (?, ?)`, id, tick); err != nil {
		return err
	}
	_, err := t.tx.Exec(`DELETE FROM node_access WHERE node_id = ? AND rowid NOT IN (
		SELECT rowid FROM node_access WHERE node_id = ? ORDER BY tick DESC, rowid DESC LIMIT ?)`, id, id, max)
	return err
}

// Accesses returns the access ticks of the given nodes.
func (t *Tx) Accesses(ids []string) (map[string][]int64, error) {
	out := map[string][]int64{}
	for _, chunk := range chunks(ids, 500) {
		rows, err := t.tx.Query(`SELECT node_id, tick FROM node_access WHERE node_id IN (`+placeholders(len(chunk))+`)`, anys(chunk)...)
		if err != nil {
			return nil, err
		}
		for rows.Next() {
			var id string
			var v int64
			if err := rows.Scan(&id, &v); err != nil {
				rows.Close()
				return nil, err
			}
			out[id] = append(out[id], v)
		}
		if err := rows.Close(); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// ---------------------------------------------------------------- traces

// TraceEntry is one activated node of a recall.
type TraceEntry struct {
	NodeID     string
	Activation float64
	// Path is the strongest activation path that reached the node, starting
	// at its seed and ending at the node itself.
	Path []string
}

// Trace is the activation pattern of one recall.
type Trace struct {
	RecallID string
	Session  string
	Tick     int64 // global recall tick of the recall
	Entries  []TraceEntry
}

// InsertTrace stores a recall trace.
func (t *Tx) InsertTrace(tr *Trace) error {
	for _, e := range tr.Entries {
		path, err := json.Marshal(e.Path)
		if err != nil {
			return err
		}
		if _, err := t.tx.Exec(`INSERT INTO traces (recall_id, session, tick, node_id, activation, path) VALUES (?,?,?,?,?,?)`,
			tr.RecallID, tr.Session, tr.Tick, e.NodeID, e.Activation, string(path)); err != nil {
			return err
		}
	}
	return nil
}

// PendingTraces returns the session's traces not yet learned from, oldest
// first.
func (t *Tx) PendingTraces(session string) ([]*Trace, error) {
	rows, err := t.tx.Query(`SELECT recall_id, tick, node_id, activation, path FROM traces
		WHERE session = ? ORDER BY tick, recall_id, activation DESC`, session)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []*Trace
	byID := map[string]*Trace{}
	for rows.Next() {
		var rid, nid, path string
		var v int64
		var a float64
		if err := rows.Scan(&rid, &v, &nid, &a, &path); err != nil {
			return nil, err
		}
		en := TraceEntry{NodeID: nid, Activation: a}
		if err := json.Unmarshal([]byte(path), &en.Path); err != nil {
			return nil, fmt.Errorf("trace %s: bad path: %w", rid, err)
		}
		tr := byID[rid]
		if tr == nil {
			tr = &Trace{RecallID: rid, Session: session, Tick: v}
			byID[rid] = tr
			out = append(out, tr)
		}
		tr.Entries = append(tr.Entries, en)
	}
	return out, rows.Err()
}

// DeleteTraces removes traces that have been learned from.
func (t *Tx) DeleteTraces(recallIDs []string) error {
	for _, chunk := range chunks(recallIDs, 500) {
		if _, err := t.tx.Exec(`DELETE FROM traces WHERE recall_id IN (`+placeholders(len(chunk))+`)`, anys(chunk)...); err != nil {
			return err
		}
	}
	return nil
}

// PurgeTraces deletes pending traces recorded before the given tick and
// returns the number of recalls they belonged to.
func (t *Tx) PurgeTraces(beforeTick int64) (int64, error) {
	var n int64
	if err := t.tx.QueryRow(`SELECT count(DISTINCT recall_id) FROM traces WHERE tick < ?`, beforeTick).Scan(&n); err != nil {
		return 0, err
	}
	_, err := t.tx.Exec(`DELETE FROM traces WHERE tick < ?`, beforeTick)
	return n, err
}

// ---------------------------------------------------------------- params

// LoadParams returns defaults overridden by stored values.
func (t *Tx) LoadParams() (model.Params, error) {
	rows, err := t.tx.Query(`SELECT key, value FROM params`)
	if err != nil {
		return model.Params{}, err
	}
	defer rows.Close()
	m := map[string]string{}
	for rows.Next() {
		var k, v string
		if err := rows.Scan(&k, &v); err != nil {
			return model.Params{}, err
		}
		m[k] = v
	}
	if err := rows.Err(); err != nil {
		return model.Params{}, err
	}
	return model.DefaultParams().With(m)
}

// SetParam stores one override (validated by the caller).
func (t *Tx) SetParam(key, value string) error {
	_, err := t.tx.Exec(`INSERT INTO params (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value`, key, value)
	return err
}

// ResetParam removes an override.
func (t *Tx) ResetParam(key string) error {
	_, err := t.tx.Exec(`DELETE FROM params WHERE key = ?`, key)
	return err
}

// ---------------------------------------------------------------- stats

// Stats summarizes the graph.
type Stats struct {
	Nodes         int            `json:"nodes"`
	DormantNodes  int            `json:"dormant_nodes"`
	NodesByType   map[string]int `json:"nodes_by_type"`
	Edges         int            `json:"edges"`
	EdgesByKind   map[string]int `json:"edges_by_kind"`
	PendingTraces int            `json:"pending_traces"`
}

// Stats computes graph statistics.
func (t *Tx) Stats() (*Stats, error) {
	s := &Stats{NodesByType: map[string]int{}, EdgesByKind: map[string]int{}}
	if err := groupCount(t.tx, `SELECT type, count(*) FROM nodes GROUP BY type`, s.NodesByType, &s.Nodes); err != nil {
		return nil, err
	}
	if err := groupCount(t.tx, `SELECT kind, count(*) FROM edges GROUP BY kind`, s.EdgesByKind, &s.Edges); err != nil {
		return nil, err
	}
	if err := t.tx.QueryRow(`SELECT count(*) FROM nodes WHERE dormant = 1`).Scan(&s.DormantNodes); err != nil {
		return nil, err
	}
	if err := t.tx.QueryRow(`SELECT count(DISTINCT recall_id) FROM traces`).Scan(&s.PendingTraces); err != nil {
		return nil, err
	}
	return s, nil
}

func groupCount(tx *sql.Tx, q string, into map[string]int, total *int) error {
	rows, err := tx.Query(q)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var k string
		var n int
		if err := rows.Scan(&k, &n); err != nil {
			return err
		}
		into[k] = n
		*total += n
	}
	return rows.Err()
}

// ---------------------------------------------------------------- helpers

func placeholders(n int) string {
	return strings.TrimSuffix(strings.Repeat("?,", n), ",")
}

func anys(ss []string) []any {
	out := make([]any, len(ss))
	for i, s := range ss {
		out[i] = s
	}
	return out
}

func chunks(ss []string, size int) [][]string {
	var out [][]string
	for len(ss) > size {
		out = append(out, ss[:size])
		ss = ss[size:]
	}
	if len(ss) > 0 {
		out = append(out, ss)
	}
	return out
}
