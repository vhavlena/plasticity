package store

// migrations[i] upgrades the schema from user_version i to i+1.
var migrations = []string{
	`
CREATE TABLE nodes (
	id         TEXT PRIMARY KEY,
	type       TEXT NOT NULL,
	title      TEXT NOT NULL,
	content    TEXT NOT NULL DEFAULT '',
	rationale  TEXT NOT NULL DEFAULT '',
	tags       TEXT NOT NULL DEFAULT '[]',
	source     TEXT NOT NULL DEFAULT '',
	importance REAL NOT NULL DEFAULT 0,
	dormant    INTEGER NOT NULL DEFAULT 0,
	fires      INTEGER NOT NULL DEFAULT 0, -- recalls this neuron fired in
	created_tick INTEGER NOT NULL DEFAULT 0,
	created_at INTEGER NOT NULL,
	updated_at INTEGER NOT NULL
);

CREATE VIRTUAL TABLE nodes_fts USING fts5(
	title, content, rationale, tags,
	content = 'nodes', content_rowid = 'rowid',
	tokenize = 'porter unicode61'
);

CREATE TRIGGER nodes_ai AFTER INSERT ON nodes BEGIN
	INSERT INTO nodes_fts(rowid, title, content, rationale, tags)
	VALUES (new.rowid, new.title, new.content, new.rationale, new.tags);
END;
CREATE TRIGGER nodes_ad AFTER DELETE ON nodes BEGIN
	INSERT INTO nodes_fts(nodes_fts, rowid, title, content, rationale, tags)
	VALUES ('delete', old.rowid, old.title, old.content, old.rationale, old.tags);
END;
CREATE TRIGGER nodes_au AFTER UPDATE OF title, content, rationale, tags ON nodes BEGIN
	INSERT INTO nodes_fts(nodes_fts, rowid, title, content, rationale, tags)
	VALUES ('delete', old.rowid, old.title, old.content, old.rationale, old.tags);
	INSERT INTO nodes_fts(rowid, title, content, rationale, tags)
	VALUES (new.rowid, new.title, new.content, new.rationale, new.tags);
END;

-- Associative edges are stored once with src < dst; typed edges are directed.
-- src_mark/dst_mark are the endpoints' fire counts at the last update; the
-- edge's age is the number of endpoint firings since then.
CREATE TABLE edges (
	src           TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
	dst           TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
	kind          TEXT NOT NULL,
	weight        REAL NOT NULL,
	stability     REAL NOT NULL,
	src_mark      INTEGER NOT NULL DEFAULT 0,
	dst_mark      INTEGER NOT NULL DEFAULT 0,
	last_update   INTEGER NOT NULL,
	coactivations INTEGER NOT NULL DEFAULT 0,
	PRIMARY KEY (src, dst, kind)
) WITHOUT ROWID;
CREATE INDEX edges_dst ON edges(dst);

-- Access times in global recall ticks (ACT-R).
CREATE TABLE node_access (
	node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
	tick    INTEGER NOT NULL
);
CREATE INDEX node_access_node ON node_access(node_id, tick);

-- Pending recall traces, deleted once learned from (or when too old).
CREATE TABLE traces (
	recall_id  TEXT NOT NULL,
	session    TEXT NOT NULL,
	tick       INTEGER NOT NULL, -- global recall tick of the recall
	node_id    TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
	activation REAL NOT NULL,
	path       TEXT NOT NULL DEFAULT '[]', -- strongest activation path, seed first (JSON)
	PRIMARY KEY (recall_id, node_id)
);
CREATE INDEX traces_session ON traces(session, tick);
CREATE INDEX traces_tick ON traces(tick);

-- Logical clocks; "tick" counts recorded recalls.
CREATE TABLE counters (
	name  TEXT PRIMARY KEY,
	value INTEGER NOT NULL
);
INSERT INTO counters (name, value) VALUES ('tick', 0);

CREATE TABLE params (
	key   TEXT PRIMARY KEY,
	value TEXT NOT NULL
);
`,
}
