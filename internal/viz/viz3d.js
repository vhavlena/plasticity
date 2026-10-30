// ------------------------------------------------------------------ 3D view
// The 3D view shares the model, filters, search, selection and details panel
// with the 2D view and replaces layout, drawing and navigation: a force
// simulation in 3D (Barnes–Hut octree), WebGL rendering of shaded spheres
// with depth fog, synapses as screen-space quads (width = weight), particles
// flowing along typed synapses in their direction, an orbit camera, optional
// community coloring, and an animated replay of spreading activation from the
// selected neuron that follows retrieval.Spread.
const view3d = (() => {
	const cv = $("gl"), lab = $("gl-labels"), lc = lab.getContext("2d");
	const gl = cv.getContext("webgl", { antialias: true, alpha: false });
	if (!gl) return { supported: false };

	const FOVY = (50 * Math.PI) / 180, TAN = Math.tan(FOVY / 2);
	const HOP_MS = 700; // replay time per hop of spreading activation
	const opts = { color: "type", rotate: true, flow: true };
	const cam = { tx: 0, ty: 0, tz: 0, yaw: 0.6, pitch: 0.3, dist: 600 };
	let heat = 0, seeded = false, layoutSeed = 1, rnd = mulberry32(1);
	let flight = null, spread = null, last = 0, idleSince = 0;
	let community = new Map(), ncomm = 0, palette = [], cache = new Map();
	let basis = null, f = 1; // camera basis and focal length in CSS pixels

	// -------------------------------------------------------------- GL setup
	const NODE_VS = `
attribute vec3 a_center; attribute vec2 a_corner; attribute float a_radius;
attribute vec4 a_color; attribute float a_kind;
uniform mat4 u_proj; uniform float u_px;
varying vec2 v_c; varying vec4 v_color; varying float v_kind, v_px, v_depth;
void main() {
	// kind 0: shaded sphere, 1: soft glow, 2: ring around the sphere.
	float s = a_kind < 0.5 ? 1.0 : a_kind < 1.5 ? 2.8 : 1.45;
	vec3 p = a_center;
	p.z += a_radius; // the sphere's front: synapses to its center end at its surface
	p.xy += a_corner * a_radius * s;
	v_c = a_corner * s;
	v_color = a_color;
	v_kind = a_kind;
	v_px = a_radius * u_px / max(-p.z, 1e-3);
	v_depth = -a_center.z;
	gl_Position = u_proj * vec4(p, 1.0);
}`;
	const NODE_FS = `
precision mediump float;
uniform vec3 u_fog; uniform vec2 u_fogRange; uniform vec3 u_light;
varying vec2 v_c; varying vec4 v_color; varying float v_kind, v_px, v_depth;
void main() {
	float r = length(v_c);
	float fog = smoothstep(u_fogRange.x, u_fogRange.y, v_depth) * 0.85;
	if (v_kind < 0.5) {
		if (r > 1.0) discard;
		vec3 n = vec3(v_c, sqrt(max(0.0, 1.0 - r * r)));
		float dif = 0.55 + 0.45 * max(dot(n, u_light), 0.0);
		float spec = pow(max(dot(n, normalize(u_light + vec3(0.0, 0.0, 1.0))), 0.0), 32.0) * 0.35;
		float rim = pow(1.0 - n.z, 3.0) * 0.35;
		vec3 c = v_color.rgb * (dif + rim) + vec3(spec);
		gl_FragColor = vec4(mix(c, u_fog, fog), v_color.a * clamp((1.0 - r) * v_px, 0.0, 1.0));
	} else if (v_kind < 1.5) {
		float g = max(0.0, 1.0 - r / 2.8);
		gl_FragColor = vec4(v_color.rgb, v_color.a * g * g * (1.0 - fog * 0.8));
	} else {
		float a = clamp(1.25 - abs(r - 1.22) * v_px, 0.0, 1.0);
		if (a <= 0.0) discard;
		gl_FragColor = vec4(v_color.rgb, v_color.a * a);
	}
}`;
	const EDGE_VS = `
attribute vec3 a_pos; attribute vec3 a_other; attribute float a_side; attribute float a_end;
attribute float a_width; attribute vec4 a_color; attribute float a_dash;
uniform mat4 u_proj; uniform vec2 u_vp;
varying vec4 v_color; varying float v_t, v_side, v_half, v_dash, v_depth;
void main() {
	// Both ends are projected; the quad is widened in screen space so the
	// width is in pixels. a_end flips the direction at the far end so both
	// ends agree on which side is which.
	vec4 a = u_proj * vec4(a_pos, 1.0), b = u_proj * vec4(a_other, 1.0);
	vec2 sa = a.xy / a.w * u_vp * 0.5, sb = b.xy / b.w * u_vp * 0.5;
	vec2 d = (sb - sa) * (a_end < 0.5 ? 1.0 : -1.0);
	float len = length(d);
	vec2 dir = len > 1e-4 ? d / len : vec2(1.0, 0.0);
	v_half = a_width * 0.5;
	v_side = a_side * (v_half + 1.0);
	gl_Position = vec4((sa + vec2(-dir.y, dir.x) * v_side) / (u_vp * 0.5) * a.w, a.z, a.w);
	v_t = a_end * len;
	v_color = a_color;
	v_dash = a_dash;
	v_depth = -a_pos.z;
}`;
	const EDGE_FS = `
precision mediump float;
uniform vec3 u_fog; uniform vec2 u_fogRange;
varying vec4 v_color; varying float v_t, v_side, v_half, v_dash, v_depth;
void main() {
	if (v_dash > 0.5 && v_dash < 1.5 && mod(v_t, 10.0) > 6.0) discard;
	if (v_dash > 1.5 && mod(v_t, 7.0) > 4.0) discard;
	float fog = smoothstep(u_fogRange.x, u_fogRange.y, v_depth) * 0.85;
	float aa = clamp(v_half + 0.5 - abs(v_side), 0.0, 1.0);
	gl_FragColor = vec4(mix(v_color.rgb, u_fog, fog), v_color.a * aa * (1.0 - fog * 0.5));
}`;

	function program(vs, fs, layout) {
		const shader = (type, src) => {
			const s = gl.createShader(type);
			gl.shaderSource(s, src);
			gl.compileShader(s);
			if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
			return s;
		};
		const p = gl.createProgram();
		gl.attachShader(p, shader(gl.VERTEX_SHADER, vs));
		gl.attachShader(p, shader(gl.FRAGMENT_SHADER, fs));
		gl.linkProgram(p);
		if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
		const u = {};
		for (let i = 0; i < gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i++) {
			const a = gl.getActiveUniform(p, i);
			u[a.name] = gl.getUniformLocation(p, a.name);
		}
		let off = 0;
		const attrs = layout.map(([name, size]) => {
			const a = { loc: gl.getAttribLocation(p, name), size, off };
			off += size;
			return a;
		});
		return { p, u, attrs, stride: off, buf: gl.createBuffer() };
	}
	let nodeProg, edgeProg;
	try {
		nodeProg = program(NODE_VS, NODE_FS, [["a_center", 3], ["a_corner", 2], ["a_radius", 1], ["a_color", 4], ["a_kind", 1]]);
		edgeProg = program(EDGE_VS, EDGE_FS, [["a_pos", 3], ["a_other", 3], ["a_side", 1], ["a_end", 1], ["a_width", 1], ["a_color", 4], ["a_dash", 1]]);
	} catch (err) {
		console.error("plasticity 3D view:", err);
		return { supported: false };
	}
	const MAX_ATTRIBS = gl.getParameter(gl.MAX_VERTEX_ATTRIBS);

	// Growable vertex arrays, refilled every frame.
	function verts(stride) {
		return {
			a: new Float32Array(stride * 6 * 256), n: 0, stride,
			room(k) {
				const need = (this.n + k) * this.stride;
				if (need > this.a.length) {
					const b = new Float32Array(Math.max(need, this.a.length * 2));
					b.set(this.a);
					this.a = b;
				}
				return this.n * this.stride;
			},
		};
	}
	const solid = verts(nodeProg.stride), clear = verts(nodeProg.stride), glow = verts(nodeProg.stride), lines = verts(edgeProg.stride);
	const CORNERS = [-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1];
	function quad(v, x, y, z, r, c, alpha, kind) {
		let o = v.room(6);
		const a = v.a;
		for (let i = 0; i < 12; i += 2) {
			a[o++] = x; a[o++] = y; a[o++] = z; a[o++] = CORNERS[i]; a[o++] = CORNERS[i + 1]; a[o++] = r;
			a[o++] = c[0]; a[o++] = c[1]; a[o++] = c[2]; a[o++] = c[3] * alpha; a[o++] = kind;
		}
		v.n += 6;
	}
	function segment(v, A, B, width, c, alpha, dash) {
		let o = v.room(6);
		const a = v.a;
		const put = (end, side) => {
			const P = end ? B : A, Q = end ? A : B;
			a[o++] = P[0]; a[o++] = P[1]; a[o++] = P[2]; a[o++] = Q[0]; a[o++] = Q[1]; a[o++] = Q[2];
			a[o++] = side; a[o++] = end; a[o++] = width;
			a[o++] = c[0]; a[o++] = c[1]; a[o++] = c[2]; a[o++] = c[3] * alpha; a[o++] = dash;
		};
		put(0, 1); put(0, -1); put(1, 1); put(1, 1); put(0, -1); put(1, -1);
		v.n += 6;
	}
	function draw(prog, v, uniforms) {
		if (!v.n) return;
		gl.useProgram(prog.p);
		uniforms(prog.u);
		gl.bindBuffer(gl.ARRAY_BUFFER, prog.buf);
		gl.bufferData(gl.ARRAY_BUFFER, v.a.subarray(0, v.n * v.stride), gl.STREAM_DRAW);
		for (let i = 0; i < MAX_ATTRIBS; i++) gl.disableVertexAttribArray(i);
		for (const a of prog.attrs) {
			if (a.loc < 0) continue;
			gl.enableVertexAttribArray(a.loc);
			gl.vertexAttribPointer(a.loc, a.size, gl.FLOAT, false, prog.stride * 4, a.off * 4);
		}
		gl.drawArrays(gl.TRIANGLES, 0, v.n);
	}

	// -------------------------------------------------------------- colors
	// CSS colors ("#rrggbb" or "rgba(...)") as [r, g, b, a] in 0..1.
	function col(s) {
		let c = cache.get(s);
		if (c) return c;
		let m = /^#([0-9a-f]{6})$/i.exec(s);
		if (m) {
			const v = parseInt(m[1], 16);
			c = [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255, 1];
		} else if ((m = /rgba?\(([^)]+)\)/.exec(s))) {
			const p = m[1].split(",").map(Number);
			c = [p[0] / 255, p[1] / 255, p[2] / 255, p.length > 3 ? p[3] : 1];
		} else c = [0.5, 0.5, 0.5, 1];
		cache.set(s, c);
		return c;
	}
	const isDark = () => { const b = col(colors.bg); return b[0] + b[1] + b[2] < 1.5; };
	function hsl(h, s, l) {
		const k = (n) => (n + h / 30) % 12, a = s * Math.min(l, 1 - l);
		const ch = (n) => l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1));
		return [ch(0), ch(8), ch(4), 1];
	}
	// Community k gets a hue by the golden angle, so neighbors in the
	// numbering (similar sizes) get well separated colors.
	function commColor(k) {
		while (palette.length <= k) palette.push(hsl((palette.length * 137.508) % 360, 0.65, isDark() ? 0.62 : 0.46));
		return palette[k];
	}
	function nodeColor(n) {
		let c = col(colors["t-" + n.type] || colors.muted);
		if (opts.color === "community") {
			const k = community.get(n);
			c = k >= 0 ? commColor(k) : col(colors.muted);
		}
		if (!n.dormant) return c;
		const m = col(colors.muted);
		return [(c[0] + m[0]) / 2, (c[1] + m[1]) / 2, (c[2] + m[2]) / 2, 1];
	}

	// Weighted label propagation over the visible graph; communities are
	// numbered by size, neurons alone in theirs get the neutral color.
	function detectCommunities() {
		const lbl = new Map(visNodes.map((n) => [n, n.i]));
		const order = [...visNodes], r = mulberry32(7);
		for (let it = 0; it < 30; it++) {
			for (let i = order.length - 1; i > 0; i--) {
				const j = Math.floor(r() * (i + 1));
				[order[i], order[j]] = [order[j], order[i]];
			}
			let changed = false;
			for (const n of order) {
				const score = new Map();
				for (const e of n.adj) {
					if (!e.vis) continue;
					const l = lbl.get(e.s === n ? e.t : e.s);
					score.set(l, (score.get(l) || 0) + e.w + 0.05);
				}
				let best = lbl.get(n), bs = score.get(best) || 0;
				for (const [l, s] of score) if (s > bs + 1e-9) { best = l; bs = s; }
				if (best !== lbl.get(n)) { lbl.set(n, best); changed = true; }
			}
			if (!changed) break;
		}
		const size = new Map();
		for (const l of lbl.values()) size.set(l, (size.get(l) || 0) + 1);
		const ids = [...size.keys()].filter((l) => size.get(l) > 1).sort((a, b) => size.get(b) - size.get(a) || a - b);
		const idx = new Map(ids.map((l, k) => [l, k]));
		community = new Map();
		for (const [n, l] of lbl) community.set(n, idx.has(l) ? idx.get(l) : -1);
		ncomm = ids.length;
		$("comm3d").textContent = opts.color === "community" ? `${ncomm} communit${ncomm === 1 ? "y" : "ies"}` : "";
	}

	// -------------------------------------------------------------- layout
	// The same forces as the 2D layout (charge grows with importance, springs
	// shorter and stiffer with weight), in 3D and for the whole visible graph
	// at once, with a weak pull to the origin that keeps components together.
	function octree(ns) {
		let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
		for (const n of ns) {
			x0 = Math.min(x0, n.x3); x1 = Math.max(x1, n.x3);
			y0 = Math.min(y0, n.y3); y1 = Math.max(y1, n.y3);
			z0 = Math.min(z0, n.z3); z1 = Math.max(z1, n.z3);
		}
		const root = cell(x0, y0, z0, Math.max(x1 - x0, y1 - y0, z1 - z0, 1) + 1);
		for (const n of ns) oinsert(root, n, 0);
		return root;
	}
	function cell(x0, y0, z0, size) { return { x0, y0, z0, size, m: 0, cx: 0, cy: 0, cz: 0, kids: null, n: null }; }
	function oinsert(q, n, depth) {
		const c = 120 + 80 * n.importance;
		q.cx = (q.cx * q.m + n.x3 * c) / (q.m + c);
		q.cy = (q.cy * q.m + n.y3 * c) / (q.m + c);
		q.cz = (q.cz * q.m + n.z3 * c) / (q.m + c);
		q.m += c;
		if (!q.kids) {
			if (!q.n) { q.n = n; return; }
			if (depth > 40) return; // coincident points
			const old = q.n;
			q.n = null;
			q.kids = [null, null, null, null, null, null, null, null];
			oplace(q, old, depth);
		}
		oplace(q, n, depth);
	}
	function oplace(q, n, depth) {
		const h = q.size / 2;
		const ix = n.x3 >= q.x0 + h ? 1 : 0, iy = n.y3 >= q.y0 + h ? 1 : 0, iz = n.z3 >= q.z0 + h ? 1 : 0, k = ix + 2 * iy + 4 * iz;
		if (!q.kids[k]) q.kids[k] = cell(q.x0 + ix * h, q.y0 + iy * h, q.z0 + iz * h, h);
		oinsert(q.kids[k], n, depth + 1);
	}
	function repel(q, n) {
		if (!q) return;
		if (q.n === n && !q.kids) return;
		let dx = q.cx - n.x3, dy = q.cy - n.y3, dz = q.cz - n.z3, d2 = dx * dx + dy * dy + dz * dz;
		if (q.kids && (q.size * q.size) / (d2 || 1e-9) >= THETA2) {
			for (const k of q.kids) if (k) repel(k, n);
			return;
		}
		if (d2 < 1e-6) {
			dx = Math.random() - 0.5; dy = Math.random() - 0.5; dz = Math.random() - 0.5;
			d2 = dx * dx + dy * dy + dz * dz;
		}
		if (q.n) {
			const d = Math.sqrt(d2), min = n.r + q.n.r + 8;
			if (d < min) {
				const s = ((min - d) / d) * 0.5;
				n.vx3 -= dx * s; n.vy3 -= dy * s; n.vz3 -= dz * s;
			}
		}
		const s = (q.m * heat) / Math.max(d2, 25);
		n.vx3 -= dx * s; n.vy3 -= dy * s; n.vz3 -= dz * s;
	}
	function tick() {
		const ns = visNodes;
		if (ns.length > 1) {
			const tree = octree(ns);
			for (const n of ns) repel(tree, n);
		}
		for (const e of visEdges) {
			const s = e.s, t = e.t;
			let dx = t.x3 + t.vx3 - s.x3 - s.vx3, dy = t.y3 + t.vy3 - s.y3 - s.vy3, dz = t.z3 + t.vz3 - s.z3 - s.vz3;
			const d = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
			const len = 70 + s.r + t.r + 70 * (1 - e.w);
			const k = (0.15 + 0.6 * e.w) / Math.min(s.deg, t.deg);
			const g = ((d - len) / d) * heat * k;
			dx *= g; dy *= g; dz *= g;
			const bs = t.deg / (s.deg + t.deg);
			t.vx3 -= dx * bs; t.vy3 -= dy * bs; t.vz3 -= dz * bs;
			s.vx3 += dx * (1 - bs); s.vy3 += dy * (1 - bs); s.vz3 += dz * (1 - bs);
		}
		for (const n of ns) {
			n.vx3 -= n.x3 * 0.02 * heat; n.vy3 -= n.y3 * 0.02 * heat; n.vz3 -= n.z3 * 0.02 * heat;
			if (n.fixed) { n.vx3 = n.vy3 = n.vz3 = 0; continue; }
			n.vx3 *= 1 - VELOCITY_DECAY; n.vy3 *= 1 - VELOCITY_DECAY; n.vz3 *= 1 - VELOCITY_DECAY;
			n.x3 += n.vx3; n.y3 += n.vy3; n.z3 += n.vz3;
		}
		heat += (0 - heat) * ALPHA_DECAY;
	}
	// New neurons start next to a placed neighbor, or anywhere in a ball.
	function place(n) {
		const e = n.adj.find((e) => e.vis && (e.s === n ? e.t : e.s).has3);
		if (e) {
			const m = e.s === n ? e.t : e.s;
			n.x3 = m.x3 + (rnd() - 0.5) * 40; n.y3 = m.y3 + (rnd() - 0.5) * 40; n.z3 = m.z3 + (rnd() - 0.5) * 40;
		} else {
			const R = 20 + 25 * Math.cbrt(visNodes.length);
			let x, y, z;
			do { x = 2 * rnd() - 1; y = 2 * rnd() - 1; z = 2 * rnd() - 1; } while (x * x + y * y + z * z > 1);
			n.x3 = x * R; n.y3 = y * R; n.z3 = z * R;
		}
		n.vx3 = n.vy3 = n.vz3 = 0;
		n.has3 = true;
	}
	// Runs the hot, chaotic start of the simulation synchronously; the rest
	// is animated.
	function warm() {
		heat = 1;
		let budget = visNodes.length > 2000 ? 30 : 100;
		while (heat > 0.12 && budget--) tick();
	}
	function relayout() {
		rnd = mulberry32(++layoutSeed);
		for (const n of nodes) n.has3 = false;
		for (const n of visNodes) place(n);
		warm();
		fit(visNodes, true);
	}

	// -------------------------------------------------------------- spreading activation
	// The strongest path from the seed to every neuron (Dijkstra on −ln a),
	// as in retrieval.Spread: crossing a synapse of effective weight w from a
	// neuron whose synapses sum to load k multiplies the activation by
	// δ · w / max(1, k/K0)^β; neurons below θ are not reached. Unlike a real
	// recall the seed starts at 1 and every synapse is followed (recall loads
	// only a neuron's strongest edges_per_node).
	const SPREAD = Object.assign({ hop_decay: 0.8, firing_threshold: 0.1, fan_k0: 8, fan_beta: 0.5, max_hops: 10 }, DATA.spread || {});
	function spreadFrom(seed) {
		const P = SPREAD, best = new Map([[seed, { a: 1, hops: 0, parent: null, edge: null }]]);
		const heap = [[1, seed]];
		const push = (it) => {
			heap.push(it);
			for (let i = heap.length - 1; i > 0;) {
				const p = (i - 1) >> 1;
				if (heap[p][0] >= heap[i][0]) break;
				[heap[p], heap[i]] = [heap[i], heap[p]];
				i = p;
			}
		};
		const pop = () => {
			const top = heap[0], end = heap.pop();
			if (heap.length) {
				heap[0] = end;
				for (let i = 0; ;) {
					const l = 2 * i + 1, r = l + 1;
					let m = i;
					if (l < heap.length && heap[l][0] > heap[m][0]) m = l;
					if (r < heap.length && heap[r][0] > heap[m][0]) m = r;
					if (m === i) break;
					[heap[m], heap[i]] = [heap[i], heap[m]];
					i = m;
				}
			}
			return top;
		};
		while (heap.length) {
			const [a, n] = pop(), cur = best.get(n);
			if (a < cur.a || cur.hops >= P.max_hops) continue;
			let load = 0;
			for (const e of n.adj) load += Math.max(0, e.w);
			const fan = Math.pow(Math.max(1, load / P.fan_k0), P.fan_beta);
			for (const e of n.adj) {
				if (e.w <= 0) continue;
				const o = e.s === n ? e.t : e.s, b = (a * P.hop_decay * e.w) / fan, prev = best.get(o);
				if (b < P.firing_threshold || (prev && b <= prev.a)) continue;
				best.set(o, { a: b, hops: cur.hops + 1, parent: n, edge: e });
				push([b, o]);
			}
		}
		return best;
	}
	function startSpread(n) {
		const res = spreadFrom(n);
		let hops = 0;
		for (const r of res.values()) hops = Math.max(hops, r.hops);
		const tree = new Map();
		for (const [m, r] of res) if (r.edge) tree.set(r.edge, { m, r });
		spread = { seed: n, res, tree, hops, t0: performance.now(), dur: (hops + 1) * HOP_MS + 1500 };
		const shown = [...res.keys()].filter((m) => m.vis);
		if (shown.length > 1) fit(shown, true);
		kick();
	}
	function details(box, n) {
		const out = el("div");
		const btn = el("button", { textContent: spread && spread.seed === n ? "↻ Replay" : "▶ Simulate recall from here",
			title: "Replay how activation spreads from this neuron when a recall seeds it" });
		btn.onclick = () => { startSpread(n); btn.textContent = "↻ Replay"; report(out); };
		const sec = el("div", {}, el("h2", { textContent: "Spreading activation" }), btn, out);
		sec.style.margin = "10px 0 12px";
		box.append(sec);
		if (spread && spread.seed === n) report(out);
	}
	function report(out) {
		const reached = [...spread.res].filter(([m]) => m !== spread.seed).sort((a, b) => b[1].a - a[1].a);
		out.textContent = "";
		const s = reached.length === 1 ? "" : "s";
		const note = el("div", { className: "sub", textContent: reached.length
			? `Reaches ${reached.length} neuron${s} within ${spread.hops} hop${spread.hops === 1 ? "" : "s"} (θ ${SPREAD.firing_threshold}, δ ${SPREAD.hop_decay}, fan effect).`
			: `No neighbor gets above the firing threshold θ ${SPREAD.firing_threshold}.` });
		note.style.margin = "6px 0 2px";
		out.append(note);
		const ul = el("ul", { className: "links" });
		for (const [m, r] of reached.slice(0, 12)) {
			const dot = el("span", { className: "dot" });
			dot.style.background = colors["t-" + m.type];
			const li = el("li", { title: m.title }, dot, el("span", { className: "t", textContent: m.title }),
				el("span", { className: "k", textContent: `${r.hops} hop${r.hops === 1 ? "" : "s"}` }),
				el("span", { className: "w", textContent: r.a.toFixed(2) }));
			if (!m.vis) li.style.opacity = 0.5;
			li.onclick = () => { if (m.vis) center(m); };
			ul.append(li);
		}
		out.append(ul);
	}

	// -------------------------------------------------------------- camera
	const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
	const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
	const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
	function camBasis() {
		const cp = Math.cos(cam.pitch);
		const eye = [cam.tx + cam.dist * cp * Math.sin(cam.yaw), cam.ty + cam.dist * Math.sin(cam.pitch), cam.tz + cam.dist * cp * Math.cos(cam.yaw)];
		const fw = norm([cam.tx - eye[0], cam.ty - eye[1], cam.tz - eye[2]]), rt = norm(cross(fw, [0, 1, 0]));
		basis = { eye, fw, rt, up: cross(rt, fw) };
		f = H / 2 / TAN;
		return basis;
	}
	function flyTo(to, animate) {
		if (!animate) { Object.assign(cam, to); flight = null; kick(); return; }
		flight = { from: { tx: cam.tx, ty: cam.ty, tz: cam.tz, dist: cam.dist }, to, t0: performance.now(), dur: 750 };
		kick();
	}
	// Frames ns: the camera looks at the center of their bounding box from
	// far enough for their bounding sphere to fit the narrower field of view.
	function fit(ns = visNodes, animate = true) {
		if (!ns || !ns.length) return;
		let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
		for (const n of ns) {
			x0 = Math.min(x0, n.x3); x1 = Math.max(x1, n.x3);
			y0 = Math.min(y0, n.y3); y1 = Math.max(y1, n.y3);
			z0 = Math.min(z0, n.z3); z1 = Math.max(z1, n.z3);
		}
		const tx = (x0 + x1) / 2, ty = (y0 + y1) / 2, tz = (z0 + z1) / 2;
		let R = 40;
		for (const n of ns) R = Math.max(R, Math.hypot(n.x3 - tx, n.y3 - ty, n.z3 - tz) + n.r);
		const half = Math.min(FOVY / 2, Math.atan((TAN * W) / Math.max(1, H)));
		flyTo({ tx, ty, tz, dist: (R / Math.sin(half)) * 1.05 }, animate);
	}
	function center(n) { flyTo({ tx: n.x3, ty: n.y3, tz: n.z3, dist: Math.min(cam.dist, 220 + 8 * n.r) }, true); }
	function zoomAt(sx, sy, factor) {
		const { rt, up } = camBasis();
		const dist = Math.min(50000, Math.max(5, cam.dist * factor)), g = 1 - dist / cam.dist;
		const ox = ((sx - W / 2) / f) * cam.dist, oy = (-(sy - H / 2) / f) * cam.dist;
		cam.tx += (rt[0] * ox + up[0] * oy) * g; cam.ty += (rt[1] * ox + up[1] * oy) * g; cam.tz += (rt[2] * ox + up[2] * oy) * g;
		cam.dist = dist;
		kick();
	}
	function pan(dx, dy) {
		const { rt, up } = camBasis(), s = cam.dist / f;
		cam.tx += (-rt[0] * dx + up[0] * dy) * s; cam.ty += (-rt[1] * dx + up[1] * dy) * s; cam.tz += (-rt[2] * dx + up[2] * dy) * s;
		kick();
	}
	function pick(sx, sy) {
		let best = null;
		for (const n of visNodes) {
			if (!(n.sr > 0)) continue;
			if (Math.hypot(n.sx - sx, n.sy - sy) < n.sr + 3 && (!best || n.qz > best.qz)) best = n;
		}
		return best;
	}

	// -------------------------------------------------------------- rendering
	function render(now) {
		const { eye, fw, rt, up } = camBasis();
		// View space (camera looks down −z) and screen positions.
		let dmin = Infinity, dmax = -Infinity;
		for (const n of visNodes) {
			const dx = n.x3 - eye[0], dy = n.y3 - eye[1], dz = n.z3 - eye[2];
			n.qx = dx * rt[0] + dy * rt[1] + dz * rt[2];
			n.qy = dx * up[0] + dy * up[1] + dz * up[2];
			n.qz = -(dx * fw[0] + dy * fw[1] + dz * fw[2]);
			dmin = Math.min(dmin, -n.qz - n.r);
			dmax = Math.max(dmax, -n.qz + n.r);
		}
		if (!visNodes.length) dmin = dmax = cam.dist;
		const near = dmin > 1 ? Math.max(0.5, Math.min(dmin * 0.7, cam.dist * 0.5)) : Math.max(0.5, cam.dist * 0.01);
		const far = Math.max(near * 2, dmax * 1.2 + 10);
		const span = Math.max(1, dmax - Math.max(dmin, near));
		const fog = [Math.max(dmin, near) + span * 0.35, dmax + span * 0.15];
		for (const n of visNodes) {
			if (-n.qz > near) {
				const s = f / -n.qz;
				n.sx = W / 2 + n.qx * s; n.sy = H / 2 - n.qy * s; n.sr = n.r * s;
				n.fog = Math.min(1, Math.max(0, (-n.qz - fog[0]) / (fog[1] - fog[0])));
			} else n.sr = 0;
		}

		const sel = state.selected || state.hover;
		const focus = spread ? null : focusSet();
		const searching = state.query !== "";
		const lit = (n) => (focus ? focus.has(n) : true) && (!searching || state.matches.has(n) || (focus && focus.has(n)));
		const st = spread ? now - spread.t0 : 0;
		const reachedAt = (n) => { const r = spread.res.get(n); return r && st >= r.hops * HOP_MS ? r : null; };
		const accent = col(colors.accent), dark = isDark();

		// Nodes: opaque ones first (they write depth), translucent ones sorted
		// back to front after the synapses, glows, rings and particles last.
		solid.n = clear.n = glow.n = lines.n = 0;
		const translucent = [];
		for (const n of visNodes) {
			let a;
			if (spread) a = n === spread.seed || reachedAt(n) ? 1 : 0.1;
			else a = lit(n) ? (n.dormant ? 0.5 : 1) : 0.12;
			n.a3 = a;
			if (a >= 0.99) quad(solid, n.qx, n.qy, n.qz, n.r, nodeColor(n), 1, 0);
			else translucent.push(n);
		}
		translucent.sort((a, b) => a.qz - b.qz);
		for (const n of translucent) quad(clear, n.qx, n.qy, n.qz, n.r, nodeColor(n), n.a3, 0);

		for (const n of visNodes) {
			if (n === state.selected) {
				quad(glow, n.qx, n.qy, n.qz, n.r, accent, dark ? 0.55 : 0.35, 1);
				quad(glow, n.qx, n.qy, n.qz, n.r, accent, 1, 2);
			} else if (searching && state.matches.has(n)) quad(glow, n.qx, n.qy, n.qz, n.r, col(colors.label), 0.9, 2);
			else if (n === state.hover) quad(glow, n.qx, n.qy, n.qz, n.r, col(colors.muted), 0.9, 2);
			if (spread && n !== spread.seed) {
				const r = reachedAt(n);
				if (r) {
					const flash = Math.exp(-(st - r.hops * HOP_MS) / 350);
					quad(glow, n.qx, n.qy, n.qz, n.r * (1 + 0.5 * flash), nodeColor(n), Math.min(1, 0.15 + 0.5 * r.a + 0.7 * flash), 1);
				}
			}
		}

		// Synapses, clipped to the near plane in view space.
		const clip = (A, B) => {
			const za = -A[2], zb = -B[2];
			if (za <= near && zb <= near) return false;
			if (za <= near) { const t = (near - za) / (zb - za); for (let i = 0; i < 3; i++) A[i] += (B[i] - A[i]) * t; }
			else if (zb <= near) { const t = (near - zb) / (za - zb); for (let i = 0; i < 3; i++) B[i] += (A[i] - B[i]) * t; }
			return true;
		};
		let flowing = false;
		for (const e of visEdges) {
			if (!e.drawn) continue;
			const A = [e.s.qx, e.s.qy, e.s.qz], B = [e.t.qx, e.t.qy, e.t.qz];
			if (!clip(A, B)) continue;
			let c = e.tentative ? col(colors.tentative) : e.kind === "associative" ? col(colors.strong) : col(colors["k-" + e.kind]);
			let width = e.tentative ? 1 : 0.8 + 3.6 * e.w, a;
			const dash = e.kind === "contradicts" ? 1 : e.tentative ? 2 : 0;
			if (spread) {
				const tr = spread.tree.get(e);
				if (tr && st >= tr.r.hops * HOP_MS) { c = accent; a = 0.9; width = 1.2 + 3 * tr.r.a; }
				else a = 0.25;
			} else {
				const on = focus ? focus.has(e.s) && focus.has(e.t) && (e.s === sel || e.t === sel) : !searching;
				a = on ? 1 : 0.15;
				if (e.kind === "associative" && !e.tentative) a *= 0.3 + 0.7 * e.strength;
				if (on && focus) {
					width *= 1.3;
					if (e.kind === "associative" && !e.tentative) c = col(colors.muted);
				}
				// Particles run from source to target: the synapse's direction.
				if (opts.flow && e.kind !== "associative") {
					flowing = true;
					const count = e.w > 0.5 ? 2 : 1, hash = ((e.s.i * 7919 + e.t.i * 104729) % 1000) / 1000;
					for (let k = 0; k < count; k++) {
						const u = (now / (2600 - 1200 * e.w) + k / count + hash) % 1;
						const x = e.s.qx + (e.t.qx - e.s.qx) * u, y = e.s.qy + (e.t.qy - e.s.qy) * u, z = e.s.qz + (e.t.qz - e.s.qz) * u;
						if (-z > near) quad(glow, x, y, z, 1.8 + 1.8 * e.w, col(colors["k-" + e.kind]), on ? 0.9 : 0.15, 1);
					}
				}
			}
			segment(lines, A, B, width, c, a, dash);
		}
		// Activation travelling along the strongest paths.
		if (spread) {
			for (const [m, r] of spread.res) {
				const p = r.parent, t0 = (r.hops - 1) * HOP_MS;
				if (!p || !p.vis || !m.vis || st < t0 || st > t0 + HOP_MS) continue;
				let u = (st - t0) / HOP_MS;
				u = u * u * (3 - 2 * u);
				const x = p.qx + (m.qx - p.qx) * u, y = p.qy + (m.qy - p.qy) * u, z = p.qz + (m.qz - p.qz) * u;
				if (-z <= near) continue;
				quad(glow, x, y, z, 3 + 4 * r.a, accent, 1, 1);
				quad(glow, x, y, z, 1.5 + 1.5 * r.a, accent, 1, 0);
			}
		}

		// Draw.
		gl.viewport(0, 0, cv.width, cv.height);
		const bg = col(colors.bg);
		gl.clearColor(bg[0], bg[1], bg[2], 1);
		gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
		gl.enable(gl.DEPTH_TEST);
		gl.depthFunc(gl.LEQUAL);
		gl.enable(gl.BLEND);
		gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
		const fa = H / Math.max(1, W) * (1 / TAN), fb = 1 / TAN;
		const proj = new Float32Array([fa, 0, 0, 0, 0, fb, 0, 0, 0, 0, (far + near) / (near - far), -1, 0, 0, (2 * far * near) / (near - far), 0]);
		const light = norm([-0.35, 0.55, 0.75]);
		const nodeU = (u) => {
			gl.uniformMatrix4fv(u.u_proj, false, proj);
			gl.uniform1f(u.u_px, f * DPR);
			gl.uniform3f(u.u_fog, bg[0], bg[1], bg[2]);
			gl.uniform2f(u.u_fogRange, fog[0], fog[1]);
			if (u.u_light) gl.uniform3f(u.u_light, light[0], light[1], light[2]);
		};
		gl.depthMask(true);
		draw(nodeProg, solid, nodeU);
		gl.depthMask(false);
		draw(edgeProg, lines, (u) => {
			gl.uniformMatrix4fv(u.u_proj, false, proj);
			gl.uniform2f(u.u_vp, W, H);
			gl.uniform3f(u.u_fog, bg[0], bg[1], bg[2]);
			gl.uniform2f(u.u_fogRange, fog[0], fog[1]);
		});
		draw(nodeProg, clear, nodeU);
		// Glows add light on a dark background and tint a light one.
		if (dark) gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
		draw(nodeProg, glow, nodeU);
		gl.depthMask(true);

		drawLabels(focus, searching, lit, reachedAt);
		return flowing;
	}

	// Labels on a 2D overlay, greedily by priority (selection, hover, focus,
	// search hits, then importance and closeness), each in the first of four
	// spots around its neuron that covers no placed label.
	function drawLabels(focus, searching, lit, reachedAt) {
		lc.setTransform(DPR, 0, 0, DPR, 0, 0);
		lc.clearRect(0, 0, W, H);
		if (state.labels === "off") return;
		const cand = [];
		for (const n of visNodes) {
			if (!(n.sr > 0) || n.sx < -300 || n.sx > W + 20 || n.sy < -20 || n.sy > H + 20) continue;
			let show, forced = false;
			if (spread) { show = n === spread.seed || !!reachedAt(n); forced = n === spread.seed; }
			else if (focus) show = forced = focus.has(n);
			else if (searching) show = forced = state.matches.has(n);
			else if (state.labels === "none") show = false;
			else { show = true; forced = state.labels === "all"; }
			if (n === state.selected || n === state.hover) show = forced = true;
			if (show) cand.push({ n, forced, p: (n === state.selected ? 3e9 : n === state.hover ? 2e9 : forced ? 1e9 : 0) + n.rank + Math.min(20, n.sr) / 4 });
		}
		cand.sort((a, b) => b.p - a.p);
		const grid = new Map(), CELL = 64, placed = [];
		const cells = (b, fn) => {
			for (let gx = Math.floor(b.l / CELL); gx <= Math.floor(b.r / CELL); gx++)
				for (let gy = Math.floor(b.t / CELL); gy <= Math.floor(b.b / CELL); gy++) fn(gx + "," + gy);
		};
		const hits = (b) => {
			let h = 0;
			cells(b, (key) => { for (const o of grid.get(key) || []) if (o.l < b.r && b.l < o.r && o.t < b.b && b.t < o.b) h++; });
			return h;
		};
		for (const c of cand) {
			const n = c.n, px = Math.round(11 + Math.min(3, n.sr / 6)), w = (n.lw * px) / LABEL_PX + 8, h = px + 6, g = 5;
			const spots = [
				{ l: n.sx + n.sr + g, t: n.sy - h / 2 },
				{ l: n.sx - n.sr - g - w, t: n.sy - h / 2 },
				{ l: n.sx - w / 2, t: n.sy + n.sr + 2 },
				{ l: n.sx - w / 2, t: n.sy - n.sr - 2 - h },
			];
			let best = null, bh = Infinity;
			for (const s of spots) {
				const b = { l: s.l, r: s.l + w, t: s.t, b: s.t + h };
				const hh = hits(b) + (b.l < 0 || b.r > W || b.t < 0 || b.b > H ? 1 : 0);
				if (hh < bh) { best = b; bh = hh; }
				if (!hh) break;
			}
			if (bh && !c.forced) continue;
			cells(best, (key) => { const l = grid.get(key); l ? l.push(best) : grid.set(key, [best]); });
			placed.push({ n, b: best, px, a: (spread || lit(n) ? 1 : 0.3) * (1 - 0.6 * n.fog) });
		}
		lc.textBaseline = "middle";
		for (const { n, b, px, a } of placed.reverse()) {
			lc.globalAlpha = a;
			lc.font = `${px}px ${FONT}`;
			lc.fillStyle = colors.halo;
			lc.beginPath();
			if (lc.roundRect) lc.roundRect(b.l, b.t, b.r - b.l, b.b - b.t, 4); else lc.rect(b.l, b.t, b.r - b.l, b.b - b.t);
			lc.fill();
			lc.fillStyle = colors.label;
			lc.fillText(n.label, b.l + 4, (b.t + b.b) / 2 + 0.5);
		}
		lc.globalAlpha = 1;
	}

	function frame(now) {
		const dt = last ? Math.min(64, now - last) : 16;
		last = now;
		let busy = false;
		if (heat > ALPHA_MIN) {
			const steps = visNodes.length > 1500 ? 1 : 2;
			for (let i = 0; i < steps; i++) tick();
			busy = true;
		}
		if (flight) {
			let u = Math.min(1, (now - flight.t0) / flight.dur);
			u = u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
			const { from, to } = flight;
			cam.tx = from.tx + (to.tx - from.tx) * u; cam.ty = from.ty + (to.ty - from.ty) * u; cam.tz = from.tz + (to.tz - from.tz) * u;
			cam.dist = from.dist * Math.pow(to.dist / from.dist, u);
			if (u >= 1) flight = null;
			busy = true;
		}
		if (opts.rotate && !state.selected && !drag && !spread && now - idleSince > 1500) {
			cam.yaw += dt * 0.00006;
			busy = true;
		}
		if (spread && spread.seed !== state.selected) spread = null;
		if (spread && now - spread.t0 < spread.dur) busy = true;
		if (render(now)) busy = true;
		if (busy) kick(); else last = 0;
	}

	// -------------------------------------------------------------- input
	const ptrs = new Map();
	let drag = null, gesture = null, moved = 0;
	cv.addEventListener("contextmenu", (ev) => ev.preventDefault());
	cv.addEventListener("pointerdown", (ev) => {
		cv.setPointerCapture(ev.pointerId);
		ptrs.set(ev.pointerId, { x: ev.offsetX, y: ev.offsetY });
		moved = 0;
		flight = null;
		idleSince = performance.now();
		if (ptrs.size === 2) {
			const [a, b] = [...ptrs.values()];
			gesture = { d: Math.hypot(a.x - b.x, a.y - b.y), x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
			if (drag && drag.node) drag.node.fixed = false;
			drag = null;
			return;
		}
		const n = ev.button === 0 && !ev.shiftKey ? pick(ev.offsetX, ev.offsetY) : null;
		if (n) { drag = { node: n, depth: -n.qz }; n.fixed = true; }
		else drag = { pan: ev.button !== 0 || ev.shiftKey, x: ev.offsetX, y: ev.offsetY };
		cv.classList.add("dragging");
	});
	cv.addEventListener("pointermove", (ev) => {
		const prev = ptrs.get(ev.pointerId);
		if (prev) ptrs.set(ev.pointerId, { x: ev.offsetX, y: ev.offsetY });
		if (gesture && ptrs.size === 2) {
			const [a, b] = [...ptrs.values()];
			const d = Math.hypot(a.x - b.x, a.y - b.y), x = (a.x + b.x) / 2, y = (a.y + b.y) / 2;
			zoomAt(x, y, (gesture.d || d) / (d || 1));
			pan(x - gesture.x, y - gesture.y);
			gesture = { d, x, y };
			idleSince = performance.now();
			return;
		}
		if (drag) {
			if (prev) moved += Math.hypot(ev.offsetX - prev.x, ev.offsetY - prev.y);
			idleSince = performance.now();
			if (drag.node) {
				// Keep the neuron under the pointer at its depth.
				const { eye, fw, rt, up } = camBasis(), k = drag.depth / f, n = drag.node;
				const x = (ev.offsetX - W / 2) * k, y = -(ev.offsetY - H / 2) * k;
				n.x3 = eye[0] + rt[0] * x + up[0] * y + fw[0] * drag.depth;
				n.y3 = eye[1] + rt[1] * x + up[1] * y + fw[1] * drag.depth;
				n.z3 = eye[2] + rt[2] * x + up[2] * y + fw[2] * drag.depth;
				heat = Math.max(heat, 0.15);
				kick();
			} else if (drag.pan) pan(ev.offsetX - drag.x, ev.offsetY - drag.y);
			else {
				cam.yaw -= (ev.offsetX - drag.x) * 0.006;
				cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch + (ev.offsetY - drag.y) * 0.006));
				kick();
			}
			if (!drag.node) { drag.x = ev.offsetX; drag.y = ev.offsetY; }
			hideTip();
			return;
		}
		const n = pick(ev.offsetX, ev.offsetY);
		if (n !== state.hover) { state.hover = n; kick(); }
		cv.classList.toggle("over", !!n);
		if (n) showTip(n, ev.offsetX, ev.offsetY); else hideTip();
	});
	function endPointer(ev) {
		ptrs.delete(ev.pointerId);
		if (ptrs.size < 2) gesture = null;
		if (!drag) return;
		if (drag.node) {
			drag.node.fixed = false;
			if (moved < 4) select(drag.node);
		} else if (moved < 4 && !drag.pan) select(null);
		drag = null;
		cv.classList.remove("dragging");
		kick();
	}
	cv.addEventListener("pointerup", endPointer);
	cv.addEventListener("pointercancel", endPointer);
	cv.addEventListener("pointerleave", () => { if (state.hover && !drag) { state.hover = null; hideTip(); kick(); } });
	cv.addEventListener("wheel", (ev) => {
		ev.preventDefault();
		idleSince = performance.now();
		flight = null;
		zoomAt(ev.offsetX, ev.offsetY, Math.exp(ev.deltaY * (ev.ctrlKey ? 0.01 : 0.0015)));
	}, { passive: false });
	cv.addEventListener("dblclick", (ev) => {
		const n = pick(ev.offsetX, ev.offsetY);
		if (n) fit([...focusSet()], true); else zoomAt(ev.offsetX, ev.offsetY, 1 / 1.6);
	});

	for (const r of document.querySelectorAll('input[name="color3d"]')) r.onchange = (ev) => {
		opts.color = ev.target.value;
		if (opts.color === "community") detectCommunities(); else $("comm3d").textContent = "";
		kick();
	};
	$("rotate3d").onchange = (ev) => { opts.rotate = ev.target.checked; idleSince = 0; kick(); };
	$("flow3d").onchange = (ev) => { opts.flow = ev.target.checked; kick(); };

	return {
		supported: true, frame, fit, center, relayout, details,
		// Called on entering the view and after the filters change.
		filtered() {
			for (const n of visNodes) if (!n.has3) place(n);
			if (!seeded) { seeded = true; warm(); fit(visNodes, false); }
			else heat = Math.max(heat, 0.3);
			if (opts.color === "community") detectCommunities();
			kick();
		},
		resize() {
			for (const c of [cv, lab]) { c.width = Math.round(W * DPR); c.height = Math.round(H * DPR); }
		},
		theme() { cache = new Map(); palette = []; },
		toggleRotate() { $("rotate3d").checked = opts.rotate = !opts.rotate; idleSince = 0; kick(); },
	};
})();
