// NanoMatAI in the browser: structure in, prediction and verdict out.
//
// A port of nanomat/graph.py and nanomat/predict.py with no dependencies, so the
// same file runs in the page's Web Worker and under Node for the parity check
// against Python (scripts/web_parity.py + scripts/web_parity.mjs). Nothing here
// touches the DOM.
//
// What it must reproduce, step by step: parse a POSCAR or CIF; find the vacuum axis
// and pad a vacuum thinner than the cutoff (ensure_vacuum); list periodic neighbours
// within 8 A; the per-atom angular histogram; the forward pass of every ensemble
// member; and the verdict, interval and derived numbers exactly as Predictor.run
// assembles them. The models are tiny, so a plain JavaScript forward pass is enough;
// the one trick is splitting each CGConv linear layer into a node part and an edge
// part, which cuts the work per edge by a factor of seven.

export const SYMBOLS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr " +
  "Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te " +
  "I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg " +
  "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr").split(" ");
const Z_OF = Object.fromEntries(SYMBOLS.map((s, i) => [s, i + 1]));

export const CUTOFF = 8.0;          // neighbour cutoff of the shipped weights
export const ANG_CUTOFF = 4.0;      // angular descriptor cutoff (graph.py)
export const MIN_LAYER_VACUUM = 5.0;
export const METAL_GAP = 0.1;       // predict.py: below this, metal-like

/* ------------------------------------------------------------ small linalg */
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm3 = a => Math.sqrt(dot3(a, a));
const det3 = m => dot3(m[0], cross3(m[1], m[2]));
function inv3(m) {
  const d = det3(m);
  if (Math.abs(d) < 1e-12) throw new Error("the lattice vectors are degenerate");
  const c0 = cross3(m[1], m[2]), c1 = cross3(m[2], m[0]), c2 = cross3(m[0], m[1]);
  // rows of m are lattice vectors; frac = cart . inv, with inv[k][j] = c_j[k] / d
  return [[c0[0] / d, c1[0] / d, c2[0] / d],
          [c0[1] / d, c1[1] / d, c2[1] / d],
          [c0[2] / d, c1[2] / d, c2[2] / d]];
}
const vecMat = (v, m) => [0, 1, 2].map(j => v[0] * m[0][j] + v[1] * m[1][j] + v[2] * m[2][j]);
const mod1 = x => { const y = x - Math.floor(x); return y >= 1 ? 0 : y; };

/* ---------------------------------------------------------------- structure */
// { lattice: 3x3 rows, frac: [[x,y,z]], z: [Z], species: [symbol] }
function makeStructure(lattice, frac, species) {
  if (!frac.length) throw new Error("no atoms found");
  const z = species.map(s => {
    const k = Z_OF[s];
    if (!k) throw new Error(`unknown element "${s}"`);
    if (k >= 100) throw new Error(`${s} (Z=${k}) is outside the model's element table`);
    return k;
  });
  return { lattice, frac, z, species };
}
export const cartCoords = st => st.frac.map(f => vecMat(f, st.lattice));
export const volume = st => Math.abs(det3(st.lattice));

function cleanSymbol(tok) {
  // "Mo_pv", "S/1a2b3c" (VASP 6 hashes), "Fe2+", "O1" (CIF labels)
  const m = /^([A-Z][a-z]?)/.exec(tok.trim());
  return m ? m[1] : tok.trim();
}

/* ------------------------------------------------------------------ POSCAR */
const isNum = s => s !== undefined && s !== "" && !isNaN(+s);
export function parsePOSCAR(text) {
  let L = text.replace(/\r/g, "").split("\n").map(l => l.trim()).filter(l => l !== "");
  if (L.length < 6) throw new Error("too short to be a POSCAR");
  // forgive a missing comment line: the first line is then the scale factor and
  // the next three are lattice vectors (a mistake language models made, too)
  const nums = l => l.split(/\s+/);
  if (nums(L[0]).every(isNum) && nums(L[0]).length <= 3 && nums(L[1]).length >= 3
      && nums(L[1]).slice(0, 3).every(isNum) && nums(L[3]).slice(0, 3).every(isNum)
      && !nums(L[4]).slice(0, 3).every(isNum)) {
    L = ["(no comment line)", ...L];
  }
  const comment = L[0];
  const sc = nums(L[1]).map(Number);
  let lat = [2, 3, 4].map(i => nums(L[i]).slice(0, 3).map(Number));
  if (lat.flat().some(v => !isFinite(v))) throw new Error("could not read the lattice vectors");
  let i = 5;
  let species;
  let counts;
  if (nums(L[i]).every(isNum)) {
    // VASP 4 style: no species line, species from the comment if it lists elements
    counts = nums(L[i]).map(Number);
    const fromComment = comment.split(/\s+/).map(cleanSymbol).filter(s => Z_OF[s]);
    if (fromComment.length !== counts.length)
      throw new Error("this POSCAR has no element line; add one above the counts");
    species = fromComment;
    i += 1;
  } else {
    species = nums(L[i]).map(cleanSymbol);
    counts = nums(L[i + 1]).map(Number);
    i += 2;
  }
  if (counts.length !== species.length || counts.some(c => !Number.isInteger(c) || c < 0))
    throw new Error("element and count lines do not match");
  if (/^s/i.test(L[i])) i += 1;                     // Selective dynamics
  const cart = /^[ck]/i.test(L[i]);
  i += 1;
  // scale: one number (negative = target volume) or three per-axis factors; it
  // applies to the lattice and to cartesian coordinates alike
  const axisScale = sc.length === 3 ? sc
    : [sc[0] < 0 ? Math.cbrt(-sc[0] / Math.abs(det3(lat))) : sc[0]].flatMap(s => [s, s, s]);
  lat = lat.map(v => v.map((x, k) => x * axisScale[k]));
  const inv = inv3(lat);
  const total = counts.reduce((a, b) => a + b, 0);
  if (L.length < i + total) throw new Error(`expected ${total} coordinate lines`);
  const frac = [], spec = [];
  let k = 0;
  for (let s = 0; s < species.length; s++) {
    for (let c = 0; c < counts[s]; c++, k++) {
      const v = nums(L[i + k]).slice(0, 3).map(Number);
      if (v.some(x => !isFinite(x))) throw new Error(`could not read coordinate line ${k + 1}`);
      const f = cart ? vecMat(v.map((x, j) => x * axisScale[j]), inv) : v;
      frac.push(f);
      spec.push(species[s]);
    }
  }
  return makeStructure(lat, frac, spec);
}

/* --------------------------------------------------------------------- CIF */
function cifTokens(text) {
  // a small CIF 1.1 tokenizer: quoted strings, ;-delimited text fields, comments
  const toks = [];
  const lines = text.replace(/\r/g, "").split("\n");
  for (let n = 0; n < lines.length; n++) {
    const line = lines[n];
    if (line.startsWith(";")) {
      const buf = [line.slice(1)];
      while (++n < lines.length && !lines[n].startsWith(";")) buf.push(lines[n]);
      toks.push(buf.join("\n").trim());
      continue;
    }
    const re = /'((?:[^']|'(?=\S))*)'(?=\s|$)|"((?:[^"]|"(?=\S))*)"(?=\s|$)|(#.*)|(\S+)/g;
    let m;
    while ((m = re.exec(line))) {
      if (m[3] !== undefined) break;
      toks.push(m[1] ?? m[2] ?? m[4]);
    }
  }
  return toks;
}
function cifBlocks(text) {
  const toks = cifTokens(text);
  const blocks = [];
  let cur = null;
  for (let i = 0; i < toks.length;) {
    const t = toks[i];
    if (/^data_/i.test(t)) { cur = { items: {}, loops: [] }; blocks.push(cur); i++; continue; }
    if (!cur) { cur = { items: {}, loops: [] }; blocks.push(cur); }
    if (/^loop_$/i.test(t)) {
      i++;
      const names = [];
      while (i < toks.length && toks[i].startsWith("_")) names.push(toks[i++].toLowerCase());
      const vals = [];
      while (i < toks.length && !toks[i].startsWith("_") && !/^(loop_|data_)/i.test(toks[i]))
        vals.push(toks[i++]);
      const rows = [];
      for (let r = 0; r + names.length <= vals.length; r += names.length)
        rows.push(Object.fromEntries(names.map((nm, k) => [nm, vals[r + k]])));
      cur.loops.push({ names, rows });
      continue;
    }
    if (t.startsWith("_")) { cur.items[t.toLowerCase()] = toks[i + 1]; i += 2; continue; }
    i++;
  }
  return blocks;
}
const cifNum = s => +String(s).replace(/\(\d+\)$/, "");
function symop(expr) {
  // "x-y+1/2, -x, z" -> function(f) returning the transformed fractional coordinates
  const parts = expr.toLowerCase().replace(/\s/g, "").split(",");
  if (parts.length !== 3) throw new Error(`cannot read symmetry operation "${expr}"`);
  const rows = parts.map(p => {
    const row = [0, 0, 0, 0];
    const terms = p.replace(/-/g, "+-").split("+").filter(Boolean);
    for (let term of terms) {
      let sign = 1;
      if (term.startsWith("-")) { sign = -1; term = term.slice(1); }
      const m = /^([0-9.]*)(?:\/([0-9.]+))?\*?([xyz])?$/.exec(term);
      if (!m) throw new Error(`cannot read symmetry operation "${expr}"`);
      let val = m[1] === "" ? 1 : +m[1];
      if (m[2]) val /= +m[2];
      if (m[3]) row["xyz".indexOf(m[3])] += sign * val; else row[3] += sign * val;
    }
    return row;
  });
  return f => rows.map(r => r[0] * f[0] + r[1] * f[1] + r[2] * f[2] + r[3]);
}
export function latticeFromParameters(a, b, c, al, be, ga) {
  // pymatgen's Lattice.from_parameters convention; any consistent one gives the same
  // graph, which depends on distances only
  const r = Math.PI / 180, ca = Math.cos(al * r), cb = Math.cos(be * r), cg = Math.cos(ga * r);
  const sa = Math.sin(al * r), sb = Math.sin(be * r);
  const val = Math.max(-1, Math.min(1, (ca * cb - cg) / (sa * sb)));
  const gs = Math.acos(val);
  return [[a * sb, 0, a * cb], [-b * sa * Math.cos(gs), b * sa * Math.sin(gs), b * ca], [0, 0, c]];
}
export function parseCIF(text) {
  const blocks = cifBlocks(text).filter(b => b.loops.some(l => l.names.includes("_atom_site_fract_x")));
  if (!blocks.length) throw new Error("no _atom_site_fract_x loop: is this a CIF of a crystal?");
  const B = blocks[0];
  const it = k => B.items[k];
  const need = ["_cell_length_a", "_cell_length_b", "_cell_length_c",
                "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma"];
  for (const k of need) if (it(k) === undefined) throw new Error(`missing ${k}`);
  const lat = latticeFromParameters(...need.map(k => cifNum(it(k))));
  const site = B.loops.find(l => l.names.includes("_atom_site_fract_x"));
  const opsLoop = B.loops.find(l => l.names.some(n =>
    /_symmetry_equiv_pos_as_xyz|_space_group_symop_operation_xyz|_space_group_symop\.operation_xyz/.test(n)));
  let ops;
  if (opsLoop) {
    const key = opsLoop.names.find(n => /as_xyz|operation_xyz/.test(n));
    ops = opsLoop.rows.map(r => symop(r[key]));
  } else {
    const hm = String(it("_symmetry_space_group_name_h-m") ?? it("_space_group_name_h-m_alt") ?? "P 1");
    if (hm.replace(/\s/g, "").toUpperCase() !== "P1")
      throw new Error(`this CIF names space group ${hm} but lists no symmetry operations; ` +
                      "export it with them, or as P1");
    ops = [f => f];
  }
  const frac = [], spec = [];
  for (const r of site.rows) {
    const occ = r["_atom_site_occupancy"] === undefined ? 1 : cifNum(r["_atom_site_occupancy"]);
    if (occ < 0.99) throw new Error("partial occupancies (disorder) are not supported");
    const sym = cleanSymbol(r["_atom_site_type_symbol"] ?? r["_atom_site_label"]);
    const f0 = ["_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"].map(k => cifNum(r[k]));
    for (const op of ops) {
      const f = op(f0).map(mod1);
      // merge images of one site, as pymatgen does (site tolerance 1e-4, fractional)
      const dup = frac.some((g, k) => spec[k] === sym && [0, 1, 2].every(j => {
        const d = Math.abs(g[j] - f[j]);
        return Math.min(d, 1 - d) < 1e-4;
      }));
      if (!dup) { frac.push(f); spec.push(sym); }
    }
  }
  return makeStructure(lat, frac, spec);
}

export function parseStructure(text) {
  if (/_cell_length_a/i.test(text) || /^\s*data_/im.test(text)) return parseCIF(text);
  return parsePOSCAR(text);
}

/* ------------------------------------------------------- vacuum (graph.py) */
function perpHeight(lat, axis) {
  const [j, k] = [0, 1, 2].filter(i => i !== axis);
  return Math.abs(det3(lat)) / norm3(cross3(lat[j], lat[k]));
}
function axisGap(st, axis) {
  const s = st.frac.map(f => mod1(f[axis])).sort((a, b) => a - b);
  const h = perpHeight(st.lattice, axis);
  if (s.length === 1) return [1.0, s[0], h];
  let best = -1, k = 0;
  for (let i = 0; i < s.length; i++) {
    const g = (i + 1 < s.length ? s[i + 1] : s[0] + 1) - s[i];
    if (g > best) { best = g; k = i; }       // first maximum, like numpy.argmax
  }
  return [best, s[(k + 1) % s.length], h];
}
export function layerInfo(st) {
  let best = null;
  for (let axis = 0; axis < 3; axis++) {
    const [gap, start, h] = axisGap(st, axis);
    const vac = gap * h;
    if (best === null || vac > best.vacuum)
      best = { axis, vacuum: vac, thickness: (1 - gap) * h, _start: start, _gap: gap, _h: h };
  }
  best.is_layer = best.vacuum >= MIN_LAYER_VACUUM;
  return best;
}
export function ensureVacuum(st, cutoff = CUTOFF, target = 20.0) {
  const info = layerInfo(st);
  info.padded = false;
  if (!info.is_layer || info.vacuum > cutoff + 1.0) return [st, info];
  const { axis, _start: start, _gap: gap, _h: h } = info;
  // 1) contiguous layer from fractional 0 along the axis
  const f1 = st.frac.map(f => f.map((x, j) => mod1(j === axis ? x - start : x)));
  const cart = f1.map(f => vecMat(f, st.lattice));
  // 2) stretch the vacuum axis, cartesian coordinates kept
  const thick = (1 - gap) * h;
  const factor = (thick + target) / h;
  const lat = st.lattice.map((v, j) => (j === axis ? v.map(x => x * factor) : v.slice()));
  const inv = inv3(lat);
  // 3) centre the layer (cosmetic: the graph does not depend on it)
  const shift = 0.5 - thick / (thick + target) / 2;
  const frac = cart.map(c => vecMat(c, inv).map((x, j) => mod1(j === axis ? x + shift : x)));
  info.padded = true;
  info.vacuum_after = target;
  return [{ ...st, lattice: lat, frac }, info];
}

/* ------------------------------------------------------ neighbours, angles */
export function neighborList(st, r = CUTOFF) {
  // every periodic pair within r, self excluded: what pymatgen's get_neighbor_list
  // returns, as flat arrays (centre, neighbour, distance, vector)
  const lat = st.lattice, inv = inv3(lat);
  const cart = cartCoords(st);
  // images needed along k: the cell's extent perpendicular to the planes of axis k
  const reach = [0, 1, 2].map(k => {
    const col = [inv[0][k], inv[1][k], inv[2][k]];
    return Math.ceil(r * norm3(col)) + 1;
  });
  const shifts = [];
  for (let a = -reach[0]; a <= reach[0]; a++)
    for (let b = -reach[1]; b <= reach[1]; b++)
      for (let c = -reach[2]; c <= reach[2]; c++)
        shifts.push(vecMat([a, b, c], lat));
  const C = [], N = [], D = [], V = [];
  const r2 = r * r;
  for (let i = 0; i < cart.length; i++) {
    for (let j = 0; j < cart.length; j++) {
      const base = [cart[j][0] - cart[i][0], cart[j][1] - cart[i][1], cart[j][2] - cart[i][2]];
      for (const s of shifts) {
        const v = [base[0] + s[0], base[1] + s[1], base[2] + s[2]];
        const d2 = dot3(v, v);
        if (d2 > r2 || d2 < 1e-16) continue;
        C.push(i); N.push(j); D.push(Math.sqrt(d2)); V.push(v);
      }
    }
  }
  return { c: C, n: N, d: D, v: V };
}

export function angleFeatures(nl, natoms, cutoff = ANG_CUTOFF, nBins = 9) {
  // graph.py angle_features: per atom, a soft histogram of cos(theta) over pairs of
  // bonds within `cutoff`, both weighted by a smooth envelope, normalised to sum 1
  const out = Array.from({ length: natoms }, () => new Float64Array(nBins));
  const centres = Array.from({ length: nBins }, (_, k) => -1 + 2 * k / (nBins - 1));
  const width = centres[1] - centres[0];
  const bonds = Array.from({ length: natoms }, () => []);
  for (let e = 0; e < nl.c.length; e++) if (nl.d[e] <= cutoff) bonds[nl.c[e]].push(e);
  for (let a = 0; a < natoms; a++) {
    const B = bonds[a];
    if (B.length < 2) continue;
    const unit = B.map(e => nl.v[e].map(x => x / Math.max(nl.d[e], 1e-9)));
    const w = B.map(e => 0.5 * (Math.cos(Math.PI * Math.min(Math.max(nl.d[e] / cutoff, 0), 1)) + 1));
    for (let i = 0; i < B.length; i++) {
      for (let j = i + 1; j < B.length; j++) {
        const cs = Math.max(-1, Math.min(1, dot3(unit[i], unit[j])));
        const pair = w[i] * w[j];
        for (let k = 0; k < nBins; k++) {
          const t = (cs - centres[k]) / width;
          out[a][k] += Math.exp(-0.5 * t * t) * pair;
        }
      }
    }
  }
  for (const row of out) {
    const s = row.reduce((x, y) => x + y, 0);
    for (let k = 0; k < nBins; k++) row[k] = s > 1e-9 ? row[k] / s : 0;
  }
  return out;
}

export function buildGraph(st, nAng = 9) {
  const nl = neighborList(st, CUTOFF);
  if (!nl.c.length) return null;
  // One neighbour seen through several periodic images at the same distance sends the
  // same message every time, and in a small cell that is most edges (MoS2: ~180 edges,
  // a few dozen distinct). So edges are grouped by (target, source, distance) and each
  // group is computed once and counted; distinct distances are kept once for the
  // radial part. Distances match to 1e-8 A, far below anything the model resolves.
  const dIndex = new Map(), gIndex = new Map();
  const ud = [], gDst = [], gSrc = [], gU = [], gCount = [];
  for (let e = 0; e < nl.c.length; e++) {
    const dk = Math.round(nl.d[e] * 1e8);
    let u = dIndex.get(dk);
    if (u === undefined) { u = ud.length; dIndex.set(dk, u); ud.push(nl.d[e]); }
    const key = nl.n[e] + "," + nl.c[e] + "," + u;
    const gi = gIndex.get(key);
    if (gi === undefined) {
      gIndex.set(key, gDst.length);
      gDst.push(nl.n[e]); gSrc.push(nl.c[e]); gU.push(u); gCount.push(1);
    } else gCount[gi]++;
  }
  return { z: st.z, nEdges: nl.c.length, ud, gDst, gSrc, gU, gCount,
           ang: nAng ? angleFeatures(nl, st.z.length, ANG_CUTOFF, nAng) : null };
}

/* ------------------------------------------------------------- the network */
const softplus = x => (x > 20 ? x : Math.log1p(Math.exp(x)));
const sigmoid = x => 1 / (1 + Math.exp(-x));

// y = W x + b for W stored row-major (out x in), as torch keeps nn.Linear.weight
function linear(W, b, x, nOut, nIn, out = new Float64Array(nOut)) {
  for (let o = 0; o < nOut; o++) {
    let s = b ? b[o] : 0;
    const row = o * nIn;
    for (let i = 0; i < nIn; i++) s += W[row + i] * x[i];
    out[o] = s;
  }
  return out;
}
// the columns [off, off+len) of W applied to x: one block of a concatenated input
function linearBlock(W, x, nOut, nIn, off, len, out) {
  for (let o = 0; o < nOut; o++) {
    let s = 0;
    const row = o * nIn + off;
    for (let i = 0; i < len; i++) s += W[row + i] * x[i];
    out[o] = s;
  }
  return out;
}

// encode(): embedding (+ angular layer) -> CGConv layers -> mean over atoms.
// CGConv(x)_i = x_i + BN( sum_j sigmoid(Wf z + bf) * softplus(Ws z + bs) ),
// z = [x_i, x_j, rbf(d_ij)], where i is the edge's TARGET (edge_index[1], the
// neighbour) and j its SOURCE (edge_index[0], the centre).
export function encode(m, g) {
  const H = m.h, R = m.n_rbf, N = g.z.length, U = g.ud.length, G = g.gDst.length, IN = 2 * H + R;
  let x = Array.from({ length: N }, (_, a) => {
    const v = new Float64Array(H);
    const row = g.z[a] * H;
    for (let k = 0; k < H; k++) v[k] = m.emb[row + k];
    if (m.ang_w) {
      const a2 = linear(m.ang_w, m.ang_b, g.ang[a], H, m.n_ang);
      for (let k = 0; k < H; k++) v[k] += a2[k];
    }
    return v;
  });
  // radial basis, once per distinct distance
  const rbf = Array.from({ length: U }, (_, u) => {
    const v = new Float64Array(R);
    for (let k = 0; k < R; k++) { const t = g.ud[u] - m.centers[k]; v[k] = Math.exp(-0.5 * t * t); }
    return v;
  });
  const tmp = new Float64Array(H);
  for (const cv of m.convs) {
    const Fi = [], Fj = [], Si = [], Sj = [];
    for (let a = 0; a < N; a++) {
      Fi.push(linearBlock(cv.wf, x[a], H, IN, 0, H, new Float64Array(H)));
      Fj.push(linearBlock(cv.wf, x[a], H, IN, H, H, new Float64Array(H)));
      Si.push(linearBlock(cv.ws, x[a], H, IN, 0, H, new Float64Array(H)));
      Sj.push(linearBlock(cv.ws, x[a], H, IN, H, H, new Float64Array(H)));
    }
    const agg = Array.from({ length: N }, () => new Float64Array(H));
    // the edge block of each linear layer, plus its bias, once per distinct distance
    const EF = [], ES = [];
    for (let u = 0; u < U; u++) {
      const ef = linearBlock(cv.wf, rbf[u], H, IN, 2 * H, R, new Float64Array(H));
      const es = linearBlock(cv.ws, rbf[u], H, IN, 2 * H, R, new Float64Array(H));
      for (let k = 0; k < H; k++) { ef[k] += cv.bf[k]; es[k] += cv.bs[k]; }
      EF.push(ef); ES.push(es);
    }
    for (let q = 0; q < G; q++) {
      const i = g.gDst[q], j = g.gSrc[q], n = g.gCount[q];
      const A = agg[i], fi = Fi[i], fj = Fj[j], si = Si[i], sj = Sj[j], ef = EF[g.gU[q]], es = ES[g.gU[q]];
      for (let k = 0; k < H; k++)
        A[k] += n * sigmoid(fi[k] + fj[k] + ef[k]) * softplus(si[k] + sj[k] + es[k]);
    }
    x = x.map((xa, a) => {
      const A = agg[a], out = new Float64Array(H);
      for (let k = 0; k < H; k++) {
        tmp[k] = (A[k] - cv.rm[k]) / Math.sqrt(cv.rv[k] + 1e-5) * cv.bnw[k] + cv.bnb[k];
        out[k] = xa[k] + tmp[k];
      }
      return out;
    });
  }
  const h = new Float64Array(H);
  for (const xa of x) for (let k = 0; k < H; k++) h[k] += xa[k];
  for (let k = 0; k < H; k++) h[k] /= N;
  return h;
}
export function regress(m, h) {           // CGCNN.head, de-normalised
  const z1 = linear(m.head0_w, m.head0_b, h, m.h, m.h).map(softplus);
  return linear(m.head2_w, m.head2_b, z1, 1, m.h)[0] * m.std + m.mean;
}
export function classify(m, h) {          // CGCNNcls: probability of the positive class
  const z1 = linear(m.body_w, m.body_b, h, m.h, m.h).map(softplus);
  return sigmoid(linear(m.cls_w, m.cls_b, z1, 1, m.h)[0]);
}

/* ----------------------------------------------------------------- weights */
function f16to32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    out[i] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return out;
}
function tensor(buf, spec) {
  if (spec.dtype === "float16") return f16to32(new Uint16Array(buf, spec.offset, spec.size));
  if (spec.dtype === "int8") {
    // one float32 scale per row (scripts/export_web_model.py refs_spec)
    const q = new Int8Array(buf, spec.offset, spec.size);
    const scale = new Float32Array(buf, spec.scale.offset, spec.scale.size);
    const dim = spec.size / spec.scale.size, out = new Float32Array(spec.size);
    for (let i = 0; i < spec.size; i++) out[i] = q[i] * scale[(i / dim) | 0];
    return out;
  }
  return new Float32Array(buf, spec.offset, spec.size);
}
function member(buf, spec, kind) {
  const t = k => tensor(buf, spec.tensors[k]);
  const m = { h: spec.h, n_rbf: spec.n_rbf, n_ang: spec.n_ang, mean: spec.mean, std: spec.std,
              emb: t("emb.weight"), centers: t("centers"), convs: [] };
  if (spec.n_ang) { m.ang_w = t("ang.weight"); m.ang_b = t("ang.bias"); }
  for (let c = 0; c < spec.n_conv; c++) {
    const p = `convs.${c}.`;
    m.convs.push({ wf: t(p + "lin_f.weight"), bf: t(p + "lin_f.bias"),
                   ws: t(p + "lin_s.weight"), bs: t(p + "lin_s.bias"),
                   bnw: t(p + "bn.weight"), bnb: t(p + "bn.bias"),
                   rm: t(p + "bn.running_mean"), rv: t(p + "bn.running_var") });
  }
  if (kind === "reg") {
    m.head0_w = t("head.0.weight"); m.head0_b = t("head.0.bias");
    m.head2_w = t("head.2.weight"); m.head2_b = t("head.2.bias");
  } else {
    m.body_w = t("body.0.weight"); m.body_b = t("body.0.bias");
    m.cls_w = t("cls.weight"); m.cls_b = t("cls.bias");
  }
  return m;
}

export class Model {
  // manifest: docs/model/manifest.json; weights, refs: ArrayBuffers of the two .bin files;
  // meta: the parsed neighbours.json, or null (then no neighbours are named)
  constructor(manifest, weights, refs, meta = null) {
    this.M = manifest;
    this.meta = meta;
    const g = manifest.models;
    this.gap = g.gap.members.map(s => member(weights, s, "reg"));
    this.metal = g.metal ? g.metal.members.map(s => member(weights, s, "cls")) : null;
    this.type = g.type ? g.type.members.map(s => member(weights, s, "cls")) : null;
    this.wf = g.wf ? g.wf.members.map(s => member(weights, s, "reg")) : null;
    this.refs = tensor(refs, manifest.refs.gap);
    this.wfRefs = manifest.refs.wf ? tensor(refs, manifest.refs.wf) : null;
    const hd = manifest.heads;
    this.heads = hd ? Object.fromEntries(Object.entries(hd.w).map(([k, s]) => [k, { w: tensor(weights, s), b: hd.b[k] }])) : null;
    const pop = manifest.population;
    this.pop = pop ? { w: tensor(weights, pop.w), b: pop.b, table: pop.table } : null;
  }

  nearest(h, refs, dim = 128) {
    // Predictor.nearest: the 10 nearest reference embeddings by cosine similarity,
    // most similar first (an exact tie keeps the lower index, as in Python), and
    // 1 - their mean similarity
    let n = 0;
    for (let k = 0; k < dim; k++) n += h[k] * h[k];
    n = Math.max(Math.sqrt(n), 1e-9);
    const e = h.map(v => v / n);
    const R = refs.length / dim, top = new Float64Array(10).fill(-Infinity), idx = new Int32Array(10).fill(-1);
    for (let r = 0; r < R; r++) {
      let s = 0;
      const o = r * dim;
      for (let k = 0; k < dim; k++) s += e[k] * refs[o + k];
      if (s > top[9]) {                       // keep the ten largest, sorted
        let p = 9;
        while (p > 0 && top[p - 1] < s) { top[p] = top[p - 1]; idx[p] = idx[p - 1]; p--; }
        top[p] = s;
        idx[p] = r;
      }
    }
    const k = Math.min(10, R);
    let m = 0;
    for (let i = 0; i < k; i++) m += top[i];
    return { distance: 1 - m / k, idx: Array.from(idx.slice(0, k)), sims: Array.from(top.slice(0, k)) };
  }

  latent(h, refs, dim = 128) {               // 1 - mean cosine similarity to the 10 nearest
    return this.nearest(h, refs, dim).distance;
  }

  neighbours(near, show = 5) {               // Predictor.neighbours
    const M = this.meta;
    if (!M) return [];
    const out = [], seen = new Set();
    for (let j = 0; j < near.idx.length && out.length < show; j++) {
      const i = near.idx[j], gap = M.gap[i];
      const key = M.formula[i] + "|" + (gap === null ? "none" : Math.floor(gap * 100 + 0.5));
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({ id: M.id[i], source: sourceOf(M.id[i]), formula: M.formula[i], dft_gap: gap,
                 flag: M.flag[String(i)] || "", distance: Math.max(0, 1 - near.sims[j]) });
    }
    return out;
  }

  verdict(unc, cal, latent, reason) {      // predict.verdict
    let tier;
    if (unc <= cal.unc_median) tier = "reliable";
    else if (unc <= cal.unc_q75) tier = "check (elevated uncertainty)";
    else tier = `out-of-domain (${reason})`;
    if (latent !== null && latent !== undefined) {
      if (latent > (cal.latent_q90 ?? Infinity)) return "out-of-domain (chemistry unlike anything in training)";
      if (latent > (cal.latent_q75 ?? Infinity) && tier === "reliable")
        return "check (chemistry sparsely covered by training data)";
    }
    return tier;
  }

  predict(stIn) {
    const M = this.M, cal = M.calibration;
    const [st, info] = ensureVacuum(stIn, CUTOFF);
    const warnings = [];
    if (!info.is_layer)
      warnings.push(`no vacuum gap >= 5 A found (max ${info.vacuum.toFixed(1)} A): does not look like ` +
                    "an isolated 2D layer; model trained on monolayers only");
    else if (info.padded)
      warnings.push(`vacuum was ${info.vacuum.toFixed(1)} A < cutoff 8 A; padded to ` +
                    `${info.vacuum_after.toFixed(0)} A to avoid spurious inter-layer edges`);
    const g = buildGraph(st, M.models.gap.members[0].n_ang);
    if (!g) throw new Error("no atom has a neighbour within 8 A: not a structure the model can read");

    const hs = this.gap.map(m => encode(m, g));
    const preds = this.gap.map((m, k) => regress(m, hs[k]));
    const gap = preds.reduce((a, b) => a + b, 0) / preds.length;
    const unc = Math.sqrt(preds.reduce((a, p) => a + (p - gap) ** 2, 0) / preds.length);
    const emb = new Float64Array(hs.length * hs[0].length);
    hs.forEach((h, k) => emb.set(h, k * h.length));
    const near = this.nearest(hs[0], this.refs);
    const ld = near.distance;

    let pMeta = null;
    if (this.pop) {
      let z = this.pop.b;
      for (let k = 0; k < emb.length; k++) z += this.pop.w[k] * emb[k];
      pMeta = 1 / (1 + Math.exp(-z));
    }
    const pMetal = this.metal
      ? this.metal.reduce((a, m) => a + classify(m, encode(m, g)), 0) / this.metal.length : null;
    const pInd = this.type
      ? this.type.reduce((a, m) => a + classify(m, encode(m, g)), 0) / this.type.length : null;

    let v = this.verdict(unc, cal, ld, "possibly metal / unusual structure");
    if (pMetal !== null && pMetal >= M.models.metal.threshold) {
      v = `out-of-domain (metal gate: p_metal=${pMetal.toFixed(2)})`;
      warnings.push("metal gate says metal; gap regressor is not trained on metals");
    }
    if (!info.is_layer) v = "out-of-domain (not a 2D layer)";
    const metalLike = gap < METAL_GAP;
    const ood = v.startsWith("out-of-domain");

    // interval: the cell of (resembles metastable?, spread quartile)
    let iv = cal.scale90 * unc;
    const T = this.pop && this.pop.table;
    if (T && pMeta !== null) {
      const col = T.unc_edges.filter(e => e <= unc).length;
      iv = T.scale[pMeta > T.p_threshold ? 1 : 0][col] * unc;
    }

    const out = { natoms: st.z.length, gap, unc, verdict: v, latent_distance: ld,
                  p_metal: pMetal, p_metastable: pMeta, interval90: iv,
                  typical_error: cal.tier_mae[tierKey(v)], is_metal_like: metalLike,
                  gap_type: pInd === null ? null : (pInd >= M.models.type.threshold ? "indirect" : "direct"),
                  p_indirect: pInd, neighbours: this.neighbours(near),
                  layer: { axis: info.axis, vacuum: info.vacuum, thickness: info.thickness,
                                             is_layer: info.is_layer, padded: info.padded },
                  warnings };
    if (!metalLike && !ood && this.heads) {      // _apply_heads
      const val = k => {
        const { w, b } = this.heads[k];
        let s = b;
        for (let i = 0; i < emb.length; i++) s += w[i] * emb[i];
        return s + w[emb.length] * gap;
      };
      const eb = val("exciton_binding"), direct = val("gap_dir_gw");
      Object.assign(out, { gap_quasiparticle: val("gap_gw"), gap_quasiparticle_direct: direct,
                           exciton_binding: eb, exp_gap_est: direct - eb });
    }
    if (this.wf) {                                 // _wf_fields
      const wcal = M.models.wf.calibration;
      const whs = this.wf.map(m => encode(m, g));
      const wp = this.wf.map((m, k) => regress(m, whs[k]));
      const wv = wp.reduce((a, b) => a + b, 0) / wp.length;
      const wu = Math.sqrt(wp.reduce((a, p) => a + (p - wv) ** 2, 0) / wp.length);
      const wl = this.wfRefs ? this.latent(whs[0], this.wfRefs) : null;
      const wverdict = wcal.unc_median !== undefined ? this.verdict(wu, wcal, wl, "unusual structure") : null;
      Object.assign(out, { work_function: wv, work_function_unc: wu, work_function_latent: wl,
                           work_function_verdict: wverdict,
                           work_function_interval90: wcal.scale90 ? wcal.scale90 * wu : null });
      if (!metalLike && !ood && !(wverdict || "").startsWith("out-of-domain")) {
        out.electron_affinity = wv - gap / 2;
        out.ionisation_potential = wv + gap / 2;
      }
    }
    return out;
  }
}
const sourceOf = id => (id.startsWith("2dm-") ? "2dmatpedia" : id.startsWith("mc2d-") ? "mc2d" : "alexandria");
export const tierKey = v => (v.startsWith("reliable") ? "reliable" : v.startsWith("check") ? "check" : "out_of_domain");
export const tierCode = v => (v.startsWith("reliable") ? 0 : v.startsWith("check") ? 1 : 2);

/* -------------------------------------------------------------- formula */
export function formulaOf(st) {
  // reduced formula, elements in order of first appearance (display only)
  const counts = new Map();
  for (const s of st.species) counts.set(s, (counts.get(s) || 0) + 1);
  const gcd = (a, b) => (b ? gcd(b, a % b) : a);
  const g = [...counts.values()].reduce(gcd);
  return [...counts].map(([s, n]) => s + (n / g === 1 ? "" : n / g)).join("");
}
