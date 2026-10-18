// Runs the predictor off the page's main thread, so a large cell does not freeze the
// interface. The models are fetched once, on the first request, with progress
// reported back; after that every prediction is local and the file never leaves
// the visitor's machine.
import { Model, parseStructure, formulaOf } from "./nanomat.js";

let model = null;
let loading = null;

async function fetchBuffer(url, expected, report) {
  const res = await fetch(url, { cache: "no-cache" });
  if (!res.ok) throw new Error(`could not load ${url} (${res.status})`);
  if (!res.body || !res.body.getReader) {
    const b = await res.arrayBuffer();
    report(b.byteLength);
    return b;
  }
  const reader = res.body.getReader();
  const out = new Uint8Array(expected);
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (got + value.length > out.length) throw new Error(`${url} is larger than the manifest says`);
    out.set(value, got);
    got += value.length;
    report(value.length);
  }
  if (got !== expected) throw new Error(`${url}: got ${got} bytes, expected ${expected}`);
  return out.buffer;
}

async function load(base) {
  const manifest = await (await fetch(base + "manifest.json", { cache: "no-cache" })).json();
  const nbBytes = manifest.bytes["neighbours.json"] || 0;
  const total = manifest.bytes["weights.bin"] + manifest.bytes["refs.bin"] + nbBytes;
  let done = 0;
  const report = n => { done += n; postMessage({ type: "progress", loaded: done, total }); };
  const [w, r, nb] = await Promise.all([
    fetchBuffer(base + "weights.bin", manifest.bytes["weights.bin"], report),
    fetchBuffer(base + "refs.bin", manifest.bytes["refs.bin"], report),
    nbBytes ? fetchBuffer(base + "neighbours.json", nbBytes, report) : null,
  ]);
  const meta = nb ? JSON.parse(new TextDecoder().decode(nb)) : null;
  model = new Model(manifest, w, r, meta);
  postMessage({ type: "ready", created: manifest.created_utc });
}

self.onmessage = async e => {
  const { id, text, base } = e.data;
  try {
    if (!model) {
      loading = loading || load(base || "model/");
      await loading;
    }
    const st = parseStructure(text);
    const t0 = performance.now();
    const res = model.predict(st);
    res.formula = formulaOf(st);
    res.elements = [...new Set(st.species)];
    res.ms = performance.now() - t0;
    postMessage({ type: "result", id, res });
  } catch (err) {
    loading = model ? loading : null;   // a failed download may be retried
    postMessage({ type: "error", id, message: err.message || String(err) });
  }
};
