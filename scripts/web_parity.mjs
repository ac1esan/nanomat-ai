#!/usr/bin/env node
// Browser-vs-Python parity: run the cases scripts/web_parity.py wrote through
// docs/js/nanomat.js and compare every field.
//
//   node scripts/web_parity.mjs --model docs/model --cases runs/web/parity.json
//
// Exits non-zero when the page would tell a user something Python would not:
// a number off by more than the tolerance, or a different verdict. A verdict may
// legitimately differ only when the signal that decides it sits within float noise
// of a threshold, and those cases are listed separately so they can be read.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const { Model, parseStructure } = await import(path.join(here, "..", "docs", "js", "nanomat.js"));

const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const modelDir = arg("--model", path.join(here, "..", "docs", "model"));
const casesPath = arg("--cases", path.join(here, "..", "runs", "web", "parity.json"));
const tol = +arg("--tol", "0.002");           // eV, for every energy-like number
const outPath = arg("--report", null);

const manifest = JSON.parse(fs.readFileSync(path.join(modelDir, "manifest.json")));
const buf = f => { const b = fs.readFileSync(path.join(modelDir, f)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const model = new Model(manifest, buf("weights.bin"), buf("refs.bin"));
const cases = JSON.parse(fs.readFileSync(casesPath));

const NUM = ["gap", "unc", "latent_distance", "p_metal", "p_metastable", "interval90", "typical_error",
             "p_indirect", "gap_quasiparticle", "gap_quasiparticle_direct", "exciton_binding",
             "exp_gap_est", "work_function", "work_function_unc", "work_function_latent",
             "work_function_interval90", "electron_affinity", "ionisation_potential"];
const EXACT = ["natoms", "verdict", "gap_type", "is_metal_like", "work_function_verdict", "is_layer", "padded"];

const diffs = Object.fromEntries(NUM.map(k => [k, []]));
const fails = [], nearThreshold = [], errors = [];
const cal = manifest.calibration, wcal = (manifest.models.wf || {}).calibration || {};
const near = (x, t, rel = 2e-3) => t !== undefined && x !== null && Math.abs(x - t) <= rel * Math.max(Math.abs(t), 1e-3);
let t0 = Date.now(), n = 0;
for (const c of cases) {
  if (!c.python) continue;
  let js;
  try {
    js = model.predict(parseStructure(c.text));
  } catch (e) {
    errors.push(`${c.name}: ${e.message}`);
    continue;
  }
  n++;
  if (n % 250 === 0) console.error(`  ${n} cases, ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  js.is_layer = js.layer.is_layer;
  js.padded = js.layer.padded;
  const py = c.python;
  for (const k of NUM) {
    const a = py[k] ?? null, b = js[k] ?? null;
    if ((a === null) !== (b === null)) { fails.push(`${c.name}: ${k} python=${a} js=${b}`); continue; }
    if (a === null) continue;
    const d = Math.abs(a - b);
    diffs[k].push(d);
    const lim = k.startsWith("p_") || k.includes("latent") ? tol / 2 : tol;
    if (d > lim) fails.push(`${c.name}: ${k} python=${a.toFixed(5)} js=${b.toFixed(5)} (diff ${d.toExponential(2)})`);
  }
  for (const k of EXACT) {
    const a = py[k] ?? null, b = js[k] ?? null;
    if (a === b) continue;
    // a verdict flip is explained only if a deciding signal sits on its threshold
    const explained = (k === "verdict" || k === "gap_type" || k === "work_function_verdict") && (
      near(js.unc, cal.unc_median) || near(js.unc, cal.unc_q75) ||
      near(js.latent_distance, cal.latent_q75) || near(js.latent_distance, cal.latent_q90) ||
      near(js.p_metal, manifest.models.metal?.threshold) || near(js.p_indirect, manifest.models.type?.threshold) ||
      near(js.work_function_unc, wcal.unc_median) || near(js.work_function_unc, wcal.unc_q75) ||
      near(js.work_function_latent, wcal.latent_q75) || near(js.work_function_latent, wcal.latent_q90));
    (explained ? nearThreshold : fails).push(`${c.name}: ${k} python="${a}" js="${b}"`);
  }
}
const q = (a, p) => { if (!a.length) return null; const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
console.log(`${n} cases compared in ${((Date.now() - t0) / 1000).toFixed(1)} s (${manifest.weights_dtype} weights); ` +
            `${errors.length} could not be read by the page`);
console.log("field                         n     median        p99        max");
for (const k of NUM) {
  const a = diffs[k];
  if (!a.length) continue;
  console.log(`${k.padEnd(26)} ${String(a.length).padStart(5)}  ${q(a, 0.5).toExponential(2).padStart(9)}  ` +
              `${q(a, 0.99).toExponential(2).padStart(9)}  ${Math.max(...a).toExponential(2).padStart(9)}`);
}
for (const [label, list] of [["parse errors", errors], ["verdict flips at a threshold (float noise)", nearThreshold],
                             ["FAILURES", fails]]) {
  if (!list.length) continue;
  console.log(`\n${label}: ${list.length}`);
  for (const l of list.slice(0, 25)) console.log("  " + l);
  if (list.length > 25) console.log(`  ... ${list.length - 25} more`);
}
if (outPath) fs.writeFileSync(outPath, JSON.stringify({ n, errors, nearThreshold, fails,
  diffs: Object.fromEntries(NUM.map(k => [k, { median: q(diffs[k], 0.5), p99: q(diffs[k], 0.99),
                                               max: diffs[k].length ? Math.max(...diffs[k]) : null }])) }, null, 1));
process.exit(fails.length || errors.length ? 1 : 0);
