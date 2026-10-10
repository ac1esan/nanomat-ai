# Model card — NanoMatAI

Three models ship together: a band-gap ensemble, a metal/semiconductor gate that
runs before it, and a direct/indirect gap classifier. All are CGCNN
(PyTorch Geometric) over the same graph: periodic neighbours within 8 Å, edge
feature = distance, atomic number embedded at the nodes.

## Band-gap ensemble — `weights/cgcnn_2d_ensemble.pt`

| | |
|---|---|
| **Task** | Band gap (eV, PBE level) of an isolated 2D layer from its structure |
| **Architecture** | Z-embedding 128 **plus a 9-bin angular descriptor**, 4 × `CGConv`, Gaussian RBF edge features (40 centres on 0–8 Å), mean pooling, MLP head, dropout 0.2 |
| **Training data** | 45 115 monolayers, all semiconductors (`gap > 0.01` eV), all PBE: the 10 733 training structures of the stable set (Alexandria 2D, `e_above_hull ≤ 0.1` eV/atom), **plus 32 736 metastable Alexandria monolayers** (0.1 < `e_above_hull` ≤ 1.0) **plus 1 646 non-magnetic 2DMatPedia monolayers** that Alexandria does not contain. No added formula occurs in the validation or test split, or in the held-out test sets of the three data experiments |
| **Split** | Composition-disjoint (`GroupShuffleSplit` on reduced formula). Validation (1 290) and test (1 326) are the stable-set ones and have not changed since the first model, so every generation is scored on the same 1 326 structures |
| **Ensemble** | 5 members, seeds 0–4, up to 200 epochs, batch 64, Adam 1e-3 with plateau decay, early stopping after 40 epochs without improvement |
| **Test performance** | **MAE 0.186 eV** (95% bootstrap CI 0.172–0.202), RMSE 0.327, R² 0.942. Members: 0.220 / 0.208 / 0.215 / 0.218 / 0.217 |
| **Why the extra data** | Three experiments, each against a control retrained in the same session. Stable set alone → + metastable to 0.2 + 2DMatPedia: 0.249 → 0.225 on the stable test. That → + metastable to 0.5: 0.230 → 0.197 (paired −0.032 eV, bootstrap 95% −0.043…−0.022). That → + metastable to 1.0: 0.191 → **0.186** (−0.004, −0.011…+0.003), 0.256 → **0.248** on held-out 0.2–0.5 eV/atom (−0.007, −0.012…−0.002). The last step passed its rule and little more: the return on metastable data has run out. Details below |
| **Why angles** | An edge carries a distance and nothing else, so two polymorphs of one composition are nearly the same graph. Against its own control, adding a per-atom bond-angle histogram took MAE from 0.271 to 0.246 eV (paired +0.026, bootstrap 95% +0.015…+0.037, Wilcoxon p = 4·10⁻⁵) and doubled how far the model separates 1H-MX₂ from 1T-MX₂ |
| **Composition baseline** | **0.430 eV** on the identical test set (`scripts/composition_baseline.py`), so structure wins by 57%. The 0.360 eV quoted in early versions came from five-fold cross-validation, a different protocol that leaks near-duplicate compositions |
| **Checksum** | SHA-256 `136efe2086f6a954d5d8c64a2da0727b7c2f4aefcac852b7dfb744795d4f6633` |

The checkpoint also carries its own `calibration` block, both gap corrections, the
latent heads, the metastability head with its interval table (`population`) and
45 115 reference embeddings, so it can judge and correct its own output without any
external file. Its checksum therefore changes whenever the calibration is refitted,
not only when the weights are retrained.

### What the stability filter was costing

The first models trained only on `e_above_hull ≤ 0.1`, because a 26k set at 0.2
scored worse than the 13k one (0.262 against 0.215). Each of those numbers was
measured on its own test split, and the 26k split was half metastable, which is
harder for any model — the comparison changed the population it measured on, not
only the data it trained on. `scripts/stability_experiment.py` asks the question
properly: four arms, one split, one environment, only the training part differs.
Its two extra test sets were cut from the added data by formula; none of their
formulas is in any added data, but about 40% of the metastable set are polymorphs of
formulas in the stable training set every arm shares (this card used to say no arm
trained on their formulas, which was wrong).

| test set | stable only | + 2DMatPedia | + metastable to 0.2 | **+ both** |
|---|---|---|---|---|
| stable Alexandria, 1 326 | 0.249 | 0.261 | 0.236 | **0.225** |
| held-out metastable Alexandria, 2 512 | 0.500 | 0.490 | 0.314 | **0.311** |
| held-out 2DMatPedia, 397 | 0.770 | 0.464 | 0.721 | **0.437** |

The decision rule was written down before the runs finished: an arm had to stay
within +0.005 eV of the control on the stable test (upper 95% bound below +0.015)
and beat it on one of the other two with the whole interval below zero. The
2DMatPedia-only arm failed the first condition (+0.011); the other two passed; the
combined arm has the larger gain. The two sources are complementary — each repairs
its own population and barely moves the other's.

Where the model was weakest the change is largest (control → both, MAE in eV):
carbon on 2DMatPedia 1.59 → 0.67 (n = 17, read it qualitatively), boron 1.82 → 0.92,
nitrogen on metastable Alexandria 1.01 → 0.38, hydrogen 1.25 → 0.83.

And the polymorphs gained more than they did from the angles, on C2DB, which none of
these models trained on:

| | no angles | angles | + metastable to 0.2 + 2DMatPedia | + to 0.5 | + to 1.0 (shipped) |
|---|---|---|---|---|---|
| 1H-MX₂ against 1T-MX₂, compression | 0.18 | 0.37 | 0.65 | 0.68 | **0.81** |
| sign of the 1H − 1T difference | — | 79% | 92% | 94% | **92%** |
| correlation of the 1H − 1T differences | — | — | 0.64 | 0.77 | **0.90** |
| within one composition | 0.37 | 0.49 | 0.69 | 0.91 | **0.81** |

In hindsight the reason is plain: a metastable structure is, nearly by definition,
another polymorph of a composition that has a stable one. The filter removed exactly
the "same formula, different structure" contrast a model needs in order to learn
what separates polymorphs. The angles made that contrast visible; the data supplied
it. The two steps further up the hull (next sections) moved the two measures in turn:
to 0.5 eV/atom the within-composition ratio (0.69 → 0.91) and hardly the 1H/1T pair,
to 1.0 the 1H/1T pair (0.68 → 0.81, correlation 0.77 → 0.90) while within-composition
fell back to 0.81. Neither is monotone in the data.

### Past 0.2 eV/atom

`scripts/hull_experiment.py` repeats the design with that combined arm as the control
and Alexandria semiconductors at 0.2–0.5 eV/atom as the addition: three arms side by
side in one session, the same validation and test, the two held-out sets above, and a
fourth, 2 524 structures at 0.2–0.5 eV/atom whose formulas no arm trained on. The
decision rule — the same two conditions, applied by the evaluation itself — is in the
script's header, written before any arm trained.

| test set | control | + 0.2–0.3 | **+ 0.2–0.5** | + 0.2–0.5 vs control, 95% CI |
|---|---|---|---|---|
| stable Alexandria, 1 326 | 0.230 | 0.210 | **0.197** | −0.032 [−0.043, −0.022] |
| held-out metastable Alexandria, 2 512 | 0.313 | 0.281 | **0.267** | −0.045 [−0.057, −0.035] |
| held-out 2DMatPedia, 397 | 0.450 | 0.436 | **0.412** | −0.038 [−0.063, −0.013] |
| held-out 0.2–0.5 eV/atom, 2 524 | 0.438 | 0.324 | **0.256** | −0.182 [−0.197, −0.167] |

Both additions passed; the larger has the larger summed gain (0.266 against 0.159 eV)
and shipped as the previous release. The response is monotone and has not saturated
at 0.5. The control scores
0.230 where the same data scored 0.225 on the previous GPU, the environment's share.
Light elements on the far set: hydrogen 1.18 → 0.58, oxygen 0.68 → 0.44, nitrogen
0.46 → 0.31, boron 0.56 → 0.40. Two slices moved the other way without significance:
boron on the metastable set (+0.15, n = 26, CI −0.03…+0.34) and hydrogen on
2DMatPedia (+0.07, n = 40, CI −0.03…+0.16).

### Past 0.5 eV/atom

`scripts/hull_1ev_experiment.py`: the previous release's exact recipe as the control,
plus 4 567 Alexandria semiconductors at 0.5–1.0 eV/atom, the four test sets above and a
fifth, 453 structures at 0.5–1.0 whose formulas no arm trained on. That fifth set is
reported but does not vote — an arm trained on a band improves on it almost by
construction. Rule as before, written into the script before training.

| test set | control | **+ 0.5–1.0 (shipped)** | Δ, 95% CI |
|---|---|---|---|
| stable Alexandria, 1 326 | 0.191 | **0.186** | −0.004 [−0.011, +0.003] |
| held-out metastable Alexandria, 2 512 | 0.263 | **0.260** | −0.003 [−0.007, +0.002] |
| held-out 2DMatPedia, 397 | 0.417 | **0.402** | −0.015 [−0.034, +0.004] |
| held-out 0.2–0.5 eV/atom, 2 524 | 0.256 | **0.248** | −0.007 [−0.012, −0.002] |
| held-out 0.5–1.0 eV/atom, 453 (no vote) | 0.327 | 0.222 | −0.105 [−0.143, −0.073] |

It passes on the far set alone. Nitrogen there gains −0.039; carbon and boron on
2DMatPedia about −0.14 (n = 17 and 21). The control, the previous recipe retrained,
scored 0.191 against the 0.197 that recipe shipped with — two trainings of one recipe
differ by about 0.005 eV, so most of the 0.011 between the two releases is not the
data. The step to 0.5 bought 0.032 on the stable test, this one 0.004: the next gain
has to come from somewhere other than metastable data. The trust layer paid for it —
Spearman of the spread against error 0.45 → 0.40, of the latent distance 0.41 → 0.39
— while the tiers still rank (0.098 / 0.167 / 0.346 eV against 0.103 / 0.169 / 0.372).

### 2DMatPedia, checked structure by structure before use

Its labels were compared with Alexandria's by **structure**, not by formula
(`scripts/audit_2dmatpedia.py`): every layer put in one frame (vacuum axis to the
layer normal, equal vacuum, primitive cell), then pymatgen's `StructureMatcher` with
tolerances tightened for slabs — the defaults normalise by a volume that is mostly
vacuum and accepted as MoS₂ a structure 1 eV/atom higher in energy. A match must
also agree on the layer's thickness within 0.2 Å, because even the tightened matcher
passes a planar and a buckled layer as one (its tolerance in a slab is ~0.5 Å). The
threshold comes from the pairs the matcher alone accepted: up to 0.2 Å the gap
disagreement is flat, past it it triples.

- 1 131 of 6 351 entries have a structural twin in Alexandria. Their PBE energies
  agree to a median 0.0014 eV/atom (94% within 0.01): one computational setup,
  including +U.
- On the same structure the gaps agree to MAE 0.088 eV (median 0.055, r = 0.995),
  0.084 on non-magnetic ones. Paired by formula instead — the comparison made before
  — the same entries disagree by 0.306: most of the apparent disagreement between
  the databases was polymorphs, not labels.
- Where they disagree by more than 0.3 eV (about 3% of pairs), C2DB and JARVIS
  dft_2d as arbiters side with neither database overall. Alexandria is wrong on
  planar honeycombs with band edges at K — graphene 1.23 eV and planar silicene 0.86,
  where the other databases have zero, planar BP 1.31 against 0.90 — and its value
  depends on the cell (the same BP layer in three larger cells gives 0.906), which
  points at k-point sampling. **Both graphene entries sit in this model's test split.**
  2DMatPedia is wrong on the ZrNCl family by about 1 eV. On magnetic materials the
  databases agree less (91% on metal or not, against 97% on non-magnetic ones) with
  neither consistently right — hence the rule: non-magnetic 2DMatPedia entries only.
  GaAs, AlAs, the ZrX₃/HfX₃ trihalides, FeCl₂ and FeI₂, listed here before as label
  errors or as "a different magnetic state", were different structures that the
  matcher let through before the thickness check.

Before any of its data was used, the ensemble of the time was run on 2DMatPedia as
an external validation — a database it had never seen: MAE 0.752 eV overall, and the
verdict still ranked the error, 0.306 / 0.580 / 0.829 eV for reliable / check /
out-of-domain, with 78% of that population flagged out-of-domain by the model itself.

### Uncertainty, and why the raw spread is not enough

| Signal | Spearman vs abs. error |
|---|---|
| MC-dropout on a single model | 0.17 |
| Ensemble spread (5 seeds) | 0.40 (0.45 when training stopped at 0.5 eV/atom) |
| Distance to training set in latent space | 0.39 (0.41 at 0.5, 0.47 at 0.2 eV/atom) |
| The two combined | 0.44 (0.47, 0.50) |

Both signals lost ground as the training set grew: covering more leaves less that is
far away, and smaller, more even errors leave less to rank. Coverage of the raw
spread is poor: ±1σ contains 44% of cases, not 68%. A scale factor fitted on
validation (×4.07 for 90%, ×2.04 for 68%) gives 91.5% coverage on the held-out test
split. Conditional coverage by uncertainty quartile is 84 / 93 / 93 / 95% under that
single scale.

**Population-scoped calibration** (`scripts/calibrate_population.py`). The validation
split is near-hull; the training set is not. On the held-out metastable sets
(0.1–0.5 eV/atom) the single scale covers 86%. Neither signal above separates the two
populations (ROC-AUC 0.56 for the spread, 0.63 for latent distance); a logistic head
on the concatenated member embeddings does, at 0.91 on held-out data. It is fitted on
training structures only: 10 733 near-hull against 32 736 metastable Alexandria
entries. 2DMatPedia records no hull distance. The 90% scale is looked up per cell of
(head > 0.5) × spread quartile, fitted on the validation split plus the held-out
metastable sets:

| | spread q1 | q2 | q3 | q4 |
|---|---|---|---|---|
| resembles near-hull | ×5.11 | ×3.57 | ×3.32 | ×3.59 |
| resembles metastable | ×8.48 | ×4.95 | ×4.35 | ×3.48 |

Scored by 20 composition-grouped halvings of the metastable sets (fit on one half,
report on the other): metastable coverage 85.8% ± 0.5 → 89.6% ± 0.8, near-hull test
91.5% → 91.1% with the median width ±0.37 → ±0.36 eV, quartiles 91 / 91 / 90 / 93%.
2DMatPedia, used by no fit here, goes the other way: 85.1% → 84.1%. On the screening
table's unseen rows the interval covers 87.9% (Alexandria 90.4%, C2DB 82.5%, JARVIS
79.2%). The reliable tier's median interval on the test split is ±0.25 eV.

**Rejected: the head as a verdict rule** (measured on the previous ensemble). Demoting reliable to check when the head
says metastable sharpened the near-hull reliable tier (0.104 → 0.094 eV) and inverted
the tier order on every unseen population: held-out metastable Alexandria reliable
0.251 / check 0.212, and the same inversion on C2DB and JARVIS rows. The structures it
kept in the reliable tier were the metastable ones the head takes for near-hull —
where the model is confidently wrong. So the head sets the interval only.

The tiers keep their order on every held-out set the model never trained on: 0.145 /
0.219 / 0.388 eV on metastable Alexandria at 0.1–0.2 eV/atom, 0.150 / 0.216 / 0.393 at
0.2–0.5, and 0.125 / 0.247 / 0.480 on 2DMatPedia.

**A retraction: phosphorene was not the failure it was presented as.** Earlier
versions of this card motivated the latent-distance signal with phosphorene: 0.82 eV
predicted, 2.0 eV measured, verdict "reliable". The structure is in the training
split (`agm2000000335`), and its PBE gap is 0.85–0.90 eV in all four databases that
carry it (Alexandria 0.901, C2DB 0.903, 2DMatPedia 0.852, JARVIS dft_2d 0.848). The
prediction was right at the level the model predicts; the 1.2 eV was PBE against an
optical measurement — physics, which is what the quasiparticle and optical outputs
are for. What latent distance flagged was a sparse neighbourhood: phosphorene is the
only elemental-phosphorus layer in training, so most of its ten nearest neighbours
are unlike it. On a training point that is a false alarm, not an error caught. The
signal stands on its statistics above, not on this anecdote. With metastable
phosphorus layers in training the neighbourhood is no longer sparse: the shipped
ensemble predicts 0.908 eV, latent distance 0.27 against a q90 of 0.32, and the
latent signal no longer fires. It reads `check` because the spread (0.102) is just
above the median — the spread's call, not the neighbourhood's.

**Looked for, not found: a better third signal** (`scripts/trust_signals.py`, rule
written before scoring — beat spread × latent distance by more than 0.02 in Spearman
on the test split and lose on none of the held-out metastable, far and 2DMatPedia
sets). Spearman against |error|, test / T_meta / T_far / T_2dmp / C2DB / JARVIS:

| signal, combined with the spread | test | T_meta | T_far | T_2dmp | C2DB | JARVIS |
|---|---|---|---|---|---|---|
| latent distance, k = 10 (shipped) | 0.437 | 0.391 | 0.367 | 0.400 | 0.307 | 0.278 |
| latent distance, k = 1 | 0.455 | 0.395 | 0.390 | 0.411 | 0.333 | 0.368 |
| Mahalanobis (Ledoit–Wolf, member-0 embeddings) | 0.405 | 0.376 | 0.342 | 0.392 | 0.313 | 0.307 |
| error head (ridge, fitted on validation) | 0.491 | 0.376 | 0.321 | 0.438 | 0.250 | 0.389 |

k = 1 misses the threshold by 0.002; the error head learns the near-hull population
and loses on the metastable sets. Fitted with half of the metastable set as well (an
exploration, 10 composition-grouped halvings, not a test under the rule) the head
reads 0.495 / 0.413 / 0.370 / 0.459 / 0.262 / 0.399 — ahead everywhere but C2DB. It
is the next experiment, with its own rule.

An intermediate attempt that did **not** work, recorded so it is not repeated:
counting how many training structures share the query's chemical system correlates
with error in the wrong direction (MAE 0.236 for unseen systems versus 0.378 for
systems seen ten or more times), because chemically rich systems are both better
represented and intrinsically harder.

### Verdict tiers

Thresholds are validation quartiles; the typical error of each tier is measured on
the held-out test split with the full verdict, spread and latent distance together.
Measured on the spread alone they would be 0.11 / 0.18 / 0.35.

| Verdict | Condition | Typical error |
|---|---|---|
| reliable | spread ≤ 0.090 eV and latent distance ≤ 0.239 | 0.10 eV |
| check | spread ≤ 0.147 eV, or a reliable spread with latent distance > 0.239 | 0.17 eV |
| out-of-domain | spread > 0.147 eV, or latent distance > 0.315, or the metal gate fires | 0.35 eV |

On held-out metastable structures the same tiers err about 0.15 / 0.22 / 0.39 eV: the
order holds, the level is about one and a half times the near-hull one in the reliable
tier, and the interval widens for them (above).

Out-of-domain results deliberately suppress the estimated experimental gap and the
calibrated interval: that interval assumes the spread is meaningful, which is
precisely what fails there.

## Metal gate — `weights/cgcnn_2d_metal.pt`

| | |
|---|---|
| **Task** | Binary: is this 2D structure a metal/semimetal (gap ≤ 0.05 eV)? Runs before the regressor |
| **Training data** | 93 238 structures, 56% metals: the 41 611 of release 1.1 (Alexandria 2D ≤ 0.1 eV/atom + C2DB + JARVIS dft_2d with metals kept, Alexandria at 0.1–0.2 eV/atom, non-magnetic 2DMatPedia) plus 51 627 Alexandria monolayers at 0.2–0.5 eV/atom, metals included |
| **Architecture** | Ensemble of 5 CGCNN classifiers (seeds 0–4) with the gap ensemble's 9-bin angular descriptor; the probability is the mean of the members' |
| **Split / performance** | The composition-disjoint validation and test splits of release 1.1. ROC-AUC **0.984** on its test split (balanced accuracy 0.934, n 2 440), **0.951** on held-out metastable Alexandria (n 2 998), **0.939** on held-out 2DMatPedia (n 478), **0.917** on held-out Alexandria at 0.2–0.5 eV/atom (n 7 910), none of whose compositions are in training |
| **Threshold** | 0.473, fitted on the validation mean of the five — `pos_weight` shifts probabilities away from 0.5 |
| **Checksum** | SHA-256 `36c4984657f29840981a4e0f62996b9717b298f73490f09442b012f18d851394` |

**Retrained again in release 1.3** (`scripts/classifier_hull_experiment.py`, rule
written before training, now with reference checks for both classifiers). Arms: the
previous recipe (one model, the control), the same data with five seeds, five seeds
with Alexandria 0.2–0.5 added. Against the control, paired: +0.014 [+0.010, +0.018]
on the old test split, +0.033 on metastable Alexandria, +0.041 on 2DMatPedia, +0.078
on the far set; the five seeds alone account for about half. Semiconductors wrongly
rejected fall to 6.5 / 13.9 / 17.9% on the first three sets (from 9.6 / 18.0 / 25.8%),
metals caught 93.3 / 89.7 / 87.8%. The control failed the reference check (MoS₂ 0.42
against its threshold 0.40). The shipped gate holds all six reference semiconductors
from −2% to +2% in-plane strain, MoSe₂ at +2% sitting at the threshold (0.47); from +3%
it rejects MoSe₂ and WSe₂, from +4% WS₂ too. The previous gate rejected WSe₂ from +2%.
A MoS₂ polymorph labelled 0.83 eV (`agm2000041223`) is rejected by every version
(shipped 0.68).

**Release 1.1** (`scripts/classifier_experiment.py`, rule written before
training). Against a control rerun of the old recipe in the same session, paired on
the same structures: +0.020 [+0.013, +0.027] on the old test split, +0.123 [+0.108,
+0.137] on metastable Alexandria, +0.108 [+0.067, +0.150] on 2DMatPedia. The control
with another seed moved by −0.004, −0.002 and −0.046: a single classifier is
seed-sensitive on unfamiliar data, and the gains are well outside it. At the
validation threshold the gate catches 92.7 / 85.6 / 87.8% of metals and wrongly
rejects 9.6 / 18.0 / 25.8% of semiconductors on the three sets; the previous gate
caught 83.8 / 60.2 / 66.8% at 8.6 / 15.8 / 21.2%.

**Departure from the written rule (release 1.1).** Arms with and without angles were
indistinguishable on all three sets, and the tie-breaker picked the one without.
That arm rejects WS₂ as a metal at +1% in-plane strain (p 0.56, threshold 0.47);
this one holds all six reference semiconductors between −1% and +1% and first
rejects WSe₂ at +2% (p 0.58, threshold 0.52). One percent is the spread of lattice
constants between functionals, so the angular arm ships, and a test checks −1%, 0
and +1%.

Mixing three DFT functionals is deliberate and limited to this model: the
metal/semiconductor distinction is far more robust across functionals than the gap
value. The same mixing would be wrong for the regressor and is not done there.

An Alexandria-only version scored a higher ROC-AUC (0.952) and was useless where it
mattered — `p(metal) = 0.00` on graphene, because only two of its 19 691 training
structures were carbon-only and both were labelled semiconductors. The merged model
returned 1.00 on graphene with no false positives on MoS₂, MoSe₂, WS₂, WSe₂ or h-BN,
and the retrained ones still do (shipped: graphene 0.94; the five at 0.00–0.21).

## Work function — `weights/cgcnn_2d_workfunction.pt`

| | |
|---|---|
| **Task** | Vacuum level minus Fermi level, in eV, from the structure |
| **Training data** | The 3 505 C2DB structures carrying a work function, values outside 1.5–8 eV dropped as failed calculations. Metals included: the quantity is defined for them too |
| **Split / performance** | Composition-disjoint, 5-model ensemble. **MAE 0.254 eV** (95% CI 0.229–0.280), RMSE 0.358, R² 0.871, n_test 337. Members 0.242–0.298 |
| **Composition baseline** | 0.314 eV on the same test set, so structure wins by 19% |
| **Calibration** | Raw ±1σ covers 25%; a validation-fitted ×7.09 gives 93% on the held-out split. Spearman against error: spread 0.351, latent distance 0.277, product 0.372 |
| **Checksum** | SHA-256 `b1d23e1645904766d0092508c198f97191aad1a095bcc600ef093ff7f8d131d3` |

**What the number is, and is not.** For a metal, vacuum minus Fermi level is the
work function proper. For an undoped semiconductor DFT places the Fermi level
mid-gap, so the value is a reference level rather than something a probe measures.
h-BN illustrates this: the model returns 3.4 eV where the literature quotes 4.5, and
C2DB itself has 3.49 — the model is right about its target, and the target is not
the quantity the literature means.

**What it unlocks.** With the gap it fixes both band edges relative to vacuum:
electron affinity = work function − gap/2, ionisation potential = work function +
gap/2. Against published monolayer values the ionisation potentials land within
about 0.1 eV (MoS₂ 6.09 vs 6.1, MoSe₂ 5.45 vs 5.5, WS₂ 5.89 vs 6.0, WSe₂ 5.20 vs 5.2)
and the affinities within 0.45 eV. The mid-gap assumption is a convention, not a law.

**The edges need both models to stand behind their half**, so they are withheld
whenever either verdict says out-of-domain. This model has its own trust layer, not
a share of the gap model's: its own uncertainty quartiles, its own ×7.09 interval and
its own latent distance against its own 2 823 training embeddings. That matters
because the two were trained on different databases — C2DB here, Alexandria for the
gap — so a structure can be routine for one and unseen for the other. On a sample of
400 Alexandria structures this model calls 56% of them out of its own distribution
and says so; on C2DB the figure is 15%.

**Known failure, now caught.** Graphene: predicted 3.18 against the database's 4.25.
The whole dataset holds one elemental-carbon structure and it sits in the test split,
so the model never saw carbon.
Its spread is 0.32 against 0.03 for the TMDs and its latent distance is 0.382 against
a q90 of 0.314, so its own verdict reads out-of-domain and no band edges are offered.
Phosphorene and h-BN are rejected the same way.

**Does that verdict rank its own error?** Measured on the 1 107 C2DB rows in the
screening table that carry a reference work function, restricted to rows this model
never trained on: MAE 0.130 eV in the reliable tier, 0.233 in check, 0.445 in
out-of-domain. The same rows split by training role give 0.102 eV for rows it
trained on against 0.263 for held-out ones, which is why the browser tags them.

## Gap type — `weights/cgcnn_2d_typed.pt`

| | |
|---|---|
| **Task** | Binary: indirect gap (`band_gap_dir − band_gap_ind ≥ 0.1` eV), 21% positive |
| **Performance** | ROC-AUC 0.758, balanced accuracy 0.702, F1(indirect) 0.486, threshold 0.327, composition-disjoint split |
| **Checksum** | SHA-256 `005bcacaac57e54d8c3fbedd8d5659ab81483c28cf7ef3cbe6fa921979885a49` |

Accuracy is meaningless here (majority baseline 79.5%). Deriving the type from the
difference of two regressed gaps was tried first and collapsed to that baseline:
the difference of two predictions with MAE ≈ 0.25 eV is noise. A separate model is
used instead of a multi-task head because multi-tasking cost the regressor 0.05 eV.

**Retrained on the metastable data, and not shipped** (`scripts/classifier_experiment.py`).
2DMatPedia records no direct gap, so only the metastable Alexandria monolayers were
added (training 10 733 → 20 457). Paired against a same-session control, ROC-AUC
moved −0.001 [−0.028, +0.025] on the stable test and +0.072 [+0.052, +0.094] on
held-out metastable structures, against a seed-noise shift of +0.019 — the written
rule accepted it. It also calls monolayer MoS₂ (0.45 against its threshold 0.38),
MoSe₂ (0.52) and phosphorene (0.93) indirect; Alexandria's labels and experiment both
call them direct. The rule had a reference-material check for the metal gate and none
here, and this is the case it would have caught. The control retrained with seed 1
flips MoS₂ too (0.47 against 0.43), so near the boundary a single classifier's call on
one material is partly seed luck even when its ranking improves. The shipped
classifier is unchanged; an ensemble of seeds was the next measurement.

**Release 1.3: seed ensembles, and still not shipped** (`scripts/classifier_hull_experiment.py`,
reference check written into the rule this time). ROC-AUC on the stable test /
held-out metastable / held-out 0.2–0.5 eV/atom: shipped 0.758 / 0.634 / 0.560,
control (old recipe) 0.762 / 0.640 / 0.573, five seeds 0.782 / 0.728 / 0.615, five
seeds with Alexandria 0.2–0.5 added 0.798 / 0.738 / 0.684. Both five-seed arms call
MoS₂, MoSe₂ and phosphorene indirect (MoS₂ 0.53 against a threshold of 0.41) and fail
the rule; the control calls h-BN direct and fails it too. All five members move
together, so the seed was not the cause: Alexandria labels MoS₂ direct by 0.07 eV
against a boundary of 0.1, and the added metastable structures move the boundary
across it. The type call stays a hint.

## Intended use and limits

Fast screening of candidate 2D semiconductors before committing DFT time. Trust the
`reliable` verdict; treat `check` as a shortlist worth verifying; ignore the number
under `out-of-domain`.

- **PBE target, and three many-body numbers on top of it.** The model predicts the
  PBE gap, which nothing measures. `scripts/fit_exciton.py` fits three linear heads
  on the ensemble's own latent space — the 128-dimensional embedding each of the five
  members computes on the way to a gap, concatenated — and all three ship in the
  checkpoint (641 weights each):

  | Head | Fitted against | n | MAE, composition-disjoint | same target from the gap alone |
  |---|---|---|---|---|
  | Quasiparticle gap (G₀W₀) | C2DB G₀W₀ | 214 | **0.249 eV** | 0.416 eV |
  | Direct quasiparticle gap | C2DB G₀W₀ | 214 | **0.266 eV** | 0.493 eV |
  | Exciton binding energy | C2DB BSE | 214 | **0.133 eV** | 0.243 eV |
  | Optical gap = direct − exciton | — | 214 | **0.229 eV** | 0.386 eV |

  Refitted on the current encoder (1 284 held-out predictions over 158 compositions).
  Against the heads of the previous encoder all four slipped (0.226 → 0.249, direct
  0.242 → 0.266, optical 0.219 → 0.229, binding 0.129 → 0.133), on a different set of
  materials (214 in-domain against 196), so not a paired comparison. The gap model is
  chosen on the gap; the heads take what that encoder gives them.

  Validated on 8 folds × 6 shuffles, split by composition because C2DB holds several
  entries per composition. The head wins in every band of predicted gap.

  **Why a head and not a correction.** As a function of the band gap alone the
  optical error sat at 0.38 eV, and feeding it C2DB's own PBE gap instead of the
  model's prediction only moved it to 0.31 — so the residual was not the network. The
  exciton binding energy is not a function of the band gap; it depends on screening,
  which is a property of the structure, and the structure is what the encoder saw.
  Training a graph network on 214 materials would fail — measured in this project at
  696 — but a ridge head on an encoder trained on 45 115 does not.

  **Known weakness.** Ridge does not extrapolate. Refit with every entry of BN and
  the four TMDs removed, the head gives h-BN 5.12 eV against a measured 6.00, where a
  polynomial in the gap gives 6.06; there are few materials above 5 eV in the fit.
  Against measurement on those five the polynomial wins overall (0.285 against 0.330
  eV) while the head wins on the four TMDs (0.193 against 0.341): h-BN carries the
  difference. The polynomial corrections stay in the checkpoint as the fallback when
  the heads are absent; the optical one may only take a shape that stays above the raw
  gap from 0 to 12 eV, since on this encoder cross-validation preferred a cubic that
  fell below it past 7 eV.

  **Retracted:** earlier versions said the difference between two separately fitted
  corrections was the exciton binding energy, 0.55 eV on the TMDs. Both had been
  trained on those same four materials, so the agreement was circular. The heads
  settle it by predicting the binding energy instead of inferring it: 0.57 / 0.50 /
  0.51 / 0.48 eV on the four TMDs against BSE's 0.55 / 0.50 / 0.52 / 0.48, and the
  optical gap is the direct gap minus that number by construction.
- **No spin-orbit coupling.** The target is Alexandria's PBE gap without SOC, and SOC
  lowers a gap. On the 651 in-domain, gapped C2DB rows (PBE+SOC, none trained on), the
  model is higher by +0.27 eV on average for compounds containing an element with
  Z ≥ 52 (median +0.20, n = 389), +0.17 for tungsten compounds (median +0.23, n = 39)
  and +0.03 without heavy elements (n = 262). Control on held-out Alexandria, computed
  without SOC: −0.02 and −0.04, so the offset is the physics, not the model (measured
  on the previous ensemble). A learned
  SOC correction was planned, but C2DB's table interface exposes only the SOC-inclusive
  gap. The structures held by both Alexandria (no SOC) and C2DB (SOC) — about 700
  in `data/structure_twins.csv` — carry exactly that difference and are the training
  set for one; the light-element offset between them is +0.02 eV, so the code
  difference (VASP against GPAW) is small next to it.
- **Polymorphs: much better, still not solved.** `scripts/polymorph_sensitivity.py`
  compares the spread of predictions against the spread of the reference at three
  levels, from the least to the most dependent on geometry alone; 1.00 would mean
  the model reproduces the variation exactly. On C2DB, which no model here trained
  on, the shipped ensemble reaches 0.96 globally, 0.81 within one composition and
  0.81 between 1H-MX₂ and 1T-MX₂, with the sign of that difference right 92% of the
  time (53 pairs; tables above). The 1H/1T case is the extreme one — same composition, same
  coordination number, nearly the same bond lengths, median X–M–X angle 85.0°
  against 91.7° — and it matters because 1H-MoS₂ is a semiconductor and 1T-MoS₂ is
  metallic. The model still compresses that difference by a fifth.
- **Data-density bias.** Error is lowest on transition-metal and heavy-element
  chemistries where the data is dense, highest on light main-group compounds, and it
  does not grow with the size of the gap. Adding metastable and 2DMatPedia data
  narrowed this considerably (nitrogen 1.01 → 0.38 eV, boron 1.82 → 0.92 on held-out
  sets, then hydrogen 1.18 → 0.58 at 0.2–0.5 eV/atom, and nitrogen 0.32 → 0.28 there from the step to 1.0) without closing it; carbon
  remains the sparsest element in training.
- **Geometry.** Inputs must be relaxed monolayers with a vacuum gap. Vacuum thinner
  than the 8 Å cutoff is padded automatically; cells with no gap ≥ 5 Å are rejected
  as not 2D. Under biaxial strain the model is usable in tension and unreliable
  below −2% compression.
- **Calibration is population-scoped.** The 90% interval is scaled per cell of
  resemblance to the metastable training structures × spread quartile, fitted on
  held-out near-hull and metastable Alexandria 2D semiconductors. On the 6 625 rows of
  the screening table this ensemble never trained on — held-out near-hull and
  metastable Alexandria, plus C2DB and JARVIS dft_2d under their own functionals — the
  interval covers 87.9%: 90.4% on Alexandria, 82.5% on C2DB and 79.2% on JARVIS,
  because part of their error is the difference in method. Restricted to the rows
  where the tool shows an interval (`scripts/trust_signals.py sources`): C2DB
  compounds without an element of Z ≥ 52 are covered 91.3%, those with one 69.7% with
  the model +0.24 eV above the SOC-inclusive reference — the whole C2DB shortfall is
  spin-orbit coupling. JARVIS would need the interval ×3.4. Neither is widened: the
  interval stays an interval on the PBE gap. The typical errors the tool
  quotes per tier (0.10 / 0.17 / 0.35 eV) are measured on the near-hull test split and
  are optimistic elsewhere. Re-calibrate before trusting absolute intervals on a
  different population.
- **The verdict ranks error against PBE and cannot be checked against JARVIS.** On
  the unseen rows the tiers keep their order on Alexandria (0.128 / 0.202 / 0.339 eV)
  and C2DB (0.188 / 0.212 / 0.355). On JARVIS dft_2d the reliable tier errs more than
  check (0.398 against 0.333), and did under every earlier ensemble as well (0.448 /
  0.406 on rows with no Alexandria twin two releases ago), hidden until the tiers were
  read per source. Its labels use OptB88vdW: on the 149 reliable JARVIS rows whose
  structure is also in Alexandria (measured on the previous ensemble), the prediction matched Alexandria's PBE label to 0.066 eV while the
  JARVIS label differs from that PBE label by 0.331. The error measured there is
  mostly the functional, not the model.
- **Through a language model.** `nanomat/mcp_server.py` serves the tool over MCP,
  verdict first, and it was tested on ten models over three runs
  ([README](README.md#does-a-language-model-pass-the-verdict-on)). The verdict reaches
  most answers. Under a neutral system prompt two of seven open models still gave
  graphene's out-of-domain number when asked for "just the number", and several models
  read a tier's typical error, which is measured against DFT, as agreement with
  experiment. The guided prompt in `scripts/llm_probe.py` fixed the first and not the
  second. Check the verdict in the tool's own output before relying on a number an
  assistant quotes.
- **Licence.** MIT for code and weights. Data: Alexandria (CC-BY 4.0), C2DB,
  JARVIS-DFT and 2DMatPedia (Zhou et al., *Sci. Data* **6**, 86 (2019)) through
  JARVIS-Tools.
