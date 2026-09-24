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
| **Training data** | 22 103 monolayers, all semiconductors (`gap > 0.01` eV), all PBE: the 10 733 training structures of the stable set (Alexandria 2D, `e_above_hull ≤ 0.1` eV/atom), **plus 9 724 metastable Alexandria monolayers** (0.1 < `e_above_hull` ≤ 0.2) **plus 1 646 non-magnetic 2DMatPedia monolayers** that Alexandria does not contain. No added formula occurs in the validation or test split |
| **Split** | Composition-disjoint (`GroupShuffleSplit` on reduced formula). Validation (1 290) and test (1 326) are the stable-set ones and have not changed since the first model, so every generation is scored on the same 1 326 structures |
| **Ensemble** | 5 members, seeds 0–4, up to 200 epochs, batch 64, Adam 1e-3 with plateau decay, early stopping after 40 epochs without improvement |
| **Test performance** | **MAE 0.225 eV** (95% bootstrap CI 0.207–0.243), RMSE 0.405, R² 0.912. Members: 0.266 / 0.243 / 0.251 / 0.253 / 0.267 |
| **Why the extra data** | Against a control trained on the stable set alone — same split, same seeds, same GPU session — MAE goes 0.249 → **0.225** on the stable test (paired −0.025 eV, bootstrap 95% −0.039…−0.011), 0.500 → **0.311** on held-out metastable Alexandria, 0.770 → **0.437** on held-out 2DMatPedia. Details below |
| **Why angles** | An edge carries a distance and nothing else, so two polymorphs of one composition are nearly the same graph. Against its own control, adding a per-atom bond-angle histogram took MAE from 0.271 to 0.246 eV (paired +0.026, bootstrap 95% +0.015…+0.037, Wilcoxon p = 4·10⁻⁵) and doubled how far the model separates 1H-MX₂ from 1T-MX₂ |
| **Composition baseline** | **0.430 eV** on the identical test set (`scripts/composition_baseline.py`), so structure wins by 48%. The 0.360 eV quoted in early versions came from five-fold cross-validation, a different protocol that leaks near-duplicate compositions |
| **Checksum** | SHA-256 `3f167dfb1a5eac369f469be530c0ce103b90a935bfad4c7a1c816a1c624f7f21` |

The checkpoint also carries its own `calibration` block, both gap corrections, the
latent heads and 22 103 reference embeddings, so it can judge and correct its own
output without any external file. Its checksum therefore changes whenever the
calibration is refitted, not only when the weights are retrained.

### What the stability filter was costing

The first models trained only on `e_above_hull ≤ 0.1`, because a 26k set at 0.2
scored worse than the 13k one (0.262 against 0.215). Each of those numbers was
measured on its own test split, and the 26k split was half metastable, which is
harder for any model — the comparison changed the population it measured on, not
only the data it trained on. `scripts/stability_experiment.py` asks the question
properly: four arms, one split, one environment, only the training part differs.

| test set | stable only | + 2DMatPedia | + metastable | **+ both (shipped)** |
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

Where the model was weakest the change is largest (control → shipped, MAE in eV):
carbon on 2DMatPedia 1.59 → 0.67 (n = 17, read it qualitatively), boron 1.82 → 0.92,
nitrogen on metastable Alexandria 1.01 → 0.38, hydrogen 1.25 → 0.83.

And the polymorphs gained more than they did from the angles, on C2DB, which none of
these models trained on:

| | no angles | angles | angles + metastable + 2DMatPedia |
|---|---|---|---|
| 1H-MX₂ against 1T-MX₂, compression | 0.18 | 0.37 | **0.65** |
| sign of the 1H − 1T difference | — | 79% | **92%** |
| within one composition | 0.37 | 0.49 | **0.69** |

In hindsight the reason is plain: a metastable structure is, nearly by definition,
another polymorph of a composition that has a stable one. The filter removed exactly
the "same formula, different structure" contrast a model needs in order to learn
what separates polymorphs. The angles made that contrast visible; the data supplied
it.

### 2DMatPedia, checked structure by structure before use

Its labels were compared with Alexandria's by **structure**, not by formula
(`scripts/audit_2dmatpedia.py`): every layer put in one frame (vacuum axis to the
layer normal, equal vacuum, primitive cell), then pymatgen's `StructureMatcher` with
tolerances tightened for slabs — the defaults normalise by a volume that is mostly
vacuum and accepted as MoS₂ a structure 1 eV/atom higher in energy.

- 1 178 of 6 351 entries have a structural twin in Alexandria. Their PBE energies
  agree to a median 0.0015 eV/atom (91% within 0.01): one computational setup,
  including +U.
- On the same structure the gaps agree to MAE 0.089 eV (median 0.055, r = 0.995),
  0.085 on non-magnetic ones. Paired by formula instead — the comparison made before
  — the same entries disagree by 0.304: most of the apparent disagreement between
  the databases was polymorphs, not labels.
- Where they disagree by more than 0.3 eV (about 4% of pairs), C2DB and JARVIS
  dft_2d as arbiters side with neither database overall. Alexandria is wrong on
  Dirac-like honeycombs — graphene 1.23 eV, silicene 0.86, GaAs 1.11, where C2DB has
  zero — and **that graphene entry sits in this model's test split**. 2DMatPedia is
  wrong on the ZrNCl family by about 1 eV and, more often, lands in a different
  magnetic state. Hence the rule: non-magnetic 2DMatPedia entries only.

Before any of its data was used, the ensemble of the time was run on 2DMatPedia as
an external validation — a database it had never seen: MAE 0.752 eV overall, and the
verdict still ranked the error, 0.306 / 0.580 / 0.829 eV for reliable / check /
out-of-domain, with 78% of that population flagged out-of-domain by the model itself.

### Uncertainty, and why the raw spread is not enough

| Signal | Spearman vs abs. error |
|---|---|
| MC-dropout on a single model | 0.17 |
| Ensemble spread (5 seeds) | 0.45 |
| Distance to training set in latent space | 0.47 |
| The two combined | 0.50 |

Coverage of the raw spread is poor: ±1σ contains 42% of cases, not 68%. A scale
factor fitted on validation (×4.19 for 90%, ×2.13 for 68%) gives 90.2% coverage on
the held-out test split. Conditional coverage by uncertainty quartile is
82 / 92 / 92 / 94%, so the most confident quartile stays slightly optimistic.

The tiers keep their order on data the model never trained on: 0.205 / 0.235 /
0.396 eV on held-out metastable Alexandria and 0.140 / 0.238 / 0.489 eV on held-out
2DMatPedia.

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
signal stands on its statistics above, not on this anecdote. The shipped ensemble
predicts 0.96 eV with a spread of 0.096, and calls it `check`.

An intermediate attempt that did **not** work, recorded so it is not repeated:
counting how many training structures share the query's chemical system correlates
with error in the wrong direction (MAE 0.236 for unseen systems versus 0.378 for
systems seen ten or more times), because chemically rich systems are both better
represented and intrinsically harder.

### Verdict tiers

Thresholds are validation quartiles; the typical error of each tier is measured on
the held-out test split.

| Verdict | Condition | Typical error |
|---|---|---|
| reliable | spread ≤ 0.092 eV and latent distance ≤ 0.235 | 0.12 eV |
| check | spread ≤ 0.144 eV, or reliable spread with latent distance > 0.235 | 0.21 eV |
| out-of-domain | spread > 0.144 eV, or latent distance > 0.299, or the metal gate fires | 0.41 eV |

Out-of-domain results deliberately suppress the estimated experimental gap and the
calibrated interval: that interval assumes the spread is meaningful, which is
precisely what fails there.

## Metal gate — `weights/cgcnn_2d_metal.pt`

| | |
|---|---|
| **Task** | Binary: is this 2D structure a metal/semimetal (gap ≤ 0.05 eV)? Runs before the regressor |
| **Training data** | Alexandria 2D + C2DB + JARVIS dft_2d, metals kept: 24 314 structures, 39.8% metals |
| **Split / performance** | Composition-disjoint. ROC-AUC 0.944, balanced accuracy 0.876, precision on metals 0.866, n_test 2 440 |
| **Threshold** | 0.648, fitted on validation — `pos_weight` on the rare class shifts probabilities away from 0.5 |
| **Checksum** | SHA-256 `1d4e77fda6fbf19390529c37d71e234d41002370c1e39fe633c586d490867b38` |

Mixing three DFT functionals is deliberate and limited to this model: the
metal/semiconductor distinction is far more robust across functionals than the gap
value. The same mixing would be wrong for the regressor and is not done there.

An Alexandria-only version scored a higher ROC-AUC (0.952) and was useless where it
mattered — `p(metal) = 0.00` on graphene, because only two of its 19 691 training
structures were carbon-only and both were labelled semiconductors. The merged model
returns 1.00 on graphene with no false positives on MoS₂, MoSe₂, WS₂, WSe₂ or h-BN.

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
0.15 eV (MoS₂ 6.10 vs 6.1, MoSe₂ 5.49 vs 5.5, WS₂ 5.92 vs 6.0, WSe₂ 5.05 vs 5.2)
and the affinities within 0.4 eV. The mid-gap assumption is a convention, not a law.

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
  | Quasiparticle gap (G₀W₀) | C2DB G₀W₀ | 184 | **0.255 eV** | 0.404 eV |
  | Direct quasiparticle gap | C2DB G₀W₀ | 184 | **0.276 eV** | 0.472 eV |
  | Exciton binding energy | C2DB BSE | 184 | **0.123 eV** | 0.240 eV |
  | Optical gap = direct − exciton | — | 184 | **0.218 eV** | 0.360 eV |

  Refitted on the current encoder. Against the heads of the previous one the binding
  energy improved (0.133 → 0.123) and the gaps got slightly worse (optical 0.198 →
  0.218): the encoder now serves a much wider population, and three heads fitted on
  184 C2DB materials paid for that. The band gap itself improved by 10–43%, so the
  trade was taken, but it is a trade.

  Validated on 8 folds × 6 shuffles, split by composition because C2DB holds several
  entries per composition. The head wins in every band of predicted gap.

  **Why a head and not a correction.** As a function of the band gap alone the
  optical error sat at 0.38 eV, and feeding it C2DB's own PBE gap instead of the
  model's prediction only moved it to 0.31 — so the residual was not the network. The
  exciton binding energy is not a function of the band gap; it depends on screening,
  which is a property of the structure, and the structure is what the encoder saw.
  Training a graph network on 184 materials would fail — measured in this project at
  696 — but a ridge head on an encoder trained on 22 103 does not.

  **Known weakness.** Ridge does not extrapolate. Refit with every entry of BN and
  the four TMDs removed, the head gives h-BN 5.43 eV against a measured 6.00, where a
  polynomial in the gap gives 6.01; there are few materials above 5 eV in the fit.
  Against measurement on those five the polynomial wins overall (0.248 against 0.278
  eV) while the head wins on the four TMDs (0.205 against 0.309): h-BN carries the
  difference. The polynomial corrections stay in the checkpoint as the fallback when
  the heads are absent.

  **Retracted:** earlier versions said the difference between two separately fitted
  corrections was the exciton binding energy, 0.55 eV on the TMDs. Both had been
  trained on those same four materials, so the agreement was circular. The heads
  settle it by predicting the binding energy instead of inferring it: 0.56 / 0.50 /
  0.53 / 0.52 eV on the four TMDs against BSE's 0.55 / 0.50 / 0.52 / 0.48, and the
  optical gap is the direct gap minus that number by construction.
- **Polymorphs: much better, still not solved.** `scripts/polymorph_sensitivity.py`
  compares the spread of predictions against the spread of the reference at three
  levels, from the least to the most dependent on geometry alone; 1.00 would mean
  the model reproduces the variation exactly. On C2DB, which no model here trained
  on, the shipped ensemble reaches 0.96 globally, 0.69 within one composition and
  0.65 between 1H-MX₂ and 1T-MX₂, with the sign of that difference right 92% of the
  time (tables above). The 1H/1T case is the extreme one — same composition, same
  coordination number, nearly the same bond lengths, median X–M–X angle 85.0°
  against 91.7° — and it matters because 1H-MoS₂ is a semiconductor and 1T-MoS₂ is
  metallic. The model still compresses that difference by a third.
- **Data-density bias.** Error is lowest on transition-metal and heavy-element
  chemistries where the data is dense, highest on light main-group compounds, and it
  does not grow with the size of the gap. Adding metastable and 2DMatPedia data
  narrowed this considerably (nitrogen 1.01 → 0.38 eV, boron 1.82 → 0.92 on held-out
  sets) without closing it; carbon remains the sparsest element in training.
- **Geometry.** Inputs must be relaxed monolayers with a vacuum gap. Vacuum thinner
  than the 8 Å cutoff is padded automatically; cells with no gap ≥ 5 Å are rejected
  as not 2D. Under biaxial strain the model is usable in tension and unreliable
  below −2% compression.
- **Calibration is population-scoped.** Fitted on stable (`e_above_hull ≤ 0.1`)
  Alexandria 2D semiconductors. On the 6 625 rows of the screening table this
  ensemble never trained on — held-out stable and metastable Alexandria, plus C2DB
  and JARVIS dft_2d under their own functionals — tier ordering survives (MAE 0.199 /
  0.236 / 0.416 eV for reliable / check / out-of-domain), but 90% intervals cover 83%
  and per-tier errors are optimistic. The previous ensemble covered 78% on its own
  unseen rows; the two populations differ, since the metastable rows it had not seen
  are now training data, so the two figures are not a like-for-like comparison.
  Re-calibrate before trusting absolute intervals on a different population.
- **Licence.** MIT for code and weights. Data: Alexandria (CC-BY 4.0), C2DB,
  JARVIS-DFT and 2DMatPedia (Zhou et al., *Sci. Data* **6**, 86 (2019)) through
  JARVIS-Tools.
