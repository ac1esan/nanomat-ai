# Model card — NanoMatAI

Three models ship together: a band-gap ensemble, a metal/semiconductor gate that
runs before it, and a direct/indirect gap classifier. All are CGCNN
(PyTorch Geometric) over the same graph: periodic neighbours within 8 Å, edge
feature = distance, atomic number embedded at the nodes.

## Band-gap ensemble — `weights/cgcnn_2d_ensemble.pt`

| | |
|---|---|
| **Task** | Band gap (eV, PBE level) of an isolated 2D layer from its structure |
| **Architecture** | Z-embedding 128, 4 × `CGConv`, Gaussian RBF edge features (40 centres on 0–8 Å), mean pooling, MLP head, dropout 0.2 |
| **Training data** | Alexandria 2D (`alex_pbe_2d_all` via JARVIS-Tools), `e_above_hull ≤ 0.1` eV/atom, `band_gap_ind > 0.01` eV → 13 349 stable 2D semiconductors |
| **Split** | Composition-disjoint (`GroupShuffleSplit` on reduced formula): 10 733 train / 1 290 val / 1 326 test. The dataset holds only 8 388 unique compositions, so a random split leaks near-duplicates |
| **Ensemble** | 5 members, seeds 0–4, 200 epochs, batch 64, Adam 1e-3 with plateau decay and early stopping |
| **Test performance** | **MAE 0.261 eV** (95% bootstrap CI 0.242–0.283), RMSE 0.454, R² 0.889. Members: 0.281 / 0.305 / 0.274 / 0.277 / 0.319 |
| **Control** | Identical code and data on a random split: MAE 0.252. The honest split costs ≈ 0.01 eV |
| **Composition baseline** | Magpie + RandomForest/XGBoost on the same data: CV MAE 0.360 eV |
| **Checksum** | SHA-256 `5697273713e2d9ecc7ac8b51b1bc11c90992f0b81779ddd7f39b6e6c68ac5f67` |

The checkpoint also carries its own `calibration` block and 10 733 reference
embeddings, so it can judge its own output without any external file.

### Uncertainty, and why the raw spread is not enough

| Signal | Spearman vs abs. error |
|---|---|
| MC-dropout on a single model | 0.17 |
| Ensemble spread (5 seeds) | 0.45 |
| Distance to training set in latent space | 0.47 |
| The two combined | 0.50 |

Coverage of the raw spread is poor: ±1σ contains 37% of cases, not 68%. A scale
factor fitted on validation (×5.09 for 90%, ×2.37 for 68%) gives 91.8% coverage on
the held-out test split. Conditional coverage by uncertainty quartile is
84 / 92 / 94 / 96%, so the most confident quartile stays slightly optimistic.

**The failure that motivated the second signal.** On phosphorene the ensemble
predicted 0.82 eV against an experimental 2.0 eV, with a spread of 0.035 eV and a
verdict of "reliable". Alexandria contains exactly one elemental-phosphorus 2D
structure; every member learned from it and they agreed. Ensemble spread measures
disagreement between initialisations, not ignorance of a chemistry. Latent distance
places phosphorene in the 98th percentile and now overrides the verdict.

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
| reliable | spread ≤ 0.089 eV and latent distance ≤ 0.226 | 0.14 eV |
| check | spread ≤ 0.142 eV, or reliable spread with latent distance > 0.226 | 0.25 eV |
| out-of-domain | spread > 0.142 eV, or latent distance > 0.288, or the metal gate fires | 0.46 eV |

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

## Gap type — `weights/cgcnn_2d_typed.pt`

| | |
|---|---|
| **Task** | Binary: indirect gap (`band_gap_dir − band_gap_ind ≥ 0.1` eV), 21% positive |
| **Performance** | ROC-AUC 0.758, balanced accuracy 0.702, F1(indirect) 0.486, threshold 0.327, composition-disjoint split |
| **Checksum** | SHA-256 `005bcacaac57e54d8c3fbedd8d5659ab81483c28cf7ef3cbe6fa921979885a49` |

Accuracy is meaningless here (majority baseline 79.5%). Deriving the type from the
difference of two regressed gaps was tried first and collapsed to that baseline:
the difference of two predictions with MAE ≈ 0.26 eV is noise. A separate model is
used instead of a multi-task head because multi-tasking cost the regressor 0.05 eV.

## Intended use and limits

Fast screening of candidate 2D semiconductors before committing DFT time. Trust the
`reliable` verdict; treat `check` as a shortlist worth verifying; ignore the number
under `out-of-domain`.

- **PBE target.** Real gaps are larger. The correction `exp ≈ 1.39·gap − 0.44` is
  fitted on five reference monolayers, four of them TMDs, using only points the
  tool calls usable (MAE 0.42 → 0.16 eV). It is a rough estimate.
- **Data-density bias.** Error is lowest on transition-metal and heavy-element
  chemistries where Alexandria is dense, highest on light main-group compounds. It
  does not grow with the size of the gap.
- **Geometry.** Inputs must be relaxed monolayers with a vacuum gap. Vacuum thinner
  than the 8 Å cutoff is padded automatically; cells with no gap ≥ 5 Å are rejected
  as not 2D. Under biaxial strain the model is usable in tension and unreliable
  below −2% compression.
- **Calibration is population-scoped.** Fitted on stable (`e_above_hull ≤ 0.1`)
  Alexandria 2D semiconductors. Verified at scale on 16 349 unseen structures
  spanning metastable entries and two other functionals: tier ordering survives
  (MAE 0.327 / 0.431 / 0.569 eV for reliable / check / out-of-domain), but 90%
  intervals cover 78% and per-tier errors are optimistic. MAE is 0.261 eV on stable
  structures and 0.509 eV on metastable ones. Re-calibrate before trusting absolute
  intervals on a different population.
- **Licence.** MIT for code and weights. Data: Alexandria (CC-BY 4.0), C2DB and
  JARVIS-DFT through JARVIS-Tools.
