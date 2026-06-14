"""
Baseline: предсказание band gap по составу (composition-only)
=============================================================

Узкая вертикаль: 2D-полупроводники (наноэлектроника — канальные материалы, контакты).

Что делает скрипт:
  1. Тянет данные из готового API (JARVIS-DFT 2D, без ключа) ИЛИ из встроенного
     датасета matminer (тоже без ключа). Никакого собственного парсера не нужно.
  2. Считает дескрипторы состава (Magpie + Stoichiometry + ValenceOrbital).
  3. Обучает два baseline: RandomForest и XGBoost.
  4. Честные метрики: MAE / RMSE / R2 на отложенном тесте + 5-fold кросс-валидация.
  5. Простая оценка неопределённости (разброс по деревьям RandomForest).
  6. Parity-plot (предсказание vs истина) -> PNG.

Запуск:
    pip install -r requirements.txt
    python baseline_2d_bandgap.py --source jarvis      # 2D-материалы (по теме)
    python baseline_2d_bandgap.py --source matminer    # запасной вариант, если jarvis недоступен

GPU не нужен. На Mac M4 всё считается на CPU за минуты.
"""

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- ML ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- материаловедение ---
from pymatgen.core import Composition
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import (
    ElementProperty,
    Stoichiometry,
    ValenceOrbital,
)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ---------------------------------------------------------------------------
# 1. ЗАГРУЗКА ДАННЫХ  (через готовые API — НЕ свой парсер)
# ---------------------------------------------------------------------------
def load_jarvis_2d() -> pd.DataFrame:
    """2D-материалы из JARVIS-DFT. Скачивается с figshare без API-ключа."""
    from jarvis.db.figshare import data as jarvis_data

    raw = jarvis_data("dft_2d")  # список словарей, ~1k 2D-материалов
    df = pd.DataFrame(raw)

    # band gap из OptB88vdW; часть значений приходит строкой 'na'
    df = df[["formula", "optb88vdw_bandgap"]].rename(
        columns={"optb88vdw_bandgap": "band_gap"}
    )
    df["band_gap"] = pd.to_numeric(df["band_gap"], errors="coerce")
    df = df.dropna(subset=["formula", "band_gap"]).reset_index(drop=True)
    return df


def load_c2db() -> pd.DataFrame:
    """2D-материалы из C2DB (через JARVIS API). Band gap — функционал PBE.
    ~3500 структур, ~1115 полупроводников. НЕ мешать с dft_2d (там OptB88vdW)."""
    from jarvis.db.figshare import data as jarvis_data
    from jarvis.core.atoms import Atoms

    raw = jarvis_data("c2db")
    rows = []
    for e in raw:
        gap = e.get("gap")
        formula = e.get("formula")
        if formula is None and "atoms" in e:
            try:
                formula = Atoms.from_dict(e["atoms"]).composition.reduced_formula
            except Exception:
                formula = None
        rows.append({"formula": formula, "band_gap": gap})
    df = pd.DataFrame(rows)
    df["band_gap"] = pd.to_numeric(df["band_gap"], errors="coerce")
    df = df.dropna(subset=["formula", "band_gap"]).reset_index(drop=True)
    return df


def load_alex_2d(ehull_max: float = 0.1) -> pd.DataFrame:
    """Alexandria 2D (через JARVIS API). Таргет band_gap_ind (непрямая щель, PBE).
    137k структур; фильтруем по стабильности e_above_hull<=ehull_max (по умолч. 0.1
    эВ/атом → ~13k стабильных). Тоже PBE → согласуется с c2db."""
    from jarvis.db.figshare import data as jarvis_data

    raw = jarvis_data("alex_pbe_2d_all")
    rows = []
    for e in raw:
        ehull = pd.to_numeric(e.get("e_above_hull"), errors="coerce")
        if pd.isna(ehull) or ehull > ehull_max:
            continue
        rows.append({"formula": e.get("formula"), "band_gap": e.get("band_gap_ind")})
    df = pd.DataFrame(rows)
    df["band_gap"] = pd.to_numeric(df["band_gap"], errors="coerce")
    df = df.dropna(subset=["formula", "band_gap"]).reset_index(drop=True)
    return df


def load_matminer_gap() -> pd.DataFrame:
    """Запасной источник: экспериментальные band gap из matminer (без ключа)."""
    from matminer.datasets import load_dataset

    df = load_dataset("matbench_expt_gap")  # колонки: 'composition', 'gap expt'
    df = df.rename(columns={"composition": "formula", "gap expt": "band_gap"})
    df = df.dropna(subset=["formula", "band_gap"]).reset_index(drop=True)
    # можно оставить только полупроводники (отсечь металлы с нулём), по желанию:
    # df = df[df["band_gap"] > 0].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. ФИЧИ ИЗ СОСТАВА
# ---------------------------------------------------------------------------
def build_featurizer() -> MultipleFeaturizer:
    return MultipleFeaturizer(
        [
            ElementProperty.from_preset("magpie"),  # ~130 признаков
            Stoichiometry(),
            ValenceOrbital(props=["frac"]),
        ]
    )


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["composition"] = df["formula"].apply(Composition)
    feat = build_featurizer()
    df = feat.featurize_dataframe(
        df, "composition", ignore_errors=True, pbar=True
    )
    feature_cols = feat.feature_labels()
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df, feature_cols


# ---------------------------------------------------------------------------
# 3. ОБУЧЕНИЕ + ЧЕСТНЫЕ МЕТРИКИ
# ---------------------------------------------------------------------------
def report(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  {name:14s}  MAE={mae:.3f} eV   RMSE={rmse:.3f} eV   R2={r2:.3f}")
    return mae, rmse, r2


def train_eval(X, y, feature_cols):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nРазмер: train={len(X_train)}  test={len(X_test)}  фичей={X.shape[1]}")

    # --- RandomForest (даёт нам бесплатную оценку неопределённости) ---
    rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    print("\nМетрики на отложенном тесте:")
    report("RandomForest", y_test, rf_pred)

    # --- XGBoost ---
    if HAS_XGB:
        xgb = XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",   # быстро на CPU
            n_jobs=-1,
            random_state=42,
        )
        xgb.fit(X_train, y_train)
        report("XGBoost", y_test, xgb.predict(X_test))

    # --- 5-fold кросс-валидация (надёжнее одного сплита) ---
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mae = -cross_val_score(
        rf, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    print(f"\n5-fold CV (RandomForest): MAE = {cv_mae.mean():.3f} ± {cv_mae.std():.3f} eV")

    # --- неопределённость: разброс предсказаний по деревьям ---
    per_tree = np.stack([t.predict(X_test) for t in rf.estimators_])
    unc = per_tree.std(axis=0)
    print(
        f"Неопределённость (std по деревьям): "
        f"медиана={np.median(unc):.3f} eV, макс={unc.max():.3f} eV"
    )
    print("  -> высокий std = модель не уверена, такие случаи стоит проверять экспериментом")

    # --- топ-10 важных признаков ---
    imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )
    print("\nТоп-10 важных признаков (физически интерпретируемо):")
    for name, val in imp.head(10).items():
        print(f"  {val:.3f}  {name}")

    return rf, X_test, y_test, rf_pred, unc


# ---------------------------------------------------------------------------
# 4. PARITY-PLOT
# ---------------------------------------------------------------------------
def parity_plot(y_true, y_pred, unc, path="parity_bandgap.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(y_true, y_pred, c=unc, cmap="viridis", s=18, alpha=0.7)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("Band gap (DFT/эксперимент), eV")
    ax.set_ylabel("Band gap (предсказание), eV")
    ax.set_title("Parity plot — baseline")
    plt.colorbar(sc, label="неопределённость, eV")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\nГрафик сохранён: {path}")


# ---------------------------------------------------------------------------
def describe_target(df: pd.DataFrame) -> None:
    """Краткая сводка по распределению band gap + доля металлов (gap≈0)."""
    y = df["band_gap"]
    n = len(y)
    n_metal = int((y <= 0.01).sum())  # металлы: щель практически ноль
    print("\nРаспределение таргета (band gap, eV):")
    print(f"  всего={n}   металлов (gap<=0.01)={n_metal} ({100*n_metal/n:.0f}%)   "
          f"полупроводников={n - n_metal}")
    print(f"  min={y.min():.3f}  median={y.median():.3f}  mean={y.mean():.3f}  "
          f"max={y.max():.3f}  std={y.std():.3f}")
    # грубая гистограмма по бинам
    bins = [0, 0.01, 0.5, 1, 2, 3, 4, 100]
    labels = ["=0", "0–0.5", "0.5–1", "1–2", "2–3", "3–4", ">4"]
    counts = pd.cut(y, bins=bins, labels=labels, include_lowest=True, right=False).value_counts(sort=False)
    print("  по интервалам:")
    for lab, c in counts.items():
        print(f"    {lab:>6s} eV : {c}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["jarvis", "c2db", "alex_2d", "matminer"], default="jarvis")
    ap.add_argument("--drop-metals", action="store_true",
                    help="отсечь металлы (band gap <= 0.01 eV) — регрессия только по полупроводникам")
    ap.add_argument("--ehull-max", type=float, default=0.1,
                    help="только для alex_2d: фильтр стабильности e_above_hull<=X эВ/атом (по умолч. 0.1)")
    args = ap.parse_args()

    print("Загрузка данных...")
    loaders = {"jarvis": load_jarvis_2d, "c2db": load_c2db,
               "alex_2d": lambda: load_alex_2d(args.ehull_max), "matminer": load_matminer_gap}
    df = loaders[args.source]()
    print(f"Получено {len(df)} материалов с band gap.")

    describe_target(df)

    if args.drop_metals:
        before = len(df)
        df = df[df["band_gap"] > 0.01].reset_index(drop=True)
        print(f"\n--drop-metals: оставлено {len(df)} полупроводников (убрано {before - len(df)} металлов).")

    print("\nСчитаю дескрипторы состава...")
    df, feature_cols = featurize(df)

    X = df[feature_cols].values
    y = df["band_gap"].values

    rf, X_test, y_test, rf_pred, unc = train_eval(X, y, feature_cols)
    parity_plot(y_test, rf_pred, unc)

    print("\nГотово. Это твоя точка отсчёта — дальше сравнивай с ней любые улучшения.")


if __name__ == "__main__":
    main()
