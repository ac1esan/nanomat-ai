"""
Screening-инструмент: структура 2D-материала -> предсказанный band gap.
=======================================================================

Берёт обученную CGCNN (дотюн на 13k стабильных 2D, Alexandria PBE, MAE~0.22 эВ),
читает кристаллическую структуру (CIF / POSCAR / .vasp) и выдаёт:
  - предсказанный band gap (эВ),
  - неопределённость (MC-dropout, std по K стохастическим проходам),
  - флаг доверия (по медиане неопределённости на тесте ~0.09 эВ).

Два режима:
  # интерактивная морда (перетащил файл -> увидел gap)
  python screen_bandgap.py --app

  # батч-скрининг: папка/файл структур -> CSV, отсортированный по щели
  python screen_bandgap.py --in structures/ --out results.csv
  python screen_bandgap.py --in MoS2.cif

Веса модели: cgcnn_2d_bandgap.pt (скачать из Kaggle Output). Лежат рядом со скриптом.

ВАЖНО: архитектура и параметры графа (CUTOFF, N_RBF) ДОЛЖНЫ совпадать с обучением.
"""

import argparse
import glob
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import CGConv, global_mean_pool
from pymatgen.core import Structure

_HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_ENS = os.path.join(_HERE, "cgcnn_2d_ensemble.pt")   # ансамбль gap (приоритет)
CKPT_ONE = os.path.join(_HERE, "cgcnn_2d_bandgap.pt")    # одиночная модель gap
CKPT_TYPE = os.path.join(_HERE, "cgcnn_2d_typed.pt")     # классификатор типа щели (опц.)

# параметры графа — переопределяются из чекпойнта, дефолты под обучение
CUTOFF, N_RBF = 8.0, 40
MED_UNC = 0.092  # медиана неопределённости на тесте — порог «доверять/проверить»


# --- граф из структуры (идентично обучению) ---
def to_graph(st: Structure):
    c, n, img, d = st.get_neighbor_list(r=CUTOFF)  # соседи с учётом периодики
    if len(c) == 0:
        return None
    z = torch.tensor([s.specie.Z for s in st], dtype=torch.long)
    ei = torch.tensor(np.vstack([c, n]), dtype=torch.long)
    ew = torch.tensor(d, dtype=torch.float)
    return Data(z=z, edge_index=ei, edge_weight=ew, num_nodes=len(z))


# --- модель (идентична Kaggle CGCNN2) ---
class CGCNN(nn.Module):
    def __init__(self, h=128, n_conv=4, p=0.2):
        super().__init__()
        self.emb = nn.Embedding(100, h)
        self.convs = nn.ModuleList(
            [CGConv(h, dim=N_RBF, batch_norm=True) for _ in range(n_conv)]
        )
        self.head = nn.Sequential(nn.Linear(h, h), nn.Softplus(), nn.Linear(h, 1))
        self.drop = nn.Dropout(p)
        self.register_buffer("centers", torch.linspace(0, CUTOFF, N_RBF))

    def forward(self, data):
        ea = torch.exp(-0.5 * (data.edge_weight.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2)
        x = self.emb(data.z)
        for c in self.convs:
            x = c(x, data.edge_index, ea)
        x = self.drop(global_mean_pool(x, data.batch))
        return self.head(x).squeeze(-1)


# --- классификатор типа щели (прямая/непрямая), идентичен Kaggle CGCNNcls ---
class CGCNNcls(nn.Module):
    def __init__(self, h=128, n_conv=4):
        super().__init__()
        self.emb = nn.Embedding(100, h)
        self.convs = nn.ModuleList([CGConv(h, dim=N_RBF, batch_norm=True) for _ in range(n_conv)])
        self.body = nn.Sequential(nn.Linear(h, h), nn.Softplus())
        self.reg = nn.Linear(h, 1)
        self.cls = nn.Linear(h, 1)
        self.register_buffer("centers", torch.linspace(0, CUTOFF, N_RBF))

    def forward(self, data):
        ea = torch.exp(-0.5 * (data.edge_weight.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2)
        x = self.emb(data.z)
        for c in self.convs:
            x = c(x, data.edge_index, ea)
        x = self.body(global_mean_pool(x, data.batch))
        return self.reg(x).squeeze(-1), self.cls(x).squeeze(-1)


def load_type_model():
    """Классификатор типа щели, если веса есть. Иначе None."""
    if not os.path.exists(CKPT_TYPE):
        return None
    ck = torch.load(CKPT_TYPE, map_location="cpu")
    m = CGCNNcls(); m.load_state_dict(ck["state_dict"]); m.eval()
    print("Загружен классификатор типа щели (прямая/непрямая).")
    return m


def load_models():
    """Грузит ансамбль (если есть) либо одиночную модель. Возвращает
    (models, mean, std): список моделей — для ансамбля >1, иначе 1."""
    global CUTOFF, N_RBF
    if os.path.exists(CKPT_ENS):
        ck = torch.load(CKPT_ENS, map_location="cpu")
        CUTOFF, N_RBF = ck.get("cutoff", CUTOFF), ck.get("n_rbf", N_RBF)
        models = []
        for sd in ck["state_dicts"]:
            m = CGCNN(); m.load_state_dict(sd); m.eval(); models.append(m)
        print(f"Загружен ансамбль из {len(models)} моделей.")
        return models, float(ck["mean"]), float(ck["std"])
    if os.path.exists(CKPT_ONE):
        ck = torch.load(CKPT_ONE, map_location="cpu")
        CUTOFF, N_RBF = ck.get("cutoff", CUTOFF), ck.get("n_rbf", N_RBF)
        m = CGCNN(); m.load_state_dict(ck["state_dict"]); m.eval()
        print("Загружена одиночная модель (неопределённость через MC-dropout).")
        return [m], float(ck["mean"]), float(ck["std"])
    raise SystemExit(
        "Не найдены веса. Скачай из Kaggle один из файлов в папку проекта:\n"
        f"  {os.path.basename(CKPT_ENS)} (ансамбль, предпочтительно) или\n"
        f"  {os.path.basename(CKPT_ONE)} (одиночная модель)."
    )


def read_structure(path: str) -> Structure:
    try:
        return Structure.from_file(path)
    except Exception:
        return Structure.from_file(path, fmt="poscar")  # POSCAR без расширения


@torch.no_grad()
def predict(models, mean, std, st: Structure, mc=30):
    """Возвращает (gap, uncertainty) в эВ.
    Ансамбль (len>1): gap=среднее, unc=std между моделями.
    Одиночная: gap=точечное, unc=std по MC-dropout."""
    g = to_graph(st)
    if g is None:
        return None, None
    batch = Batch.from_data_list([g])

    if len(models) > 1:
        preds = [(m(batch) * std + mean).item() for m in models]
        return float(np.mean(preds)), float(np.std(preds))

    model = models[0]
    model.eval()
    gap = (model(batch) * std + mean).item()
    model.train()  # dropout ВКЛ
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm1d):
            mod.eval()
    preds = [(model(batch) * std + mean).item() for _ in range(mc)]
    model.eval()
    return gap, float(np.std(preds))


@torch.no_grad()
def predict_type(type_model, st: Structure):
    """Возвращает (label, p_непрямозонный) или (None, None)."""
    if type_model is None:
        return None, None
    g = to_graph(st)
    if g is None:
        return None, None
    _, logit = type_model(Batch.from_data_list([g]))
    p_ind = torch.sigmoid(logit).item()
    return ("непрямозонный" if p_ind >= 0.5 else "прямозонный"), p_ind


# PBE->эксперимент: грубая линейная поправка из validate_experiment.py (exp≈1.26·gap−0.12)
A_CORR, B_CORR = 1.26, -0.12


def corrected_gap(gap):
    return A_CORR * gap + B_CORR


def verdict(unc):
    if unc <= 1.5 * MED_UNC:
        return "надёжно"
    if unc <= 3.0 * MED_UNC:
        return "проверить (повышенная неопределённость)"
    return "вне области применимости (возможно металл / нетипичная структура)"


# ---------------------------------------------------------------------------
# Режим 1: батч-скрининг папки/файла -> CSV
# ---------------------------------------------------------------------------
def run_batch(in_path, out_path):
    import pandas as pd

    models, mean, std = load_models()
    type_model = load_type_model()

    if os.path.isdir(in_path):
        files = []
        for ext in ("*.cif", "*.vasp", "*.poscar", "POSCAR*"):
            files += glob.glob(os.path.join(in_path, "**", ext), recursive=True)
    else:
        files = [in_path]
    files = sorted(set(files))
    if not files:
        raise SystemExit(f"Структур не найдено в {in_path}")

    rows = []
    for i, f in enumerate(files, 1):
        try:
            st = read_structure(f)
            gap, unc = predict(models, mean, std, st)
            if gap is None:
                continue
            gap_type, p_ind = predict_type(type_model, st)
            row = {
                "file": os.path.basename(f),
                "formula": st.composition.reduced_formula,
                "band_gap_PBE_eV": round(gap, 3),
                "exp_gap_est_eV": round(corrected_gap(gap), 3) if gap > 0.1 else 0.0,
                "uncertainty_eV": round(unc, 3),
                "verdict": verdict(unc),
            }
            if gap_type is not None:
                row["gap_type"] = gap_type
                row["p_indirect"] = round(p_ind, 2)
            rows.append(row)
            extra = f"  [{gap_type}]" if gap_type else ""
            print(f"[{i}/{len(files)}] {st.composition.reduced_formula:14s} "
                  f"gap={gap:.3f} ± {unc:.3f} эВ{extra}")
        except Exception as e:
            print(f"[{i}/{len(files)}] ПРОПУСК {os.path.basename(f)}: {e}")

    df = pd.DataFrame(rows).sort_values("band_gap_PBE_eV").reset_index(drop=True)
    out_path = out_path or "results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nГотово: {len(df)} структур -> {out_path} (отсортировано по band gap)")


# ---------------------------------------------------------------------------
# Режим 2: Gradio-морда
# ---------------------------------------------------------------------------
def run_app():
    import gradio as gr

    models, mean, std = load_models()
    type_model = load_type_model()

    def infer(file):
        if file is None:
            return "Загрузи файл структуры (CIF / POSCAR / .vasp)."
        try:
            st = read_structure(file)
        except Exception as e:
            return f"Не удалось прочитать структуру: {e}"
        gap, unc = predict(models, mean, std, st)
        if gap is None:
            return "Не удалось построить граф (нет соседей в радиусе обрезки)."
        kind = "металл / полуметалл" if gap < 0.1 else "полупроводник / диэлектрик"
        gap_type, p_ind = predict_type(type_model, st)
        type_line = ""
        if gap_type is not None:
            conf = p_ind if gap_type == "непрямозонный" else 1 - p_ind
            type_line = f"Тип щели:         {gap_type}  (p={conf:.2f})\n"
        corr_line = (f"≈ эксперимент:    {corrected_gap(gap):.2f} эВ  (PBE-поправка)\n"
                     if gap > 0.1 else "")
        return (
            f"Состав:           {st.composition.reduced_formula}\n"
            f"Атомов в ячейке:  {len(st)}\n"
            f"─────────────────────────────\n"
            f"Band gap (PBE):   {gap:.3f} эВ   ({kind})\n"
            f"{corr_line}"
            f"{type_line}"
            f"Неопределённость: ± {unc:.3f} эВ\n"
            f"Оценка:           {verdict(unc)}\n"
            f"─────────────────────────────\n"
            f"Модель: CGCNN-ансамбль на 2D-полупроводниках (Alexandria PBE), "
            f"MAE gap ≈ 0.23 эВ, тип щели ROC-AUC ≈ 0.80. PBE занижает истинную щель "
            f"(поправка калибрована на 6 эталонах)."
        )

    demo = gr.Interface(
        fn=infer,
        inputs=gr.File(label="Структура 2D-материала (CIF / POSCAR / .vasp)", type="filepath"),
        outputs=gr.Textbox(label="Предсказание band gap", lines=10),
        title="NanoMatAI — предсказание band gap 2D-материалов",
        description="Загрузи кристаллическую структуру → получи band gap и неопределённость. "
                    "Инференс на CPU, без DFT.",
    )
    demo.launch()


def main():
    ap = argparse.ArgumentParser(description="Screening band gap 2D-материалов по структуре")
    ap.add_argument("--app", action="store_true", help="запустить Gradio-морду")
    ap.add_argument("--in", dest="inp", help="файл или папка со структурами (батч-режим)")
    ap.add_argument("--out", help="куда писать CSV (по умолч. results.csv)")
    args = ap.parse_args()

    if args.app:
        run_app()
    elif args.inp:
        run_batch(args.inp, args.out)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
