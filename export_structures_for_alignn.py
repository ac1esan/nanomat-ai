"""
Экспорт 2D-структур JARVIS-DFT в формат, который понимает ALIGNN.
=================================================================

Зачем: baseline по составу упёрся в ~R²=0.56 (см. CLAUDE.md). Дальше нужна
СТРУКТУРА, а её даёт GNN (ALIGNN). ALIGNN/DGL не ставится под Python 3.14 на маке,
поэтому обучение гоним на Kaggle. Этот скрипт локально готовит данные для заливки.

Что делает:
  1. Тянет dft_2d из JARVIS (тот же источник, что и baseline).
  2. Берёт band gap (optb88vdw_bandgap) и кристаллическую структуру (atoms).
  3. По умолчанию отсекает металлы (gap<=0.01) — как в baseline c --drop-metals,
     чтобы сравнение GNN vs baseline было честным (та же выборка).
  4. Пишет папку POSCAR-файлов + id_prop.csv в формате ALIGNN.

Формат ALIGNN (train_folder / train_alignn):
  <out_dir>/
    <jid>.vasp        # структура в POSCAR-формате
    ...
    id_prop.csv       # строки: <jid>.vasp,<band_gap>[,<extra_target>]   (без заголовка)

Запуск (локально, БЕЗ прокси, если он мешает сети):
    env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
        ./venv/bin/python export_structures_for_alignn.py
    # с металлами:        ... --keep-metals
    # другая папка/таргет: ... --out-dir alignn_data --target optb88vdw_bandgap

Дальше: заливаешь папку <out_dir> в Kaggle Dataset и в ноутбуке запускаешь ALIGNN
(см. CLAUDE.md, раздел про GNN).
"""

import argparse
import csv
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="dft_2d", choices=["dft_2d", "c2db", "alex_2d"],
                    help="датасет JARVIS: dft_2d (OptB88vdW), c2db (PBE) или alex_2d (Alexandria 2D, PBE, ~137k)")
    ap.add_argument("--out-dir", default=None,
                    help="куда писать (по умолчанию alignn_data_<source>)")
    ap.add_argument("--target", default=None,
                    help="поле band gap (дефолт: dft_2d->optb88vdw_bandgap, c2db->gap, alex_2d->band_gap_ind)")
    ap.add_argument("--ehull-max", type=float, default=0.1,
                    help="только для alex_2d: фильтр стабильности e_above_hull<=X эВ/атом (по умолч. 0.1)")
    ap.add_argument("--keep-metals", action="store_true",
                    help="не отсекать металлы (по умолчанию убираем gap<=0.01). "
                         "Действует ТОЛЬКО когда таргет — щель: отсечение по значению "
                         "другого свойства (напр. работы выхода) не имеет смысла")
    ap.add_argument("--target-min", type=float, default=None,
                    help="отбросить структуры со значением таргета ниже этого. Нужно для "
                         "свойств с физичным диапазоном: у работы выхода значения вне "
                         "1–8 эВ — сорвавшиеся расчёты, а на MSE один такой выброс весит "
                         "как сотни нормальных точек")
    ap.add_argument("--target-max", type=float, default=None,
                    help="то же сверху")
    ap.add_argument("--extra-target", default=None,
                    help="второе поле в id_prop.csv (напр. band_gap_dir для классификатора типа щели)")
    args = ap.parse_args()

    # дефолты под источник
    src_map = {"dft_2d": "dft_2d", "c2db": "c2db", "alex_2d": "alex_pbe_2d_all"}
    jarvis_tag = src_map[args.source]
    if args.target is None:
        args.target = {"dft_2d": "optb88vdw_bandgap", "c2db": "gap",
                       "alex_2d": "band_gap_ind"}[args.source]
    if args.out_dir is None:
        args.out_dir = f"alignn_data_{args.source}"
    gap_like = "gap" in args.target.lower()
    if not gap_like:
        print(f"Таргет «{args.target}» — не щель, фильтр металлов выключен.")

    from jarvis.db.figshare import data as jarvis_data
    from jarvis.core.atoms import Atoms

    print(f"Загрузка {args.source} из JARVIS (таргет: {args.target})...")
    raw = jarvis_data(jarvis_tag)
    print(f"Всего записей: {len(raw)}")

    os.makedirs(args.out_dir, exist_ok=True)

    rows = []          # (filename, target) для id_prop.csv
    n_skip_target = 0  # пропущено из-за плохого таргета ('na' и т.п.)
    n_skip_metal = 0
    n_skip_struct = 0
    n_skip_ehull = 0   # только alex_2d: нестабильные
    n_skip_range = 0   # вне --target-min/--target-max

    for e in raw:
        # --- фильтр стабильности (только Alexandria) ---
        if args.source == "alex_2d":
            ehull = pd.to_numeric(e.get("e_above_hull"), errors="coerce")
            if pd.isna(ehull) or ehull > args.ehull_max:
                n_skip_ehull += 1
                continue

        # --- таргет ---
        gap = pd.to_numeric(e.get(args.target), errors="coerce")
        if pd.isna(gap):
            n_skip_target += 1
            continue
        if (args.target_min is not None and gap < args.target_min) or \
           (args.target_max is not None and gap > args.target_max):
            n_skip_range += 1
            continue
        # Фильтр металлов осмыслен только для таргета-щели. Для работы выхода или
        # любого другого свойства «значение <= 0.01» ничего про металличность не
        # говорит, поэтому фильтр молча выключается.
        if gap_like and not args.keep_metals and gap <= 0.01:
            n_skip_metal += 1
            continue

        # --- структура ---
        try:
            atoms = Atoms.from_dict(e["atoms"])
            poscar = atoms.get_string()
        except Exception:
            n_skip_struct += 1
            continue

        jid = e.get("jid") or e.get("mat_id") or e.get("id") or f"id-{len(rows)}"
        fname = f"{jid}.vasp"
        with open(os.path.join(args.out_dir, fname), "w") as f:
            f.write(poscar)
        if args.extra_target:
            extra = pd.to_numeric(e.get(args.extra_target), errors="coerce")
            if pd.isna(extra):
                n_skip_target += 1
                os.remove(os.path.join(args.out_dir, fname))
                continue
            rows.append((fname, float(gap), float(extra)))
        else:
            rows.append((fname, float(gap)))

    # --- id_prop.csv (без заголовка, как ждёт ALIGNN) ---
    with open(os.path.join(args.out_dir, "id_prop.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print("\nГотово.")
    print(f"  Экспортировано структур : {len(rows)}")
    if args.source == "alex_2d":
        print(f"  Пропущено (нестабильные) : {n_skip_ehull} (e_above_hull>{args.ehull_max})")
    print(f"  Пропущено (плохой таргет): {n_skip_target}")
    if args.target_min is not None or args.target_max is not None:
        print(f"  Пропущено (вне диапазона): {n_skip_range}")
    if gap_like and not args.keep_metals:
        print(f"  Пропущено (металлы)      : {n_skip_metal}")
    print(f"  Пропущено (структура)    : {n_skip_struct}")
    print(f"  Папка                    : {args.out_dir}/")
    print(f"  id_prop.csv              : {os.path.join(args.out_dir, 'id_prop.csv')}")
    print("\nДальше: залить папку в Kaggle Dataset и обучить ALIGNN (см. CLAUDE.md).")


if __name__ == "__main__":
    main()
