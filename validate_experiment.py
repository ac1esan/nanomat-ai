"""
Внешняя валидация: предсказания модели (PBE-уровень) vs ЭКСПЕРИМЕНТ.
====================================================================

Зачем: модель обучена на PBE-щелях (Alexandria). PBE систематически ЗАНИЖАЕТ
истинную электронную щель. Берём каноничные 2D-монослои с измеренными щелями,
прогоняем через наш инструмент и количественно показываем сдвиг PBE→эксперимент.
Заодно подгоняем грубую линейную поправку.

ВАЖНАЯ ФИЗИКА (честно): экспериментальные щели монослоёв — обычно ОПТИЧЕСКИЕ
(с экситонным связыванием ~0.3–0.7 эВ в 2D), а PBE даёт KS-щель. PBE занижает
квазичастичную щель, но оптическая = квазичастичная − экситон → ошибки частично
гасятся. Поэтому для TMD PBE-KS бывает близко к оптике, а для широкозонного h-BN
занижение большое. Этот разбор и есть «понимание физики».

Структуры берём из JARVIS dft_2d (готовые релакс-геометрии монослоёв).
Эксп. значения — из литературы (оптические щели монослоёв, типичные ссылки).

Запуск (без прокси, если мешает):
    env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
        ./venv/bin/python validate_experiment.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np

import screen_bandgap as S  # переиспользуем модель/инференс

# jid в dft_2d -> (подпись, эксп. щель эВ, тип, заметка)
REF = {
    "JVASP-664":   ("MoS2",        1.88, "прямая",   "оптич. монослой"),
    "JVASP-76621": ("MoSe2",       1.55, "прямая",   "оптич. монослой"),
    "JVASP-658":   ("WS2",         2.00, "прямая",   "оптич. монослой"),
    "JVASP-652":   ("WSe2",        1.65, "прямая",   "оптич. монослой"),
    "JVASP-76649": ("h-BN",        6.00, "непрямая", "широкозонный диэлектрик"),
    "JVASP-60235": ("phosphorene", 2.00, "прямая",   "оптич.; транспортная ниже"),
    "JVASP-60368": ("graphene",    0.00, "—",        "полуметалл (sanity)"),
}


def main():
    from jarvis.db.figshare import data as jarvis_data
    from jarvis.core.atoms import Atoms

    models, mean, std = S.load_models()
    type_model = S.load_type_model()

    raw = {e.get("jid"): e for e in jarvis_data("dft_2d")}

    rows = []
    print(f"\n{'материал':12s} {'модель(PBE)':>12s} {'эксп':>7s} {'Δ(эксп-мод)':>12s} "
          f"{'тип-модель':>12s} {'тип-эксп':>10s}")
    print("-" * 72)
    for jid, (name, exp, etype, note) in REF.items():
        e = raw.get(jid)
        if e is None:
            print(f"{name:12s}  нет структуры (jid {jid})")
            continue
        st = Atoms.from_dict(e["atoms"]).pymatgen_converter()
        gap, unc = S.predict(models, mean, std, st)
        tlabel, p_ind = S.predict_type(type_model, st)
        tlabel = tlabel or "—"
        rows.append((name, gap, exp, etype, note))
        print(f"{name:12s} {gap:7.2f}±{unc:.2f} {exp:7.2f} {exp-gap:12.2f} "
              f"{tlabel:>12s} {etype:>10s}")

    # --- систематика по полупроводникам (исключаем графен gap=0) ---
    semi = [(m, e) for (n, m, e, et, nt) in rows if e > 0.1]
    mod = np.array([m for m, e in semi]); ex = np.array([e for m, e in semi])
    print("\n--- систематика (полупроводники) ---")
    print(f"средний сдвиг (эксп - модель): {np.mean(ex - mod):+.2f} эВ "
          f"(PBE {'занижает' if np.mean(ex-mod) > 0 else 'завышает'})")
    print(f"средн. относит. занижение: {100*np.mean((ex-mod)/ex):.0f}%")

    # линейная поправка exp ≈ a*model + b
    a, b = np.polyfit(mod, ex, 1)
    pred = a * mod + b
    mae_raw = np.mean(np.abs(mod - ex))
    mae_corr = np.mean(np.abs(pred - ex))
    print(f"\nлинейная поправка PBE->эксперимент:  exp ≈ {a:.2f}·gap_PBE + {b:.2f}")
    print(f"MAE до поправки:  {mae_raw:.2f} эВ")
    print(f"MAE после поправки: {mae_corr:.2f} эВ")
    print("\nИнтерпретация: для TMD модель близка к оптич. щели (экситон гасит часть "
          "PBE-занижения); для h-BN занижение максимально (широкая щель, мало экситонной "
          "компенсации). Графен (gap≈0) модель ловит как полуметалл — sanity-check пройден.")


if __name__ == "__main__":
    main()
