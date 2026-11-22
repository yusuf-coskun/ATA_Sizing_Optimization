#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
İlk 500 uçak adayı için 3B Vortex Lattice Method (VLM / VSPAERO) analizi,
kalkış ve iniş fazları için flaplı 3B VLM analizi ve tüm performans modüllerinin
(kalkış, tırmanış, seyir, yük bırakma, iniş) VLM çıktılarıyla baştan çalıştırılması.

Akış:
1. `vlm_ilk_500_aday.csv` okunur (yoksa `aday_siralama.py` çalıştırılarak üretilir).
2. İlk 500 adaydaki benzersiz kanat-kuyruk geometrileri için:
   - Flapsız 3B VLM analizi (tırmanış, seyir ve yük bırakma için)
   - Flaplı 3B VLM analizi (`b_flap_m = min(1.0, b_w_m)` iç açıklıkta `<airfoil>_flap`
     kambur/sıfır-taşıma açısı ve kesit CL_max sınırlarıyla; kalkış ve iniş için)
   çalıştırılır.
3. Flapsız ve flaplı LLT + OpenVSP polar/özet tabloları VLM sonuçlarıyla güncellenerek
   `vlm_sonuclari/` dizinine kaydedilir:
   - `vlm_flapsiz_coklu_re.csv`, `vlm_flapsiz_polar_detay.csv`, `vlm_flapsiz_aday_ozet.csv`
   - `vlm_flapli_coklu_re.csv`, `vlm_flapli_polar_detay.csv`, `vlm_flapli_aday_ozet.csv`
4. İlk 500 adayın kütle satırları süzülerek `vlm_sonuclari/vlm_kutle_500_aday.csv` oluşturulur
   ve 5 performans modülü gerçekten baştan çalıştırılır:
   - `kalkis_hesabi.py`      -> Flaplı VLM tablolarıyla (`vlm_kalkis_sonuclari.csv`)
   - `tirmanis_hesabi.py`    -> Flapsız VLM tablolarıyla (`vlm_tirmanis_sonuclari.csv`)
   - `seyir_deneme.py`       -> Flapsız VLM + VLM kalkış/tırmanış çıktılarıyla (`vlm_seyir_optimizasyon_sonuclari.csv`)
   - `yuk_birakma.py`        -> Flapsız VLM tablolarıyla (`vlm_yuk_birakma_sonuclari.csv`)
   - `inis_hesabi_deneme.py` -> Flaplı ve flapsız VLM tablolarıyla (`vlm_inis_sonuclari.csv`)
5. Üretilen gerçek VLM performans tabloları `aday_siralama.adaylari_degerlendir_ve_sirala`
   ile birleştirilip puanlanır ve `vlm_nihai_ucak_siralamasi.csv` dosyasına kaydedilir.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from aday_siralama import (
    BATARYA_KAPASITE_MAH,
    GERILIM_V,
    KULLANILABILIR_MAH,
    MAX_GOREV_SURESI_SN,
    adaylari_degerlendir_ve_sirala,
    argumanlari_coz as siralama_argumanlari_coz,
    bool_al as _ham_bool_al,
    csv_oku,
    csv_yaz,
    dosya_yolu_coz,
    metin_al as _ham_metin_al,
    sayi_al as _ham_sayi_al,
)

# [DEĞİŞİKLİK 2026-11-18 | Görev A] Yeni VSPAERO modülü (OpenVSP Python API ile).
# Eski elle yazılan .vspaero akışının yerine geçti; ayrıntılar vspaero_cozucu.py başında.
from vspaero_cozucu import vspaero_api_coz, vspaero_hazir_mi
from vspaero_cozucu import vspaero_api_coz_richardson  # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson]

# [DEĞİŞİKLİK 2026-11-18 | Görev A] VSPAERO davranış ayarları (main() içinde komut satırından doldurulur).
# VSPAERO_ZORUNLU=True -> VSPAERO hata verirse program DURUR (NumPy'a geçmez).
# VSPAERO_KUYRUK_DAHIL=True -> modele yatay kuyruk da eklenir (varsayılan: sadece kanat).
VSPAERO_ZORUNLU = False
VSPAERO_KUYRUK_DAHIL = False
VSPAERO_MODEL_KLASORU: Optional[Path] = None
VSPAERO_HATALARI: List[str] = []  # hangi geometride neden NumPy'a düşüldü
VSPAERO_SPAN_PANEL: Optional[int] = None
VSPAERO_UC_SIK: Optional[float] = None
VSPAERO_RICHARDSON: Optional[Tuple[int, int]] = (24, 48)  # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson] None = tek ağ (VSPAERO_SPAN_PANEL)
VSPAERO_E_KAYNAGI: str = "iz"                               # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson] "iz" | "yuzey"
VSPAERO_VETER_TESS: Optional[int] = None
# [DEĞİŞİKLİK 2026-11-19 v10 | eğri tutarlılığı] Flaplı α0 için flap verimi (eta) ve CL eğrisi kurma yöntemi.
#   FLAP_VERIMI: VSPAERO (sürtünmesiz) 30° flabın α0 kaymasını abartıyor; XFOIL/ince profil karşılaştırması
#                (flap_alpha0_kontrol.py, 29 profil) ortanca eta: Re=200k 0.530, Re=250k (kalkış, v11) 0.531 -> 0.53.
#                α0_flapli = α0_flapsiz + eta·(α0_flapli_VSPAERO − α0_flapsiz). 1.0 = düzeltme yok (ham VSPAERO).
#   EGRI_YONTEMI: "otomatik" (VSPAERO: flapsız "kaydir", flaplı "dogrusal";
#                 diğer çözücüler "dogrusal"), "kaydir", "dogrusal".
FLAP_VERIMI: float = 0.53
EGRI_YONTEMI: str = "otomatik"
VSPAERO_GOVDE: Optional[Dict[str, Any]] = None  # [DEĞİŞİKLİK 2026-11-19 v9 | tam akışta gövde] None = sadece kanat; dolu = her adaya gövde (tam akış)
VSPAERO_E_VISKOZ: float = 0.995                 # [DEĞİŞİKLİK 2026-11-19 v9 | tam akışta gövde] e viskoz düzeltmesindeki k çarpanı                    # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson] veter yönü panel (Tess_W)  # [DEĞİŞİKLİK 2026-11-18 v6 | inceleme-2 düzeltmeleri] kanat ucuna panel sıklaştırma (OutCluster)  # [DEĞİŞİKLİK 2026-11-18 v5 | gövde] açıklık yönü panel sayısı (None = varsayılan)


def _anahtarlari_duzlestir(
    anahtarlar: Tuple[Any, ...],
) -> Tuple[List[str], Any, bool]:
    """
    Hem `sayi_al(row, "a", "b", varsayilan=0.0)` hem de
    `sayi_al(row, ("a", "b"), 0.0)` biçimindeki çağrıları güvenli şekilde ayrıştırır.
    """
    duz_anahtarlar: List[str] = []
    konumsal_varsayilan: Any = None
    konumsal_varsayilan_var = False

    if (
        len(anahtarlar) == 2
        and isinstance(anahtarlar[0], (tuple, list))
        and not isinstance(anahtarlar[1], (str, tuple, list))
    ):
        for k in anahtarlar[0]:
            if isinstance(k, str):
                duz_anahtarlar.append(k)
        konumsal_varsayilan = anahtarlar[1]
        konumsal_varsayilan_var = True
        return duz_anahtarlar, konumsal_varsayilan, konumsal_varsayilan_var

    for item in anahtarlar:
        if isinstance(item, (tuple, list)):
            for k in item:
                if isinstance(k, str):
                    duz_anahtarlar.append(k)
        elif isinstance(item, str):
            duz_anahtarlar.append(item)
        else:
            konumsal_varsayilan = item
            konumsal_varsayilan_var = True

    return duz_anahtarlar, konumsal_varsayilan, konumsal_varsayilan_var


def sayi_al(
    row: Optional[Dict[str, Any]],
    *anahtarlar: Any,
    varsayilan: Optional[float] = None,
) -> float:
    """
    `aday_siralama.sayi_al` sarmalayıcısı:
    Hem `sayi_al(row, "a", "b", varsayilan=0.0)` hem de `sayi_al(row, ("a", "b"), 0.0)`
    çağrılarını destekler ve `None` yerine her zaman `float` döndürür.
    """
    duz_anahtarlar, konumsal_vars, konumsal_var = _anahtarlari_duzlestir(anahtarlar)
    etkin_varsayilan = (
        varsayilan
        if varsayilan is not None
        else (float(konumsal_vars) if konumsal_var and konumsal_vars is not None else 0.0)
    )
    deger = _ham_sayi_al(row, *duz_anahtarlar, varsayilan=etkin_varsayilan)
    if deger is None or (isinstance(deger, float) and math.isnan(deger)):
        return float(etkin_varsayilan)
    return float(deger)


def metin_al(
    row: Optional[Dict[str, Any]],
    *anahtarlar: Any,
    varsayilan: str = "",
) -> str:
    """
    `aday_siralama.metin_al` sarmalayıcısı:
    Hem çoklu string argümanları hem de tuple/list anahtarları destekler.
    """
    duz_anahtarlar, konumsal_vars, konumsal_var = _anahtarlari_duzlestir(anahtarlar)
    etkin_varsayilan = (
        varsayilan
        if varsayilan != ""
        else (str(konumsal_vars) if konumsal_var and konumsal_vars is not None else "")
    )
    deger = _ham_metin_al(row, *duz_anahtarlar, varsayilan=etkin_varsayilan)
    return str(deger if deger is not None else etkin_varsayilan)


def bool_al(
    row: Optional[Dict[str, Any]],
    *anahtarlar: Any,
    varsayilan: bool = False,
) -> bool:
    """
    `aday_siralama.bool_al` sarmalayıcısı:
    Hem çoklu string argümanları hem de tuple/list anahtarları destekler.
    """
    duz_anahtarlar, konumsal_vars, konumsal_var = _anahtarlari_duzlestir(anahtarlar)
    etkin_varsayilan = (
        bool(konumsal_vars) if (konumsal_var and varsayilan is False) else bool(varsayilan)
    )
    return bool(_ham_bool_al(row, *duz_anahtarlar, varsayilan=etkin_varsayilan))


# ---------------------------------------------------------------------------
# 1. VLM Motor Seçimi ve 3B VLM Çözücüleri (Flapsız + Flaplı Destekli)
# ---------------------------------------------------------------------------

def kullanilabilir_vlm_motoru_sec(tercih: str = "auto") -> str:
    """Sistemde kurulu olan en yüksek doğruluklu VLM motorunu seçer."""
    tercih_norm = (tercih or "auto").strip().lower()

    if tercih_norm in ("vspaero", "auto"):
        # [DEĞİŞİKLİK 2026-11-18 | Görev A] Eskiden yalnızca PATH'te "vspaero" arıyordu (Windows'ta genelde
        # PATH'te olmaz) ve nedenini söylemeden yedeğe geçiyordu. Artık kontrol
        # vspaero_cozucu.vspaero_hazir_mi() ile yapılıyor ve sebep ekrana yazılıyor.
        # --cozucu vspaero açıkça seçildiyse VSPAERO yoksa program durur.
        hazir, mesaj = vspaero_hazir_mi()
        if hazir:
            print(f"[VSPAERO] Hazır: {mesaj}")
            return "vspaero"
        print(f"[UYARI] VSPAERO kullanılamıyor: {mesaj}")
        if tercih_norm == "vspaero":
            raise RuntimeError(f"--cozucu vspaero seçildi ama VSPAERO hazır değil: {mesaj}")

    if tercih_norm in ("aerosandbox", "vspaero", "auto"):
        try:
            import aerosandbox  # type: ignore # noqa: F401
            return "aerosandbox"
        except ImportError:
            if tercih_norm == "aerosandbox":
                print("[UYARI] AeroSandbox bulunamadı; dahili NumPy 3B VLM motoruna geçiliyor.")

    return "numpy_vlm"


def _biot_savart_segment(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    core_radius: float = 1e-5,
) -> np.ndarray:
    """Sonlu vorteks segmentinin (a -> b) p noktalarında indüklediği hız (Gamma=1 için)."""
    r1 = p - a
    r2 = p - b
    r0 = b - a
    cross = np.cross(r1, r2)
    cross_sq = np.sum(cross * cross, axis=-1, keepdims=True)
    r1_norm = np.linalg.norm(r1, axis=-1, keepdims=True)
    r2_norm = np.linalg.norm(r2, axis=-1, keepdims=True)

    denom = 4.0 * math.pi * np.maximum(cross_sq, core_radius ** 2)
    dot_term = (
        np.sum(r0 * r1, axis=-1, keepdims=True) / np.maximum(r1_norm, core_radius)
        - np.sum(r0 * r2, axis=-1, keepdims=True) / np.maximum(r2_norm, core_radius)
    )
    return (cross / denom) * dot_term


def _yari_sonsuz_iz_hizi(
    p: np.ndarray,
    a: np.ndarray,
    u_inf_dir: np.ndarray,
    core_radius: float = 1e-5,
) -> np.ndarray:
    """a noktasından u_inf_dir yönünde sonsuza uzanan vorteks izinin indüklediği hız."""
    r1 = p - a
    r1_norm = np.linalg.norm(r1, axis=-1, keepdims=True)
    cross = np.cross(u_inf_dir, r1)
    cross_sq = np.sum(cross * cross, axis=-1, keepdims=True)
    denom = 4.0 * math.pi * np.maximum(cross_sq, core_radius ** 2)
    factor = 1.0 + np.sum(r1 * u_inf_dir, axis=-1, keepdims=True) / np.maximum(r1_norm, core_radius)
    return (cross / denom) * factor


def numpy_3d_vlm_coz(
    b_w: float,
    c_w: float,
    s_w: float,
    b_ht: float = 0.0,
    c_ht: float = 0.0,
    tail_arm: float = 0.0,
    alpha_list_deg: Tuple[float, ...] = (-2.0, 0.0, 2.0, 4.0, 6.0, 8.0),
    n_span: int = 24,
    n_chord: int = 4,
    alpha_l0_2d_deg: float = 0.0,
    cl_max_2d: float = 1.45,
    flap_aciklik_orani: float = 0.0,
    alpha_l0_flap_2d_deg: Optional[float] = None,
    cl_max_flap_2d: Optional[float] = None,
) -> Dict[str, Any]:
    """
    3B At Nalı Vorteks (Horseshoe VLM) + Trefftz Düzlemi çözücüsü.
    `flap_aciklik_orani > 0` verildiğinde, kanadın iç açıklık bölgesindeki
    (`|y| <= 0.5 * b_w * flap_aciklik_orani`) panellere flaplı kesitin sıfır taşıma
    açısını (`alpha_l0_flap_2d_deg`) ve 2B maksimum taşıma katsayısını (`cl_max_flap_2d`)
    uygulayarak 3B flaplı sirkülasyon dağılımını, flap ucu vorteks kayıplarını (Oswald e)
    ve 3B flaplı CL_max değerini doğrudan çözer.
    """
    b_w = max(float(b_w), 0.1)
    c_w = max(float(c_w), 0.05)
    s_w = max(float(s_w), b_w * c_w)
    ar_w = (b_w ** 2) / s_w

    flap_orani = float(np.clip(flap_aciklik_orani, 0.0, 1.0))
    flap_yari_aciklik = 0.5 * b_w * flap_orani
    alpha_l0_flap_deg = (
        float(alpha_l0_flap_2d_deg)
        if alpha_l0_flap_2d_deg is not None
        else float(alpha_l0_2d_deg)
    )
    cl_max_flap_kesit = (
        float(cl_max_flap_2d)
        if (cl_max_flap_2d is not None and cl_max_flap_2d > 0.1)
        else float(cl_max_2d)
    )

    n_span_even = max(6, int(n_span) + (int(n_span) % 2))
    n_chord = max(1, int(n_chord))

    y_edges = np.linspace(-0.5 * b_w, 0.5 * b_w, n_span_even + 1)
    x_edges = np.linspace(0.0, c_w, n_chord + 1)

    alpha_0_flapsiz_rad = math.radians(float(alpha_l0_2d_deg))
    alpha_0_flapli_rad = math.radians(float(alpha_l0_flap_deg))

    n_wing_flapsiz = np.array(
        [-math.sin(alpha_0_flapsiz_rad), 0.0, math.cos(alpha_0_flapsiz_rad)],
        dtype=float,
    )
    n_wing_flapli = np.array(
        [-math.sin(alpha_0_flapli_rad), 0.0, math.cos(alpha_0_flapli_rad)],
        dtype=float,
    )

    vort_a_list: List[np.ndarray] = []
    vort_b_list: List[np.ndarray] = []
    coll_list: List[np.ndarray] = []
    norm_list: List[np.ndarray] = []
    panel_span_idx: List[int] = []
    panel_is_wing: List[bool] = []
    strip_is_flapped = np.zeros(n_span_even, dtype=bool)

    for i in range(n_span_even):
        y1, y2 = float(y_edges[i]), float(y_edges[i + 1])
        ym = 0.5 * (y1 + y2)
        is_flap_strip = bool(flap_orani > 1e-4 and abs(ym) <= flap_yari_aciklik + 1e-9)
        strip_is_flapped[i] = is_flap_strip
        n_panel = n_wing_flapli if is_flap_strip else n_wing_flapsiz

        for j in range(n_chord):
            x1, x2 = float(x_edges[j]), float(x_edges[j + 1])
            dx = x2 - x1
            x_vort = x1 + 0.25 * dx
            x_coll = x1 + 0.75 * dx

            vort_a_list.append(np.array([x_vort, y1, 0.0], dtype=float))
            vort_b_list.append(np.array([x_vort, y2, 0.0], dtype=float))
            coll_list.append(np.array([x_coll, ym, 0.0], dtype=float))
            norm_list.append(n_panel)
            panel_span_idx.append(i)
            panel_is_wing.append(True)

    if b_ht > 0.05 and c_ht > 0.02 and tail_arm > 0.1:
        n_ht_span = max(6, (n_span_even // 3) + ((n_span_even // 3) % 2))
        y_ht_edges = np.linspace(-0.5 * b_ht, 0.5 * b_ht, n_ht_span + 1)
        x_ht_edges = np.linspace(tail_arm, tail_arm + c_ht, 3)
        n_ht = np.array([0.0, 0.0, 1.0], dtype=float)
        for i in range(n_ht_span):
            y1, y2 = float(y_ht_edges[i]), float(y_ht_edges[i + 1])
            ym = 0.5 * (y1 + y2)
            for j in range(2):
                x1, x2 = float(x_ht_edges[j]), float(x_ht_edges[j + 1])
                dx = x2 - x1
                vort_a_list.append(np.array([x1 + 0.25 * dx, ym - 0.5 * (y2 - y1), 0.0], dtype=float))
                vort_b_list.append(np.array([x1 + 0.25 * dx, ym + 0.5 * (y2 - y1), 0.0], dtype=float))
                coll_list.append(np.array([x1 + 0.75 * dx, ym, 0.0], dtype=float))
                norm_list.append(n_ht)
                panel_span_idx.append(-1)
                panel_is_wing.append(False)

    vort_a = np.asarray(vort_a_list, dtype=float)
    vort_b = np.asarray(vort_b_list, dtype=float)
    coll = np.asarray(coll_list, dtype=float)
    normals = np.asarray(norm_list, dtype=float)
    span_idx_arr = np.asarray(panel_span_idx, dtype=int)
    is_wing_arr = np.asarray(panel_is_wing, dtype=bool)

    u_trail = np.array([1.0, 0.0, 0.0], dtype=float)
    p_grid = coll[:, None, :]
    a_grid = vort_a[None, :, :]
    b_grid = vort_b[None, :, :]

    v_bound = _biot_savart_segment(p_grid, a_grid, b_grid)
    v_trail_left = -_yari_sonsuz_iz_hizi(p_grid, a_grid, u_trail)
    v_trail_right = _yari_sonsuz_iz_hizi(p_grid, b_grid, u_trail)
    v_total = v_bound + v_trail_left + v_trail_right

    aic = np.sum(v_total * normals[:, None, :], axis=-1)
    dy_wing = np.diff(y_edges)
    y_centers_wing = 0.5 * (y_edges[:-1] + y_edges[1:])
    theta_wing = np.arccos(np.clip((2.0 * y_centers_wing) / b_w, -0.9999, 0.9999))

    alphas_deg: List[float] = []
    cls_wing: List[float] = []
    e_values: List[float] = []
    peak_ratios: List[float] = []
    strip_limit_ratios: List[float] = []

    cl_max_2d_strip = np.where(strip_is_flapped, cl_max_flap_kesit, float(cl_max_2d))

    for alpha_deg in alpha_list_deg:
        alpha_rad = math.radians(float(alpha_deg))
        u_inf = np.array([math.cos(alpha_rad), 0.0, math.sin(alpha_rad)], dtype=float)
        rhs = -np.sum(normals * u_inf[None, :], axis=-1)

        try:
            gamma = np.linalg.solve(aic, rhs)
        except np.linalg.LinAlgError:
            gamma, _, _, _ = np.linalg.lstsq(aic, rhs, rcond=None)

        gamma_strip = np.zeros(n_span_even, dtype=float)
        for idx_p in np.where(is_wing_arr)[0]:
            s_i = span_idx_arr[idx_p]
            if 0 <= s_i < n_span_even:
                gamma_strip[s_i] += float(gamma[idx_p])

        cl_wing = float((2.0 / s_w) * np.sum(gamma_strip * dy_wing))
        alphas_deg.append(float(alpha_deg))
        cls_wing.append(cl_wing)

        if abs(cl_wing) > 0.08:
            n_harmonics = min(13, n_span_even // 2)
            harmonics = [2 * k + 1 for k in range(n_harmonics)]
            sin_mat = np.column_stack([np.sin(n * theta_wing) for n in harmonics])
            a_coeffs, _, _, _ = np.linalg.lstsq(sin_mat, gamma_strip / (2.0 * b_w), rcond=None)
            if abs(a_coeffs[0]) > 1e-7:
                delta_ind = float(
                    np.sum([n * ((a_coeffs[idx] / a_coeffs[0]) ** 2) for idx, n in enumerate(harmonics[1:], start=1)])
                )
                e_inv = (1.0 + delta_ind) / 0.995 + 0.007 * math.pi * ar_w
                e_vlm = float(np.clip(1.0 / max(e_inv, 1e-6), 0.45, 0.99))
            else:
                e_vlm = 0.80
            e_values.append(e_vlm)

            cl_local = (2.0 * gamma_strip) / c_w
            mean_cl = max(abs(cl_wing), 1e-6)
            peak_ratio = float(np.max(np.abs(cl_local)) / mean_cl)
            peak_ratios.append(peak_ratio)

            positive_cl = np.maximum(cl_local, 0.0)
            if cl_wing > 0.15 and np.max(positive_cl) > 1e-5:
                critical_fraction = float(np.max(positive_cl / np.maximum(cl_max_2d_strip, 0.2)))
                if critical_fraction > 1e-6:
                    strip_limit_ratios.append(cl_wing / critical_fraction)

    alphas_arr = np.asarray(alphas_deg, dtype=float)
    cls_arr = np.asarray(cls_wing, dtype=float)
    slope_per_deg, intercept = np.polyfit(alphas_arr, cls_arr, 1)
    slope_per_deg = float(max(slope_per_deg, 1e-4))
    alpha_l0_vlm = float(-intercept / slope_per_deg)

    e_mid = [
        e_values[idx]
        for idx, a in enumerate(alphas_deg[: len(e_values)])
        if 1.5 <= a <= 6.5
    ]
    e_oswald_vlm = float(np.mean(e_mid)) if e_mid else (float(np.mean(e_values)) if e_values else 0.82)
    peak_cl_ratio = float(np.mean(peak_ratios)) if peak_ratios else 1.07

    if flap_orani > 1e-4 and strip_limit_ratios:
        cl_max_3d_vlm = float(np.mean(strip_limit_ratios[-2:]))
    else:
        cl_max_eff_2d = flap_orani * cl_max_flap_kesit + (1.0 - flap_orani) * float(cl_max_2d)
        cl_max_3d_vlm = float(cl_max_eff_2d / max(peak_cl_ratio, 1.02))

    return {
        "cozucu": "numpy_3d_vlm",
        "CL_alpha_3d_per_deg": round(slope_per_deg, 6),
        "alpha_L0_deg": round(alpha_l0_vlm, 4),
        "e_oswald": round(e_oswald_vlm, 5),
        "CL_max_3d_tahmini": round(cl_max_3d_vlm, 5),
        "peak_cl_ratio": round(peak_cl_ratio, 4),
        "flap_aciklik_orani": round(flap_orani, 4),
    }


def aerosandbox_vlm_coz(
    b_w: float,
    c_w: float,
    s_w: float,
    b_ht: float = 0.0,
    c_ht: float = 0.0,
    tail_arm: float = 0.0,
    alpha_l0_2d_deg: float = 0.0,
    cl_max_2d: float = 1.45,
    flap_aciklik_orani: float = 0.0,
    alpha_l0_flap_2d_deg: Optional[float] = None,
    cl_max_flap_2d: Optional[float] = None,
) -> Dict[str, Any]:
    """AeroSandbox VortexLatticeMethod çözücüsü (flapsız ve flaplı kesit destekli)."""
    try:
        import aerosandbox as asb  # type: ignore
    except ImportError:
        return numpy_3d_vlm_coz(
            b_w=b_w,
            c_w=c_w,
            s_w=s_w,
            b_ht=b_ht,
            c_ht=c_ht,
            tail_arm=tail_arm,
            alpha_l0_2d_deg=alpha_l0_2d_deg,
            cl_max_2d=cl_max_2d,
            flap_aciklik_orani=flap_aciklik_orani,
            alpha_l0_flap_2d_deg=alpha_l0_flap_2d_deg,
            cl_max_flap_2d=cl_max_flap_2d,
        )

    flap_orani = float(np.clip(flap_aciklik_orani, 0.0, 1.0))
    alpha_l0_flap_deg = (
        float(alpha_l0_flap_2d_deg)
        if alpha_l0_flap_2d_deg is not None
        else float(alpha_l0_2d_deg)
    )
    cl_max_flap_kesit = (
        float(cl_max_flap_2d)
        if (cl_max_flap_2d is not None and cl_max_flap_2d > 0.1)
        else float(cl_max_2d)
    )

    try:
        delta_flap_twist = float(alpha_l0_2d_deg - alpha_l0_flap_deg) if flap_orani > 1e-4 else 0.0
        if 0.05 < flap_orani < 0.98:
            y_flap = 0.5 * b_w * flap_orani
            xsecs = [
                asb.WingXSec(xyz_le=[0.0, 0.0, 0.0], chord=c_w, twist=delta_flap_twist),
                asb.WingXSec(xyz_le=[0.0, y_flap, 0.0], chord=c_w, twist=delta_flap_twist),
                asb.WingXSec(xyz_le=[0.0, 0.5 * b_w, 0.0], chord=c_w, twist=0.0),
            ]
        else:
            xsecs = [
                asb.WingXSec(xyz_le=[0.0, 0.0, 0.0], chord=c_w, twist=delta_flap_twist),
                asb.WingXSec(xyz_le=[0.0, 0.5 * b_w, 0.0], chord=c_w, twist=delta_flap_twist),
            ]

        wings = [asb.Wing(name="MainWing", symmetric=True, xsecs=xsecs)]
        if b_ht > 0.05 and c_ht > 0.02 and tail_arm > 0.1:
            wings.append(
                asb.Wing(
                    name="HTail",
                    symmetric=True,
                    xsecs=[
                        asb.WingXSec(xyz_le=[tail_arm, 0.0, 0.0], chord=c_ht, twist=0.0),
                        asb.WingXSec(xyz_le=[tail_arm, 0.5 * b_ht, 0.0], chord=c_ht, twist=0.0),
                    ],
                )
            )
        airplane = asb.Airplane(name="CandidateAircraft", wings=wings)
        alphas = (0.0, 4.0)
        cls: List[float] = []
        cdi_list: List[float] = []
        for a_deg in alphas:
            vlm = asb.VortexLatticeMethod(
                airplane=airplane,
                op_point=asb.OperatingPoint(velocity=20.0, alpha=float(a_deg - alpha_l0_2d_deg)),
                spanwise_resolution=12,
                chordwise_resolution=4,
            )
            res = vlm.run()
            cls.append(float(res["CL"]))
            cdi_list.append(float(res["CDi"]))

        cl_alpha_per_deg = max((cls[1] - cls[0]) / (alphas[1] - alphas[0]), 1e-4)
        alpha_l0_eff = float(alpha_l0_2d_deg - flap_orani * delta_flap_twist)
        ar_w = (b_w ** 2) / max(s_w, 1e-6)
        if abs(cls[1]) > 0.05 and cdi_list[1] > 1e-6:
            e_oswald = float(np.clip((cls[1] ** 2) / (math.pi * ar_w * cdi_list[1]), 0.50, 0.98))
        else:
            e_oswald = 0.82

        cl_max_eff_2d = flap_orani * cl_max_flap_kesit + (1.0 - flap_orani) * float(cl_max_2d)
        cl_max_3d_tahmini = cl_max_eff_2d * (0.90 if flap_orani > 1e-4 else 0.92)

        return {
            "cozucu": "aerosandbox_vlm",
            "CL_alpha_3d_per_deg": round(cl_alpha_per_deg, 6),
            "alpha_L0_deg": round(alpha_l0_eff, 4),
            "e_oswald": round(e_oswald, 5),
            "CL_max_3d_tahmini": round(cl_max_3d_tahmini, 5),
            "peak_cl_ratio": 1.087,
            "flap_aciklik_orani": round(flap_orani, 4),
        }
    except Exception:
        return numpy_3d_vlm_coz(
            b_w=b_w,
            c_w=c_w,
            s_w=s_w,
            b_ht=b_ht,
            c_ht=c_ht,
            tail_arm=tail_arm,
            alpha_l0_2d_deg=alpha_l0_2d_deg,
            cl_max_2d=cl_max_2d,
            flap_aciklik_orani=flap_aciklik_orani,
            alpha_l0_flap_2d_deg=alpha_l0_flap_2d_deg,
            cl_max_flap_2d=cl_max_flap_2d,
        )


def vspaero_cli_coz(
    b_w: float,
    c_w: float,
    s_w: float,
    b_ht: float = 0.0,
    c_ht: float = 0.0,
    tail_arm: float = 0.0,
    alpha_l0_2d_deg: float = 0.0,
    cl_max_2d: float = 1.45,
    flap_aciklik_orani: float = 0.0,
    alpha_l0_flap_2d_deg: Optional[float] = None,
    cl_max_flap_2d: Optional[float] = None,
    ek_geometri: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    [DEĞİŞİKLİK 2026-11-18 | Görev A] Fonksiyon gövdesi tamamen yenilendi.
    ESKİ HAL (yedekler/vlm_entegrasyonu_YEDEK_2026-11-18_gorevA_oncesi.py) sorunları:
      - .polar dosyasında AoA yerine Mach, CL yerine AoA sütununu okuyordu,
      - kuyruk ve kanat profili modele eklenmiyordu,
      - flaba gerçek sapma açısı yerine Δα0 veriliyordu,
      - her hatada SESSİZCE NumPy çözücüsüne dönüyordu.
    YENİ HAL: işi vspaero_cozucu.vspaero_api_coz() yapar (OpenVSP API, profil
    dosyaları, flaplı profil, sonuçları sütun ADIYLA okuma). Hata olursa ekrana
    yazılır, VSPAERO_HATALARI listesine eklenir; VSPAERO_ZORUNLU ise program durur.
    alpha_l0_2d_deg / alpha_l0_flap_2d_deg artık kullanılmaz: VSPAERO α0'ı profil
    kamburluğundan kendisi hesaplar (imza geriye dönük uyumluluk için korundu).
    """
    ek = dict(ek_geometri or {})
    profil = str(ek.get("profil", "")).strip()
    geo_etiket = f"{profil} b={b_w:.3f} S={s_w:.3f} flap_orani={flap_aciklik_orani:.3f}"
    try:
        if not profil:
            raise ValueError("ek_geometri içinde 'profil' yok")
        # [DEĞİŞİKLİK 2026-11-19 v9 | tam akışta gövde] Tam akışta gövde: ölçüler adayın kütle satırından, bilinmeyenler komut satırı parametrelerinden
        _govde_kw: Dict[str, Any] = {}
        if VSPAERO_GOVDE:
            _D = float(ek.get("govde_D", 0.0) or 0.0)
            _L = float(ek.get("govde_L", 0.0) or 0.0)
            if _D <= 0 or _L <= 0:
                raise ValueError("Gövde ölçüleri (govde_D_m, govde_L_toplam_m) kütle tablosunda bulunamadı")
            _g = {"D": _D, "L_toplam": _L}
            _g.update({k: v for k, v in VSPAERO_GOVDE.items()
                       if k in ("L_aviyonik", "L_kuyruk", "D_min", "kanat_x", "kanat_z", "gomme")})
            _govde_kw = {"govde": _g, "zcg": VSPAERO_GOVDE.get("zcg"),
                         "flap_ic_y": (0.5 * _D if VSPAERO_GOVDE.get("flap_baslangic") == "govde_kenari" else 0.0)}
        _coz = vspaero_api_coz  # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson] Richardson açıksa iki ağla çöz
        _ek: Dict[str, Any] = {"span_panel": VSPAERO_SPAN_PANEL}
        if VSPAERO_RICHARDSON:
            _coz = vspaero_api_coz_richardson
            _ek = {"n_kaba": VSPAERO_RICHARDSON[0], "n_ince": VSPAERO_RICHARDSON[1]}
        return _coz(
            **_ek,
            e_kaynagi=VSPAERO_E_KAYNAGI,    # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson]
            viskoz_e_carpani=VSPAERO_E_VISKOZ,  # [DEĞİŞİKLİK 2026-11-19 v9 | tam akışta gövde]
            **_govde_kw,                        # [DEĞİŞİKLİK 2026-11-19 v9 | tam akışta gövde]
            veter_tess=VSPAERO_VETER_TESS,  # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson]
            profil=profil,
            b_w=b_w,
            c_w=c_w,
            s_w=s_w,
            cl_max_2d=cl_max_2d,
            flap_aciklik_orani=flap_aciklik_orani,
            cl_max_flap_2d=cl_max_flap_2d,
            kuyruk_dahil=VSPAERO_KUYRUK_DAHIL,
            b_ht=float(ek.get("b_ht", b_ht) or 0.0),
            c_ht=float(ek.get("c_ht", c_ht) or 0.0),
            tail_arm=float(ek.get("tail_arm", tail_arm) or 0.0),
            kuyruk_profil=str(ek.get("kuyruk_profil", "naca0010") or "naca0010"),
            model_klasoru=VSPAERO_MODEL_KLASORU,
            uc_sik=VSPAERO_UC_SIK,          # [DEĞİŞİKLİK 2026-11-18 v6 | inceleme-2 düzeltmeleri]
        )
    except Exception as exc:
        mesaj = f"{geo_etiket}: {type(exc).__name__}: {exc}"
        VSPAERO_HATALARI.append(mesaj)
        print(f"[VSPAERO-HATA] {mesaj}")
        if VSPAERO_ZORUNLU:
            raise
        print("[VSPAERO-HATA] Bu geometri için NumPy VLM yedeğine geçiliyor (sonuçta 'cozucu' sütununa bakın).")
        sonuc = numpy_3d_vlm_coz(
            b_w=b_w,
            c_w=c_w,
            s_w=s_w,
            b_ht=b_ht,
            c_ht=c_ht,
            tail_arm=tail_arm,
            alpha_l0_2d_deg=alpha_l0_2d_deg,
            cl_max_2d=cl_max_2d,
            flap_aciklik_orani=flap_aciklik_orani,
            alpha_l0_flap_2d_deg=alpha_l0_flap_2d_deg,
            cl_max_flap_2d=cl_max_flap_2d,
        )
        sonuc["cozucu"] = "numpy_3d_vlm_VSPAERO_HATASI_SONRASI_YEDEK"
        return sonuc


def _secilen_vlm_cozucu_cagir(
    motor: str,
    b_w: float,
    c_w: float,
    s_w: float,
    b_ht: float,
    c_ht: float,
    tail_arm: float,
    alpha_l0_2d_deg: float,
    cl_max_2d: float,
    n_span: int = 24,
    n_chord: int = 4,
    flap_aciklik_orani: float = 0.0,
    alpha_l0_flap_2d_deg: Optional[float] = None,
    cl_max_flap_2d: Optional[float] = None,
    ek_geometri: Optional[Dict[str, Any]] = None,  # [DEĞİŞİKLİK 2026-11-18 | Görev A] VSPAERO için profil/kuyruk bilgisi
) -> Dict[str, Any]:
    """Seçilen VLM motorunu flapsız veya flaplı parametrelerle çalıştırır."""
    if motor == "vspaero":
        return vspaero_cli_coz(
            ek_geometri=ek_geometri,  # [DEĞİŞİKLİK 2026-11-18 | Görev A]
            b_w=b_w,
            c_w=c_w,
            s_w=s_w,
            b_ht=b_ht,
            c_ht=c_ht,
            tail_arm=tail_arm,
            alpha_l0_2d_deg=alpha_l0_2d_deg,
            cl_max_2d=cl_max_2d,
            flap_aciklik_orani=flap_aciklik_orani,
            alpha_l0_flap_2d_deg=alpha_l0_flap_2d_deg,
            cl_max_flap_2d=cl_max_flap_2d,
        )
    if motor == "aerosandbox":
        return aerosandbox_vlm_coz(
            b_w=b_w,
            c_w=c_w,
            s_w=s_w,
            b_ht=b_ht,
            c_ht=c_ht,
            tail_arm=tail_arm,
            alpha_l0_2d_deg=alpha_l0_2d_deg,
            cl_max_2d=cl_max_2d,
            flap_aciklik_orani=flap_aciklik_orani,
            alpha_l0_flap_2d_deg=alpha_l0_flap_2d_deg,
            cl_max_flap_2d=cl_max_flap_2d,
        )
    return numpy_3d_vlm_coz(
        b_w=b_w,
        c_w=c_w,
        s_w=s_w,
        b_ht=b_ht,
        c_ht=c_ht,
        tail_arm=tail_arm,
        n_span=n_span,
        n_chord=n_chord,
        alpha_l0_2d_deg=alpha_l0_2d_deg,
        cl_max_2d=cl_max_2d,
        flap_aciklik_orani=flap_aciklik_orani,
        alpha_l0_flap_2d_deg=alpha_l0_flap_2d_deg,
        cl_max_flap_2d=cl_max_flap_2d,
    )


# ---------------------------------------------------------------------------
# 2. Benzersiz Geometri Anahtarı ve Flapsız + Flaplı VLM Analizi
# ---------------------------------------------------------------------------

def _profil_normalize(profil: str) -> str:
    """Airfoil adını küçük harfe çevirir, uzantıyı ve olası `_flap` sonekini temizler."""
    p = Path(str(profil or "").strip()).stem.lower().strip()
    if p.endswith("_flap"):
        return p[:-5]
    return p


def _geometri_anahtari(row: Dict[str, Any]) -> Tuple[str, float, float, float]:
    """
    Aday satırından `(temel_profil, b_w, s_w, c_w)` geometrik anahtarını üretir.
    Bazı tablolarda `c_w_m` kolonu bulunmadığından `b_w > 0` ve `s_w > 0` olduğunda
    `c_w = round(s_w / b_w, 4)` üzerinden tüm CSV'ler arasında %100 tutarlı anahtar üretilir.
    """
    profil = _profil_normalize(metin_al(row, "kanat_airfoil", "profil", varsayilan=""))
    b_w = round(sayi_al(row, "b_w_m", "aciklik_m", varsayilan=0.0), 4)
    s_w = round(sayi_al(row, "S_w_m2", "alan_m2", varsayilan=0.0), 4)
    c_w_raw = sayi_al(row, "c_w_m", "c_root_w_m", "veter_m", varsayilan=0.0)
    if b_w > 0.0 and s_w > 0.0:
        c_w = round(s_w / b_w, 4)
    else:
        c_w = round(c_w_raw, 4)
    return (profil, b_w, s_w, c_w)


def _vlm_eslesmesi_bul(
    vlm_haritasi: Dict[Tuple[str, float, float, float], Dict[str, Any]],
    g_key: Tuple[str, float, float, float],
) -> Optional[Dict[str, Any]]:
    """Geometri anahtarını doğrudan veya `(profil, round(b_w, 3), round(s_w, 3))` toleransıyla bulur."""
    dogrudan = vlm_haritasi.get(g_key)
    if dogrudan is not None:
        return dogrudan
    p, bw, sw, _ = g_key
    for k, v in vlm_haritasi.items():
        if k[0] == p and abs(k[1] - bw) <= 1e-3 and abs(k[2] - sw) <= 1e-3:
            return v
    return None


def ilk_500_aday_vlm_analizi(
    vlm_aday_satirlari: List[Dict[str, Any]],
    llt_flapsiz_satirlar: List[Dict[str, Any]],
    llt_flapli_satirlar: Optional[List[Dict[str, Any]]] = None,
    cozucu_tercihi: str = "auto",
    n_span: int = 24,
    n_chord: int = 4,
    kutle_satirlari: Optional[List[Dict[str, Any]]] = None,  # [DEĞİŞİKLİK 2026-11-18 | Görev A] VSPAERO kuyruk/profil bilgisi için
    geometri_limiti: int = 0,  # [DEĞİŞİKLİK 2026-11-18 | Görev A] >0 ise sadece ilk N geometri çözülür (hızlı test)
) -> Dict[Tuple[str, float, float, float], Dict[str, Any]]:
    """
    İlk 500 adaydaki benzersiz kanat-kuyruk geometrileri için hem flapsız 3B VLM
    hem de kalkış/iniş fazlarında kullanılan flaplı 3B VLM analizini çalıştırır.
    """
    motor = kullanilabilir_vlm_motoru_sec(cozucu_tercihi)
    print(f"[VLM] Seçilen 3B VLM çözücü motoru: {motor}")

    llt_flapsiz_indeks: Dict[Tuple[str, float, float, float], Dict[str, Any]] = {}
    for r in llt_flapsiz_satirlar:
        k = _geometri_anahtari(r)
        if k[0] and k not in llt_flapsiz_indeks:
            llt_flapsiz_indeks[k] = r

    llt_flapli_indeks: Dict[Tuple[str, float, float, float], Dict[str, Any]] = {}
    for r in (llt_flapli_satirlar or []):
        k = _geometri_anahtari(r)
        if k[0] and k not in llt_flapli_indeks:
            llt_flapli_indeks[k] = r

    vlm_sonuc_haritasi: Dict[Tuple[str, float, float, float], Dict[str, Any]] = {}

    # [DEĞİŞİKLİK 2026-11-18 | Görev A] Kütle tablosundan (kanat_kuyruk_govde_kutleli.csv) gerçek kuyruk kolu,
    # kuyruk profili ve yatay kuyruk ölçüleri alınır. Sadece VSPAERO kullanır;
    # NumPy/AeroSandbox çözücülerinin girdileri DEĞİŞMEDİ (test tabanı bozulmasın diye).
    kutle_geo_indeks: Dict[Tuple[str, float, float, float], Dict[str, Any]] = {}
    for kr in (kutle_satirlari or []):
        kk = _geometri_anahtari(kr)
        if kk[0] and kk not in kutle_geo_indeks:
            kutle_geo_indeks[kk] = kr

    for row in vlm_aday_satirlari:
        g_key = _geometri_anahtari(row)
        if not g_key[0] or g_key[1] <= 0.0 or g_key in vlm_sonuc_haritasi:
            continue
        if geometri_limiti > 0 and len(vlm_sonuc_haritasi) >= geometri_limiti:  # [DEĞİŞİKLİK 2026-11-18 | Görev A]
            break

        profil, b_w, s_w, c_w = g_key
        if c_w <= 0.0:
            c_w = round(s_w / max(b_w, 1e-6), 4)

        llt_ref = llt_flapsiz_indeks.get(g_key, row)
        llt_flap_ref = llt_flapli_indeks.get(g_key, {})

        s_ht = sayi_al(
            row,
            "S_HT_m2",
            varsayilan=sayi_al(llt_ref, "S_HT_m2", varsayilan=0.0),
        )
        c_ht = sayi_al(
            row,
            "c_HT_m",
            "c_root_HT_m",
            varsayilan=sayi_al(llt_ref, "c_HT_m", "c_root_HT_m", varsayilan=0.0),
        )
        b_ht = (s_ht / c_ht) if (s_ht > 0.0 and c_ht > 0.0) else sayi_al(row, "b_HT_m", varsayilan=0.0)
        govde_l = sayi_al(row, "govde_L_m", "govde_uzunlugu_m", "L_govde_m", varsayilan=1.5)
        if govde_l <= 0.0:
            govde_l = 1.5
        tail_arm = 0.55 * govde_l

        # Flapsız LLT referans parametreleri
        alpha_l0_deg = sayi_al(llt_ref, "alpha_L0_deg", varsayilan=-2.5)
        cl_alpha_llt = sayi_al(llt_ref, "CL_alpha_3d_per_deg", varsayilan=0.078)
        if cl_alpha_llt <= 0.0:
            cl_alpha_llt = 0.078
        e_llt = sayi_al(
            llt_ref,
            "e_oswald",
            "e_w",
            varsayilan=sayi_al(row, "e_oswald", varsayilan=0.84),
        )
        if e_llt <= 0.0:
            e_llt = 0.84
        cl_max_3d_llt = sayi_al(llt_ref, "CL_max_3d_tahmini", varsayilan=1.35)
        if cl_max_3d_llt <= 0.0:
            cl_max_3d_llt = 1.35
        cl_max_2d_tahmin = max(cl_max_3d_llt / 0.92, 0.90)

        # [DEĞİŞİKLİK 2026-11-18 | Görev A] VSPAERO'ya giden ek geometri bilgisi
        kutle_ref = kutle_geo_indeks.get(g_key, {})
        ek_geometri = {
            "profil": profil,
            "tail_arm": sayi_al(kutle_ref, "tail_arm_m", varsayilan=tail_arm),
            "b_ht": sayi_al(kutle_ref, "b_HT_m", varsayilan=b_ht),
            "c_ht": sayi_al(kutle_ref, "c_root_HT_m", "c_HT_m", varsayilan=c_ht),
            "kuyruk_profil": Path(metin_al(kutle_ref, "kuyruk_airfoil", varsayilan="naca0010")).stem,
            "govde_D": sayi_al(kutle_ref, "govde_D_m", varsayilan=0.0),           # [DEĞİŞİKLİK 2026-11-19 v9 | tam akışta gövde]
            "govde_L": sayi_al(kutle_ref, "govde_L_toplam_m", "govde_L_m", varsayilan=0.0),    # [DEĞİŞİKLİK 2026-11-19 v9 | tam akışta gövde]
        }

        # 1) Flapsız 3B VLM Çözümü
        vlm_flapsiz = _secilen_vlm_cozucu_cagir(
            motor=motor,
            b_w=b_w,
            c_w=c_w,
            s_w=s_w,
            b_ht=b_ht,
            c_ht=c_ht,
            tail_arm=tail_arm,
            alpha_l0_2d_deg=alpha_l0_deg,
            cl_max_2d=cl_max_2d_tahmin,
            n_span=n_span,
            n_chord=n_chord,
            flap_aciklik_orani=0.0,
            ek_geometri=ek_geometri,  # [DEĞİŞİKLİK 2026-11-18 | Görev A]
        )

        cl_alpha_vlm = float(vlm_flapsiz["CL_alpha_3d_per_deg"])
        e_vlm = float(vlm_flapsiz["e_oswald"])
        cl_max_vlm = float(vlm_flapsiz["CL_max_3d_tahmini"])

        cl_alpha_orani = cl_alpha_vlm / max(cl_alpha_llt, 1e-5)
        e_orani = e_vlm / max(e_llt, 1e-5)
        cl_max_orani = cl_max_vlm / max(cl_max_3d_llt, 1e-5)

        # 2) Flaplı 3B VLM Çözümü (Kalkış ve İniş İçin)
        b_flap = sayi_al(
            row,
            "b_flap_m",
            varsayilan=sayi_al(llt_flap_ref, "b_flap_m", varsayilan=min(1.0, b_w)),
        )
        if b_flap <= 0.0:
            b_flap = min(1.0, b_w)
        oran_flap = float(np.clip(b_flap / max(b_w, 1e-6), 0.0, 1.0))

        alpha_l0_flap_llt = sayi_al(
            llt_flap_ref,
            "alpha_L0_deg",
            varsayilan=alpha_l0_deg - 4.5 * oran_flap,
        )
        cl_alpha_flap_llt = sayi_al(
            llt_flap_ref,
            "CL_alpha_3d_per_deg",
            varsayilan=cl_alpha_llt,
        )
        if cl_alpha_flap_llt <= 0.0:
            cl_alpha_flap_llt = cl_alpha_llt
        e_flap_llt = sayi_al(
            llt_flap_ref,
            "e_oswald",
            "e_w",
            varsayilan=max(e_llt * 0.94, 0.55),
        )
        if e_flap_llt <= 0.0:
            e_flap_llt = max(e_llt * 0.94, 0.55)
        cl_max_3d_flap_llt = sayi_al(
            llt_flap_ref,
            "CL_max_3d_tahmini",
            varsayilan=cl_max_3d_llt * (1.0 + 0.30 * oran_flap),
        )
        if cl_max_3d_flap_llt <= 0.0:
            cl_max_3d_flap_llt = cl_max_3d_llt * (1.0 + 0.30 * oran_flap)

        if oran_flap > 1e-4:
            alpha_l0_flap_kesit_2d = (
                alpha_l0_flap_llt - (1.0 - oran_flap) * alpha_l0_deg
            ) / oran_flap
            cl_max_flap_3d_kesit = (
                cl_max_3d_flap_llt - (1.0 - oran_flap) * cl_max_3d_llt
            ) / oran_flap
            cl_max_flap_kesit_2d = max(cl_max_flap_3d_kesit / 0.92, cl_max_2d_tahmin * 1.15)
        else:
            alpha_l0_flap_kesit_2d = alpha_l0_deg
            cl_max_flap_kesit_2d = cl_max_2d_tahmin

        vlm_flapli = _secilen_vlm_cozucu_cagir(
            motor=motor,
            b_w=b_w,
            c_w=c_w,
            s_w=s_w,
            b_ht=b_ht,
            c_ht=c_ht,
            tail_arm=tail_arm,
            alpha_l0_2d_deg=alpha_l0_deg,
            cl_max_2d=cl_max_2d_tahmin,
            n_span=n_span,
            n_chord=n_chord,
            flap_aciklik_orani=oran_flap,
            alpha_l0_flap_2d_deg=alpha_l0_flap_kesit_2d,
            cl_max_flap_2d=cl_max_flap_kesit_2d,
            ek_geometri=ek_geometri,  # [DEĞİŞİKLİK 2026-11-18 | Görev A]
        )

        cl_alpha_flap_vlm = float(vlm_flapli["CL_alpha_3d_per_deg"])
        e_flap_vlm = float(vlm_flapli["e_oswald"])
        cl_max_flap_vlm = float(vlm_flapli["CL_max_3d_tahmini"])
        alpha_l0_flap_vlm = float(vlm_flapli["alpha_L0_deg"])
        # [DEĞİŞİKLİK 2026-11-19 v10 | eğri tutarlılığı] Flap verimi düzeltmesi (yalnız VSPAERO sonucuna):
        # sürtünmesiz çözücü 30° flabın α0 kaymasını abartır (≈ −11.6° yerine XFOIL'e göre ≈ −6°).
        alpha_l0_flap_ham = alpha_l0_flap_vlm
        flap_verimi_uygulanan = 1.0
        if str(vlm_flapli.get("cozucu", "")).startswith("vspaero") and oran_flap > 1e-4:
            _a0_temiz = float(vlm_flapsiz["alpha_L0_deg"])
            flap_verimi_uygulanan = float(FLAP_VERIMI)
            alpha_l0_flap_vlm = _a0_temiz + flap_verimi_uygulanan * (alpha_l0_flap_ham - _a0_temiz)

        cl_alpha_flap_orani = cl_alpha_flap_vlm / max(cl_alpha_flap_llt, 1e-5)
        e_flap_orani = e_flap_vlm / max(e_flap_llt, 1e-5)
        cl_max_flap_orani = cl_max_flap_vlm / max(cl_max_3d_flap_llt, 1e-5)

        vlm_sonuc_haritasi[g_key] = {
            **vlm_flapsiz,
            "CL_alpha_llt": round(cl_alpha_llt, 6),
            "e_oswald_llt": round(e_llt, 5),
            "CL_max_3d_llt": round(cl_max_3d_llt, 5),
            "cl_alpha_orani": round(cl_alpha_orani, 5),
            "e_oswald_orani": round(e_orani, 5),
            "cl_max_orani": round(cl_max_orani, 5),
            # Flaplı VLM sonuçları
            "b_flap_m": round(b_flap, 4),
            "flap_aciklik_orani": round(oran_flap, 4),
            "CL_alpha_3d_flapli_per_deg": round(cl_alpha_flap_vlm, 6),
            "alpha_L0_flapli_deg": round(alpha_l0_flap_vlm, 4),
            "alpha_L0_flapli_ham_deg": round(alpha_l0_flap_ham, 4),        # [DEĞİŞİKLİK 2026-11-19 v10 | eğri tutarlılığı] düzeltme öncesi
            "flap_verimi": round(flap_verimi_uygulanan, 4),                # [DEĞİŞİKLİK 2026-11-19 v10 | eğri tutarlılığı] 1.0 = düzeltme yok
            "e_oswald_flapli": round(e_flap_vlm, 5),
            "CL_max_3d_flapli_tahmini": round(cl_max_flap_vlm, 5),
            "CL_alpha_flapli_llt": round(cl_alpha_flap_llt, 6),
            "e_oswald_flapli_llt": round(e_flap_llt, 5),
            "CL_max_3d_flapli_llt": round(cl_max_3d_flap_llt, 5),
            "cl_alpha_flapli_orani": round(cl_alpha_flap_orani, 5),
            "e_oswald_flapli_orani": round(e_flap_orani, 5),
            "cl_max_flapli_orani": round(cl_max_flap_orani, 5),
            # [DEĞİŞİKLİK 2026-11-18 | Görev A] Hangi çözücünün kullanıldığı + VSPAERO model dosyaları (Görev B karşılaştırması için)
            "cozucu_flapli": vlm_flapli.get("cozucu", ""),
            "vsp3_flapsiz": vlm_flapsiz.get("vsp3_dosyasi", ""),
            "vsp3_flapli": vlm_flapli.get("vsp3_dosyasi", ""),
        }

    print(
        f"[VLM] Toplam {len(vlm_sonuc_haritasi)} benzersiz kanat-kuyruk geometrisi "
        f"(flapsız + flaplı) 3B VLM ile çözüldü."
    )
    return vlm_sonuc_haritasi


# ---------------------------------------------------------------------------
# 3. Flapsız ve Flaplı LLT / Drag Tablolarının VLM İle Güncellenmesi
# ---------------------------------------------------------------------------

def _tek_mod_tablolari_guncelle(
    llt_satirlar: List[Dict[str, Any]],
    drag_polar_satirlar: List[Dict[str, Any]],
    drag_ozet_satirlar: List[Dict[str, Any]],
    vlm_haritasi: Dict[Tuple[str, float, float, float], Dict[str, Any]],
    flapli_mi: bool = False,
    sadece_vlm_adaylari: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Verilen LLT, drag polar ve drag özet tablolarını VLM (flapsız veya flaplı) sonuçlarıyla günceller."""
    cl_alpha_key = "CL_alpha_3d_flapli_per_deg" if flapli_mi else "CL_alpha_3d_per_deg"
    alpha_l0_key = "alpha_L0_flapli_deg" if flapli_mi else "alpha_L0_deg"
    e_key = "e_oswald_flapli" if flapli_mi else "e_oswald"
    cl_max_key = "CL_max_3d_flapli_tahmini" if flapli_mi else "CL_max_3d_tahmini"

    # VLM doğrusu -2..+2 derecede tam kullanılır; LLT eğriliği bu bölgenin
    # üstünde stall sınırına sönümlenerek eklenir. CLmax/stall aynı modeldendir.
    cozucu_alan = "cozucu_flapli" if flapli_mi else "cozucu"
    egri_gruplari: Dict[Tuple[Any, ...], List[Tuple[float, float]]] = {}
    egri_parametreleri: Dict[Tuple[Any, ...], Optional[Tuple[float, float, float, float]]] = {}

    def _egri_anahtari(row: Dict[str, Any], g_key: Tuple[str, float, float, float]) -> Tuple[Any, ...]:
        return (g_key, metin_al(row, "temel_aday_id", varsayilan=""), metin_al(row, "konfigurasyon_id", varsayilan=""),
                metin_al(row, "hiz_ms", varsayilan=""), metin_al(row, "Re_aday", varsayilan=""))

    def _yontem(vlm: Dict[str, Any]) -> str:
        if EGRI_YONTEMI == "olcekle":
            raise ValueError("Yalnız oranla CL ölçekleme tutarsızdır; kaydir veya dogrusal kullanın.")
        if EGRI_YONTEMI in ("kaydir", "dogrusal"):
            return EGRI_YONTEMI
        if not str(vlm.get(cozucu_alan, vlm.get("cozucu", ""))).startswith("vspaero"):
            return "dogrusal"
        # Flaplı LLT eğrilerinde düşük açıda düz tabanlar var; flapsızda LLT
        # kıvrımı sınırlar içinde korunur, flaplıda VLM doğrusu kullanılır.
        return "dogrusal" if flapli_mi else "kaydir"

    for row in llt_satirlar:
        g_key = _geometri_anahtari(row)
        if _vlm_eslesmesi_bul(vlm_haritasi, g_key) is None or str(row.get("CL", "")).strip() == "":
            continue
        egri_gruplari.setdefault(_egri_anahtari(row, g_key), []).append(
            (sayi_al(row, "alpha_deg", varsayilan=0.0), sayi_al(row, "CL", varsayilan=0.0)))

    def _egri_dogrusu(noktalar: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float, float]]:
        """(eğim, α0, en küçük α, en büyük α) — eğrinin doğrusal bölgesi: baştan CL ≤ 0.7·CL_max'a kadar."""
        nk = sorted(set(noktalar))
        while len(nk) >= 2 and abs(nk[1][1] - nk[0][1]) < 1e-9:  # baştaki düz tabanı (LLT kırpması) at
            nk = nk[1:]
        if len(nk) < 3:
            return None
        cl_tepe = max(c for _, c in nk)
        sec: List[Tuple[float, float]] = []
        for a, c in nk:
            if c > 0.7 * cl_tepe and len(sec) >= 3:
                break
            sec.append((a, c))
        egim, kes = np.polyfit([a for a, _ in sec], [c for _, c in sec], 1)
        if egim <= 1e-6:
            return None
        return float(egim), float(-kes / egim), nk[0][0], nk[-1][0]

    def _egri_deger(noktalar: List[Tuple[float, float]], prm: Tuple[float, float, float, float], a: float) -> float:
        """Eğriden ara değer; aralık altında doğrusal uzatma, üstünde son değer (stall platosu)."""
        egim, a0, a_min, a_max = prm
        nk = sorted(set(noktalar))
        if a <= a_min:
            return egim * (a - a0) + (nk[0][1] - egim * (a_min - a0))
        if a >= a_max:
            return nk[-1][1]
        return float(np.interp(a, [p for p, _ in nk], [c for _, c in nk]))

    def _yeni_cl(alpha: float, vlm: Dict[str, Any], ek: Tuple[Any, ...]) -> float:
        """VLM doğrusu ve sınırıyla, mümkünse LLT eğriliğini koruyan tek CL modeli."""
        cla = float(vlm[cl_alpha_key])
        a0 = float(vlm[alpha_l0_key])
        clmax = float(vlm[cl_max_key])
        if not all(math.isfinite(x) for x in (alpha, cla, a0, clmax)) or cla <= 0 or clmax <= 0:
            raise ValueError("VLM CL modeli için sonlu, pozitif eğim ve CLmax gerekli.")
        stall = a0 + clmax / cla
        linear = cla * (alpha - a0)
        if alpha >= stall:
            return clmax
        if _yontem(vlm) != "kaydir" or alpha <= 2.0 or stall <= 2.0:
            return linear
        if ek not in egri_parametreleri:
            egri_parametreleri[ek] = _egri_dogrusu(egri_gruplari.get(ek, []))
        prm = egri_parametreleri[ek]
        if prm is None:
            raise ValueError(f"LLT eğrisinin doğrusal bölgesi çözülemedi: {ek}")
        old_slope, old_a0, _, _ = prm
        points = egri_gruplari[ek]
        def residual(a: float) -> float:
            old_a = old_a0 + a - a0
            old_cl = _egri_deger(points, prm, old_a)
            return (cla / old_slope) * (old_cl - old_slope * (old_a - old_a0))
        # Düz bölgede tam VLM doğrusu; stall'da tam CLmax. Arada LLT kıvrımı.
        taper = (stall - alpha) / (stall - 2.0)
        correction = (residual(alpha) - residual(2.0)) * taper
        correction = max(-0.5 * max(linear - cla * (2.0 - a0), 0.0),
                         min(correction, 0.5 * max(clmax - linear, 0.0)))
        return linear + correction

    yeni_llt: List[Dict[str, Any]] = []
    for row in llt_satirlar:
        g_key = _geometri_anahtari(row)
        vlm = _vlm_eslesmesi_bul(vlm_haritasi, g_key)
        if vlm is None:
            if not sadece_vlm_adaylari:
                yeni_llt.append(dict(row))
            continue

        r_copy = dict(row)
        r_copy["e_oswald"] = f"{float(vlm[e_key]):.6f}"
        r_copy["CL_alpha_3d_per_deg"] = f"{float(vlm[cl_alpha_key]):.6f}"
        if "CL_alpha_3d_kanat_per_deg" in r_copy:
            r_copy["CL_alpha_3d_kanat_per_deg"] = r_copy["CL_alpha_3d_per_deg"]
        r_copy["alpha_L0_deg"] = f"{float(vlm[alpha_l0_key]):.4f}"
        if "e_oswald_kanat" in r_copy:
            r_copy["e_oswald_kanat"] = r_copy["e_oswald"]
        r_copy["CL_max_3d_tahmini"] = f"{float(vlm[cl_max_key]):.6f}"
        r_copy["alpha_stall_2d_deg"] = f"{float(vlm[alpha_l0_key]) + float(vlm[cl_max_key]) / float(vlm[cl_alpha_key]):.6f}"
        r_copy["aero_modeli"] = "vlm_tek_egri_v1"
        if flapli_mi:
            if "flap_airfoil" not in r_copy or not str(r_copy.get("flap_airfoil", "")).strip():
                r_copy["flap_airfoil"] = f"{g_key[0]}_flap"
            if "b_flap_m" not in r_copy or not str(r_copy.get("b_flap_m", "")).strip():
                r_copy["b_flap_m"] = f"{float(vlm['b_flap_m']):.4f}"
        if "CL" in r_copy and str(r_copy["CL"]).strip() != "":
            r_copy["CL"] = f"{_yeni_cl(sayi_al(r_copy, 'alpha_deg'), vlm, _egri_anahtari(row, g_key)):.6f}"
        for alan in ("CL_flapsiz_kisim", "CL_flapli_kisim", "CL_kanat"):
            r_copy.pop(alan, None)
        yeni_llt.append(r_copy)

    # Drag polarları aynı LLT eğrisinden, ilgili Reynolds düğümleri arasında hesaplanır.
    egri_indeksi: Dict[Tuple[str, float, float, float], Dict[float, Dict[float, float]]] = {}
    for row in yeni_llt:
        if str(row.get("CL", "")).strip():
            curve = egri_indeksi.setdefault(_geometri_anahtari(row), {}).setdefault(float(row["Re_aday"]), {})
            alpha = float(row["alpha_deg"])
            cl = float(row["CL"])
            if alpha in curve and not math.isclose(curve[alpha], cl, abs_tol=2e-6):
                raise ValueError("Aynı geometri/Re/alpha için çelişen VLM CL eğrileri.")
            curve[alpha] = cl

    def _polar_cl(row: Dict[str, Any], g_key: Tuple[str, float, float, float]) -> float:
        nodes = egri_indeksi.get(g_key)
        if not nodes:
            raise ValueError(f"Drag polar için VLM LLT eğrisi yok: {g_key}")
        re = sayi_al(row, "Re_aday")
        alpha = sayi_al(row, "alpha_deg")
        def at(re_node: float) -> float:
            curve = nodes[re_node]
            angles = sorted(curve)
            if not angles[0] <= alpha <= angles[-1]:
                raise ValueError(f"Drag polar alpha aralık dışında: {alpha}")
            return float(np.interp(alpha, angles, [curve[a] for a in angles]))
        reynolds = sorted(nodes)
        if not reynolds[0] <= re <= reynolds[-1]:
            raise ValueError(f"Drag polar Re, LLT eğri aralığı dışında: {re}")
        if re == reynolds[0] or re == reynolds[-1]:
            return at(re)
        upper = next(i for i, value in enumerate(reynolds) if value >= re)
        lo, hi = reynolds[upper - 1], reynolds[upper]
        return at(lo) + (at(hi) - at(lo)) * (re - lo) / (hi - lo)

    yeni_drag_polar: List[Dict[str, Any]] = []
    for row in drag_polar_satirlar:
        g_key = _geometri_anahtari(row)
        vlm = _vlm_eslesmesi_bul(vlm_haritasi, g_key)
        if vlm is None:
            if not sadece_vlm_adaylari:
                yeni_drag_polar.append(dict(row))
            continue

        r_copy = dict(row)
        ar_w = sayi_al(r_copy, "AR_w", varsayilan=(g_key[1] ** 2) / max(g_key[2], 1e-6))
        if ar_w <= 0.0:
            ar_w = 6.0
        cd0 = sayi_al(r_copy, "CD_0", "CD0_toplam", varsayilan=0.025)
        cl_yeni = _polar_cl(r_copy, g_key)
        cdi_yeni = (cl_yeni ** 2) / max(math.pi * ar_w * float(vlm[e_key]), 1e-6)
        cd_yeni = cd0 + cdi_yeni
        ld_yeni = cl_yeni / max(cd_yeni, 1e-6)

        r_copy["e_oswald"] = f"{float(vlm[e_key]):.6f}"
        r_copy["CL_alpha_3d_per_deg"] = f"{float(vlm[cl_alpha_key]):.6f}"
        r_copy["alpha_L0_deg"] = f"{float(vlm[alpha_l0_key]):.4f}"
        r_copy["CL_max_3d_tahmini"] = f"{float(vlm[cl_max_key]):.6f}"
        r_copy["aero_modeli"] = "vlm_tek_egri_v1"
        if "CL" in r_copy:
            r_copy["CL"] = f"{cl_yeni:.6f}"
        if "CD_i" in r_copy:
            r_copy["CD_i"] = f"{cdi_yeni:.6f}"
        if "CD" in r_copy:
            r_copy["CD"] = f"{cd_yeni:.6f}"
        if "L_D" in r_copy:
            r_copy["L_D"] = f"{ld_yeni:.4f}"
        yeni_drag_polar.append(r_copy)

    yeni_drag_ozet: List[Dict[str, Any]] = []
    for row in drag_ozet_satirlar:
        g_key = _geometri_anahtari(row)
        vlm = _vlm_eslesmesi_bul(vlm_haritasi, g_key)
        if vlm is None:
            if not sadece_vlm_adaylari:
                yeni_drag_ozet.append(dict(row))
            continue

        r_copy = dict(row)
        ar_w = sayi_al(r_copy, "AR_w", varsayilan=(g_key[1] ** 2) / max(g_key[2], 1e-6))
        if ar_w <= 0.0:
            ar_w = 6.0
        cd0 = sayi_al(r_copy, "CD_0_seyir", "CD0_toplam", "CD_0", varsayilan=0.025)
        # CL_seyir görev yükünden gelen hedef taşıma katsayısıdır; eğim oranıyla değişmez.
        cl_seyir = sayi_al(r_copy, "CL_seyir", varsayilan=0.55)
        if cl_seyir > float(vlm[cl_max_key]) + 1e-6:
            raise ValueError(f"Seyir hedef CL, VLM CLmax sınırını aşıyor: {g_key}")
        cdi_seyir = (cl_seyir ** 2) / max(math.pi * ar_w * float(vlm[e_key]), 1e-6)
        cd_seyir = cd0 + cdi_seyir
        ld_seyir = cl_seyir / max(cd_seyir, 1e-6)

        r_copy["e_oswald"] = f"{float(vlm[e_key]):.6f}"
        r_copy["CL_alpha_3d_per_deg"] = f"{float(vlm[cl_alpha_key]):.6f}"
        r_copy["alpha_L0_deg"] = f"{float(vlm[alpha_l0_key]):.4f}"
        r_copy["CL_max_3d_tahmini"] = f"{float(vlm[cl_max_key]):.6f}"
        r_copy["aero_modeli"] = "vlm_tek_egri_v1"
        if "CL_seyir" in r_copy:
            r_copy["CL_seyir"] = f"{cl_seyir:.6f}"
        if "CD_i_seyir" in r_copy:
            r_copy["CD_i_seyir"] = f"{cdi_seyir:.6f}"
        if "CD_seyir" in r_copy:
            r_copy["CD_seyir"] = f"{cd_seyir:.6f}"
        if "L_D_seyir" in r_copy:
            r_copy["L_D_seyir"] = f"{ld_seyir:.4f}"
        yeni_drag_ozet.append(r_copy)

    return yeni_llt, yeni_drag_polar, yeni_drag_ozet


def _flapsizdan_flapli_llt_olustur(
    llt_flapsiz_satirlar: List[Dict[str, Any]],
    vlm_haritasi: Dict[Tuple[str, float, float, float], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Diskte `llt_flapli_coklu_re.csv` bulunmadığında veya eksik olduğunda,
    flapsız LLT satırlarını `llt_csv_okuyucu.LLTTable(flapped=True)` formatına
    (`kanat_airfoil=<airfoil>`, `flap_airfoil=<airfoil>_flap`, `b_flap_m`) ve
    flaplı VLM katsayılarına dönüştürür.
    """
    uretilen: List[Dict[str, Any]] = []
    for row in llt_flapsiz_satirlar:
        g_key = _geometri_anahtari(row)
        vlm = _vlm_eslesmesi_bul(vlm_haritasi, g_key)
        if vlm is None:
            continue
        r_copy = dict(row)
        profil_temel = g_key[0]
        r_copy["flap_airfoil"] = f"{profil_temel}_flap"
        r_copy["b_flap_m"] = f"{float(vlm['b_flap_m']):.4f}"
        r_copy["flap_aciklik_orani"] = f"{float(vlm['flap_aciklik_orani']):.4f}"
        r_copy["e_oswald"] = f"{float(vlm['e_oswald_flapli']):.6f}"
        r_copy["CL_alpha_3d_per_deg"] = f"{float(vlm['CL_alpha_3d_flapli_per_deg']):.6f}"
        if "CL_alpha_3d_kanat_per_deg" in r_copy:
            r_copy["CL_alpha_3d_kanat_per_deg"] = r_copy["CL_alpha_3d_per_deg"]
        r_copy["alpha_L0_deg"] = f"{float(vlm['alpha_L0_flapli_deg']):.4f}"
        if "e_oswald_kanat" in r_copy:
            r_copy["e_oswald_kanat"] = r_copy["e_oswald"]
        r_copy["CL_max_3d_tahmini"] = f"{float(vlm['CL_max_3d_flapli_tahmini']):.6f}"
        r_copy["alpha_stall_2d_deg"] = f"{float(vlm['alpha_L0_flapli_deg']) + float(vlm['CL_max_3d_flapli_tahmini']) / float(vlm['CL_alpha_3d_flapli_per_deg']):.6f}"
        r_copy["aero_modeli"] = "vlm_tek_egri_v1"
        if "CL" in r_copy and str(r_copy["CL"]).strip() != "":
            alpha_deg = sayi_al(r_copy, "alpha_deg", varsayilan=0.0)
            cl_yeni = min(float(vlm["CL_max_3d_flapli_tahmini"]), float(vlm["CL_alpha_3d_flapli_per_deg"]) * (
                alpha_deg - float(vlm["alpha_L0_flapli_deg"])
            ))
            r_copy["CL"] = f"{cl_yeni:.6f}"
        for alan in ("CL_flapsiz_kisim", "CL_flapli_kisim", "CL_kanat"):
            r_copy.pop(alan, None)
        uretilen.append(r_copy)
    return uretilen


def llt_ve_drag_tablolari_vlm_ile_guncelle(
    vlm_haritasi: Dict[Tuple[str, float, float, float], Dict[str, Any]],
    llt_satirlari: List[Dict[str, Any]],
    drag_polar_satirlari: List[Dict[str, Any]],
    drag_ozet_satirlari: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Geriye dönük uyumluluk için yalnızca flapsız 3 tabloyu döndüren yardımcı fonksiyon."""
    return _tek_mod_tablolari_guncelle(
        llt_satirlar=llt_satirlari,
        drag_polar_satirlar=drag_polar_satirlari,
        drag_ozet_satirlar=drag_ozet_satirlari,
        vlm_haritasi=vlm_haritasi,
        flapli_mi=False,
        sadece_vlm_adaylari=True,
    )


def llt_ve_drag_tablolarini_vlm_ile_guncelle(
    llt_flapsiz_satirlar: List[Dict[str, Any]],
    drag_polar_satirlar: List[Dict[str, Any]],
    drag_ozet_satirlar: List[Dict[str, Any]],
    vlm_haritasi: Dict[Tuple[str, float, float, float], Dict[str, Any]],
    llt_flapli_satirlar: Optional[List[Dict[str, Any]]] = None,
    drag_polar_flapli_satirlar: Optional[List[Dict[str, Any]]] = None,
    drag_ozet_flapli_satirlar: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Hem flapsız (tırmanış, seyir, yük bırakma) hem de flaplı (kalkış, iniş)
    LLT ve OpenVSP drag tablolarını VLM sonuçlarıyla günceller.
    """
    yeni_llt_flapsiz, yeni_polar_flapsiz, yeni_ozet_flapsiz = _tek_mod_tablolari_guncelle(
        llt_satirlar=llt_flapsiz_satirlar,
        drag_polar_satirlar=drag_polar_satirlar,
        drag_ozet_satirlar=drag_ozet_satirlar,
        vlm_haritasi=vlm_haritasi,
        flapli_mi=False,
        sadece_vlm_adaylari=True,
    )

    if llt_flapli_satirlar:
        yeni_llt_flapli, _, _ = _tek_mod_tablolari_guncelle(
            llt_satirlar=llt_flapli_satirlar,
            drag_polar_satirlar=[],
            drag_ozet_satirlar=[],
            vlm_haritasi=vlm_haritasi,
            flapli_mi=True,
            sadece_vlm_adaylari=True,
        )
    else:
        yeni_llt_flapli = _flapsizdan_flapli_llt_olustur(llt_flapsiz_satirlar, vlm_haritasi)

    kaynak_polar_flapli = drag_polar_flapli_satirlar if drag_polar_flapli_satirlar else drag_polar_satirlar
    kaynak_ozet_flapli = drag_ozet_flapli_satirlar if drag_ozet_flapli_satirlar else drag_ozet_satirlar

    _, yeni_polar_flapli, yeni_ozet_flapli = _tek_mod_tablolari_guncelle(
        llt_satirlar=llt_flapli_satirlar or yeni_llt_flapli,
        drag_polar_satirlar=kaynak_polar_flapli,
        drag_ozet_satirlar=kaynak_ozet_flapli,
        vlm_haritasi=vlm_haritasi,
        flapli_mi=True,
        sadece_vlm_adaylari=True,
    )

    return {
        "llt_flapsiz": yeni_llt_flapsiz,
        "drag_polar_flapsiz": yeni_polar_flapsiz,
        "drag_ozet_flapsiz": yeni_ozet_flapsiz,
        "llt_flapli": yeni_llt_flapli,
        "drag_polar_flapli": yeni_polar_flapli,
        "drag_ozet_flapli": yeni_ozet_flapli,
    }


# ---------------------------------------------------------------------------
# 4. Performans Modüllerinin VLM Çıktılarıyla Gerçekten Baştan Çalıştırılması
# ---------------------------------------------------------------------------

def _ilk_500_icin_kutle_tablosunu_suz(
    tum_kutle_satirlari: List[Dict[str, Any]],
    ilk_500_satirlar: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    `kanat_kuyruk_govde_kutleli.csv` içinden yalnızca `vlm_ilk_500_aday.csv`
    listesinde yer alan adayların kütle satırlarını süzer.
    """
    hedef_konfig_idler: Set[str] = set()
    hedef_aday_nolar: Set[str] = set()
    hedef_geometriler: Set[Tuple[str, float, float]] = set()

    for r in ilk_500_satirlar:
        konf_id = metin_al(r, "konfigurasyon_id", varsayilan="").strip()
        if konf_id:
            hedef_konfig_idler.add(konf_id)
        for id_kol in ("temel_aday_id", "aday_no", "aday_id"):
            deger = metin_al(r, id_kol, varsayilan="").strip()
            if deger:
                hedef_aday_nolar.add(deger)
        g_key = _geometri_anahtari(r)
        if g_key[0]:
            hedef_geometriler.add((g_key[0], round(g_key[1], 3), round(g_key[2], 3)))

    suzulen: List[Dict[str, Any]] = []
    for row in tum_kutle_satirlari:
        konf_id = metin_al(row, "konfigurasyon_id", varsayilan="").strip()
        aday_no = metin_al(row, "temel_aday_id", "aday_no", "aday_id", varsayilan="").strip()
        g_key = _geometri_anahtari(row)
        g_uclu = (g_key[0], round(g_key[1], 3), round(g_key[2], 3))

        if (
            (konf_id and konf_id in hedef_konfig_idler)
            or (aday_no and aday_no in hedef_aday_nolar)
            or (g_key[0] and g_uclu in hedef_geometriler)
        ):
            suzulen.append(row)

    return suzulen if suzulen else list(tum_kutle_satirlari)


@contextlib.contextmanager
def _seyir_dosya_yonlendirici(
    seyir_modulu: Any,
    yonlendirme_haritasi: Dict[str, Path],
) -> Iterator[None]:
    """
    `seyir_deneme.py` içindeki `find_csv_file` fonksiyonunu geçici olarak sarmalayarak
    sabit kodlanmış dosya adlarını (`kanat_kuyruk_govde_kutleli.csv`,
    `openvsp_llt_flapsiz_polar_detay.csv`, `kalkis_sonuclari.csv`, `tirmanis_sonuclari.csv` vb.)
    VLM sonuç dosyalarına yönlendirir.
    """
    orijinal_find: Optional[Callable[..., Any]] = getattr(seyir_modulu, "find_csv_file", None)
    if orijinal_find is None:
        yield
        return

    norm_harita = {Path(k).name.lower(): v for k, v in yonlendirme_haritasi.items()}

    def _vlm_find_csv_file(filename: str, required: bool = True) -> Any:
        aranan_ad = Path(str(filename)).name.lower()
        if aranan_ad in norm_harita and norm_harita[aranan_ad].is_file():
            return norm_harita[aranan_ad]
        return orijinal_find(filename, required=required)

    setattr(seyir_modulu, "find_csv_file", _vlm_find_csv_file)
    try:
        yield
    finally:
        setattr(seyir_modulu, "find_csv_file", orijinal_find)


def performans_modullerini_vlm_ile_calistir(
    vlm_klasoru: Path,
    vlm_dosya_yollari: Dict[str, Path],
    ilk_500_satirlar: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Path]:
    """
    Kalkış, tırmanış, seyir, yük bırakma ve iniş performans modüllerini
    VLM çıktılarıyla (kalkış ve iniş için flaplı VLM; tırmanış, seyir ve
    yük bırakma için flapsız VLM) gerçekten baştan çalıştırır.
    """
    import inis_hesabi_deneme  # type: ignore
    import kalkis_hesabi  # type: ignore
    import seyir_deneme  # type: ignore
    import tirmanis_hesabi  # type: ignore
    import yuk_birakma  # type: ignore

    kutle_kaynak_yolu = dosya_yolu_coz(args.kutle)
    itki_yolu = dosya_yolu_coz(args.itki)

    tum_kutle = csv_oku(kutle_kaynak_yolu)
    vlm_kutle_satirlari = _ilk_500_icin_kutle_tablosunu_suz(tum_kutle, ilk_500_satirlar)
    vlm_kutle_yolu = vlm_klasoru / "vlm_kutle_500_aday.csv"
    csv_yaz(vlm_kutle_satirlari, vlm_kutle_yolu)
    print(f"[VLM-MODÜL] Kütle tablosu {len(vlm_kutle_satirlari)} aday için süzüldü: {vlm_kutle_yolu}")

    drag_flapli_yolu = (
        vlm_dosya_yollari["drag_ozet_flapli"]
        if vlm_dosya_yollari["drag_ozet_flapli"].is_file()
        else vlm_dosya_yollari["drag_polar_flapli"]
    )
    drag_flapsiz_yolu = (
        vlm_dosya_yollari["drag_ozet_flapsiz"]
        if vlm_dosya_yollari["drag_ozet_flapsiz"].is_file()
        else vlm_dosya_yollari["drag_polar_flapsiz"]
    )

    cikti_kalkis = vlm_klasoru / "vlm_kalkis_sonuclari.csv"
    cikti_tirmanis = vlm_klasoru / "vlm_tirmanis_sonuclari.csv"
    cikti_seyir = vlm_klasoru / "vlm_seyir_optimizasyon_sonuclari.csv"
    cikti_yuk_birakma = vlm_klasoru / "vlm_yuk_birakma_sonuclari.csv"
    cikti_yuk_faz = vlm_klasoru / "vlm_yuk_birakma_sonuclari_fazlar.csv"
    cikti_yuk_saniye = vlm_klasoru / "vlm_yuk_birakma_sonuclari_saniyelik.csv"
    cikti_inis = vlm_klasoru / "vlm_inis_sonuclari.csv"

    # 1. Kalkış Modülü (Flaplı VLM ile)
    print("[VLM-MODÜL] 1/5 Kalkış modülü (flaplı VLM) çalıştırılıyor...")
    kalkis_argv = [
        "--kutle", str(vlm_kutle_yolu),
        "--llt", str(vlm_dosya_yollari["llt_flapli"]),
        "--drag", str(drag_flapli_yolu),
        "--itki", str(itki_yolu),
        "--cikti", str(cikti_kalkis),
    ]
    kalkis_rc = kalkis_hesabi.main(kalkis_argv)
    if kalkis_rc not in (0, None) or not cikti_kalkis.is_file():
        raise RuntimeError(f"kalkis_hesabi başarısız oldu (kod={kalkis_rc})")

    # 2. Tırmanış Modülü (Flapsız VLM ile)
    print("[VLM-MODÜL] 2/5 Tırmanış modülü (flapsız VLM) çalıştırılıyor...")
    tirmanis_argv = [
        "--kutle", str(vlm_kutle_yolu),
        "--llt", str(vlm_dosya_yollari["llt_flapsiz"]),
        "--drag", str(drag_flapsiz_yolu),
        "--itki", str(itki_yolu),
        "--cikti", str(cikti_tirmanis),
    ]
    tirmanis_rc = tirmanis_hesabi.main(tirmanis_argv)
    if tirmanis_rc not in (0, None) or not cikti_tirmanis.is_file():
        raise RuntimeError(f"tirmanis_hesabi başarısız oldu (kod={tirmanis_rc})")

    # 3. Seyir Modülü (Flapsız VLM + VLM Kalkış ve Tırmanış sonuçlarıyla)
    print("[VLM-MODÜL] 3/5 Seyir ve görev optimizasyon modülü (flapsız VLM) çalıştırılıyor...")
    seyir_yonlendirme = {
        "kanat_kuyruk_govde_kutleli.csv": vlm_kutle_yolu,
        "itki_tablosu.csv": itki_yolu,
        "llt_flapsiz_coklu_re.csv": vlm_dosya_yollari["llt_flapsiz"],
        "openvsp_llt_flapsiz_polar_detay.csv": vlm_dosya_yollari["drag_polar_flapsiz"],
        "openvsp_llt_flapsiz_aday_ozet.csv": vlm_dosya_yollari["drag_ozet_flapsiz"],
        "kalkis_sonuclari.csv": cikti_kalkis,
        "tirmanis_sonuclari.csv": cikti_tirmanis,
    }
    with _seyir_dosya_yonlendirici(seyir_deneme, seyir_yonlendirme):
        seyir_deneme.run_full_cruise_mission_optimization(
            tail_csv_path=str(vlm_dosya_yollari["llt_flapsiz"]),
            llt_csv_path=str(vlm_dosya_yollari["llt_flapsiz"]),
            strict_re=False,
            output_path=str(cikti_seyir),
        )
    if not cikti_seyir.is_file():
        raise RuntimeError("seyir_deneme çıktı dosyası oluşturulamadı")

    # 4. Yük Bırakma Modülü (Flapsız VLM ile)
    print("[VLM-MODÜL] 4/5 Yük bırakma modülü (flapsız VLM) çalıştırılıyor...")
    yuk_argv = [
        "--kutle", str(vlm_kutle_yolu),
        "--llt", str(vlm_dosya_yollari["llt_flapsiz"]),
        "--drag", str(drag_flapsiz_yolu),
        "--itki", str(itki_yolu),
        "--cikti", str(cikti_yuk_birakma),
        "--faz-cikti", str(cikti_yuk_faz),
        "--saniye-cikti", str(cikti_yuk_saniye),
    ]
    yuk_rc = yuk_birakma.main(yuk_argv)
    if yuk_rc not in (0, None) or not cikti_yuk_birakma.is_file():
        raise RuntimeError(f"yuk_birakma başarısız oldu (kod={yuk_rc})")

    # 5. İniş Modülü (Flaplı + Flapsız VLM ile)
    print("[VLM-MODÜL] 5/5 İniş modülü (flaplı + flapsız VLM) çalıştırılıyor...")
    inis_argv = [
        "--kutle", str(vlm_kutle_yolu),
        "--llt-flap", str(vlm_dosya_yollari["llt_flapli"]),
        "--drag-flap", str(drag_flapli_yolu),
        "--llt-flapsiz", str(vlm_dosya_yollari["llt_flapsiz"]),
        "--drag-flapsiz", str(drag_flapsiz_yolu),
        "--cikti", str(cikti_inis),
    ]
    inis_rc = inis_hesabi_deneme.main(inis_argv)
    if inis_rc not in (0, None) or not cikti_inis.is_file():
        raise RuntimeError(f"inis_hesabi_deneme başarısız oldu (kod={inis_rc})")

    return {
        "kutle": vlm_kutle_yolu,
        "kalkis": cikti_kalkis,
        "tirmanis": cikti_tirmanis,
        "seyir": cikti_seyir,
        "yuk_birakma": cikti_yuk_birakma,
        "inis": cikti_inis,
    }


def _aday_kimlik_anahtari(row: Dict[str, Any]) -> Tuple[str, str]:
    """Aday satırlarını LLT ve VLM sıralamaları arasında eşleştirmek için kimlik anahtarı üretir."""
    konf_id = metin_al(row, "konfigurasyon_id", varsayilan="").strip().lower()
    if konf_id:
        return ("konf", konf_id)
    aday_id = metin_al(row, "temel_aday_id", "aday_no", "aday_id", varsayilan="").strip().lower()
    pervane = metin_al(row, "pervane", varsayilan="").strip().lower()
    if aday_id:
        return ("aday_perv", f"{aday_id}::{pervane}")
    g_key = _geometri_anahtari(row)
    return ("geom_perv", f"{g_key[0]}_{g_key[1]}_{g_key[2]}_{g_key[3]}::{pervane}")


def _analitik_vlm_yeniden_puanla(
    ilk_500_satirlar: List[Dict[str, Any]],
    vlm_haritasi: Dict[Tuple[str, float, float, float], Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """
    Yalnızca `--analitik-olcekleme` bayrağı verildiğinde veya performans modülleri
    çağrılamadığında kullanılan yedek analitik yeniden puanlama fonksiyonu.
    Kalkış ve iniş için flaplı VLM katsayılarını, diğer fazlar için flapsız VLM
    katsayılarını kullanır.
    """
    yeniden_hesaplanan: List[Dict[str, Any]] = []

    for idx, row in enumerate(ilk_500_satirlar, start=1):
        g_key = _geometri_anahtari(row)
        vlm = _vlm_eslesmesi_bul(vlm_haritasi, g_key) or {}
        kopya = dict(row)

        eski_sira = int(sayi_al(row, "vlm_sira", "sira", varsayilan=float(idx)))
        eski_toplam_puan = sayi_al(row, "toplam_performans_puani", varsayilan=0.0)
        eski_m2_puan = sayi_al(row, "mission2_400_puani", varsayilan=0.0)

        e_llt = sayi_al(row, "e_oswald", varsayilan=0.84)
        ld_yuklu_llt = sayi_al(row, "LD_yuklu_duz", varsayilan=12.0)
        ld_bos_llt = sayi_al(row, "LD_bos_duz", varsayilan=12.0)

        e_vlm = float(vlm.get("e_oswald", e_llt))
        cl_alpha_vlm = float(vlm.get("CL_alpha_3d_per_deg", 0.078))
        cl_max_vlm = float(vlm.get("CL_max_3d_tahmini", 1.35))
        e_orani = float(vlm.get("e_oswald_orani", 1.0))
        cl_max_flap_orani = float(vlm.get("cl_max_flapli_orani", vlm.get("cl_max_orani", 1.0)))
        cozucu_adi = str(vlm.get("cozucu", "numpy_3d_vlm"))

        cdi_carpani = 1.0 / max(e_orani, 0.5)
        toplam_drag_carpani = 0.60 + 0.40 * cdi_carpani

        ld_yuklu_vlm = round(ld_yuklu_llt / max(toplam_drag_carpani, 0.5), 4)
        ld_bos_vlm = round(ld_bos_llt / max(toplam_drag_carpani, 0.5), 4)

        kalkis_mesafe_llt = sayi_al(row, "kalkis_mesafe_m", "M2_S_G_m", varsayilan=15.0)
        inis_mesafe_llt = sayi_al(row, "inis_mesafe_m", "S_LDG_toplam_m", varsayilan=18.0)

        kalkis_mesafe_vlm = round(kalkis_mesafe_llt / max(cl_max_flap_orani, 0.7), 3)
        inis_mesafe_vlm = round(inis_mesafe_llt / max(cl_max_flap_orani, 0.7), 3)

        kalkis_20m_vlm = kalkis_mesafe_vlm <= 20.0
        inis_20m_vlm = inis_mesafe_vlm <= 20.0

        puan_kalkis_60_llt = sayi_al(row, "resmi_kalkis_puani_60", varsayilan=60.0)
        puan_kalkis_m2_20_llt = sayi_al(row, "puan_M2_yuklu_kalkis_20", varsayilan=20.0)
        if kalkis_20m_vlm:
            puan_kalkis_60_vlm = max(puan_kalkis_60_llt, 60.0) if puan_kalkis_60_llt >= 45.0 else puan_kalkis_60_llt
            puan_kalkis_m2_20_vlm = 20.0
        else:
            puan_kalkis_m2_20_vlm = 5.0
            puan_kalkis_60_vlm = max(15.0, puan_kalkis_60_llt - (puan_kalkis_m2_20_llt - 5.0))

        puan_inis_140_llt = sayi_al(row, "resmi_inis_puani_140", varsayilan=140.0)
        puan_inis_m2_60_llt = sayi_al(row, "puan_M2_inis_60", varsayilan=60.0)
        if inis_20m_vlm:
            puan_inis_m2_60_vlm = 60.0
            puan_inis_140_vlm = 140.0 if puan_inis_140_llt >= 100.0 else puan_inis_140_llt
        else:
            puan_inis_m2_60_vlm = 40.0 if puan_inis_m2_60_llt >= 40.0 else puan_inis_m2_60_llt
            puan_inis_140_vlm = 100.0 if puan_inis_140_llt >= 100.0 else puan_inis_140_llt

        kalkis_mah = sayi_al(row, "kalkis_tuketim_mAh", varsayilan=0.0) / max(cl_max_flap_orani, 0.7)
        tirmanis_mah = sayi_al(row, "tirmanis_tuketim_mAh", varsayilan=0.0) * (0.75 + 0.25 * toplam_drag_carpani)
        seyir_mah = sayi_al(row, "seyir_tuketim_mAh", varsayilan=0.0) * toplam_drag_carpani
        yuk_mah = sayi_al(row, "yuk_birakma_tuketim_mAh", varsayilan=0.0) * (0.75 + 0.25 * toplam_drag_carpani)

        toplam_mah_vlm = round(kalkis_mah + tirmanis_mah + seyir_mah + yuk_mah + float(args.inis_ek_butce_mah), 3)
        toplam_enerji_wh_vlm = round(toplam_mah_vlm * GERILIM_V / 1000.0, 4)
        kalan_batarya_vlm = round(100.0 * (BATARYA_KAPASITE_MAH - toplam_mah_vlm) / BATARYA_KAPASITE_MAH, 2)

        batarya_uygun_vlm = toplam_mah_vlm <= KULLANILABILIR_MAH
        kalkis_uygun_vlm = bool_al(row, "kalkis_uygun", varsayilan=True) and (
            kalkis_20m_vlm if args.kalkis_20m_zorunlu else True
        )
        inis_uygun_vlm = bool_al(row, "inis_uygun", varsayilan=True) and (
            inis_20m_vlm if args.inis_20m_zorunlu else True
        )
        tum_fazlar_uygun_vlm = (
            kalkis_uygun_vlm
            and bool_al(row, "tirmanis_uygun", varsayilan=True)
            and bool_al(row, "seyir_uygun", varsayilan=True)
            and bool_al(row, "yuk_birakma_uygun", varsayilan=True)
            and inis_uygun_vlm
            and batarya_uygun_vlm
            and bool_al(row, "sure_uygun", varsayilan=True)
        )

        su_kg = sayi_al(row, "su_kutlesi_kg", varsayilan=0.0)
        ham_enerji_endeksi_vlm = (
            (su_kg * 20.0) / max(toplam_enerji_wh_vlm, 1e-6)
            if su_kg > 0.0
            else sayi_al(row, "ham_enerji_endeksi", varsayilan=0.0) / max(toplam_drag_carpani, 0.5)
        )

        kopya.update(
            {
                "vlm_cozucu": cozucu_adi,
                "llt_onc_sira": eski_sira,
                "llt_sira": eski_sira,
                "e_oswald_llt": e_llt,
                "e_oswald_vlm": e_vlm,
                "CL_alpha_3d_vlm": cl_alpha_vlm,
                "CL_max_3d_vlm": cl_max_vlm,
                "b_flap_m_vlm": vlm.get("b_flap_m", ""),
                "flap_aciklik_orani_vlm": vlm.get("flap_aciklik_orani", ""),
                "CL_alpha_3d_flapli_vlm": vlm.get("CL_alpha_3d_flapli_per_deg", ""),
                "alpha_L0_flapli_vlm": vlm.get("alpha_L0_flapli_deg", ""),
                "e_oswald_flapli_vlm": vlm.get("e_oswald_flapli", ""),
                "CL_max_3d_flapli_vlm": vlm.get("CL_max_3d_flapli_tahmini", ""),
                "LD_yuklu_duz_vlm": ld_yuklu_vlm,
                "LD_bos_duz_vlm": ld_bos_vlm,
                "kalkis_mesafe_m_vlm": kalkis_mesafe_vlm,
                "kalkis_20m_icinde_vlm": kalkis_20m_vlm,
                "inis_mesafe_m_vlm": inis_mesafe_vlm,
                "inis_20m_icinde_vlm": inis_20m_vlm,
                "gercek_toplam_tuketim_mAh_vlm": toplam_mah_vlm,
                "kalan_batarya_yuzde_vlm": kalan_batarya_vlm,
                "gercek_toplam_enerji_Wh_vlm": toplam_enerji_wh_vlm,
                "tum_fazlar_uygun_vlm": tum_fazlar_uygun_vlm,
                "resmi_kalkis_puani_60_vlm": round(puan_kalkis_60_vlm, 4),
                "puan_M2_yuklu_kalkis_20_vlm": round(puan_kalkis_m2_20_vlm, 4),
                "resmi_inis_puani_140_vlm": round(puan_inis_140_vlm, 4),
                "puan_M2_inis_60_vlm": round(puan_inis_m2_60_vlm, 4),
                "ham_enerji_endeksi_vlm": round(ham_enerji_endeksi_vlm, 6),
                "llt_onc_toplam_puan": eski_toplam_puan,
                "llt_onc_mission2_400_puani": eski_m2_puan,
            }
        )
        yeniden_hesaplanan.append(kopya)

    uygun_endeksler = [
        float(r["ham_enerji_endeksi_vlm"])
        for r in yeniden_hesaplanan
        if r["tum_fazlar_uygun_vlm"] and float(r["ham_enerji_endeksi_vlm"]) > 0.0
    ]
    if not uygun_endeksler:
        uygun_endeksler = [
            float(r["ham_enerji_endeksi_vlm"])
            for r in yeniden_hesaplanan
            if float(r["ham_enerji_endeksi_vlm"]) > 0.0
        ]
    best_index_vlm = max(uygun_endeksler) if uygun_endeksler else 1.0

    for r in yeniden_hesaplanan:
        puan_enerji_vlm = round(60.0 * float(r["ham_enerji_endeksi_vlm"]) / max(best_index_vlm, 1e-9), 4)
        puan_nav = sayi_al(r, "puan_navigasyon_140", "puan_navigasyon", varsayilan=0.0)
        puan_yuk = sayi_al(r, "puan_yuk_birakma_120", varsayilan=0.0)

        toplam_puan_vlm = round(
            float(r["resmi_kalkis_puani_60_vlm"])
            + puan_nav
            + puan_enerji_vlm
            + puan_yuk
            + float(r["resmi_inis_puani_140_vlm"]),
            4,
        )
        mission2_puan_vlm = round(
            float(r["puan_M2_yuklu_kalkis_20_vlm"])
            + puan_nav
            + puan_enerji_vlm
            + puan_yuk
            + float(r["puan_M2_inis_60_vlm"]),
            4,
        )

        r["puan_enerji_60_vlm"] = puan_enerji_vlm
        r["vlm_sonrasi_toplam_puan"] = toplam_puan_vlm
        r["vlm_sonrasi_mission2_400_puani"] = mission2_puan_vlm
        r["puan_farki_vlm_eksi_llt"] = round(toplam_puan_vlm - float(r["llt_onc_toplam_puan"]), 4)

    puan_anahtari = (
        "vlm_sonrasi_mission2_400_puani"
        if args.siralama_puani == "mission2"
        else "vlm_sonrasi_toplam_puan"
    )
    yeniden_hesaplanan.sort(
        key=lambda r: (
            0 if r["tum_fazlar_uygun_vlm"] else 1,
            -float(r.get(puan_anahtari, 0.0) or 0.0),
            -sayi_al(r, "tasarim_kalkis_puani_100", varsayilan=0.0),
            float(r.get("gercek_toplam_enerji_Wh_vlm", 1e9) or 1e9),
            sayi_al(r, "toplam_kutle_kg", varsayilan=99.0),
        )
    )

    for yeni_sira, r in enumerate(yeniden_hesaplanan, start=1):
        r["vlm_nihai_sira"] = yeni_sira
        r["sira_degisimi"] = int(r["llt_onc_sira"]) - yeni_sira

    return yeniden_hesaplanan


def vlm_adaylarini_yeniden_puanla(
    ilk_500_satirlar: List[Dict[str, Any]],
    vlm_haritasi: Dict[Tuple[str, float, float, float], Dict[str, Any]],
    args: argparse.Namespace,
    vlm_klasoru: Optional[Path] = None,
    vlm_dosya_yollari: Optional[Dict[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """
    Güncellenen flapsız ve flaplı VLM aerodinamik tablolarıyla tüm performans
    modüllerini (`kalkis_hesabi`, `tirmanis_hesabi`, `seyir_deneme`, `yuk_birakma`,
    `inis_hesabi_deneme`) gerçekten baştan çalıştırır ve `aday_siralama` üzerinden
    nihai VLM puanlamasını üretir.
    """
    if getattr(args, "analitik_olcekleme", False) or vlm_klasoru is None or vlm_dosya_yollari is None:
        print("[VLM] Analitik ölçekleme modu ile yeniden puanlama yapılıyor...")
        return _analitik_vlm_yeniden_puanla(ilk_500_satirlar, vlm_haritasi, args)

    try:
        modul_ciktilari = performans_modullerini_vlm_ile_calistir(
            vlm_klasoru=vlm_klasoru,
            vlm_dosya_yollari=vlm_dosya_yollari,
            ilk_500_satirlar=ilk_500_satirlar,
            args=args,
        )
    except Exception as exc:
        print(
            f"[UYARI] Performans modülleri doğrudan çalıştırılırken hata oluştu ({exc}); "
            "yedek analitik VLM yeniden puanlama moduna geçiliyor."
        )
        return _analitik_vlm_yeniden_puanla(ilk_500_satirlar, vlm_haritasi, args)

    llt_gecmis_haritasi: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for idx, r in enumerate(ilk_500_satirlar, start=1):
        kimlik = _aday_kimlik_anahtari(r)
        llt_gecmis_haritasi[kimlik] = {
            "llt_sira": int(sayi_al(r, "vlm_sira", "sira", varsayilan=float(idx))),
            "llt_toplam_puan": sayi_al(r, "toplam_performans_puani", varsayilan=0.0),
            "llt_mission2_puan": sayi_al(r, "mission2_400_puani", varsayilan=0.0),
            "e_oswald_llt": sayi_al(r, "e_oswald", varsayilan=0.84),
        }

    siralama_argv = [
        "--kalkis", str(modul_ciktilari["kalkis"]),
        "--tirmanis", str(modul_ciktilari["tirmanis"]),
        "--seyir", str(modul_ciktilari["seyir"]),
        "--yuk-birakma", str(modul_ciktilari["yuk_birakma"]),
        "--inis", str(modul_ciktilari["inis"]),
        "--kutle", str(modul_ciktilari["kutle"]),
        "--ilk-n", str(len(ilk_500_satirlar) or 500),
        "--siralama-puani", str(args.siralama_puani),
        "--inis-ek-butce-mah", str(args.inis_ek_butce_mah),
    ]
    if args.kalkis_20m_zorunlu:
        siralama_argv.append("--kalkis-20m-zorunlu")
    if args.inis_20m_zorunlu:
        siralama_argv.append("--inis-20m-zorunlu")

    siralama_args = siralama_argumanlari_coz(siralama_argv)
    vlm_sirali_tum, _ = adaylari_degerlendir_ve_sirala(siralama_args)

    nihai_liste: List[Dict[str, Any]] = []
    for yeni_sira, row in enumerate(vlm_sirali_tum, start=1):
        g_key = _geometri_anahtari(row)
        vlm = _vlm_eslesmesi_bul(vlm_haritasi, g_key) or {}
        kimlik = _aday_kimlik_anahtari(row)
        eski_bilgi = llt_gecmis_haritasi.get(kimlik, {})

        eski_sira = int(eski_bilgi.get("llt_sira", yeni_sira))
        eski_toplam_puan = float(eski_bilgi.get("llt_toplam_puan", 0.0))
        eski_m2_puan = float(eski_bilgi.get("llt_mission2_puan", 0.0))
        e_llt = float(eski_bilgi.get("e_oswald_llt", vlm.get("e_oswald_llt", 0.84)))

        yeni_toplam_puan = sayi_al(row, "toplam_performans_puani", varsayilan=0.0)
        yeni_m2_puan = sayi_al(row, "mission2_400_puani", varsayilan=0.0)
        toplam_mah_vlm = sayi_al(row, "gercek_toplam_tuketim_mAh", varsayilan=0.0)
        kalan_batarya_vlm = sayi_al(row, "kalan_batarya_yuzde", varsayilan=0.0)
        toplam_wh_vlm = sayi_al(row, "gercek_toplam_enerji_Wh", varsayilan=0.0)
        tum_uygun_vlm = bool_al(row, "tum_fazlar_uygun", varsayilan=False)

        r_copy = dict(row)
        r_copy.update(
            {
                "vlm_nihai_sira": yeni_sira,
                "llt_onc_sira": eski_sira,
                "llt_sira": eski_sira,
                "sira_degisimi": eski_sira - yeni_sira,
                "vlm_cozucu": vlm.get("cozucu", ""),
                "vlm_cozucu_flapli": vlm.get("cozucu_flapli", ""),          # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson]
                "e_kaynagi_vlm": vlm.get("e_kaynagi", ""),                  # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson]
                "e_span_ham_vlm": vlm.get("e_span_vspaero_ham", ""),        # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson]
                "e_yuzey_vlm": vlm.get("e_yuzey", ""),                      # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson]
                "richardson_vlm": vlm.get("richardson", ""),                # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson]
                "vlm_kalite": vlm.get("kalite", ""),                        # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson]
                "vlm_govde_dahil": vlm.get("govde_dahil", ""),              # [DEĞİŞİKLİK 2026-11-19 v9 | tam akışta gövde]
                "vlm_kanat_x_ceyrek_m": vlm.get("kanat_x_ceyrek_m", ""),    # [DEĞİŞİKLİK 2026-11-19 v9 | tam akışta gövde]
                "vlm_kanat_z_m": vlm.get("kanat_z_m", ""),                  # [DEĞİŞİKLİK 2026-11-19 v9 | tam akışta gövde]
                "vlm_Cm_alpha_per_deg": vlm.get("Cm_alpha_per_deg", ""),    # [DEĞİŞİKLİK 2026-11-19 v9 | tam akışta gövde]
                "alpha_L0_flapli_ham_vlm": vlm.get("alpha_L0_flapli_ham_deg", ""),  # [DEĞİŞİKLİK 2026-11-19 v10 | eğri tutarlılığı]
                "flap_verimi_vlm": vlm.get("flap_verimi", ""),                   # [DEĞİŞİKLİK 2026-11-19 v10 | eğri tutarlılığı]
                "egri_yontemi_vlm": EGRI_YONTEMI,                                # [DEĞİŞİKLİK 2026-11-19 v10 | eğri tutarlılığı]
                "CL_max_kaynak": vlm.get("CL_max_kaynak", ""),              # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson]
                "vsp3_flapsiz": vlm.get("vsp3_flapsiz", ""),                # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson]
                "vsp3_flapli": vlm.get("vsp3_flapli", ""),                  # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson]
                "e_oswald_llt": e_llt,
                "CL_alpha_3d_vlm": vlm.get("CL_alpha_3d_per_deg", ""),
                "alpha_L0_vlm": vlm.get("alpha_L0_deg", ""),
                "e_oswald_vlm": vlm.get("e_oswald", ""),
                "CL_max_3d_vlm": vlm.get("CL_max_3d_tahmini", ""),
                "b_flap_m_vlm": vlm.get("b_flap_m", ""),
                "flap_aciklik_orani_vlm": vlm.get("flap_aciklik_orani", ""),
                "CL_alpha_3d_flapli_vlm": vlm.get("CL_alpha_3d_flapli_per_deg", ""),
                "alpha_L0_flapli_vlm": vlm.get("alpha_L0_flapli_deg", ""),
                "e_oswald_flapli_vlm": vlm.get("e_oswald_flapli", ""),
                "CL_max_3d_flapli_vlm": vlm.get("CL_max_3d_flapli_tahmini", ""),
                "gercek_toplam_tuketim_mAh_vlm": round(toplam_mah_vlm, 3),
                "kalan_batarya_yuzde_vlm": round(kalan_batarya_vlm, 2),
                "gercek_toplam_enerji_Wh_vlm": round(toplam_wh_vlm, 4),
                "tum_fazlar_uygun_vlm": tum_uygun_vlm,
                "vlm_sonrasi_toplam_puan": round(yeni_toplam_puan, 4),
                "vlm_sonrasi_mission2_400_puani": round(yeni_m2_puan, 4),
                "llt_onc_toplam_puan": round(eski_toplam_puan, 4),
                "llt_onc_mission2_400_puani": round(eski_m2_puan, 4),
                "puan_farki_vlm_eksi_llt": round(yeni_toplam_puan - eski_toplam_puan, 4),
            }
        )
        nihai_liste.append(r_copy)

    return nihai_liste


# ---------------------------------------------------------------------------
# 5. CLI Argümanları ve Ana Akış (`main`)
# ---------------------------------------------------------------------------

def argumanlari_coz(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "İlk 500 aday için flapsız + flaplı 3B VLM analizi yapar ve tüm performans "
            "modüllerini VLM çıktılarıyla baştan çalıştırarak yeniden puanlar."
        )
    )
    parser.add_argument(
        "--girdi",
        default="vlm_ilk_500_aday.csv",
        help="İlk 500 aday girdi CSV dosyası (varsayılan: vlm_ilk_500_aday.csv)",
    )
    parser.add_argument(
        "--kutle",
        default="kanat_kuyruk_govde_kutleli.csv",
        help="Aday kütle CSV dosyası (varsayılan: kanat_kuyruk_govde_kutleli.csv)",
    )
    parser.add_argument(
        "--itki",
        default="itki_tablosu.csv",
        help="Pervane/motor itki CSV dosyası (varsayılan: itki_tablosu.csv)",
    )
    parser.add_argument(
        "--llt-flapsiz",
        dest="llt_flapsiz",
        default="llt_flapsiz_coklu_re.csv",
        help="Flapsız LLT çoklu Reynolds CSV dosyası",
    )
    parser.add_argument(
        "--drag-polar",
        dest="drag_polar",
        default="openvsp_llt_flapsiz_polar_detay.csv",
        help="Flapsız OpenVSP/LLT drag polar detay CSV dosyası",
    )
    parser.add_argument(
        "--drag-ozet",
        dest="drag_ozet",
        default="openvsp_llt_flapsiz_aday_ozet.csv",
        help="Flapsız OpenVSP/LLT drag aday özet CSV dosyası",
    )
    parser.add_argument(
        "--llt-flapli",
        dest="llt_flapli",
        default="llt_flapli_coklu_re.csv",
        help="Kalkış ve iniş için flaplı LLT çoklu Reynolds CSV dosyası",
    )
    parser.add_argument(
        "--drag-polar-flapli",
        dest="drag_polar_flapli",
        default="openvsp_llt_flapli_polar_detay.csv",
        help="Kalkış ve iniş için flaplı OpenVSP/LLT drag polar detay CSV dosyası",
    )
    parser.add_argument(
        "--drag-ozet-flapli",
        dest="drag_ozet_flapli",
        default="openvsp_llt_flapli_aday_ozet.csv",
        help="Kalkış ve iniş için flaplı OpenVSP/LLT drag aday özet CSV dosyası",
    )
    parser.add_argument(
        "--cikti",
        default="vlm_nihai_ucak_siralamasi.csv",
        help="VLM sonrası nihai uçak sıralaması çıktı CSV dosyası",
    )
    parser.add_argument(
        "--vlm-klasoru",
        dest="vlm_klasoru",
        default="vlm_sonuclari",
        help="Ara VLM aerodinamik ve performans CSV dosyalarının kaydedileceği klasör",
    )
    parser.add_argument(
        "--cozucu",
        choices=("auto", "vspaero", "aerosandbox", "numpy_vlm"),
        default="auto",
        help="Tercih edilen 3B VLM çözücü motoru (varsayılan: auto)",
    )
    parser.add_argument(
        "--n-span",
        dest="n_span",
        type=int,
        default=24,
        help="Kanat açıklığı yönündeki VLM panel sayısı (varsayılan: 24)",
    )
    parser.add_argument(
        "--n-chord",
        dest="n_chord",
        type=int,
        default=4,
        help="Kanat veteri yönündeki VLM panel sayısı (varsayılan: 4)",
    )
    parser.add_argument(
        "--siralama-puani",
        dest="siralama_puani",
        choices=("modul_toplamlari", "mission2"),
        default="modul_toplamlari",
        help="Sıralamada baz alınacak ana puan türü",
    )
    parser.add_argument(
        "--kalkis-20m-zorunlu",
        dest="kalkis_20m_zorunlu",
        action="store_true",
        default=True,
        help="20 m kalkış şartını uygunluk kriteri olarak zorunlu tut",
    )
    parser.add_argument(
        "--inis-20m-zorunlu",
        dest="inis_20m_zorunlu",
        action="store_true",
        default=False,
        help="20 m iniş şartını uygunluk kriteri olarak zorunlu tut",
    )
    parser.add_argument(
        "--inis-ek-butce-mah",
        dest="inis_ek_butce_mah",
        type=float,
        default=0.0,
        help="İniş fazı için ek batarya tüketim bütçesi (mAh)",
    )
    # [DEĞİŞİKLİK 2026-11-18 | Görev A] Görev A için yeni komut satırı seçenekleri
    parser.add_argument(
        "--vspaero-zorunlu",
        dest="vspaero_zorunlu",
        action="store_true",
        default=False,
        help="VSPAERO herhangi bir geometride hata verirse NumPy'a geçmeden programı durdur.",
    )
    parser.add_argument(
        "--vspaero-kuyruk",
        dest="vspaero_kuyruk",
        action="store_true",
        default=False,
        help="VSPAERO modeline yatay kuyruğu da ekle (varsayılan: sadece kanat).",
    )
    parser.add_argument(
        "--vlm-test",
        dest="vlm_test",
        type=int,
        default=0,
        help="Sadece ilk N geometriyi seçilen çözücü + NumPy ile çöz, karşılaştırma CSV'si yaz ve çık "
             "(performans modülleri çalışmaz, ana çıktı dosyalarına dokunulmaz).",
    )
    # [DEĞİŞİKLİK 2026-11-18 v5 | gövde] Gövde ve panel seçenekleri
    parser.add_argument(
        "--vspaero-govde",
        dest="vspaero_govde",
        action="store_true",
        default=False,
        help="[DEĞİŞİKLİK 2026-11-19 v9 | tam akışta gövde] VSPAERO modeline gövdeyi ekle. --vlm-test N ile: gövdeli/gövdesiz KARŞILAŞTIRMA testi. "
             "Tek başına (tam akış): HER adaya gövde eklenir ve tablolara KANAT+GÖVDE katsayıları gider. "
             "Bilinmeyenler: --kanat-x, --kanat-z, --govde-gomme, --zcg, --flap-baslangic, --govde-l-aviyonik, "
             "--govde-l-kuyruk, --govde-d-min, --e-viskoz-carpan.",
    )
    parser.add_argument(
        "--kanat-x",
        dest="kanat_x",
        type=float,
        default=None,
        help="Kanat 1/4 veterinin burundan uzaklığı [m]. Verilmezse su haznesinin ortası.",
    )
    parser.add_argument(
        "--kanat-z",
        dest="kanat_z",
        default="alt_gomulu",  # [DEĞİŞİKLİK 2026-11-18 v7 | ekip kararı: kanat yeri] ekip: kanat gövdenin içinde, alt yüzeye olabildiğince yakın (eskiden "ust")
        help="[DEĞİŞİKLİK 2026-11-18 v6 | inceleme-2 düzeltmeleri] Kanadın gövdeye göre dikey yeri: ust (gerçek yüksek kanat) | ust_gomulu | orta | alt "
             "(gerçek alçak kanat) | alt_gomulu | hepsi | virgüllü liste (ör. ust,alt) veya sayı [m]. "
             "Konum profil kalınlığından hesaplanır (bkz. vspaero_cozucu.kanat_z_hesapla).",
    )
    parser.add_argument(
        "--govde-gomme",
        dest="govde_gomme",
        type=float,
        default=None,
        help="[DEĞİŞİKLİK 2026-11-18 v6 | inceleme-2 düzeltmeleri] Kanadın gövdeye gömülme payı [m] (varsayılan 0.005).",
    )
    parser.add_argument(
        "--zcg",
        dest="zcg",
        default=None,
        help="[DEĞİŞİKLİK 2026-11-18 v6 | inceleme-2 düzeltmeleri] Moment referans yüksekliği: sayı [m] (gövde ekseni = 0, varsayılan) veya 'kanat'.",
    )
    parser.add_argument(
        "--flap-baslangic",
        dest="flap_baslangic",
        choices=("merkez", "govde_kenari"),
        default="merkez",  # [DEĞİŞİKLİK 2026-11-19 v11 | flap ve k diğer kodlardan] kanat_boyutlandirma.py ile aynı: flap kanadın ortasında, y ∈ [−b_flap/2, +b_flap/2] (v8'de govde_kenari idi)
        help="[DEĞİŞİKLİK 2026-11-18 v6 | inceleme-2 düzeltmeleri] Flabın başladığı yer (flap boyu 1 m sabit; govde_kenari'nda flap dışa kayar). Varsayılan: merkez (kanat_boyutlandirma.py / LLT ile aynı).",  # [DEĞİŞİKLİK 2026-11-19 v11 | flap ve k diğer kodlardan]  # [DEĞİŞİKLİK 2026-11-19 v10 | eğri tutarlılığı] yardım metni düzeltildi
    )
    parser.add_argument(
        "--vspaero-richardson",
        dest="vspaero_richardson",
        default="evet",
        help="[DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson] e için iki ağlı Richardson: 'evet' (VARSAYILAN: ince = --vspaero-span-panel, kaba = yarısı; "
             "48 -> 24/48), 'kaba,ince' (ör. 32,64) veya 'yok' (tek ağ). Süreyi ~1.5 katına çıkarır.",
    )
    parser.add_argument(
        "--vspaero-e-kaynagi",
        dest="vspaero_e_kaynagi",
        choices=("iz", "yuzey"),
        default="iz",
        help="[DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson] e kaynağı: iz (Trefftz, CLiw/CDiw — önerilen) veya yuzey (eski; panel arttıkça ıraksıyor).",
    )
    parser.add_argument(
        "--vspaero-veter-tess",
        dest="vspaero_veter_tess",
        type=int,
        default=None,
        help="[DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson] Kanat veter yönü panel ayarı (OpenVSP Tess_W; varsayılan 33). Test değerleri: 17, 33, 65.",
    )
    parser.add_argument(
        "--govde-l-aviyonik", dest="govde_l_aviyonik", type=float, default=0.30,
        help="[DEĞİŞİKLİK 2026-11-19 v9 | tam akışta gövde] Gövde aviyonik silindiri boyu [m] (govde_boyutlandirma.py: 0.30).",
    )
    parser.add_argument(
        "--govde-l-kuyruk", dest="govde_l_kuyruk", type=float, default=1.00,
        help="[DEĞİŞİKLİK 2026-11-19 v9 | tam akışta gövde] Gövde kuyruk konisi boyu [m] (govde_boyutlandirma.py: 1.0).",
    )
    parser.add_argument(
        "--govde-d-min", dest="govde_d_min", type=float, default=0.08,
        help="[DEĞİŞİKLİK 2026-11-19 v9 | tam akışta gövde] Kuyruk konisi uç çapı [m] (govde_boyutlandirma.py: 0.08).",
    )
    parser.add_argument(
        "--e-viskoz-carpan", dest="e_viskoz_carpan", type=float, default=None,  # [DEĞİŞİKLİK 2026-11-19 v11 | flap ve k diğer kodlardan] None = gövde modelliyse 1.0, değilse 0.995
        help="[DEĞİŞİKLİK 2026-11-19 v9 | tam akışta gövde] e düzeltmesindeki k: 1/e = 1/(k·e_span) + 0.007·π·AR. Gövde VSPAERO'da modellenince "
             "Verilmezse: gövde VSPAERO'da modelliyse 1.0, değilse 0.995 (lifting_line_cozucu.py'de k_fus=0.995 'gövde blokajı düzeltmesi'; "
             "gövde modellenince iki kez sayılmasın).",
    )
    parser.add_argument(
        "--vspaero-uc-sik",
        dest="vspaero_uc_sik",
        type=float,
        default=None,
        help="[DEĞİŞİKLİK 2026-11-18 v6 | inceleme-2 düzeltmeleri] Kanat ucuna doğru panel sıklaştırma (OutCluster, ör. 0.5). Varsayılan: OpenVSP varsayılanı.",
    )
    parser.add_argument(
        "--vspaero-yontem",
        dest="vspaero_yontem",
        choices=("vlm", "panel", "hepsi"),
        default="vlm",
        help="Gövde testinde VSPAERO yöntemi: vlm (ince yüzey), panel (kalın yüzey) veya hepsi.",
    )
    parser.add_argument(
        "--vspaero-span-panel",
        dest="vspaero_span_panel",
        type=int,
        default=48,  # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson] eskiden None (OpenVSP varsayılanı 6 şerit/bölüm — çok kaba)
        help="[DEĞİŞİKLİK 2026-11-18 v6 | inceleme-2 düzeltmeleri] YARIM KANADIN toplam açıklık yönü şerit sayısı (bölümlere boyla orantılı dağıtılır). "
             "Panel yakınsama testi için (ör. 6, 12, 24, 48).",
    )
    parser.add_argument(  # [DEĞİŞİKLİK 2026-11-19 v10 | eğri tutarlılığı]
        "--flap-verimi",
        dest="flap_verimi",
        type=float,
        default=0.53,
        help="[DEĞİŞİKLİK 2026-11-19 v10 | eğri tutarlılığı] VSPAERO flaplı α0 kaymasına uygulanan flap verimi (eta). "
             "0.53 = XFOIL/ince profil karşılaştırmasının ortancası (flap_alpha0_kontrol.py). 1.0 = ham VSPAERO.",
    )
    parser.add_argument(  # [DEĞİŞİKLİK 2026-11-19 v10 | eğri tutarlılığı]
        "--egri-yontemi",
        dest="egri_yontemi",
        choices=("otomatik", "kaydir", "dogrusal"),
        default="otomatik",
        help="[DEĞİŞİKLİK 2026-11-19 v10 | eğri tutarlılığı] LLT CL eğrisinin VLM ile güncellenmesi: kaydir = VLM α0'ına kaydır + "
             "VLM eğimine uyan LLT kıvrımı; dogrusal = CLα·(α−α0), CL_max'ta sınırlı; "
             "otomatik = VSPAERO'da flapsız kaydir + flaplı dogrusal, diğer çözücülerde dogrusal.",
    )
    parser.add_argument(
        "--analitik-olcekleme",
        dest="analitik_olcekleme",
        action="store_true",
        default=False,
        help="Performans modüllerini baştan çalıştırmak yerine hızlı analitik ölçekleme kullan",
    )
    return parser.parse_args(argv)


def _hedef_geometriler_icin_csv_oku(
    dosya_adi: str,
    hedef_uclu_anahtarlar: Optional[Set[Tuple[str, float, float]]] = None,
) -> List[Dict[str, Any]]:
    """
    Büyük LLT / drag CSV dosyalarını (`llt_flapli_coklu_re.csv` vb.) akış halinde (streaming)
    okur ve yalnızca ilk 500 adayın kanat geometrileriyle eşleşen satırları belleğe alır.
    """
    try:
        yol = dosya_yolu_coz(dosya_adi)
        if not yol.is_file():
            return []
        if not hedef_uclu_anahtarlar:
            return csv_oku(yol)

        suzulen: List[Dict[str, Any]] = []
        with yol.open("r", encoding="utf-8-sig", newline="") as f:
            ilk_satir = f.readline()
            f.seek(0)
            ayirici = ";" if ilk_satir.count(";") > ilk_satir.count(",") else ","
            okuyucu = csv.DictReader(f, delimiter=ayirici)
            for row in okuyucu:
                g_key = _geometri_anahtari(row)
                if g_key[0] and (g_key[0], round(g_key[1], 3), round(g_key[2], 3)) in hedef_uclu_anahtarlar:
                    suzulen.append(row)
        return suzulen
    except Exception:
        return []


def _vspaero_hata_ozeti_yaz() -> None:
    """[DEĞİŞİKLİK 2026-11-18 | Görev A] VSPAERO'nun hata verip NumPy'a düştüğü geometrileri özetler (sessiz yedek yok)."""
    if VSPAERO_HATALARI:
        print(f"\n[VSPAERO-ÖZET] {len(VSPAERO_HATALARI)} çözümde VSPAERO hata verdi, NumPy yedeği kullanıldı:")
        for m in VSPAERO_HATALARI[:10]:
            print(f"   - {m}")
        if len(VSPAERO_HATALARI) > 10:
            print(f"   ... (+{len(VSPAERO_HATALARI) - 10} tane daha)")


def _vlm_test_modu(
    args: argparse.Namespace,
    ilk_500: List[Dict[str, Any]],
    llt_flapsiz_satirlar: List[Dict[str, Any]],
    llt_flapli_satirlar: List[Dict[str, Any]],
    kutle_satirlari: List[Dict[str, Any]],
) -> int:
    """
    [DEĞİŞİKLİK 2026-11-18 | Görev A] Hızlı test: ilk N geometri seçilen çözücüyle VE NumPy ile çözülür,
    sonuçlar yan yana `vlm_sonuclari/vlm_test_karsilastirma.csv` dosyasına yazılır.
    Ana çıktı dosyalarına (vlm_nihai_ucak_siralamasi.csv vb.) DOKUNULMAZ.
    Görev B (arayüzlü OpenVSP) sonuçları bu dosyadaki değerlerle karşılaştırılır.
    """
    n = int(args.vlm_test)
    print(f"[VLM-TEST] İlk {n} geometri '{args.cozucu}' ve 'numpy_vlm' ile çözülüyor...")
    ortak = dict(
        vlm_aday_satirlari=ilk_500,
        llt_flapsiz_satirlar=llt_flapsiz_satirlar,
        llt_flapli_satirlar=llt_flapli_satirlar,
        n_span=args.n_span,
        n_chord=args.n_chord,
        kutle_satirlari=kutle_satirlari,
        geometri_limiti=n,
    )
    secilen = ilk_500_aday_vlm_analizi(cozucu_tercihi=args.cozucu, **ortak)
    _vspaero_hata_ozeti_yaz()
    numpy_sonuc = ilk_500_aday_vlm_analizi(cozucu_tercihi="numpy_vlm", **ortak)

    alanlar = [
        ("CL_alpha_3d_per_deg", "CLa_flapsiz"),
        ("alpha_L0_deg", "a0_flapsiz"),
        ("e_oswald", "e_flapsiz"),
        ("CL_max_3d_tahmini", "CLmax_flapsiz"),
        ("CL_alpha_3d_flapli_per_deg", "CLa_flapli"),
        ("alpha_L0_flapli_deg", "a0_flapli"),
        ("e_oswald_flapli", "e_flapli"),
        ("CL_max_3d_flapli_tahmini", "CLmax_flapli"),
    ]
    satirlar: List[Dict[str, Any]] = []
    for g_key, s in secilen.items():
        npy = numpy_sonuc.get(g_key, {})
        r: Dict[str, Any] = {
            "profil": g_key[0], "b_w_m": g_key[1], "S_w_m2": g_key[2], "c_w_m": g_key[3],
            "AR_w": round(g_key[1] ** 2 / max(g_key[2], 1e-9), 4),
            "b_flap_m": s.get("b_flap_m", ""),
            "cozucu_flapsiz": s.get("cozucu", ""), "cozucu_flapli": s.get("cozucu_flapli", ""),
        }
        for k, kisa in alanlar:
            r[f"{kisa}_secilen"] = s.get(k, "")
            r[f"{kisa}_numpy"] = npy.get(k, "")
            r[f"{kisa}_llt"] = s.get(
                {"CL_alpha_3d_per_deg": "CL_alpha_llt", "e_oswald": "e_oswald_llt",
                 "CL_max_3d_tahmini": "CL_max_3d_llt", "CL_alpha_3d_flapli_per_deg": "CL_alpha_flapli_llt",
                 "e_oswald_flapli": "e_oswald_flapli_llt", "CL_max_3d_flapli_tahmini": "CL_max_3d_flapli_llt"}.get(k, "_"), "")
        r["vsp3_flapsiz"] = s.get("vsp3_flapsiz", "")
        r["vsp3_flapli"] = s.get("vsp3_flapli", "")
        satirlar.append(r)

    vk = Path(args.vlm_klasoru)
    vk = vk if vk.is_absolute() else SCRIPT_DIR / vk
    cikti = vk / "vlm_test_karsilastirma.csv"
    csv_yaz(satirlar, cikti)
    print(f"\n[VLM-TEST] Karşılaştırma yazıldı: {cikti}")
    for r in satirlar:
        print(
            f"  {r['profil']:<14} b={r['b_w_m']:.3f}  "
            f"CLa {r['CLa_flapsiz_secilen']} / numpy {r['CLa_flapsiz_numpy']} | "
            f"a0 {r['a0_flapsiz_secilen']} / {r['a0_flapsiz_numpy']} | "
            f"e {r['e_flapsiz_secilen']} / {r['e_flapsiz_numpy']} | "
            f"flapli a0 {r['a0_flapli_secilen']} / {r['a0_flapli_numpy']}  [{r['cozucu_flapsiz']}]"
        )
    return 0


def _govde_test_modu(
    args: argparse.Namespace,
    ilk_500: List[Dict[str, Any]],
    kutle_satirlari: List[Dict[str, Any]],
) -> int:
    """
    [DEĞİŞİKLİK 2026-11-18 v5 | gövde] Kanat + gövde VSPAERO testi (sadece --vlm-test ile).
    [DEĞİŞİKLİK 2026-11-18 v6 | inceleme-2 düzeltmeleri] Yeniden yazıldı (önceki hali:
    yedekler/vlm_entegrasyonu_YEDEK_2026-11-18_v5_duzeltme_oncesi.py):
      - Referans (sadece kanat) modeline de gövde kenarında (y=R) bölüm sınırı konur → iki model AYNI ağla
        çözülür, fark yalnızca gövdeden gelir.
      - Sonuç hesaplanamasa bile satır yazılır ve "kalite" sütununda işaretlenir (eskiden atılıyordu).
      - İz tabanlı (CLwtot/CDiw) CLα, α0, e ve farkları da yazılır; kanat yüzey/alt-üst z, Zcg, panel ayarları,
        flap başlangıcı, sonuç kaynağı sütunları eklendi.
      - Yeni seçenekler: --kanat-z (ust/ust_gomulu/orta/alt/alt_gomulu/hepsi), --govde-gomme, --zcg,
        --flap-baslangic, --vspaero-span-panel (yarım kanat toplam), --vspaero-uc-sik.
    Sonuçlar vlm_sonuclari/vlm_test_govde_karsilastirma.csv dosyasına yazılır; performans tablolarına İŞLENMEZ.
    """
    from vspaero_cozucu import KANAT_Z_SECENEKLERI

    hazir, mesaj = vspaero_hazir_mi()
    if not hazir:
        print(f"[HATA] Gövde testi VSPAERO gerektirir: {mesaj}")
        return 1
    print(f"[VSPAERO] Hazır: {mesaj}")

    ham_z = str(args.kanat_z or "alt_gomulu").strip().lower()  # [DEĞİŞİKLİK 2026-11-18 v7 | ekip kararı: kanat yeri]
    z_listesi: List[Any] = list(KANAT_Z_SECENEKLERI) if ham_z == "hepsi" else []
    if not z_listesi:
        for parca in ham_z.split(","):
            parca = parca.strip()
            if not parca:
                continue
            try:
                z_listesi.append(float(parca))
            except ValueError:
                if parca not in KANAT_Z_SECENEKLERI:
                    print(f"[HATA] --kanat-z değeri anlaşılamadı: {parca} (geçerli: {', '.join(KANAT_Z_SECENEKLERI)}, hepsi, sayı)")
                    return 1
                z_listesi.append(parca)
    yontemler = ["vlm", "panel"] if args.vspaero_yontem == "hepsi" else [args.vspaero_yontem]
    zcg: Any = None
    if args.zcg is not None:
        zcg = "kanat" if str(args.zcg).strip().lower() == "kanat" else float(args.zcg)

    kutle_geo: Dict[Tuple[str, float, float, float], Dict[str, Any]] = {}
    for kr in kutle_satirlari:
        kk = _geometri_anahtari(kr)
        if kk[0] and kk not in kutle_geo:
            kutle_geo[kk] = kr

    geometriler: List[Tuple[str, float, float, float]] = []
    for r in ilk_500:
        gk = _geometri_anahtari(r)
        if gk[0] and gk[1] > 0 and gk not in geometriler:
            geometriler.append(gk)
        if len(geometriler) >= int(args.vlm_test):
            break

    def _f(v: Any) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    satirlar: List[Dict[str, Any]] = []
    for gk in geometriler:
        profil, b_w, s_w, c_w = gk
        kr = kutle_geo.get(gk, {})
        D = sayi_al(kr, "govde_D_m", varsayilan=0.14)
        L = sayi_al(kr, "govde_L_toplam_m", "govde_L_m", varsayilan=1.56)
        R = 0.5 * D
        b_flap = min(1.0, b_w)
        flap_ic = R if args.flap_baslangic == "govde_kenari" else 0.0
        for durum, oran in (("flapsiz", 0.0), ("flapli", b_flap / max(b_w, 1e-9))):
            for yontem in yontemler:
                vakalar: List[Tuple[str, Optional[Dict[str, Any]]]] = [("sadece_kanat", None)]
                for z in z_listesi:
                    etiket = z if isinstance(z, str) else f"z{z:+.3f}"
                    vakalar.append((etiket, {"D": D, "L_toplam": L, "kanat_x": args.kanat_x, "kanat_z": z,
                                             "gomme": args.govde_gomme,
                                             "L_aviyonik": args.govde_l_aviyonik, "L_kuyruk": args.govde_l_kuyruk,
                                             "D_min": args.govde_d_min}))  # [DEĞİŞİKLİK 2026-11-19 v9 | tam akışta gövde]
                referans: Optional[Dict[str, Any]] = None
                for yerlesim, gv in vakalar:
                    print(f"[GÖVDE-TEST] {profil} {durum} {yontem} {yerlesim} ...")
                    try:
                        r = vspaero_api_coz(
                            profil=profil, b_w=b_w, c_w=c_w, s_w=s_w, cl_max_2d=1.4,
                            flap_aciklik_orani=oran, model_klasoru=VSPAERO_MODEL_KLASORU,
                            govde=gv, yontem=yontem, span_panel=args.vspaero_span_panel,
                            uc_sik=args.vspaero_uc_sik, zcg=zcg, flap_ic_y=flap_ic,
                            ek_kesit_y=([] if gv else [R]),  # referans da aynı bölüm sınırını alır
                            gecersize_izin=True,
                            e_kaynagi=args.vspaero_e_kaynagi,       # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson]
                            viskoz_e_carpani=(args.e_viskoz_carpan if args.e_viskoz_carpan is not None
                                              else (1.0 if gv else 0.995)),  # [DEĞİŞİKLİK 2026-11-19 v11 | flap ve k diğer kodlardan] gövdeli 1.0, gövdesiz 0.995
                            veter_tess=args.vspaero_veter_tess,     # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson]
                        )
                    except Exception as exc:
                        print(f"[VSPAERO-HATA] {profil} {durum} {yontem} {yerlesim}: {type(exc).__name__}: {exc}")
                        if VSPAERO_ZORUNLU:
                            raise
                        satirlar.append({"profil": profil, "durum": durum, "yontem": yontem, "yerlesim": yerlesim,
                                         "kalite": f"HATA: {type(exc).__name__}: {exc}"})
                        continue
                    satir: Dict[str, Any] = {
                        "profil": profil, "b_w_m": b_w, "S_w_m2": s_w, "c_w_m": c_w,
                        "durum": durum, "yontem": yontem, "yerlesim": yerlesim, "kalite": r["kalite"],
                        "govde_D_m": r.get("govde_D_m", ""), "govde_L_m": r.get("govde_L_m", ""),
                        "kanat_x_ceyrek_m": r.get("kanat_x_ceyrek_m", ""), "kanat_z_m": r.get("kanat_z_m", ""),
                        "kanat_ust_z_m": r.get("kanat_ust_z_m", ""), "kanat_alt_z_m": r.get("kanat_alt_z_m", ""),
                        "govde_tepe_z_m": r.get("govde_tepe_z_m", ""), "gomme_m": r.get("gomme_m", ""),
                        "zcg_m": r["zcg_m"], "flap_ic_y_m": r["flap_ic_y_m"],
                        "CLa_per_deg": r["CL_alpha_3d_per_deg"], "a0_deg": r["alpha_L0_deg"],
                        "e_span_ham": r["e_span_vspaero_ham"], "e_kirpildi": r["e_kirpildi"], "e_oswald": r["e_oswald"],
                        "CLa_iz_per_deg": r["CL_alpha_iz_per_deg"], "a0_iz_deg": r["alpha_L0_iz_deg"], "e_iz": r["e_iz"],
                        "CL_alfa4": r["CL_alfa4"], "CDi_alfa4": r["CDi_alfa4"],
                        "Cm_alpha_per_deg": r["Cm_alpha_per_deg"], "CMy_alfa4": r["CMy_alfa4"],
                        "span_panel": r["span_panel"], "uc_sik": r["uc_sik"],
                        "sonuc_kaynagi": r["sonuc_kaynagi"], "vsp3": r["vsp3_dosyasi"],
                    }
                    if yerlesim == "sadece_kanat":
                        referans = satir
                    if referans is not None:
                        for k in ("CLa_per_deg", "a0_deg", "e_span_ham", "CLa_iz_per_deg", "e_iz",
                                  "CL_alfa4", "CDi_alfa4", "Cm_alpha_per_deg"):
                            satir[f"fark_{k}"] = round(_f(satir[k]) - _f(referans[k]), 6)
                        ref_cla = _f(referans["CLa_per_deg"])
                        satir["fark_CLa_yuzde"] = round(100.0 * (_f(satir["CLa_per_deg"]) - ref_cla) / ref_cla, 3) \
                            if ref_cla == ref_cla and abs(ref_cla) > 1e-9 else float("nan")
                        ref_e = _f(referans["e_iz"])
                        satir["fark_e_iz_yuzde"] = round(100.0 * (_f(satir["e_iz"]) - ref_e) / ref_e, 3) \
                            if ref_e == ref_e and abs(ref_e) > 1e-9 else float("nan")
                    satirlar.append(satir)

    vk = Path(args.vlm_klasoru)
    vk = vk if vk.is_absolute() else SCRIPT_DIR / vk
    cikti = vk / "vlm_test_govde_karsilastirma.csv"
    csv_yaz(satirlar, cikti)
    print(f"\n[GÖVDE-TEST] Karşılaştırma yazıldı: {cikti}")
    print(f"  {'durum':<8} {'yontem':<6} {'yerlesim':<13} {'CLa':>7} {'dCLa%':>6} {'a0':>6} "
          f"{'e_yuz':>6} {'e_iz':>6} {'de_iz%':>7} {'Cm_a':>8}  kalite")
    for r in satirlar:
        if "CLa_per_deg" not in r:
            print(f"  {r['durum']:<8} {r['yontem']:<6} {r['yerlesim']:<13} {r['kalite']}")
            continue
        print(f"  {r['durum']:<8} {r['yontem']:<6} {r['yerlesim']:<13} {_f(r['CLa_per_deg']):>7.4f} "
              f"{_f(r.get('fark_CLa_yuzde', 0.0)):>6.2f} {_f(r['a0_deg']):>6.2f} {_f(r['e_span_ham']):>6.3f} "
              f"{_f(r['e_iz']):>6.3f} {_f(r.get('fark_e_iz_yuzde', 0.0)):>7.2f} {_f(r['Cm_alpha_per_deg']):>8.4f}  {r['kalite']}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = argumanlari_coz(argv)

    # [DEĞİŞİKLİK 2026-11-18 | Görev A] Komut satırı ayarlarını VSPAERO modülüne aktar
    global VSPAERO_ZORUNLU, VSPAERO_KUYRUK_DAHIL, VSPAERO_MODEL_KLASORU
    VSPAERO_ZORUNLU = bool(args.vspaero_zorunlu)
    global VSPAERO_SPAN_PANEL  # [DEĞİŞİKLİK 2026-11-18 v5 | gövde]
    VSPAERO_SPAN_PANEL = args.vspaero_span_panel
    global VSPAERO_UC_SIK  # [DEĞİŞİKLİK 2026-11-18 v6 | inceleme-2 düzeltmeleri]
    VSPAERO_UC_SIK = args.vspaero_uc_sik
    global VSPAERO_RICHARDSON, VSPAERO_E_KAYNAGI, VSPAERO_VETER_TESS  # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson]
    VSPAERO_E_KAYNAGI = args.vspaero_e_kaynagi
    global VSPAERO_E_VISKOZ  # [DEĞİŞİKLİK 2026-11-19 v9 | tam akışta gövde]
    # [DEĞİŞİKLİK 2026-11-19 v11 | flap ve k diğer kodlardan] k varsayılanı: lifting_line_cozucu.py'deki k_fus = 0.995 GÖVDE düzeltmesi -> gövde VSPAERO'da modelliyse 1.0
    VSPAERO_E_VISKOZ = float(args.e_viskoz_carpan) if args.e_viskoz_carpan is not None else (1.0 if args.vspaero_govde else 0.995)
    VSPAERO_VETER_TESS = args.vspaero_veter_tess
    global FLAP_VERIMI, EGRI_YONTEMI  # [DEĞİŞİKLİK 2026-11-19 v10 | eğri tutarlılığı]
    FLAP_VERIMI = float(args.flap_verimi)
    EGRI_YONTEMI = str(args.egri_yontemi)
    if not (0.0 < FLAP_VERIMI <= 1.0):
        print(f"[HATA] --flap-verimi 0 ile 1 arasında olmalı (verilen: {FLAP_VERIMI}).")
        return 1
    print(f"[VLM] Flap verimi (VSPAERO flaplı α0 düzeltmesi): {FLAP_VERIMI} | CL eğrisi yöntemi: {EGRI_YONTEMI}")
    _r =str(args.vspaero_richardson or "yok").strip().lower()
    if _r in ("yok", "none", "0", ""):
        VSPAERO_RICHARDSON = None
    elif _r in ("evet", "var", "1"):
        _n = int(args.vspaero_span_panel or 48)
        VSPAERO_RICHARDSON = (max(2, _n // 2), _n)
    else:
        _a, _b = (int(x) for x in _r.split(","))
        VSPAERO_RICHARDSON = (_a, _b)
    if VSPAERO_RICHARDSON and VSPAERO_E_KAYNAGI == "yuzey":
        print("[UYARI] Richardson + yüzey e'si birlikte güvenilmez (yüzey e'si yakınsamıyor).")
    VSPAERO_KUYRUK_DAHIL = bool(args.vspaero_kuyruk)
    _vk = Path(args.vlm_klasoru)
    VSPAERO_MODEL_KLASORU = (_vk if _vk.is_absolute() else SCRIPT_DIR / _vk) / "vspaero_modelleri"

    girdi_yolu = dosya_yolu_coz(args.girdi)
    if not girdi_yolu.is_file():
        print(f"[VLM] '{args.girdi}' bulunamadı; 'aday_siralama.py' üzerinden ilk 500 aday üretiliyor...")
        siralama_args = siralama_argumanlari_coz(["--vlm-cikti", args.girdi])
        tum_sirali, ilk_500 = adaylari_degerlendir_ve_sirala(siralama_args)
        girdi_yolu = SCRIPT_DIR / args.girdi
        csv_yaz(tum_sirali, SCRIPT_DIR / siralama_args.cikti)
        csv_yaz(ilk_500, girdi_yolu)
    else:
        ilk_500 = csv_oku(girdi_yolu)

    if not ilk_500:
        print("[HATA] İlk 500 aday listesi boş!")
        return 1

    hedef_uclu_geometriler: Set[Tuple[str, float, float]] = set()
    for r in ilk_500:
        gk = _geometri_anahtari(r)
        if gk[0] and gk[1] > 0.0:
            hedef_uclu_geometriler.add((gk[0], round(gk[1], 3), round(gk[2], 3)))

    print(
        f"[VLM] İlk {len(ilk_500)} aday içindeki {len(hedef_uclu_geometriler)} benzersiz "
        "kanat geometrisi için LLT ve drag tabloları yükleniyor..."
    )

    llt_flapsiz_satirlar = _hedef_geometriler_icin_csv_oku(args.llt_flapsiz, hedef_uclu_geometriler)
    drag_polar_satirlar = _hedef_geometriler_icin_csv_oku(args.drag_polar, hedef_uclu_geometriler)
    drag_ozet_satirlar = _hedef_geometriler_icin_csv_oku(args.drag_ozet, hedef_uclu_geometriler)

    llt_flapli_satirlar = _hedef_geometriler_icin_csv_oku(args.llt_flapli, hedef_uclu_geometriler)
    drag_polar_flapli_satirlar = _hedef_geometriler_icin_csv_oku(args.drag_polar_flapli, hedef_uclu_geometriler)
    drag_ozet_flapli_satirlar = _hedef_geometriler_icin_csv_oku(args.drag_ozet_flapli, hedef_uclu_geometriler)

    # [DEĞİŞİKLİK 2026-11-18 | Görev A] Kuyruk kolu / kuyruk profili için kütle tablosu (yalnızca VSPAERO kullanır)
    _kutle_yolu = dosya_yolu_coz(args.kutle)
    kutle_satirlari = csv_oku(_kutle_yolu) if _kutle_yolu.is_file() else []

    # [DEĞİŞİKLİK 2026-11-18 | Görev A] Hızlı test modu: ilk N geometriyi seçilen çözücü + NumPy ile çöz, karşılaştır, çık.
    if args.vlm_test > 0:
        if args.vspaero_govde:  # [DEĞİŞİKLİK 2026-11-18 v5 | gövde]
            return _govde_test_modu(args, ilk_500, kutle_satirlari)
        return _vlm_test_modu(args, ilk_500, llt_flapsiz_satirlar, llt_flapli_satirlar, kutle_satirlari)
    if args.vspaero_govde:  # [DEĞİŞİKLİK 2026-11-19 v9 | tam akışta gövde] tam akışta gövde (eskiden: göz ardı ediliyordu)
        global VSPAERO_GOVDE
        _zcg: Any = None
        if args.zcg is not None:
            _zcg = "kanat" if str(args.zcg).strip().lower() == "kanat" else float(args.zcg)
        _kz: Any = str(args.kanat_z or "alt_gomulu").strip().lower()
        if "," in _kz or _kz == "hepsi":
            print("[HATA] Tam akışta --kanat-z tek değer olmalı (ör. alt_gomulu); liste/hepsi yalnız --vlm-test ile.")
            return 1
        try:
            _kz = float(_kz)
        except ValueError:
            pass
        VSPAERO_GOVDE = {
            "kanat_x": args.kanat_x, "kanat_z": _kz, "gomme": args.govde_gomme, "zcg": _zcg,
            "flap_baslangic": args.flap_baslangic, "L_aviyonik": args.govde_l_aviyonik,
            "L_kuyruk": args.govde_l_kuyruk, "D_min": args.govde_d_min,
        }
        print("[GÖVDE] Tam akışta her adaya gövde eklenecek; tablolara KANAT+GÖVDE katsayıları gidecek.")
        print(f"[GÖVDE] Parametreler: kanat_x={args.kanat_x if args.kanat_x is not None else 'su haznesi ortası'}, "
              f"kanat_z={_kz}, gömme={args.govde_gomme if args.govde_gomme is not None else 0.005} m, "
              f"Zcg={_zcg if _zcg is not None else 0.0}, flap={args.flap_baslangic}, "
              f"L_av={args.govde_l_aviyonik}, L_kuyruk={args.govde_l_kuyruk}, D_min={args.govde_d_min}, "
              f"e_viskoz_k={VSPAERO_E_VISKOZ}")  # [DEĞİŞİKLİK 2026-11-19 v11 | flap ve k diğer kodlardan]
        if args.cozucu not in ("vspaero", "auto"):
            print("[UYARI] Gövde yalnızca VSPAERO ile modellenir; NumPy/AeroSandbox gövdesiz çözer.")

    # 1. Flapsız ve Flaplı 3B VLM Analizi
    vlm_haritasi = ilk_500_aday_vlm_analizi(
        vlm_aday_satirlari=ilk_500,
        llt_flapsiz_satirlar=llt_flapsiz_satirlar,
        llt_flapli_satirlar=llt_flapli_satirlar,
        cozucu_tercihi=args.cozucu,
        n_span=args.n_span,
        n_chord=args.n_chord,
        kutle_satirlari=kutle_satirlari,  # [DEĞİŞİKLİK 2026-11-18 | Görev A]
    )
    _vspaero_hata_ozeti_yaz()  # [DEĞİŞİKLİK 2026-11-18 | Görev A]

    # 2. LLT ve Drag Tablolarını (Flapsız + Flaplı) Güncelle ve Kaydet
    vlm_klasoru = Path(args.vlm_klasoru)
    if not vlm_klasoru.is_absolute():
        vlm_klasoru = SCRIPT_DIR / vlm_klasoru
    vlm_klasoru.mkdir(parents=True, exist_ok=True)

    guncel_tablolar = llt_ve_drag_tablolarini_vlm_ile_guncelle(
        llt_flapsiz_satirlar=llt_flapsiz_satirlar,
        drag_polar_satirlar=drag_polar_satirlar,
        drag_ozet_satirlar=drag_ozet_satirlar,
        vlm_haritasi=vlm_haritasi,
        llt_flapli_satirlar=llt_flapli_satirlar,
        drag_polar_flapli_satirlar=drag_polar_flapli_satirlar,
        drag_ozet_flapli_satirlar=drag_ozet_flapli_satirlar,
    )

    vlm_dosya_yollari = {
        "llt_flapsiz": vlm_klasoru / "vlm_flapsiz_coklu_re.csv",
        "drag_polar_flapsiz": vlm_klasoru / "vlm_flapsiz_polar_detay.csv",
        "drag_ozet_flapsiz": vlm_klasoru / "vlm_flapsiz_aday_ozet.csv",
        "llt_flapli": vlm_klasoru / "vlm_flapli_coklu_re.csv",
        "drag_polar_flapli": vlm_klasoru / "vlm_flapli_polar_detay.csv",
        "drag_ozet_flapli": vlm_klasoru / "vlm_flapli_aday_ozet.csv",
    }

    for anahtar, yol in vlm_dosya_yollari.items():
        if guncel_tablolar.get(anahtar):
            csv_yaz(guncel_tablolar[anahtar], yol)
            print(f"[VLM] Güncellenmiş tablo yazıldı ({len(guncel_tablolar[anahtar])} satır): {yol}")

    # 3. Performans Modüllerini Baştan Çalıştır ve Nihai Puanlamayı Üret
    nihai_sirali = vlm_adaylarini_yeniden_puanla(
        ilk_500_satirlar=ilk_500,
        vlm_haritasi=vlm_haritasi,
        args=args,
        vlm_klasoru=vlm_klasoru,
        vlm_dosya_yollari=vlm_dosya_yollari,
    )

    cikti_yolu = Path(args.cikti)
    if not cikti_yolu.is_absolute():
        cikti_yolu = SCRIPT_DIR / cikti_yolu
    csv_yaz(nihai_sirali, cikti_yolu)

    uygun_vlm = sum(1 for r in nihai_sirali if r.get("tum_fazlar_uygun_vlm"))
    print(f"VLM sonrası tüm fazları uygun aday  : {uygun_vlm}")
    print(f"Nihai VLM sıralama çıktısı          : {cikti_yolu}")

    if nihai_sirali:
        lider = nihai_sirali[0]
        print(
            "\nVLM Sonrası En İyi 1. Uçak:\n"
            f"  vlm_nihai_sira (Önceki) : {lider.get('vlm_nihai_sira')} "
            f"(LLT Sırası: {lider.get('llt_onc_sira')}, Değişim: {int(lider.get('sira_degisimi', 0)):+d})\n"
            f"  konfigurasyon_id        : {metin_al(lider, 'konfigurasyon_id', varsayilan='?')}\n"
            f"  kanat_airfoil / pervane : {metin_al(lider, 'kanat_airfoil', 'profil', varsayilan='?')} / "
            f"{metin_al(lider, 'pervane', varsayilan='?')}\n"
            f"  e_oswald (LLT -> VLM)   : {lider.get('e_oswald_llt')} -> {lider.get('e_oswald_vlm')}\n"
            f"  CL_max_flapli (VLM)     : {lider.get('CL_max_3d_flapli_vlm')}\n"
            f"  vlm_sonrasi_toplam_puan : {sayi_al(lider, 'vlm_sonrasi_toplam_puan', 'toplam_performans_puani', varsayilan=0.0):.4f} "
            f"(LLT Öncesi: {sayi_al(lider, 'llt_onc_toplam_puan', varsayilan=0.0):.4f}, "
            f"Fark: {sayi_al(lider, 'puan_farki_vlm_eksi_llt', varsayilan=0.0):+.4f})\n"
            f"  vlm_mission2_400_puani  : {sayi_al(lider, 'vlm_sonrasi_mission2_400_puani', 'mission2_400_puani', varsayilan=0.0):.4f}\n"
            f"  toplam_tuketim_vlm      : {sayi_al(lider, 'gercek_toplam_tuketim_mAh_vlm', 'gercek_toplam_tuketim_mAh', varsayilan=0.0):.2f} mAh "
            f"(Kalan: %{sayi_al(lider, 'kalan_batarya_yuzde_vlm', 'kalan_batarya_yuzde', varsayilan=0.0):.2f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
