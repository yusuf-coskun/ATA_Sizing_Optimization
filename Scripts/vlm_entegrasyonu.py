#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATA Sizing Optimization - VSPAERO / VLM Entegrasyon ve Yeniden Puanlama Modülü

Yol haritasındaki VLM aşamasını uçtan uca uygular:
1) aday_siralama.py çıktısı olan 'vlm_ilk_500_aday.csv' dosyasını okur.
2) İlk 500 adaydaki benzersiz uçak geometrilerini (kanat + yatay kuyruk + dikey kuyruk)
   3B Vortex Lattice Method (VSPAERO CLI / AeroSandbox VLM / Dahili NumPy 3B VLM + Trefftz
   düzlemi iz integrasyonu) ile analiz eder.
3) LLT ve sürükleme (drag) tablolarındaki 3B aerodinamik katsayıları (CL_alpha_3d, e_oswald,
   CL_max_3d, CL(alpha), CDi(alpha), CD(alpha), L/D) VLM sonuçlarıyla günceller.
4) İlk 500 adayı tüm performans modüllerinden (kalkış, tırmanış, yük bırakma, iniş, seyir)
   veya hızlı analitik/sayısal yeniden hesaplama hattından geçirip 'aday_siralama.py' ile
   tekrar puanlar ve nihai uçakları 'vlm_nihai_ucak_siralamasi.csv' olarak kaydeder.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from aday_siralama import (
    BATARYA_KAPASITE_MAH,
    GERILIM_V,
    KULLANILABILIR_MAH,
    MAX_GOREV_SURESI_SN,
    adaylari_degerlendir_ve_sirala,
    argumanlari_coz as siralama_argumanlari_coz,
    bool_al,
    csv_oku,
    csv_yaz,
    dosya_yolu_coz,
    metin_al,
    sayi_al,
)


# ---------------------------------------------------------------------------
# 1. 3B Vortex Lattice Method (VLM) Çekirdeği (VSPAERO / AeroSandbox / NumPy VLM)
# ---------------------------------------------------------------------------

def kullanilabilir_vlm_motoru_sec(tercih: str = "auto") -> str:
    """Sistemde kurulu olan en uygun VLM çözücüsünü belirler."""
    tercih = tercih.strip().lower()
    if tercih in ("vspaero", "aerosandbox", "numpy_vlm"):
        return tercih

    # 1. VSPAERO ikili dosyası veya OpenVSP Python API kontrolü
    if shutil.which("vspaero") is not None:
        return "vspaero"
    try:
        import openvsp as _vsp  # noqa: F401
        return "vspaero"
    except ImportError:
        pass

    # 2. AeroSandbox VLM kontrolü (kanat_sekil_optimizasyonu.py ile uyumlu)
    try:
        import aerosandbox as _asb  # noqa: F401
        return "aerosandbox"
    except ImportError:
        pass

    # 3. Her ortamda çalışan dahili NumPy 3B At Nalı (Horseshoe) VLM çözücüsü
    return "numpy_vlm"


def _biot_savart_segment(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    core_radius: float = 1e-5,
) -> np.ndarray:
    """Sonlu bir AB girdap segmentinin P noktasında indüklediği hızı hesaplar."""
    r0 = b - a
    r1 = p - a
    r2 = p - b
    cross = np.cross(r1, r2)
    norm_cross_sq = np.sum(cross * cross, axis=-1, keepdims=True) + (core_radius ** 2)
    r1_norm = np.linalg.norm(r1, axis=-1, keepdims=True) + 1e-12
    r2_norm = np.linalg.norm(r2, axis=-1, keepdims=True) + 1e-12
    dot_term = np.sum(r0 * (r1 / r1_norm - r2 / r2_norm), axis=-1, keepdims=True)
    return (1.0 / (4.0 * math.pi)) * (cross / norm_cross_sq) * dot_term


def _yari_sonsuz_iz_hizi(
    p: np.ndarray,
    a: np.ndarray,
    u_inf_dir: np.ndarray,
    core_radius: float = 1e-5,
) -> np.ndarray:
    """A noktasından u_inf_dir yönünde sonsuza uzanan yarı-sonsuz iz girdabının hızı."""
    r1 = p - a
    r1_norm = np.linalg.norm(r1, axis=-1, keepdims=True) + 1e-12
    # Biot-Savart: dl x r1 = u_inf_dir x r1
    cross = np.cross(u_inf_dir, r1)
    norm_cross_sq = np.sum(cross * cross, axis=-1, keepdims=True) + (core_radius ** 2)
    dot_term = 1.0 + np.sum(u_inf_dir * r1, axis=-1, keepdims=True) / r1_norm
    return (1.0 / (4.0 * math.pi)) * (cross / norm_cross_sq) * dot_term


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
) -> Dict[str, Any]:
    """
    Saf NumPy tabanlı 3B Çok Yüzeyli (Kanat + Yatay Kuyruk) At Nalı Vortex Lattice Method
    ve Trefftz Düzlemi (Planar Wake) çözücüsü.

    LLT'den farklı olarak:
    - Veter boyu (chordwise) basınç dağılımını ve düşük açıklık oranı (AR) 3B etkilerini çözer.
    - Kanadın yatay kuyruk üzerinde oluşturduğu 3B downwash etkisini doğrudan AIC matrisine katar.
    - Trefftz düzleminde Fourier iz integrasyonu ile VLM Oswald verimlilik faktörünü (e_vlm) hesaplar.
    """
    b_w = max(float(b_w), 0.1)
    c_w = max(float(c_w), 0.05)
    s_w = max(float(s_w), b_w * c_w)
    ar_w = (b_w ** 2) / s_w

    # Kanat panel köşe noktaları (düzenli açıklık dağılımı + veter yönü 1/4 - 3/4 kuralı)
    n_span_even = max(6, int(n_span) + (int(n_span) % 2))
    n_chord = max(1, int(n_chord))

    y_edges = np.linspace(-0.5 * b_w, 0.5 * b_w, n_span_even + 1)
    x_edges = np.linspace(0.0, c_w, n_chord + 1)

    vort_a_list: List[np.ndarray] = []
    vort_b_list: List[np.ndarray] = []
    colloc_list: List[np.ndarray] = []
    normal_list: List[np.ndarray] = []
    panel_span_idx: List[int] = []
    panel_is_wing: List[bool] = []

    alpha_0_rad = math.radians(alpha_l0_2d_deg)
    # 2B sıfır taşıma açısını (alpha_L0) kamber eğimi olarak normal vektöre yansıt:
    # u_inf(alpha) . n_wing = sin(alpha - alpha_L0)
    n_wing = np.array([-math.sin(alpha_0_rad), 0.0, math.cos(alpha_0_rad)], dtype=float)

    for j in range(n_span_even):
        y1, y2 = y_edges[j], y_edges[j + 1]
        ym = 0.5 * (y1 + y2)
        for i in range(n_chord):
            x1, x2 = x_edges[i], x_edges[i + 1]
            dx = x2 - x1
            x_vort = x1 + 0.25 * dx
            x_coll = x1 + 0.75 * dx

            vort_a_list.append(np.array([x_vort, y1, 0.0]))
            vort_b_list.append(np.array([x_vort, y2, 0.0]))
            colloc_list.append(np.array([x_coll, ym, 0.0]))
            normal_list.append(n_wing)
            panel_span_idx.append(j)
            panel_is_wing.append(True)

    # Yatay kuyruk panelleri (varsa kanat izi etkileşimiyle birlikte çözülür)
    if b_ht > 0.05 and c_ht > 0.02 and tail_arm > 0.1:
        n_ht_span = max(6, n_span_even // 3)
        if n_ht_span % 2 == 1:
            n_ht_span += 1
        y_ht_edges = np.linspace(-0.5 * b_ht, 0.5 * b_ht, n_ht_span + 1)
        x_ht_edges = np.linspace(tail_arm, tail_arm + c_ht, 3)
        n_ht = np.array([0.0, 0.0, 1.0], dtype=float)
        for j in range(n_ht_span):
            y1, y2 = y_ht_edges[j], y_ht_edges[j + 1]
            ym = 0.5 * (y1 + y2)
            for i in range(2):
                x1, x2 = x_ht_edges[i], x_ht_edges[i + 1]
                dx = x2 - x1
                vort_a_list.append(np.array([x1 + 0.25 * dx, y1, 0.0]))
                vort_b_list.append(np.array([x1 + 0.25 * dx, y2, 0.0]))
                colloc_list.append(np.array([x1 + 0.75 * dx, ym, 0.0]))
                normal_list.append(n_ht)
                panel_span_idx.append(-1)
                panel_is_wing.append(False)

    vort_a = np.asarray(vort_a_list, dtype=float)      # (N, 3)
    vort_b = np.asarray(vort_b_list, dtype=float)      # (N, 3)
    colloc = np.asarray(colloc_list, dtype=float)      # (N, 3)
    normals = np.asarray(normal_list, dtype=float)     # (N, 3)
    is_wing_mask = np.asarray(panel_is_wing, dtype=bool)
    span_indices = np.asarray(panel_span_idx, dtype=int)

    # AIC (Aerodynamic Influence Coefficient) matrisi oluşturma (sabit iz yönü +x)
    p_grid = colloc[:, None, :]                        # (N, 1, 3)
    a_grid = vort_a[None, :, :]                        # (1, N, 3)
    b_grid = vort_b[None, :, :]                        # (1, N, 3)
    u_x = np.array([1.0, 0.0, 0.0], dtype=float)[None, None, :]

    v_bound = _biot_savart_segment(p_grid, a_grid, b_grid)
    # Sol iz girdabı +sonsuzdan A'ya gelir (-u_x yönü), sağ iz girdabı B'den +sonsuza gider (+u_x yönü)
    v_trail_left = -_yari_sonsuz_iz_hizi(p_grid, a_grid, u_x)
    v_trail_right = _yari_sonsuz_iz_hizi(p_grid, b_grid, u_x)
    v_total = v_bound + v_trail_left + v_trail_right  # (N, N, 3)

    aic = np.sum(v_total * normals[:, None, :], axis=-1)  # (N, N)

    dy_wing = np.diff(y_edges)
    y_mid_wing = 0.5 * (y_edges[:-1] + y_edges[1:])
    theta_mid = np.arccos(np.clip(-2.0 * y_mid_wing / b_w, -0.999999, 0.999999))

    cl_wing_list: List[float] = []
    cdi_wing_list: List[float] = []
    e_wing_list: List[float] = []
    span_peak_ratio_list: List[float] = []

    for alpha_deg in alpha_list_deg:
        alpha_rad = math.radians(alpha_deg)
        u_inf = np.array([math.cos(alpha_rad), 0.0, math.sin(alpha_rad)], dtype=float)
        rhs = -np.sum(normals * u_inf[None, :], axis=-1)

        gamma = np.linalg.solve(aic, rhs)

        # Kanat şerit (strip) sirkülasyon toplamı
        gamma_strip = np.zeros(n_span_even, dtype=float)
        for idx_p in range(len(gamma)):
            if is_wing_mask[idx_p]:
                gamma_strip[span_indices[idx_p]] += gamma[idx_p]

        cl_wing = float((2.0 / s_w) * np.sum(gamma_strip * dy_wing))
        cl_wing_list.append(cl_wing)

        # Trefftz düzlemi Fourier ayrıştırması ile indüklenmiş sürükleme (CDi) ve Oswald e
        n_harmonics = min(13, n_span_even // 2)
        harmonics = np.arange(1, 2 * n_harmonics, 2, dtype=float)  # Tek harmonikler (simetrik yük)
        basis = np.sin(np.outer(theta_mid, harmonics))
        gamma_norm = gamma_strip / (2.0 * b_w)
        a_coeffs, *_ = np.linalg.lstsq(basis, gamma_norm, rcond=None)

        if abs(a_coeffs[0]) > 1e-9 and abs(cl_wing) > 1e-5:
            delta_ind = float(np.sum((harmonics[1:] / harmonics[0]) * ((a_coeffs[1:] / a_coeffs[0]) ** 2)))
            # Kroo/Raymer gövde-kanat etkileşimi ve viskoz indüklenmiş düzeltmesi
            e_inv = (1.0 + delta_ind) / 0.995 + 0.007 * math.pi * ar_w
            e_vlm = float(np.clip(1.0 / max(e_inv, 1e-6), 0.50, 0.99))
            cdi_wing = float((cl_wing ** 2) / (math.pi * ar_w * e_vlm))
        else:
            e_vlm = 0.82
            cdi_wing = float((cl_wing ** 2) / (math.pi * ar_w * e_vlm))

        e_wing_list.append(e_vlm)
        cdi_wing_list.append(cdi_wing)

        # Yerel kesit Cl_local / CL_wing tepe oranı (3B stall tahmini için)
        cl_local = (2.0 * gamma_strip) / c_w
        mean_cl = abs(cl_wing)
        if mean_cl > 0.05:
            span_peak_ratio_list.append(float(np.max(np.abs(cl_local)) / mean_cl))

    # Doğrusal en küçük kareler ile CL_alpha_3d (1/deg) ve alpha_L0_deg
    alphas_arr = np.asarray(alpha_list_deg, dtype=float)
    cls_arr = np.asarray(cl_wing_list, dtype=float)
    slope_per_deg, intercept = np.polyfit(alphas_arr, cls_arr, 1)
    alpha_l0_vlm = float(-intercept / slope_per_deg) if abs(slope_per_deg) > 1e-6 else alpha_l0_2d_deg

    # Tasarım hücum açılarındaki (2°..6°) ortalama Oswald verimliliği
    valid_e = [e for a, e in zip(alpha_list_deg, e_wing_list) if 1.5 <= a <= 6.5]
    e_oswald_vlm = float(np.mean(valid_e)) if valid_e else float(np.mean(e_wing_list))

    # 3B kritik kesit tepe yükleme faktörüne göre CL_max_3d düzeltmesi
    peak_ratio = float(np.median(span_peak_ratio_list)) if span_peak_ratio_list else 1.07
    cl_max_3d_vlm = float(cl_max_2d / max(peak_ratio, 1.02))

    return {
        "cozucu": "numpy_3d_vlm",
        "CL_alpha_3d_per_deg": round(float(slope_per_deg), 6),
        "alpha_L0_deg": round(alpha_l0_vlm, 4),
        "e_oswald": round(e_oswald_vlm, 5),
        "CL_max_3d_tahmini": round(cl_max_3d_vlm, 5),
        "peak_cl_ratio": round(peak_ratio, 4),
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
    speed: float = 20.0,
) -> Dict[str, Any]:
    """
    AeroSandbox VortexLatticeMethod çözücüsü (kanat_sekil_optimizasyonu.py ile uyumlu).
    Kurulu değilse otomatik olarak numpy_3d_vlm_coz fonksiyonuna geri döner.
    """
    try:
        import aerosandbox as asb
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
        )

    half_b = 0.5 * float(b_w)
    wing = asb.Wing(
        name="MainWing",
        symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0.0, 0.0, 0.0], chord=float(c_w), twist=0.0),
            asb.WingXSec(xyz_le=[0.0, half_b, 0.0], chord=float(c_w), twist=0.0),
        ],
    )
    wings = [wing]
    if b_ht > 0.05 and c_ht > 0.02 and tail_arm > 0.1:
        ht = asb.Wing(
            name="HTail",
            symmetric=True,
            xsecs=[
                asb.WingXSec(xyz_le=[float(tail_arm), 0.0, 0.0], chord=float(c_ht), twist=0.0),
                asb.WingXSec(xyz_le=[float(tail_arm), 0.5 * float(b_ht), 0.0], chord=float(c_ht), twist=0.0),
            ],
        )
        wings.append(ht)

    airplane = asb.Airplane(name="CandidateAircraft", wings=wings)
    alphas = (0.0, 4.0)
    cls = []
    es = []
    ar_w = (b_w ** 2) / max(s_w, 1e-6)

    for a_deg in alphas:
        vlm = asb.VortexLatticeMethod(
            airplane=airplane,
            op_point=asb.OperatingPoint(velocity=float(speed), alpha=float(a_deg - alpha_l0_2d_deg)),
            spanwise_resolution=12,
            chordwise_resolution=4,
        )
        res = vlm.run()
        cl_val = float(res["CL"])
        cdi_val = max(float(res["CDi"]), 1e-6)
        cls.append(cl_val)
        if abs(cl_val) > 0.05:
            e_val = (cl_val ** 2) / (math.pi * ar_w * cdi_val)
            es.append(float(np.clip(e_val, 0.55, 0.98)))

    cl_alpha_per_deg = (cls[1] - cls[0]) / (alphas[1] - alphas[0])
    e_oswald = float(np.mean(es)) if es else 0.82
    return {
        "cozucu": "aerosandbox_vlm",
        "CL_alpha_3d_per_deg": round(float(cl_alpha_per_deg), 6),
        "alpha_L0_deg": round(float(alpha_l0_2d_deg), 4),
        "e_oswald": round(e_oswald, 5),
        "CL_max_3d_tahmini": round(float(cl_max_2d * 0.92), 5),
        "peak_cl_ratio": 1.087,
    }


def vspaero_cli_coz(
    b_w: float,
    c_w: float,
    s_w: float,
    b_ht: float = 0.0,
    c_ht: float = 0.0,
    tail_arm: float = 0.0,
    alpha_l0_2d_deg: float = 0.0,
    cl_max_2d: float = 1.45,
    speed: float = 20.0,
) -> Dict[str, Any]:
    """
    Sistemde OpenVSP Python API + VSPAERO CLI kuruluysa DegenGeom üzerinden VSPAERO
    koşturur; kurulu değilse kesintisiz olarak 3B VLM çözücüsüne döner.
    """
    vspaero_bin = shutil.which("vspaero")
    try:
        import openvsp as vsp  # type: ignore
    except ImportError:
        vsp = None

    if vsp is None or vspaero_bin is None:
        return aerosandbox_vlm_coz(
            b_w=b_w,
            c_w=c_w,
            s_w=s_w,
            b_ht=b_ht,
            c_ht=c_ht,
            tail_arm=tail_arm,
            alpha_l0_2d_deg=alpha_l0_2d_deg,
            cl_max_2d=cl_max_2d,
            speed=speed,
        )

    try:
        with tempfile.TemporaryDirectory(prefix="vspaero_run_") as tmpdir:
            tmp_path = Path(tmpdir)
            vsp.VSPCheckSetup()
            vsp.ClearVSPModel()

            wid = vsp.AddGeom("WING")
            vsp.SetParmVal(wid, "Span", "XSec_1", float(b_w * 0.5))
            vsp.SetParmVal(wid, "Root_Chord", "XSec_1", float(c_w))
            vsp.SetParmVal(wid, "Tip_Chord", "XSec_1", float(c_w))
            vsp.Update()

            degen_file = str(tmp_path / "aday")
            vsp.SetComputationFileName(vsp.DEGEN_GEOM_CSV_TYPE, degen_file + "_DegenGeom.csv")
            vsp.ComputeDegenGeom(vsp.SET_ALL, vsp.DEGEN_GEOM_CSV_TYPE)

            setup_path = tmp_path / "aday_DegenGeom.vspaero"
            setup_path.write_text(
                f"Sref = {s_w:.6f}\n"
                f"Cref = {c_w:.6f}\n"
                f"Bref = {b_w:.6f}\n"
                "X_cg = 0.0\nY_cg = 0.0\nZ_cg = 0.0\n"
                "Mach = 0.0588\n"
                "AoA = 0.0, 4.0\n"
                "Beta = 0.0\n"
                f"Vinf = {speed:.2f}\n"
                "Rho = 1.175\n"
                "ReCref = 250000\n"
                "ClMax = -1\n"
                "MaxTurningAngle = -1\n"
                "Symmetry = Y\n"
                "FarDist = -1\n"
                "NumWakeNodes = 32\n"
                "WakeIters = 3\n",
                encoding="utf-8",
            )
            subprocess.run(
                [vspaero_bin, "-omp", "4", str(tmp_path / "aday_DegenGeom")],
                cwd=tmpdir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
            polar_file = tmp_path / "aday_DegenGeom.polar"
            if polar_file.is_file():
                lines = [ln.strip() for ln in polar_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
                data_rows = [ln.split() for ln in lines if not ln.startswith("#") and (ln[0].isdigit() or ln.startswith("-"))]
                if len(data_rows) >= 2:
                    aoa0, cl0 = float(data_rows[0][2]), float(data_rows[0][4])
                    aoa1, cl1 = float(data_rows[1][2]), float(data_rows[1][4])
                    e1 = float(data_rows[1][9]) if len(data_rows[1]) > 9 else 0.82
                    slope = (cl1 - cl0) / max(aoa1 - aoa0, 1e-6)
                    return {
                        "cozucu": "vspaero_cli",
                        "CL_alpha_3d_per_deg": round(slope, 6),
                        "alpha_L0_deg": round(alpha_l0_2d_deg, 4),
                        "e_oswald": round(float(np.clip(e1, 0.55, 0.98)), 5),
                        "CL_max_3d_tahmini": round(float(cl_max_2d * 0.92), 5),
                        "peak_cl_ratio": 1.087,
                    }
    except Exception:
        pass

    return numpy_3d_vlm_coz(
        b_w=b_w,
        c_w=c_w,
        s_w=s_w,
        b_ht=b_ht,
        c_ht=c_ht,
        tail_arm=tail_arm,
        alpha_l0_2d_deg=alpha_l0_2d_deg,
        cl_max_2d=cl_max_2d,
    )


# ---------------------------------------------------------------------------
# 2. İlk 500 Aday İçin VLM Analizi ve LLT/Drag Tablolarının Güncellenmesi
# ---------------------------------------------------------------------------

def _geometri_anahtari(row: Dict[str, Any]) -> Tuple[str, float, float, float]:
    """Kanat profili ve (b_w, S_w, c_w) üzerinden benzersiz geometri anahtarı üretir."""
    profil = metin_al(row, "kanat_airfoil", "profil").strip().lower()
    b_w = round(sayi_al(row, "b_w_m", "aciklik_m", varsayilan=0.0) or 0.0, 4)
    s_w = round(sayi_al(row, "S_w_m2", "alan_m2", varsayilan=0.0) or 0.0, 4)
    c_w = round(sayi_al(row, "c_w_m", "veter_m", varsayilan=0.0) or 0.0, 4)
    return (profil, b_w, s_w, c_w)


def ilk_500_aday_vlm_analizi(
    vlm_aday_satirlari: List[Dict[str, Any]],
    llt_flapsiz_satirlar: List[Dict[str, Any]],
    cozucu_tercihi: str = "auto",
    n_span: int = 24,
    n_chord: int = 4,
) -> Dict[Tuple[str, float, float, float], Dict[str, Any]]:
    """
    vlm_ilk_500_aday.csv içindeki benzersiz kanat-kuyruk geometrilerini VLM ile çözer.
    Her geometri için LLT referans değerlerini ve VLM güncellenmiş değerlerini döndürür.
    """
    motor = kullanilabilir_vlm_motoru_sec(cozucu_tercihi)

    # LLT tablosundan mevcut (alpha_L0_deg, CL_max_3d_tahmini, kuyruk boyutları) bilgilerini indeksle
    llt_geom_bilgi: Dict[Tuple[str, float, float, float], Dict[str, Any]] = {}
    for r in llt_flapsiz_satirlar:
        g_key = _geometri_anahtari(r)
        if g_key not in llt_geom_bilgi:
            llt_geom_bilgi[g_key] = r

    vlm_sonuc_haritasi: Dict[Tuple[str, float, float, float], Dict[str, Any]] = {}

    for aday in vlm_aday_satirlari:
        g_key = _geometri_anahtari(aday)
        if g_key in vlm_sonuc_haritasi or g_key[1] <= 0.0:
            continue

        llt_ref = llt_geom_bilgi.get(g_key, {})
        b_w = g_key[1]
        s_w = g_key[2]
        c_w = g_key[3] if g_key[3] > 0 else (s_w / max(b_w, 1e-6))

        s_ht = sayi_al(llt_ref, "S_HT_m2", varsayilan=0.0) or 0.0
        c_ht = sayi_al(llt_ref, "c_HT_m", varsayilan=0.0) or 0.0
        b_ht = (s_ht / c_ht) if (s_ht > 0 and c_ht > 0) else 0.0
        govde_l = sayi_al(aday, "govde_L_m", varsayilan=1.5) or 1.5
        tail_arm = 0.55 * govde_l

        alpha_l0_llt = sayi_al(llt_ref, "alpha_L0_deg", varsayilan=-2.5) or -2.5
        cl_alpha_llt = sayi_al(llt_ref, "CL_alpha_3d_per_deg", varsayilan=0.078) or 0.078
        e_llt = sayi_al(llt_ref, "e_oswald", "e_w", varsayilan=sayi_al(aday, "e_oswald", varsayilan=0.84)) or 0.84
        cl_max_3d_llt = sayi_al(llt_ref, "CL_max_3d_tahmini", varsayilan=1.35) or 1.35
        cl_max_2d_tahmin = cl_max_3d_llt / 0.92

        if motor == "vspaero":
            vlm_out = vspaero_cli_coz(
                b_w=b_w,
                c_w=c_w,
                s_w=s_w,
                b_ht=b_ht,
                c_ht=c_ht,
                tail_arm=tail_arm,
                alpha_l0_2d_deg=alpha_l0_llt,
                cl_max_2d=cl_max_2d_tahmin,
            )
        elif motor == "aerosandbox":
            vlm_out = aerosandbox_vlm_coz(
                b_w=b_w,
                c_w=c_w,
                s_w=s_w,
                b_ht=b_ht,
                c_ht=c_ht,
                tail_arm=tail_arm,
                alpha_l0_2d_deg=alpha_l0_llt,
                cl_max_2d=cl_max_2d_tahmin,
            )
        else:
            vlm_out = numpy_3d_vlm_coz(
                b_w=b_w,
                c_w=c_w,
                s_w=s_w,
                b_ht=b_ht,
                c_ht=c_ht,
                tail_arm=tail_arm,
                n_span=n_span,
                n_chord=n_chord,
                alpha_l0_2d_deg=alpha_l0_llt,
                cl_max_2d=cl_max_2d_tahmin,
            )

        cl_alpha_vlm = vlm_out["CL_alpha_3d_per_deg"]
        e_vlm = vlm_out["e_oswald"]
        cl_max_vlm = vlm_out["CL_max_3d_tahmini"]

        vlm_sonuc_haritasi[g_key] = {
            **vlm_out,
            "CL_alpha_llt": cl_alpha_llt,
            "e_oswald_llt": e_llt,
            "CL_max_3d_llt": cl_max_3d_llt,
            "cl_alpha_orani": cl_alpha_vlm / max(cl_alpha_llt, 1e-6),
            "e_oswald_orani": e_vlm / max(e_llt, 1e-6),
            "cl_max_orani": cl_max_vlm / max(cl_max_3d_llt, 1e-6),
        }

    return vlm_sonuc_haritasi


def llt_ve_drag_tablolarini_vlm_ile_guncelle(
    vlm_haritasi: Dict[Tuple[str, float, float, float], Dict[str, Any]],
    llt_satirlari: List[Dict[str, Any]],
    drag_polar_satirlari: List[Dict[str, Any]],
    drag_ozet_satirlari: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    İlk 500 adayın geometrilerine ait LLT ve OpenVSP+LLT sürükleme tablolarını
    VLM sonuçlarıyla günceller (performans modüllerinin doğrudan okuyabileceği formatta).
    """
    yeni_llt: List[Dict[str, Any]] = []
    for row in llt_satirlari:
        g_key = _geometri_anahtari(row)
        if g_key not in vlm_haritasi:
            continue
        vlm = vlm_haritasi[g_key]
        kopya = dict(row)
        kopya["e_oswald"] = f"{vlm['e_oswald']:.6f}"
        kopya["CL_alpha_3d_per_deg"] = f"{vlm['CL_alpha_3d_per_deg']:.6f}"
        kopya["CL_max_3d_tahmini"] = f"{vlm['CL_max_3d_tahmini']:.6f}"

        cl_eski = sayi_al(row, "CL")
        if cl_eski is not None:
            cl_yeni = cl_eski * vlm["cl_alpha_orani"]
            kopya["CL"] = f"{cl_yeni:.6f}"
        yeni_llt.append(kopya)

    yeni_drag_polar: List[Dict[str, Any]] = []
    for row in drag_polar_satirlari:
        g_key = _geometri_anahtari(row)
        if g_key not in vlm_haritasi:
            continue
        vlm = vlm_haritasi[g_key]
        kopya = dict(row)
        ar_w = sayi_al(row, "AR_w", varsayilan=(g_key[1] ** 2) / max(g_key[2], 1e-6)) or 6.0
        cd0 = sayi_al(row, "CD_0", "CD0_toplam", varsayilan=0.025) or 0.025
        cl_eski = sayi_al(row, "CL", varsayilan=0.0) or 0.0

        cl_yeni = cl_eski * vlm["cl_alpha_orani"]
        cdi_yeni = (cl_yeni ** 2) / (math.pi * ar_w * vlm["e_oswald"])
        cd_yeni = cd0 + cdi_yeni
        ld_yeni = cl_yeni / max(cd_yeni, 1e-6)

        kopya["e_oswald"] = f"{vlm['e_oswald']:.6f}"
        kopya["CL_alpha_3d_per_deg"] = f"{vlm['CL_alpha_3d_per_deg']:.6f}"
        kopya["CL_max_3d_tahmini"] = f"{vlm['CL_max_3d_tahmini']:.6f}"
        kopya["CL"] = f"{cl_yeni:.6f}"
        kopya["CD_i"] = f"{cdi_yeni:.6f}"
        kopya["CD"] = f"{cd_yeni:.6f}"
        kopya["L_D"] = f"{ld_yeni:.4f}"
        yeni_drag_polar.append(kopya)

    yeni_drag_ozet: List[Dict[str, Any]] = []
    for row in drag_ozet_satirlari:
        g_key = _geometri_anahtari(row)
        if g_key not in vlm_haritasi:
            continue
        vlm = vlm_haritasi[g_key]
        kopya = dict(row)
        ar_w = sayi_al(row, "AR_w", varsayilan=(g_key[1] ** 2) / max(g_key[2], 1e-6)) or 6.0
        cd0 = sayi_al(row, "CD_0_seyir", "CD0_toplam", varsayilan=0.025) or 0.025
        cl_seyir = sayi_al(row, "CL_seyir", varsayilan=0.5) or 0.5

        cl_yeni = cl_seyir * vlm["cl_alpha_orani"]
        cdi_yeni = (cl_yeni ** 2) / (math.pi * ar_w * vlm["e_oswald"])
        cd_yeni = cd0 + cdi_yeni
        ld_yeni = cl_yeni / max(cd_yeni, 1e-6)

        kopya["e_oswald"] = f"{vlm['e_oswald']:.6f}"
        kopya["CL_alpha_3d_per_deg"] = f"{vlm['CL_alpha_3d_per_deg']:.6f}"
        kopya["CL_max_3d_tahmini"] = f"{vlm['CL_max_3d_tahmini']:.6f}"
        kopya["CL_seyir"] = f"{cl_yeni:.6f}"
        kopya["CD_i_seyir"] = f"{cdi_yeni:.6f}"
        kopya["CD_seyir"] = f"{cd_yeni:.6f}"
        kopya["L_D_seyir"] = f"{ld_yeni:.4f}"
        yeni_drag_ozet.append(kopya)

    return yeni_llt, yeni_drag_polar, yeni_drag_ozet


# ---------------------------------------------------------------------------
# 3. VLM Sonuçlarıyla İlk 500 Adayın Yeniden Performans ve Puan Hesabı
# ---------------------------------------------------------------------------

def vlm_adaylarini_yeniden_puanla(
    ilk_500_satirlar: List[Dict[str, Any]],
    vlm_haritasi: Dict[Tuple[str, float, float, float], Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """
    VLM analizinden çıkan yeni 3B aerodinamik parametreleri (e_oswald_vlm, CL_alpha_3d_vlm,
    CL_max_3d_vlm) kullanarak ilk 500 adayın:
    - Seyir ve tırmanış indüklenmiş sürükleme (CDi) ve enerji (Wh / mAh) tüketimlerini,
    - Kalkış ve iniş stall hızı (V_stall) ile yer koşusu mesafelerini (S_G),
    - Enerji verimliliği puanını (puan_enerji_60_vlm) ve toplam puanını
    yeniden hesaplayıp sıralar.
    """
    yeniden_hesaplanan: List[Dict[str, Any]] = []

    for row in ilk_500_satirlar:
        g_key = _geometri_anahtari(row)
        vlm = vlm_haritasi.get(g_key)
        kopya = dict(row)

        eski_sira = int(sayi_al(row, "vlm_sira", "sira", varsayilan=0) or 0)
        eski_toplam_puan = sayi_al(row, "toplam_performans_puani", varsayilan=0.0) or 0.0
        eski_m2_puan = sayi_al(row, "mission2_400_puani", varsayilan=0.0) or 0.0

        ar_w = sayi_al(row, "AR_w", varsayilan=6.0) or 6.0
        cd0 = sayi_al(row, "CD_0", varsayilan=0.025) or 0.025
        e_llt = sayi_al(row, "e_oswald", varsayilan=0.84) or 0.84
        ld_yuklu_llt = sayi_al(row, "LD_yuklu_duz", varsayilan=12.0) or 12.0
        ld_bos_llt = sayi_al(row, "LD_bos_duz", varsayilan=12.0) or 12.0

        if vlm is not None:
            e_vlm = vlm["e_oswald"]
            cl_alpha_vlm = vlm["CL_alpha_3d_per_deg"]
            cl_max_vlm = vlm["CL_max_3d_tahmini"]
            e_orani = vlm["e_oswald_orani"]
            cl_max_orani = vlm["cl_max_orani"]
            cozucu_adi = vlm["cozucu"]
        else:
            e_vlm = e_llt
            cl_alpha_vlm = 0.078
            cl_max_vlm = 1.35
            e_orani = 1.0
            cl_max_orani = 1.0
            cozucu_adi = "mevcut_llt"

        # 1. VLM Oswald faktörü değişiminin indüklenmiş sürükleme ve seyir/tırmanış enerjisine etkisi
        # Seyirde toplam sürüklemenin yaklaşık %40'ı indüklenmiş sürüklemedir (CDi ~ 1 / e_oswald)
        cdi_carpani = 1.0 / max(e_orani, 0.5)
        toplam_drag_carpani = 0.60 + 0.40 * cdi_carpani

        ld_yuklu_vlm = round(ld_yuklu_llt / toplam_drag_carpani, 4)
        ld_bos_vlm = round(ld_bos_llt / toplam_drag_carpani, 4)

        # 2. VLM CL_max değişiminin kalkış ve iniş mesafelerine etkisi (S_G ~ 1 / CL_max)
        kalkis_mesafe_llt = sayi_al(row, "kalkis_mesafe_m", varsayilan=15.0) or 15.0
        inis_mesafe_llt = sayi_al(row, "inis_mesafe_m", varsayilan=18.0) or 18.0

        kalkis_mesafe_vlm = round(kalkis_mesafe_llt / max(cl_max_orani, 0.7), 3)
        inis_mesafe_vlm = round(inis_mesafe_llt / max(cl_max_orani, 0.7), 3)

        kalkis_20m_vlm = kalkis_mesafe_vlm <= 20.0
        inis_20m_vlm = inis_mesafe_vlm <= 20.0

        # 3. Kalkış ve İniş Puanlarının VLM Mesafelerine Göre Güncellenmesi
        puan_kalkis_60_llt = sayi_al(row, "resmi_kalkis_puani_60", varsayilan=60.0) or 0.0
        puan_kalkis_m2_20_llt = sayi_al(row, "puan_M2_yuklu_kalkis_20", varsayilan=20.0) or 0.0
        if kalkis_20m_vlm:
            puan_kalkis_60_vlm = max(puan_kalkis_60_llt, 60.0) if puan_kalkis_60_llt >= 45.0 else puan_kalkis_60_llt
            puan_kalkis_m2_20_vlm = 20.0
        else:
            # Yüklü kalkış 20m'yi aşarsa M2 kalkış puanı 20'den 5'e düşer
            puan_kalkis_m2_20_vlm = 5.0
            puan_kalkis_60_vlm = max(15.0, puan_kalkis_60_llt - (puan_kalkis_m2_20_llt - 5.0))

        puan_inis_140_llt = sayi_al(row, "resmi_inis_puani_140", varsayilan=140.0) or 0.0
        puan_inis_m2_60_llt = sayi_al(row, "puan_M2_inis_60", varsayilan=60.0) or 0.0
        if inis_20m_vlm:
            puan_inis_m2_60_vlm = 60.0
            puan_inis_140_vlm = 140.0 if puan_inis_140_llt >= 100.0 else puan_inis_140_llt
        else:
            # 20m kutuda teker koyup dışarıda durma: M1 ve M2'de 60p yerine 40p (M3 20p sabit -> toplam 100p)
            puan_inis_m2_60_vlm = 40.0 if puan_inis_m2_60_llt >= 40.0 else puan_inis_m2_60_llt
            puan_inis_140_vlm = 100.0 if puan_inis_140_llt >= 100.0 else puan_inis_140_llt

        # 4. Enerji ve Batarya Tüketimlerinin VLM Sürüklemesiyle Güncellenmesi
        kalkis_mah = (sayi_al(row, "kalkis_tuketim_mAh", varsayilan=0.0) or 0.0) / max(cl_max_orani, 0.7)
        tirmanis_mah = (sayi_al(row, "tirmanis_tuketim_mAh", varsayilan=0.0) or 0.0) * (0.75 + 0.25 * toplam_drag_carpani)
        seyir_mah = (sayi_al(row, "seyir_tuketim_mAh", varsayilan=0.0) or 0.0) * toplam_drag_carpani
        yuk_mah = (sayi_al(row, "yuk_birakma_tuketim_mAh", varsayilan=0.0) or 0.0) * (0.75 + 0.25 * toplam_drag_carpani)

        toplam_mah_vlm = round(kalkis_mah + tirmanis_mah + seyir_mah + yuk_mah + args.inis_ek_butce_mah, 3)
        toplam_enerji_wh_vlm = round(toplam_mah_vlm * GERILIM_V / 1000.0, 4)
        kalan_batarya_vlm = round(100.0 * (BATARYA_KAPASITE_MAH - toplam_mah_vlm) / BATARYA_KAPASITE_MAH, 2)

        batarya_uygun_vlm = toplam_mah_vlm <= KULLANILABILIR_MAH
        kalkis_uygun_vlm = bool_al(row, "kalkis_uygun", varsayilan=True) and (kalkis_20m_vlm if args.kalkis_20m_zorunlu else True)
        inis_uygun_vlm = bool_al(row, "inis_uygun", varsayilan=True) and (inis_20m_vlm if args.inis_20m_zorunlu else True)
        tum_fazlar_uygun_vlm = (
            kalkis_uygun_vlm
            and bool_al(row, "tirmanis_uygun", varsayilan=True)
            and bool_al(row, "seyir_uygun", varsayilan=True)
            and bool_al(row, "yuk_birakma_uygun", varsayilan=True)
            and inis_uygun_vlm
            and batarya_uygun_vlm
            and bool_al(row, "sure_uygun", varsayilan=True)
        )

        su_kg = sayi_al(row, "su_kutlesi_kg", varsayilan=0.0) or 0.0
        # Enerji verimliliği endeksi (M3.3.6: 20 yüklü waypoint * su_kg / Wh)
        ham_enerji_endeksi_vlm = (su_kg * 20.0) / max(toplam_enerji_wh_vlm, 1e-6) if su_kg > 0 else 0.0

        kopya.update({
            "vlm_cozucu": cozucu_adi,
            "llt_onc_sira": eski_sira,
            "e_oswald_llt": e_llt,
            "e_oswald_vlm": e_vlm,
            "CL_alpha_3d_vlm": cl_alpha_vlm,
            "CL_max_3d_vlm": cl_max_vlm,
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
        })
        yeniden_hesaplanan.append(kopya)

    # En yüksek enerji endeksine göre puan_enerji_60_vlm normalizasyonu
    uygun_endeksler = [
        r["ham_enerji_endeksi_vlm"]
        for r in yeniden_hesaplanan
        if r["tum_fazlar_uygun_vlm"] and r["ham_enerji_endeksi_vlm"] > 0
    ]
    if not uygun_endeksler:
        uygun_endeksler = [r["ham_enerji_endeksi_vlm"] for r in yeniden_hesaplanan if r["ham_enerji_endeksi_vlm"] > 0]
    best_index_vlm = max(uygun_endeksler) if uygun_endeksler else 1.0

    for r in yeniden_hesaplanan:
        puan_enerji_vlm = round(60.0 * r["ham_enerji_endeksi_vlm"] / max(best_index_vlm, 1e-9), 4)
        puan_nav = sayi_al(r, "puan_navigasyon_140", varsayilan=0.0) or 0.0
        puan_yuk = sayi_al(r, "puan_yuk_birakma_120", varsayilan=0.0) or 0.0

        toplam_puan_vlm = round(
            r["resmi_kalkis_puani_60_vlm"]
            + puan_nav
            + puan_enerji_vlm
            + puan_yuk
            + r["resmi_inis_puani_140_vlm"],
            4,
        )
        mission2_puan_vlm = round(
            r["puan_M2_yuklu_kalkis_20_vlm"]
            + puan_nav
            + puan_enerji_vlm
            + puan_yuk
            + r["puan_M2_inis_60_vlm"],
            4,
        )

        r["puan_enerji_60_vlm"] = puan_enerji_vlm
        r["vlm_sonrasi_toplam_puan"] = toplam_puan_vlm
        r["vlm_sonrasi_mission2_400_puani"] = mission2_puan_vlm
        r["puan_farki_vlm_eksi_llt"] = round(toplam_puan_vlm - r["llt_onc_toplam_puan"], 4)

    # VLM sonrası nihai sıralama
    puan_anahtari = (
        "vlm_sonrasi_mission2_400_puani"
        if args.siralama_puani == "mission2"
        else "vlm_sonrasi_toplam_puan"
    )
    yeniden_hesaplanan.sort(
        key=lambda r: (
            0 if r["tum_fazlar_uygun_vlm"] else 1,
            -(r[puan_anahtari] or 0.0),
            -(sayi_al(r, "tasarim_kalkis_puani_100", varsayilan=0.0) or 0.0),
            r["gercek_toplam_enerji_Wh_vlm"],
            sayi_al(r, "toplam_kutle_kg", varsayilan=99.0) or 99.0,
        )
    )

    for yeni_sira, r in enumerate(yeniden_hesaplanan, start=1):
        r["vlm_nihai_sira"] = yeni_sira
        r["sira_degisimi"] = r["llt_onc_sira"] - yeni_sira

    return yeniden_hesaplanan


def argumanlari_coz(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="İlk 500 adayı VSPAERO / VLM ile analiz edip tekrar puanlar ve nihai uçak sıralamasını kaydeder."
    )
    parser.add_argument(
        "--girdi",
        default="vlm_ilk_500_aday.csv",
        help="aday_siralama.py tarafından üretilen ilk 500 aday CSV dosyası (varsayılan: vlm_ilk_500_aday.csv)",
    )
    parser.add_argument(
        "--llt-flapsiz",
        dest="llt_flapsiz",
        default="llt_flapsiz_coklu_re.csv",
        help="Flapsız LLT çoklu Re CSV dosyası (varsayılan: llt_flapsiz_coklu_re.csv)",
    )
    parser.add_argument(
        "--drag-polar",
        dest="drag_polar",
        default="openvsp_llt_flapsiz_polar_detay.csv",
        help="Flapsız sürükleme polar detay CSV dosyası (varsayılan: openvsp_llt_flapsiz_polar_detay.csv)",
    )
    parser.add_argument(
        "--drag-ozet",
        dest="drag_ozet",
        default="openvsp_llt_flapsiz_aday_ozet.csv",
        help="Flapsız sürükleme aday özet CSV dosyası (varsayılan: openvsp_llt_flapsiz_aday_ozet.csv)",
    )
    parser.add_argument(
        "--cikti",
        default="vlm_nihai_ucak_siralamasi.csv",
        help="VLM sonrası nihai puanlanmış uçak sıralaması CSV çıktısı (varsayılan: vlm_nihai_ucak_siralamasi.csv)",
    )
    parser.add_argument(
        "--vlm-klasoru",
        dest="vlm_klasoru",
        default="vlm_sonuclari",
        help="VLM ile güncellenen aerodinamik tabloların kaydedileceği klasör (varsayılan: vlm_sonuclari)",
    )
    parser.add_argument(
        "--cozucu",
        choices=("auto", "vspaero", "aerosandbox", "numpy_vlm"),
        default="auto",
        help="Kullanılacak VLM çözücüsü (varsayılan: auto -> vspaero > aerosandbox > numpy_vlm)",
    )
    parser.add_argument(
        "--n-span",
        dest="n_span",
        type=int,
        default=24,
        help="VLM açıklık yönü (spanwise) panel sayısı (varsayılan: 24)",
    )
    parser.add_argument(
        "--n-chord",
        dest="n_chord",
        type=int,
        default=4,
        help="VLM veter yönü (chordwise) panel sayısı (varsayılan: 4)",
    )
    parser.add_argument(
        "--siralama-puani",
        dest="siralama_puani",
        choices=("modul_toplamlari", "mission2"),
        default="modul_toplamlari",
        help="Sıralamada baz alınacak toplam puan: 'modul_toplamlari' (520p) veya 'mission2' (400p)",
    )
    parser.add_argument(
        "--kalkis-20m-zorunlu",
        dest="kalkis_20m_zorunlu",
        action="store_true",
        default=True,
        help="20m kutu içi kalkışı zorunlu fizibilite kriteri olarak uygular (varsayılan: True).",
    )
    parser.add_argument(
        "--inis-20m-zorunlu",
        dest="inis_20m_zorunlu",
        action="store_true",
        default=False,
        help="20m kutu içi duruşu zorunlu fizibilite kriteri olarak uygular (varsayılan: False).",
    )
    parser.add_argument(
        "--inis-ek-butce-mah",
        dest="inis_ek_butce_mah",
        type=float,
        default=0.0,
        help="İniş/yaklaşma için toplam tüketime eklenecek ek mAh bütçesi (varsayılan: 0.0)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = argumanlari_coz(argv)

    girdi_yolu = dosya_yolu_coz(args.girdi)
    if not girdi_yolu.is_file():
        print(f"'{args.girdi}' bulunamadı; önce aday_siralama.py çalıştırılarak ilk 500 aday üretiliyor...")
        siralama_args = siralama_argumanlari_coz(["--vlm-cikti", args.girdi])
        tum_sirali, ilk_500 = adaylari_degerlendir_ve_sirala(siralama_args)
        girdi_yolu = SCRIPT_DIR / args.girdi
        csv_yaz(tum_sirali, SCRIPT_DIR / siralama_args.cikti)
        csv_yaz(ilk_500, girdi_yolu)
    else:
        ilk_500 = csv_oku(girdi_yolu)

    llt_yolu = dosya_yolu_coz(args.llt_flapsiz)
    llt_satirlari = csv_oku(llt_yolu) if llt_yolu.is_file() else []

    secilen_motor = kullanilabilir_vlm_motoru_sec(args.cozucu)
    print(f"VLM analizine alınan aday sayısı      : {len(ilk_500)}")
    print(f"Aktif 3B VLM çözücü motoru           : {secilen_motor}")

    vlm_haritasi = ilk_500_aday_vlm_analizi(
        vlm_aday_satirlari=ilk_500,
        llt_flapsiz_satirlar=llt_satirlari,
        cozucu_tercihi=args.cozucu,
        n_span=args.n_span,
        n_chord=args.n_chord,
    )
    print(f"Çözülen benzersiz 3B uçak geometrisi : {len(vlm_haritasi)}")

    # VLM ile güncellenen aerodinamik tabloları kaydet (isteğe bağlı tam modül tekrarı için)
    vlm_klasor_yolu = Path(args.vlm_klasoru)
    if not vlm_klasor_yolu.is_absolute():
        vlm_klasor_yolu = SCRIPT_DIR / vlm_klasor_yolu
    vlm_klasor_yolu.mkdir(parents=True, exist_ok=True)

    drag_polar_yolu = dosya_yolu_coz(args.drag_polar)
    drag_ozet_yolu = dosya_yolu_coz(args.drag_ozet)
    if llt_satirlari:
        yeni_llt, yeni_polar, yeni_ozet = llt_ve_drag_tablolarini_vlm_ile_guncelle(
            vlm_haritasi=vlm_haritasi,
            llt_satirlari=llt_satirlari,
            drag_polar_satirlari=csv_oku(drag_polar_yolu) if drag_polar_yolu.is_file() else [],
            drag_ozet_satirlari=csv_oku(drag_ozet_yolu) if drag_ozet_yolu.is_file() else [],
        )
        csv_yaz(yeni_llt, vlm_klasor_yolu / "vlm_flapsiz_coklu_re.csv")
        if yeni_polar:
            csv_yaz(yeni_polar, vlm_klasor_yolu / "vlm_flapsiz_polar_detay.csv")
        if yeni_ozet:
            csv_yaz(yeni_ozet, vlm_klasor_yolu / "vlm_flapsiz_aday_ozet.csv")

    # İlk 500 adayı VLM sonuçlarıyla yeniden puanla ve sırala
    nihai_sirali = vlm_adaylarini_yeniden_puanla(
        ilk_500_satirlar=ilk_500,
        vlm_haritasi=vlm_haritasi,
        args=args,
    )

    cikti_yolu = Path(args.cikti)
    if not cikti_yolu.is_absolute():
        cikti_yolu = SCRIPT_DIR / cikti_yolu
    csv_yaz(nihai_sirali, cikti_yolu)

    uygun_vlm = sum(1 for r in nihai_sirali if r["tum_fazlar_uygun_vlm"])
    print(f"VLM sonrası tüm fazları uygun aday   : {uygun_vlm}")
    print(f"Nihai VLM sıralama çıktısı           : {cikti_yolu}")

    if nihai_sirali:
        lider = nihai_sirali[0]
        print(
            "\nVLM Sonrası En İyi 1. Uçak:\n"
            f"  vlm_nihai_sira (Önceki) : {lider['vlm_nihai_sira']} (LLT Sırası: {lider['llt_onc_sira']}, Değişim: {lider['sira_degisimi']:+d})\n"
            f"  konfigurasyon_id        : {lider['konfigurasyon_id']}\n"
            f"  kanat_airfoil / pervane : {lider['kanat_airfoil']} / {lider['pervane']}\n"
            f"  e_oswald (LLT -> VLM)   : {lider['e_oswald_llt']} -> {lider['e_oswald_vlm']}\n"
            f"  vlm_sonrasi_toplam_puan : {lider['vlm_sonrasi_toplam_puan']} "
            f"(LLT Öncesi: {lider['llt_onc_toplam_puan']}, Fark: {lider['puan_farki_vlm_eksi_llt']:+.4f})\n"
            f"  vlm_mission2_400_puani  : {lider['vlm_sonrasi_mission2_400_puani']}\n"
            f"  toplam_tuketim_vlm      : {lider['gercek_toplam_tuketim_mAh_vlm']} mAh "
            f"(Kalan: %{lider['kalan_batarya_yuzde_vlm']})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())