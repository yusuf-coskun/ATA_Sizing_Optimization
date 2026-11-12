"""
ATA_Sizing_Optimization - Bellek İçi OpenVSP Parazit Sürükleme (CD0) ve LLT (CDi) Motoru
========================================================================================
Yeni '<kanat_airfoil>_flap.dat' adlandırma standardına göre güncellenmiştir:
- Flaplı kanat profil kalınlığı (t/c) doğrudan `airfoils_flap/<kanat_airfoil>_flap.dat`
  dosyasından okunur.
"""
from __future__ import annotations

import contextlib
import csv
import io
import math
import re
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJE_KLASORU = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "scripts" else SCRIPT_DIR
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJE_KLASORU) not in sys.path:
    sys.path.insert(0, str(PROJE_KLASORU))

try:
    with contextlib.redirect_stdout(io.StringIO()):
        import govde_boyutlandirma as gb
        import kanat_boyutlandirma as kb
except Exception:
    gb = None
    kb = None

import lifting_line_cozucu as llt

AIRFOIL_KLASORU = (
    PROJE_KLASORU / "airfoils"
    if (PROJE_KLASORU / "airfoils").exists()
    else SCRIPT_DIR / "airfoils"
)
AIRFOIL_FLAP_KLASORU = (
    PROJE_KLASORU / "airfoils_flap"
    if (PROJE_KLASORU / "airfoils_flap").exists()
    else SCRIPT_DIR / "airfoils_flap"
)

SEYIR_HIZI_MS = 18.0
KALKIS_HIZI_MS = 12.0
RHO = 1.175
MU = 1.789e-5
A_SOUND = 340.294
ALPHA_MIN, ALPHA_MAX, ALPHA_STEP = -5.0, 15.0, 0.5

Q_KANAT = 1.00
Q_HT = 1.20
Q_VT = 1.20
Q_GOVDE = 1.00

CIKTI_FLAPSIZ_OZET_CSV = PROJE_KLASORU / "openvsp_llt_flapsiz_aday_ozet.csv"
CIKTI_FLAPSIZ_POLAR_CSV = PROJE_KLASORU / "openvsp_llt_flapsiz_polar_detay.csv"
CIKTI_FLAPLI_OZET_CSV = PROJE_KLASORU / "openvsp_llt_flapli_aday_ozet.csv"
CIKTI_FLAPLI_POLAR_CSV = PROJE_KLASORU / "openvsp_llt_flapli_polar_detay.csv"

_tc_onbellek: dict[tuple[str, bool], float] = {}


def _temizle_profil_adi(ad: str) -> str:
    return re.sub(r"[^a-z0-9]", "", ad.lower())


def profil_dat_dosyasi_bul(dat_veya_ad: str | Path, flapli_klasor_mu: bool = False) -> Path:
    """
    Verilen profil adına (ör. 'clarkysm-il' veya 'clarkysm-il_flap') karşılık gelen
    .dat dosyasını 'airfoils' veya 'airfoils_flap' klasöründen doğrudan bulur.
    """
    kok_ad = Path(str(dat_veya_ad).strip()).stem
    if flapli_klasor_mu and not kok_ad.lower().endswith("_flap"):
        kok_ad_flap = f"{kok_ad}_flap"
    else:
        kok_ad_flap = kok_ad

    hedef_klasor = AIRFOIL_FLAP_KLASORU if flapli_klasor_mu else AIRFOIL_KLASORU

    # 1. Doğrudan yeni standart adla kontrol et (<kanat_airfoil>_flap.dat veya <kanat_airfoil>.dat)
    for aday_ad in (kok_ad_flap, kok_ad):
        dogrudan = hedef_klasor / f"{aday_ad}.dat"
        if dogrudan.is_file():
            return dogrudan

    # 2. Tam dosya yolu verilmişse ve mevcutsa
    yol = Path(str(dat_veya_ad).strip())
    if yol.is_file():
        return yol

    # 3. Normalize edilmiş isimle klasörde ara
    norm_aranan = _temizle_profil_adi(kok_ad_flap)
    if hedef_klasor.is_dir():
        for f in hedef_klasor.glob("*.dat"):
            if _temizle_profil_adi(f.stem) == norm_aranan:
                return f

    raise FileNotFoundError(f"Airfoil .dat dosyasi bulunamadi: {hedef_klasor / (kok_ad_flap + '.dat')}")


# -------------------------------------------------------------------
# 1) AIRFOIL .DAT DOSYASINDAN ÖNBELLEKLİ (t/c) KALINLIK ORANI OKUMA
# -------------------------------------------------------------------
def dat_dosyasindan_t_bolu_c_oku(
    dat_veya_ad: str | Path,
    flapli_klasor_mu: bool = False,
) -> float:
    kok_anahtar = Path(str(dat_veya_ad).strip()).stem.lower()
    anahtar = (kok_anahtar, bool(flapli_klasor_mu))
    if anahtar in _tc_onbellek:
        return _tc_onbellek[anahtar]

    yol = profil_dat_dosyasi_bul(dat_veya_ad, flapli_klasor_mu=flapli_klasor_mu)

    noktalar: list[tuple[float, float]] = []
    with yol.open("r", encoding="utf-8", errors="ignore") as f:
        for satir in f:
            parca = satir.strip().split()
            if len(parca) >= 2:
                try:
                    x, y = float(parca[0]), float(parca[1])
                    if -0.05 <= x <= 1.05 and -0.6 <= y <= 0.6:
                        noktalar.append((x, y))
                except ValueError:
                    continue

    if not noktalar:
        _tc_onbellek[anahtar] = 0.12
        return 0.12

    xs = np.array([p[0] for p in noktalar])
    ys = np.array([p[1] for p in noktalar])

    le_idx = int(np.argmin(xs))
    if 2 <= le_idx <= len(noktalar) - 3:
        x1, y1 = xs[: le_idx + 1], ys[: le_idx + 1]
        x2, y2 = xs[le_idx:], ys[le_idx:]
        s1, s2 = np.argsort(x1), np.argsort(x2)
        x_grid = np.linspace(0.05, 0.95, 91)
        y_ust = np.interp(x_grid, x1[s1], y1[s1])
        y_alt = np.interp(x_grid, x2[s2], y2[s2])
        tc_deger = float(np.max(np.abs(y_ust - y_alt)))
    else:
        tc_deger = float(2.0 * np.max(np.abs(ys)))

    _tc_onbellek[anahtar] = tc_deger
    return tc_deger


def govde_ozelliklerini_bul(aday: dict) -> tuple[float, float, float]:
    if all(k in aday and aday[k] not in (None, "") for k in ("govde_uzunlugu_m", "govde_D_m", "govde_S_toplam_m2")):
        return (
            float(aday["govde_uzunlugu_m"]),
            float(aday["govde_D_m"]),
            float(aday["govde_S_toplam_m2"]),
        )
    hedef_l = float(aday.get("govde_uzunlugu_m") or (getattr(gb, "L_total", 1.2) if gb else 1.2))
    if gb is not None and hasattr(gb, "L_total"):
        l_dizisi = np.ravel(gb.L_total).astype(float)
        d_dizisi = np.ravel(gb.D_arr).astype(float)
        s_dizisi = np.ravel(gb.S_toplam).astype(float)
        idx = int(np.argmin(np.abs(l_dizisi - hedef_l)))
        return float(l_dizisi[idx]), float(d_dizisi[idx]), float(s_dizisi[idx])
    return hedef_l, 0.119, 0.3707


# -------------------------------------------------------------------
# 2) OPENVSP EŞDEĞERİ PARAZİT SÜRÜKLEME (CD_0) HESABI
# -------------------------------------------------------------------
def hesapla_openvsp_cd0(
    S_w: float,
    Re_w: float,
    t_c_w: float,
    S_HT: float,
    Re_HT: float,
    t_c_HT: float,
    S_VT: float,
    Re_VT: float,
    t_c_VT: float,
    L_govde: float,
    D_govde: float,
    S_wet_govde: float,
    V_ref: float = SEYIR_HIZI_MS,
    rho: float = RHO,
    mu: float = MU,
    a_sound: float = A_SOUND,
    sweep_deg: float = 0.0,
) -> dict[str, float]:
    v_safe = max(float(V_ref), 0.5)
    S_wet_w = 2.0 * S_w * (1.0 + 0.2 * t_c_w)
    S_wet_HT = 2.0 * S_HT * (1.0 + 0.2 * t_c_HT)
    S_wet_VT = 2.0 * S_VT * (1.0 + 0.2 * t_c_VT)

    Re_govde = (rho * v_safe * L_govde) / mu
    Mach = v_safe / a_sound

    sikistirilabilirlik = (1.0 + 0.144 * (Mach ** 2)) ** 0.65

    Cf_w = 0.455 / ((math.log10(max(Re_w, 10.0)) ** 2.58) * sikistirilabilirlik)
    Cf_HT = 0.455 / ((math.log10(max(Re_HT, 10.0)) ** 2.58) * sikistirilabilirlik)
    Cf_VT = 0.455 / ((math.log10(max(Re_VT, 10.0)) ** 2.58) * sikistirilabilirlik)
    Cf_govde = 0.455 / ((math.log10(max(Re_govde, 10.0)) ** 2.58) * sikistirilabilirlik)

    cos2_sweep = math.cos(math.radians(sweep_deg)) ** 2
    FF_w = 1.0 + (3.3 * t_c_w - 0.008 * (t_c_w ** 2) + 27.0 * (t_c_w ** 3)) * cos2_sweep
    FF_HT = 1.0 + (3.52 * t_c_HT) * cos2_sweep
    FF_VT = 1.0 + (3.52 * t_c_VT) * cos2_sweep

    f_govde_narinlik = L_govde / D_govde
    FF_govde = 1.0 + 2.2 * (f_govde_narinlik ** -1.5) - 0.9 * (f_govde_narinlik ** -3.0)

    f_w = Cf_w * FF_w * Q_KANAT * S_wet_w
    f_HT = Cf_HT * FF_HT * Q_HT * S_wet_HT
    f_VT = Cf_VT * FF_VT * Q_VT * S_wet_VT
    f_govde = Cf_govde * FF_govde * Q_GOVDE * S_wet_govde
    f_toplam = f_w + f_HT + f_VT + f_govde

    CD0_w = f_w / S_w
    CD0_HT = f_HT / S_w
    CD0_VT = f_VT / S_w
    CD0_govde = f_govde / S_w
    CD0_toplam = f_toplam / S_w

    return {
        "t_c_w": t_c_w,
        "t_c_HT": t_c_HT,
        "t_c_VT": t_c_VT,
        "S_wet_w_m2": S_wet_w,
        "S_wet_HT_m2": S_wet_HT,
        "S_wet_VT_m2": S_wet_VT,
        "S_wet_govde_m2": S_wet_govde,
        "Re_w": Re_w,
        "Re_HT": Re_HT,
        "Re_VT": Re_VT,
        "Re_govde": Re_govde,
        "Cf_w": Cf_w,
        "Cf_HT": Cf_HT,
        "Cf_VT": Cf_VT,
        "Cf_govde": Cf_govde,
        "FF_w": FF_w,
        "FF_HT": FF_HT,
        "FF_VT": FF_VT,
        "FF_govde": FF_govde,
        "f_w_m2": f_w,
        "f_HT_m2": f_HT,
        "f_VT_m2": f_VT,
        "f_govde_m2": f_govde,
        "f_toplam_m2": f_toplam,
        "CD0_w": CD0_w,
        "CD0_HT": CD0_HT,
        "CD0_VT": CD0_VT,
        "CD0_govde": CD0_govde,
        "CD0_0": CD0_toplam,
        "CD0_toplam": CD0_toplam,
    }


def aday_cd0_hesapla(
    aday: dict,
    hiz_ms: float,
    rho: float = RHO,
    mu: float = MU,
    flapli_mi: bool = False,
    flap_profil_adi: str | None = None,
    oran_flap: float | None = None,
) -> dict[str, float]:
    kanat_profil = Path(str(aday.get("kanat_airfoil") or aday.get("profil"))).stem.strip()
    kuyruk_profil = Path(str(aday.get("kuyruk_airfoil") or aday.get("ht_airfoil") or "naca0010")).stem.strip()

    t_c_w_flapsiz = dat_dosyasindan_t_bolu_c_oku(kanat_profil, flapli_klasor_mu=False)
    t_c_kuyruk = dat_dosyasindan_t_bolu_c_oku(kuyruk_profil, flapli_klasor_mu=False)

    b_w = llt.aday_degeri_al(aday, "b_w_m")
    s_w = llt.aday_degeri_al(aday, "S_w_m2")
    c_w = llt.aday_degeri_al(aday, "c_root_w_m", "c_w_m")

    if flapli_mi:
        f_prof = flap_profil_adi or f"{kanat_profil}_flap"
        t_c_w_flapli = dat_dosyasindan_t_bolu_c_oku(f_prof, flapli_klasor_mu=True)
        if oran_flap is None:
            b_flap = llt.aday_degeri_al(aday, "b_flap_m", varsayilan=min(1.0, b_w))
            oran_flap = b_flap / b_w
        t_c_w = oran_flap * t_c_w_flapli + (1.0 - oran_flap) * t_c_w_flapsiz
    else:
        t_c_w = t_c_w_flapsiz

    s_ht = llt.aday_degeri_al(aday, "S_HT_m2", "S_h_m2")
    c_ht = llt.aday_degeri_al(aday, "c_root_HT_m", "c_root_h_m", "c_HT_m")
    s_vt = llt.aday_degeri_al(aday, "S_VT_m2", "S_v_m2")
    c_vt = llt.aday_degeri_al(aday, "c_root_VT_m", "c_root_v_m", "c_VT_m")

    L_govde, D_govde, S_wet_govde = govde_ozelliklerini_bul(aday)

    v_safe = max(float(hiz_ms), 0.5)
    re_w = (rho * v_safe * c_w) / mu
    re_ht = (rho * v_safe * c_ht) / mu
    re_vt = (rho * v_safe * c_vt) / mu

    return hesapla_openvsp_cd0(
        S_w=s_w,
        Re_w=re_w,
        t_c_w=t_c_w,
        S_HT=s_ht,
        Re_HT=re_ht,
        t_c_HT=t_c_kuyruk,
        S_VT=s_vt,
        Re_VT=re_vt,
        t_c_VT=t_c_kuyruk,
        L_govde=L_govde,
        D_govde=D_govde,
        S_wet_govde=S_wet_govde,
        V_ref=v_safe,
        rho=rho,
        mu=mu,
    )


def csv_yaz(satirlar: list[dict], yol: Path) -> None:
    if not satirlar:
        return
    yol.parent.mkdir(parents=True, exist_ok=True)
    with yol.open("w", newline="", encoding="utf-8-sig") as f:
        y = csv.DictWriter(f, fieldnames=list(satirlar[0].keys()))
        y.writeheader()
        for s in satirlar:
            y.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in s.items()})


def grubu_hesapla_ve_kaydet(
    adaylar: list[dict],
    cikti_ozet_csv: Path,
    cikti_polar_csv: Path,
    flapli_mi: bool = False,
) -> tuple[int, int]:
    v_ref = KALKIS_HIZI_MS if flapli_mi else SEYIR_HIZI_MS
    alfa_dizisi = llt.ALFA_DIZISI_DEG

    ozet_satirlari: list[dict] = []
    polar_detay_satirlari: list[dict] = []

    for no, aday in enumerate(adaylar, start=1):
        kanat_profil = Path(str(aday.get("kanat_airfoil") or aday.get("profil"))).stem.strip()
        kuyruk_profil = Path(str(aday.get("kuyruk_airfoil") or aday.get("ht_airfoil") or "naca0010")).stem.strip()
        flap_profil = f"{kanat_profil}_flap" if flapli_mi else ""

        b_w = llt.aday_degeri_al(aday, "b_w_m")
        AR_w = llt.aday_degeri_al(aday, "AR_w")
        S_w = llt.aday_degeri_al(aday, "S_w_m2")
        c_w = llt.aday_degeri_al(aday, "c_root_w_m", "c_w_m")
        b_flap = llt.aday_degeri_al(aday, "b_flap_m", varsayilan=min(1.0, b_w))
        oran_flap = b_flap / b_w

        S_HT = llt.aday_degeri_al(aday, "S_HT_m2", "S_h_m2")
        b_HT = llt.aday_degeri_al(aday, "b_HT_m", "b_h_m")
        c_HT = llt.aday_degeri_al(aday, "c_root_HT_m", "c_root_h_m", "c_HT_m")

        S_VT = llt.aday_degeri_al(aday, "S_VT_m2", "S_v_m2")
        h_VT = llt.aday_degeri_al(aday, "h_VT_m", "b_v_m")
        c_VT = llt.aday_degeri_al(aday, "c_root_VT_m", "c_root_v_m", "c_VT_m")
        tail_arm = llt.aday_degeri_al(aday, "tail_arm_m", "l_h_m", varsayilan=0.85)

        L_govde, D_govde, S_wet_govde = govde_ozelliklerini_bul(aday)

        vsp = aday_cd0_hesapla(
            aday, hiz_ms=v_ref, rho=RHO, mu=MU,
            flapli_mi=flapli_mi, flap_profil_adi=flap_profil, oran_flap=oran_flap,
        )
        Re_w = vsp["Re_w"]

        if flapli_mi:
            llt_veri = llt.flapli_kanat_coz_re(kanat_profil, flap_profil, AR_w, oran_flap, Re_w)
        else:
            llt_veri = llt.flapsiz_kanat_coz_re(kanat_profil, AR_w, Re_w)

        CL_alpha_3d = llt_veri["CL_alpha_3d_per_deg"]
        alpha_L0_deg = llt_veri["alpha_L0_deg"]
        e_oswald = llt_veri["e_oswald"]
        CL_seyir = llt_veri["CL_seyir"]
        CL_max_3d = llt_veri["CL_max_3d_tahmini"]
        curve = llt_veri["curve"]

        CDi_seyir = (CL_seyir ** 2) / (math.pi * AR_w * e_oswald)
        CD_0 = vsp["CD0_0"]
        CD_seyir_toplam = CD_0 + CDi_seyir
        L_D_seyir_toplam = CL_seyir / CD_seyir_toplam if CD_seyir_toplam > 0 else np.nan

        temel_bilgi = {
            "aday_no": no,
            "kanat_airfoil": kanat_profil,
            "flap_airfoil": flap_profil,
            "kuyruk_airfoil": kuyruk_profil,
            "V_seyir_ms": v_ref,
            "b_w_m": b_w,
            "AR_w": AR_w,
            "S_w_m2": S_w,
            "c_w_m": c_w,
            "S_HT_m2": S_HT,
            "b_HT_m": b_HT,
            "c_HT_m": c_HT,
            "S_VT_m2": S_VT,
            "h_VT_m": h_VT,
            "c_VT_m": c_VT,
            "L_govde_m": L_govde,
            "D_govde_m": D_govde,
            "tail_arm_m": tail_arm,
            "Re_aday": Re_w,
            **vsp,
            "CL_alpha_3d_per_deg": CL_alpha_3d,
            "alpha_L0_deg": alpha_L0_deg,
            "e_oswald": e_oswald,
            "CL_max_3d_tahmini": CL_max_3d,
        }

        ozet_satirlari.append({
            **temel_bilgi,
            "CL_seyir": CL_seyir,
            "CD_i_seyir": CDi_seyir,
            "CD_0_seyir": CD_0,
            "CD_seyir": CD_seyir_toplam,
            "L_D_seyir": L_D_seyir_toplam,
        })

        for alfa_deg in alfa_dizisi:
            cl_alfa = curve[float(alfa_deg)]
            cd_i_alfa = (cl_alfa ** 2) / (math.pi * AR_w * e_oswald)
            cd_toplam_alfa = CD_0 + cd_i_alfa
            l_d_alfa = cl_alfa / cd_toplam_alfa if cd_toplam_alfa > 0 else np.nan

            polar_detay_satirlari.append({
                **temel_bilgi,
                "alpha_deg": float(alfa_deg),
                "CL": cl_alfa,
                "CD_i": cd_i_alfa,
                "CD_0": CD_0,
                "CD": cd_toplam_alfa,
                "L_D": l_d_alfa,
            })

    csv_yaz(ozet_satirlari, cikti_ozet_csv)
    csv_yaz(polar_detay_satirlari, cikti_polar_csv)
    return len(ozet_satirlari), len(polar_detay_satirlari)


def main() -> None:
    print("1) Flapsiz ucak adaylari icin OpenVSP CD0 + LLT CDi bellek icinde hesaplaniyor...")
    flapsiz_adaylar = llt.adaylari_yukle(llt.FLAPSIZ_ADAY_CSV_ADI, flapli_mi=False)
    n_flapsiz_ozet, n_flapsiz_polar = grubu_hesapla_ve_kaydet(
        flapsiz_adaylar, CIKTI_FLAPSIZ_OZET_CSV, CIKTI_FLAPSIZ_POLAR_CSV, flapli_mi=False
    )
    print(f"  - Kaydedildi: {CIKTI_FLAPSIZ_OZET_CSV} ({n_flapsiz_ozet} aday)")
    print(f"  - Kaydedildi: {CIKTI_FLAPSIZ_POLAR_CSV} ({n_flapsiz_polar} satir)")

    print("\n2) Flapli ucak adaylari icin OpenVSP CD0 + LLT CDi bellek icinde hesaplaniyor...")
    flapli_adaylar = llt.adaylari_yukle(llt.FLAPLI_ADAY_CSV_ADI, flapli_mi=True)
    n_flapli_ozet, n_flapli_polar = grubu_hesapla_ve_kaydet(
        flapli_adaylar, CIKTI_FLAPLI_OZET_CSV, CIKTI_FLAPLI_POLAR_CSV, flapli_mi=True
    )
    print(f"  - Kaydedildi: {CIKTI_FLAPLI_OZET_CSV} ({n_flapli_ozet} aday)")
    print(f"  - Kaydedildi: {CIKTI_FLAPLI_POLAR_CSV} ({n_flapli_polar} satir)")


if __name__ == "__main__":
    main()