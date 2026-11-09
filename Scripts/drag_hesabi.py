"""
ATA_Sizing_Optimization - OpenVSP Eşdeğeri Parazit Sürükleme (CD0) ve LLT (CDi) Hesaplayıcı
==========================================================================================
Bu betik:
1. 'govde_boyutlandirma.py' içindeki global dizilerden (D_arr, L_total, S_toplam) her gövde
   konfigürasyonunun çapını, uzunluğunu ve ıslak alanını doğrudan bellekte okur.
2. Flapsız adayları ('kanat_boyutlandirma.tum_adaylari_uret()') ve flaplı adayları
   ('kanat_boyutlandirma.tum_flapli_adaylari_uret()') alır.
3. LLT sonuçlarını doğrudan 'llt_flapsiz_aday_re.csv' ve 'llt_flapli_aday_re.csv'
   dosyalarından okur (dosyalar yoksa 'lifting line cozucu.py'yi çalıştırarak üretir).
4. 'CDi_seyir' değerini LLT dosyasından okumaz; doğrudan bu dosyada
   CDi_seyir = CL_seyir^2 / (pi * AR_w * e_oswald) formülüyle hesaplar.
5. OpenVSP (Schlichting Compressible Cf + Jenkinson Form Factor + Q) yöntemiyle iniş takımsız
   CD_0 değerini ve -5° .. +15° aralığındaki CL, CD_i, CD_0, CD, L_D değerlerini hesaplayıp:
   - openvsp_llt_flapsiz_aday_ozet.csv
   - openvsp_llt_flapsiz_polar_detay.csv
   - openvsp_llt_flapli_aday_ozet.csv
   - openvsp_llt_flapli_polar_detay.csv
   dosyalarını oluşturur.
"""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import math
import sys
from pathlib import Path

import numpy as np

# 1. PROJE DİZİNLERİ VE MEVCUT MODÜLLERİN BELLEKTEN İMPORT EDİLMESİ
SCRIPT_DIR = Path(__file__).resolve().parent
PROJE_KLASORU = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "scripts" else SCRIPT_DIR
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJE_KLASORU) not in sys.path:
    sys.path.insert(0, str(PROJE_KLASORU))

with contextlib.redirect_stdout(io.StringIO()):
    import govde_boyutlandirma as gb
    import kanat_boyutlandirma as kb

AIRFOIL_KLASORU = kb.AIRFOIL_KLASORU
AIRFOIL_FLAP_KLASORU = kb.AIRFOIL_FLAP_KLASORU

# lifting line cozucu.py ile birebir aynı akışkan ve açı sabitleri
SEYIR_HIZI_MS = 18.0
RHO = 1.175
MU = 1.789e-5
A_SOUND = 340.294
ALPHA_MIN, ALPHA_MAX, ALPHA_STEP = -5.0, 15.0, 0.5

# OpenVSP Etkileşim Faktörleri (Q) - Temiz Geometri (İniş Takımsız)
Q_KANAT = 1.00
Q_HT = 1.20
Q_VT = 1.20
Q_GOVDE = 1.00

# Girdi LLT CSV dosyaları
LLT_FLAPSIZ_ADAY_RE_CSV = PROJE_KLASORU / "llt_flapsiz_aday_re.csv"
LLT_FLAPLI_ADAY_RE_CSV = PROJE_KLASORU / "llt_flapli_aday_re.csv"

# Çıktı OpenVSP + LLT CSV dosyaları
CIKTI_FLAPSIZ_OZET_CSV = PROJE_KLASORU / "openvsp_llt_flapsiz_aday_ozet.csv"
CIKTI_FLAPSIZ_POLAR_CSV = PROJE_KLASORU / "openvsp_llt_flapsiz_polar_detay.csv"
CIKTI_FLAPLI_OZET_CSV = PROJE_KLASORU / "openvsp_llt_flapli_aday_ozet.csv"
CIKTI_FLAPLI_POLAR_CSV = PROJE_KLASORU / "openvsp_llt_flapli_polar_detay.csv"


# 2. AIRFOIL .DAT DOSYASINDAN KALINLIK ORANI (t/c) HESAPLAMA
def dat_dosyasindan_t_bolu_c_oku(
    dat_veya_ad: str | Path,
    varsayilan_klasor: Path = AIRFOIL_KLASORU,
) -> float:
    """
    Airfoil .dat dosyasındaki koordinatlardan profilin maksimum kalınlık/veter
    oranını (t/c) hesaplar.
    """
    yol = Path(dat_veya_ad)
    if not yol.is_file():
        aday_yollar = [
            PROJE_KLASORU / yol,
            varsayilan_klasor / f"{yol.stem}.dat",
            AIRFOIL_KLASORU / f"{yol.stem}.dat",
            AIRFOIL_FLAP_KLASORU / f"{yol.stem}.dat",
        ]
        for a_yol in aday_yollar:
            if a_yol.is_file():
                yol = a_yol
                break

    if not yol.is_file():
        raise FileNotFoundError(f"Airfoil .dat dosyası bulunamadı: {yol}")

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
        return float(np.max(np.abs(y_ust - y_alt)))

    return float(2.0 * np.max(np.abs(ys)))


# 3. GÖVDE BOYUTLANDIRMA VERİLERİNİ BELLEKTEN EŞLEŞTİRME
def govde_ozelliklerini_bul(hedef_l_govde: float) -> tuple[float, float, float]:
    """
    'govde_boyutlandirma.py' içinde bellekte tutulan global dizilerden
    (D_arr, L_total, S_toplam) adayın gövde uzunluğuna (govde_uzunlugu_m)
    karşılık gelen (L_govde, D_govde, S_wet_govde) değerlerini döndürür.
    """
    l_dizisi = np.ravel(gb.L_total).astype(float)
    d_dizisi = np.ravel(gb.D_arr).astype(float)
    s_dizisi = np.ravel(gb.S_toplam).astype(float)

    idx = int(np.argmin(np.abs(l_dizisi - float(hedef_l_govde))))
    return float(l_dizisi[idx]), float(d_dizisi[idx]), float(s_dizisi[idx])


# 4. OPENVSP EŞDEĞERİ (İNİŞ TAKIMSIZ) PARAZİT SÜRÜKLEME (CD_0) HESABI
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
    V_seyir: float = SEYIR_HIZI_MS,
    rho: float = RHO,
    mu: float = MU,
    a_sound: float = A_SOUND,
    sweep_deg: float = 0.0,
) -> dict[str, float]:
    """
    OpenVSP Parasite Drag modülü ile aynı denklemleri kullanarak her bileşenin
    ıslak alanını (S_wet), sürtünme katsayısını (Cf - Schlichting Compressible),
    şekil faktörünü (FF - Jenkinson) ve parazit sürükleme katsayısını (CD_0) hesaplar.
    """
    # 1. Kanat ve Kuyruk Islak Alanları
    S_wet_w = 2.0 * S_w * (1.0 + 0.2 * t_c_w)
    S_wet_HT = 2.0 * S_HT * (1.0 + 0.2 * t_c_HT)
    S_wet_VT = 2.0 * S_VT * (1.0 + 0.2 * t_c_VT)

    # 2. Gövde Reynolds Sayısı ve Mach Sayısı
    Re_govde = (rho * V_seyir * L_govde) / mu
    Mach = V_seyir / a_sound

    sikistirilabilirlik = (1.0 + 0.144 * (Mach ** 2)) ** 0.65

    # 3. Türbülanslı Yüzey Sürtünme Katsayısı (OpenVSP: Schlichting Compressible)
    Cf_w = 0.455 / ((math.log10(Re_w) ** 2.58) * sikistirilabilirlik)
    Cf_HT = 0.455 / ((math.log10(Re_HT) ** 2.58) * sikistirilabilirlik)
    Cf_VT = 0.455 / ((math.log10(Re_VT) ** 2.58) * sikistirilabilirlik)
    Cf_govde = 0.455 / ((math.log10(Re_govde) ** 2.58) * sikistirilabilirlik)

    # 4. Şekil Faktörleri (OpenVSP: Jenkinson Wing, Tail, Fuselage)
    cos2_sweep = math.cos(math.radians(sweep_deg)) ** 2
    FF_w = 1.0 + (3.3 * t_c_w - 0.008 * (t_c_w ** 2) + 27.0 * (t_c_w ** 3)) * cos2_sweep
    FF_HT = 1.0 + (3.52 * t_c_HT) * cos2_sweep
    FF_VT = 1.0 + (3.52 * t_c_VT) * cos2_sweep

    f_govde_narinlik = L_govde / D_govde
    FF_govde = 1.0 + 2.2 * (f_govde_narinlik ** -1.5) - 0.9 * (f_govde_narinlik ** -3.0)

    # 5. Eşdeğer Düz Plaka Alanları (f = Cf * FF * Q * S_wet) ve CD_0 Katkıları
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
        "S_wet_w_m2": S_wet_w,
        "S_wet_HT_m2": S_wet_HT,
        "S_wet_VT_m2": S_wet_VT,
        "S_wet_govde_m2": S_wet_govde,
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
        "CD_0": CD0_toplam,
    }


# 5. LLT SONUÇLARINI CSV'DEN OKUMA (CDi_seyir BURADA OKUNMAZ, DRAG HESABINDA HESAPLANIR)
def _llt_dosyalarini_kontrol_et_veya_uret() -> None:
    if LLT_FLAPSIZ_ADAY_RE_CSV.is_file() and LLT_FLAPLI_ADAY_RE_CSV.is_file():
        return

    olasi_llt_betikleri = [
        SCRIPT_DIR / "lifting line cozucu.py",
        SCRIPT_DIR / "lifting_line_cozucu.py",
    ]
    llt_dosya = next((p for p in olasi_llt_betikleri if p.is_file()), None)
    if llt_dosya is None:
        raise FileNotFoundError(
            f"LLT aday Re dosyalari ({LLT_FLAPSIZ_ADAY_RE_CSV.name}, "
            f"{LLT_FLAPLI_ADAY_RE_CSV.name}) bulunamadi ve LLT cozucu betigi mevcut degil."
        )

    spec = importlib.util.spec_from_file_location("llt_cozucu_mod", llt_dosya)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"LLT betigi yuklenemedi: {llt_dosya}")
    llt_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(llt_mod)
    llt_mod.main()


def llt_aday_re_csv_oku(
    csv_yolu: Path,
    flapli_mi: bool = False,
) -> dict[tuple[str, float, float], dict[str, float]]:
    """
    'llt_flapsiz_aday_re.csv' veya 'llt_flapli_aday_re.csv' dosyasından
    yalnızca taşıma parametrelerini (CL_alpha_3d_per_deg, alpha_L0_deg, e_oswald, CL_seyir)
    okur. CDi_seyir LLT dosyasından okunmaz; drag_hesabi.py içinde hesaplanır.
    """
    _llt_dosyalarini_kontrol_et_veya_uret()

    llt_harita: dict[tuple[str, float, float], dict[str, float]] = {}
    with csv_yolu.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for row in csv.DictReader(f):
            prof = (row.get("profil") or row.get("kanat_airfoil") or "").strip()
            b_w = round(float(row["b_w_m"]), 4)
            ar_w = round(float(row["AR_w"]), 4)

            if flapli_mi:
                cl_alpha_3d = float(
                    row.get("CL_alpha_3d_per_deg")
                    or row.get("CL_alpha_3d_kanat_per_deg")
                )
                e_oswald = float(
                    row.get("e_oswald") or row.get("e_oswald_kanat")
                )
                cl_seyir = float(row["CL_seyir"])
                if "alpha_L0_deg" in row and row["alpha_L0_deg"] != "":
                    alpha_l0_deg = float(row["alpha_L0_deg"])
                else:
                    alpha_l0_deg = -cl_seyir / cl_alpha_3d if cl_alpha_3d > 0 else 0.0
            else:
                cl_alpha_3d = float(row["CL_alpha_3d_per_deg"])
                alpha_l0_deg = float(row["alpha_L0_deg"])
                e_oswald = float(row["e_oswald"])
                cl_seyir = float(row["CL_seyir"])

            llt_harita[(prof, b_w, ar_w)] = {
                "CL_alpha_3d_per_deg": cl_alpha_3d,
                "alpha_L0_deg": alpha_l0_deg,
                "e_oswald": e_oswald,
                "CL_seyir": cl_seyir,
            }

    return llt_harita


# 6. CSV YAZMA FONKSİYONU
def csv_yaz(satirlar: list[dict], yol: Path) -> None:
    if not satirlar:
        return
    yol.parent.mkdir(parents=True, exist_ok=True)
    with yol.open("w", newline="", encoding="utf-8-sig") as f:
        y = csv.DictWriter(f, fieldnames=list(satirlar[0].keys()))
        y.writeheader()
        for s in satirlar:
            y.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in s.items()})


# 7. FLAPSIZ VE FLAPLI ADAYLAR İÇİN OPENVSP + LLT SÜRÜKLEME HESABI
def grubu_hesapla_ve_kaydet(
    adaylar: list[dict],
    llt_harita: dict[tuple[str, float, float], dict[str, float]],
    cikti_ozet_csv: Path,
    cikti_polar_csv: Path,
    flapli_mi: bool = False,
) -> tuple[int, int]:
    t_c_onbellek: dict[str, float] = {}
    nu = MU / RHO
    alfa_dizisi = np.arange(ALPHA_MIN, ALPHA_MAX + 1e-9, ALPHA_STEP)

    ozet_satirlari: list[dict] = []
    polar_detay_satirlari: list[dict] = []

    for no, aday in enumerate(adaylar, start=1):
        kanat_profil = str(aday["kanat_airfoil"])
        kuyruk_profil = str(aday["kuyruk_airfoil"])

        anahtar = (kanat_profil, round(float(aday["b_w_m"]), 4), round(float(aday["AR_w"]), 4))
        if anahtar not in llt_harita:
            continue
        llt_veri = llt_harita[anahtar]

        # Kanat ve kuyruk .dat dosyalarından (t/c) değerlerini oku (önbellekli)
        if kanat_profil not in t_c_onbellek:
            t_c_onbellek[kanat_profil] = dat_dosyasindan_t_bolu_c_oku(
                aday["kanat_airfoil_dat"], AIRFOIL_KLASORU
            )
        if kuyruk_profil not in t_c_onbellek:
            t_c_onbellek[kuyruk_profil] = dat_dosyasindan_t_bolu_c_oku(
                aday["kuyruk_airfoil_dat"], AIRFOIL_KLASORU
            )

        t_c_w_flapsiz = t_c_onbellek[kanat_profil]
        t_c_kuyruk = t_c_onbellek[kuyruk_profil]

        if flapli_mi and "flap_airfoil" in aday:
            flap_profil = str(aday["flap_airfoil"])
            if flap_profil not in t_c_onbellek:
                t_c_onbellek[flap_profil] = dat_dosyasindan_t_bolu_c_oku(
                    aday["flap_airfoil_dat"], AIRFOIL_FLAP_KLASORU
                )
            t_c_w_flapli = t_c_onbellek[flap_profil]
            oran_flap = float(aday.get("flap_aciklik_orani", float(aday["S_flap_m2"]) / float(aday["S_w_m2"])))
            t_c_w = oran_flap * t_c_w_flapli + (1.0 - oran_flap) * t_c_w_flapsiz
        else:
            t_c_w = t_c_w_flapsiz

        # Gövde özelliklerini eşleştir
        L_govde, D_govde, S_wet_govde = govde_ozelliklerini_bul(aday["govde_uzunlugu_m"])

        # Kanat ve kuyruk boyutları ile Reynolds sayıları
        b_w = float(aday["b_w_m"])
        AR_w = float(aday["AR_w"])
        S_w = float(aday["S_w_m2"])
        c_w = float(aday["c_root_w_m"])
        Re_w = SEYIR_HIZI_MS * c_w / nu

        S_HT = float(aday["S_HT_m2"])
        b_HT = float(aday["b_HT_m"])
        c_HT = float(aday["c_root_HT_m"])
        Re_HT = SEYIR_HIZI_MS * c_HT / nu

        S_VT = float(aday["S_VT_m2"])
        h_VT = float(aday["h_VT_m"])
        c_VT = float(aday["c_root_VT_m"])
        Re_VT = SEYIR_HIZI_MS * c_VT / nu

        # LLT'den gelen 3B taşıma parametreleri
        CL_alpha_3d = llt_veri["CL_alpha_3d_per_deg"]
        alpha_L0_deg = llt_veri["alpha_L0_deg"]
        e_oswald = llt_veri["e_oswald"]
        CL_seyir = llt_veri["CL_seyir"]

        # CDi_seyir LLT dosyasından okunmaz; doğrudan burada hesaplanır
        CDi_seyir = (CL_seyir ** 2) / (math.pi * AR_w * e_oswald)

        # OpenVSP eşdeğeri (iniş takımsız) CD_0 hesabı
        vsp = hesapla_openvsp_cd0(
            S_w=S_w,
            Re_w=Re_w,
            t_c_w=t_c_w,
            S_HT=S_HT,
            Re_HT=Re_HT,
            t_c_HT=t_c_kuyruk,
            S_VT=S_VT,
            Re_VT=Re_VT,
            t_c_VT=t_c_kuyruk,
            L_govde=L_govde,
            D_govde=D_govde,
            S_wet_govde=S_wet_govde,
        )

        CD_0 = vsp["CD_0"]
        CD_seyir_toplam = CD_0 + CDi_seyir
        L_D_seyir_toplam = CL_seyir / CD_seyir_toplam if CD_seyir_toplam > 0 else np.nan

        temel_bilgi = {
            "aday_no": no,
            "kanat_airfoil": kanat_profil,
            "kuyruk_airfoil": kuyruk_profil,
            "V_seyir_ms": SEYIR_HIZI_MS,
            "b_w_m": b_w,
            "AR_w": AR_w,
            "S_w_m2": S_w,
            "c_w_m": c_w,
            "t_c_w": t_c_w,
            "S_HT_m2": S_HT,
            "b_HT_m": b_HT,
            "c_HT_m": c_HT,
            "t_c_HT": t_c_kuyruk,
            "S_VT_m2": S_VT,
            "h_VT_m": h_VT,
            "c_VT_m": c_VT,
            "t_c_VT": t_c_kuyruk,
            "L_govde_m": L_govde,
            "D_govde_m": D_govde,
            "tail_arm_m": float(aday["tail_arm_m"]),
            "Re_aday": Re_w,
            "Re_HT": Re_HT,
            "Re_VT": Re_VT,
            **vsp,
            "CL_alpha_3d_per_deg": CL_alpha_3d,
            "alpha_L0_deg": alpha_L0_deg,
            "e_oswald": e_oswald,
        }

        # 1) Aday Özet Satırı (0° Seyir / Referans Koşulu)
        ozet_satirlari.append(
            {
                **temel_bilgi,
                "CL_seyir": CL_seyir,
                "CD_i_seyir": CDi_seyir,
                "CD_0_seyir": CD_0,
                "CD_seyir": CD_seyir_toplam,
                "L_D_seyir": L_D_seyir_toplam,
            }
        )

        # 2) -5° ile +15° Arası Açı Taraması
        for alfa_deg in alfa_dizisi:
            cl_alfa = CL_alpha_3d * (float(alfa_deg) - alpha_L0_deg)
            cd_i_alfa = (cl_alfa ** 2) / (math.pi * AR_w * e_oswald)
            cd_toplam_alfa = CD_0 + cd_i_alfa
            l_d_alfa = cl_alfa / cd_toplam_alfa if cd_toplam_alfa > 0 else np.nan

            polar_detay_satirlari.append(
                {
                    **temel_bilgi,
                    "alpha_deg": float(alfa_deg),
                    "CL": cl_alfa,
                    "CD_i": cd_i_alfa,
                    "CD_0": CD_0,
                    "CD": cd_toplam_alfa,
                    "L_D": l_d_alfa,
                }
            )

    csv_yaz(ozet_satirlari, cikti_ozet_csv)
    csv_yaz(polar_detay_satirlari, cikti_polar_csv)
    return len(ozet_satirlari), len(polar_detay_satirlari)


def main() -> None:
    print("1) Flapsiz ucak adaylari icin OpenVSP CD0 + LLT CDi hesaplaniyor...")
    flapsiz_adaylar = kb.tum_adaylari_uret()
    llt_flapsiz_harita = llt_aday_re_csv_oku(LLT_FLAPSIZ_ADAY_RE_CSV, flapli_mi=False)
    n_flapsiz_ozet, n_flapsiz_polar = grubu_hesapla_ve_kaydet(
        flapsiz_adaylar,
        llt_flapsiz_harita,
        CIKTI_FLAPSIZ_OZET_CSV,
        CIKTI_FLAPSIZ_POLAR_CSV,
        flapli_mi=False,
    )
    print(f"   - Kaydedildi: {CIKTI_FLAPSIZ_OZET_CSV} ({n_flapsiz_ozet} aday)")
    print(f"   - Kaydedildi: {CIKTI_FLAPSIZ_POLAR_CSV} ({n_flapsiz_polar} satir)")

    print("\n2) Flapli ucak adaylari icin OpenVSP CD0 + LLT CDi hesaplaniyor...")
    flapli_adaylar = kb.tum_flapli_adaylari_uret()
    llt_flapli_harita = llt_aday_re_csv_oku(LLT_FLAPLI_ADAY_RE_CSV, flapli_mi=True)
    n_flapli_ozet, n_flapli_polar = grubu_hesapla_ve_kaydet(
        flapli_adaylar,
        llt_flapli_harita,
        CIKTI_FLAPLI_OZET_CSV,
        CIKTI_FLAPLI_POLAR_CSV,
        flapli_mi=True,
    )
    print(f"   - Kaydedildi: {CIKTI_FLAPLI_OZET_CSV} ({n_flapli_ozet} aday)")
    print(f"   - Kaydedildi: {CIKTI_FLAPLI_POLAR_CSV} ({n_flapli_polar} satir)")


if __name__ == "__main__":
    main()