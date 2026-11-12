"""
LLT ÇÖZÜCÜ - BELLEK İÇİ (IN-MEMORY) MOTOR VE ADAY RE (-5°..+15°) CSV ÜRETİCİ
====================================================================
Yeni '<kanat_airfoil>_flap' adlandırma standardına göre güncellenmiştir:
- Örneğin 'clarkysm-il' kanat profilinin flaplı karşılığı doğrudan 'clarkysm-il_flap'
  olarak eşleştirilir.
- Eğer 'airfoils_polar_flap' klasöründeki .pol dosyaları henüz eski isimlerindeyse
  (ör. 'CLARK-Y 11.7%_flap smoothed_re100000.pol') veya yeni isimlerine ('clarkysm-il_flap_re100000.pol')
  geçirildiyse her iki durumu da otomatik tanır ve birbirine bağlar.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import re
import sys
from pathlib import Path

import numpy as np


# ==================================================== AYARLAR ====================================================
SEYIR_HIZI_MS = 18.0        # Flapsız aday_re.csv referans seyir hızı (m/s)
KALKIS_HIZI_MS = 12.0       # Flaplı aday_re.csv referans kalkış hızı (m/s)
RHO = 1.175                 # Hava yoğunluğu (kg/m^3)
MU = 1.789e-5               # Havanın dinamik viskozitesi (Pa.s)
SEYIR_ALPHA_DEG = 0.0       # CL_seyir referans hücum açısı (0 derece)

ALPHA_MIN = -5.0
ALPHA_MAX = 15.0
ALPHA_STEP = 0.5
LINEER_ALPHA_MIN = -4.0
LINEER_ALPHA_MAX = 6.0
LLT_TERIM_SAYISI = 25

KUYRUK_AIRFOIL_ADI = "naca0010"

POLAR_KLASORU_ADI = "airfoils_polar"
FLAP_POLAR_KLASORU_ADI = "airfoils_polar_flap"

FLAPSIZ_ADAY_CSV_ADI = "kanat_kuyruk_adaylari.csv"
FLAPLI_ADAY_CSV_ADI = "kanat_kuyruk_adaylari_flapli.csv"
# =================================================================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJE_KLASORU = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "scripts" else SCRIPT_DIR
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJE_KLASORU) not in sys.path:
    sys.path.insert(0, str(PROJE_KLASORU))

try:
    with contextlib.redirect_stdout(io.StringIO()):
        import kanat_boyutlandirma as kb
except Exception:
    kb = None

POLAR_KLASORU = (
    PROJE_KLASORU / POLAR_KLASORU_ADI
    if (PROJE_KLASORU / POLAR_KLASORU_ADI).exists()
    else SCRIPT_DIR / POLAR_KLASORU_ADI
)
FLAP_POLAR_KLASORU = (
    PROJE_KLASORU / FLAP_POLAR_KLASORU_ADI
    if (PROJE_KLASORU / FLAP_POLAR_KLASORU_ADI).exists()
    else SCRIPT_DIR / FLAP_POLAR_KLASORU_ADI
)

ALFA_DIZISI_DEG = np.round(
    np.arange(ALPHA_MIN, ALPHA_MAX + 1e-9, ALPHA_STEP), 4
)

# 29 kanat profili: yeni standart '<kanat_airfoil>_flap'
KANAT_PROFILLERI_29: list[str] = [
    "clarkysm-il", "cr001sm-il", "dae51-il", "e174-il", "e212-il",
    "e387-il", "e392-il", "hq2590sm-il", "hq3010-il", "mh70-il",
    "n9-il", "naca2410-il", "naca4412-il", "psu94097-il", "s2046-il",
    "s2091-il", "s3010-il", "s4062-il", "s4110-il", "s4310-il",
    "s7075-il", "sd6080-il", "sd7032-il", "sd7037-il", "sd7043-il",
    "sd8040-il", "sg6041-il", "sg6042-il", "waspsm-il",
]

FLAPLI_ESLESME_SOZLUGU: dict[str, str] = {
    prof: f"{prof}_flap" for prof in KANAT_PROFILLERI_29
}

# Eğer airfoils_polar_flap içindeki .pol dosyaları henüz eski adlarındaysa otomatik köprü kurar
ESKI_POLAR_TAKMA_ADLARI: dict[str, str] = {
    "clarkysm-il_flap": "CLARK-Y 11.7%_flap smoothed",
    "cr001sm-il_flap": "cr001sm_flap",
    "dae51-il_flap": "DAE-51 AIRFOIL_flap",
    "e174-il_flap": "E174 (Dicke 8.92%)_flap",
    "e212-il_flap": "E212 (10.55%)_flap",
    "e387-il_flap": "E387_flap",
    "e392-il_flap": "E392 (10.15%)_flap",
    "hq2590sm-il_flap": "HQ 2.5 9.0 smoothed by Eppler_flap",
    "hq3010-il_flap": "HQ 3.0 10 AIRFOIL_flap",
    "mh70-il_flap": "MH 70 11.08%_flap",
    "n9-il_flap": "N-9_flap",
    "naca2410-il_flap": "NACA 2410_flap",
    "naca4412-il_flap": "NACA 4412_flap",
    "psu94097-il_flap": "PSU 94-097 (AIAA 2001-2478)_flap",
    "s2046-il_flap": "S2046_flap",
    "s2091-il_flap": "S2091-101-83_flap",
    "s3010-il_flap": "S3010-103-84_flap",
    "s4062-il_flap": "S4062-095-87_flap",
    "s4110-il_flap": "S4110_flap",
    "s4310-il_flap": "S4310_flap",
    "s7075-il_flap": "S7075 (9%)_flap",
    "sd6080-il_flap": "SD6080 (9.2%)_flap",
    "sd7032-il_flap": "SD7032-099-88_flap",
    "sd7037-il_flap": "SD7037-092-88_flap",
    "sd7043-il_flap": "SD7043 (9.1%)_flap",
    "sd8040-il_flap": "SD8040 (10%)_flap",
    "sg6041-il_flap": "SG6041_flap",
    "sg6042-il_flap": "SG6042_flap",
    "waspsm-il_flap": "WASP (smoothed)_flap",
}


# ---------------------------------------------------------------------------------
# 1) GEOMETRİ VE PROFİL ADLARINI NORMALIZE EDEN YARDIMCI FONKSİYONLAR
# ---------------------------------------------------------------------------------
def aday_degeri_al(aday: dict, *anahtarlar: str, varsayilan: float | None = None) -> float:
    for k in anahtarlar:
        if k in aday and aday[k] not in (None, ""):
            return float(aday[k])
    if varsayilan is not None:
        return float(varsayilan)
    raise KeyError(f"Aday sozlugunde su anahtarlardan hicbiri bulunamadi: {anahtarlar}")


def flap_airfoil_bul(aday: dict) -> str:
    """Kanat profil adına doğrudan '_flap' ekleyerek yeni standart flap profil adını döndürür."""
    kanat = Path(str(aday.get("kanat_airfoil") or aday.get("profil") or "")).stem.strip().lower()
    if kanat:
        if kanat.endswith("_flap"):
            return kanat
        return f"{kanat}_flap"
    mevcut = Path(str(aday.get("flap_airfoil") or "")).stem.strip()
    if mevcut:
        return mevcut
    raise KeyError("Aday icinde kanat_airfoil veya flap_airfoil bulunamadi.")


# ---------------------------------------------------------------------------------
# 2) YARDIMCI VE POLAR İNDEKSLEME FONKSİYONLARI (CD İÇERMEZ)
# ---------------------------------------------------------------------------------
def reynolds_hesapla(
    veter_m: float,
    hiz_ms: float = SEYIR_HIZI_MS,
    rho: float = RHO,
    mu: float = MU,
) -> float:
    if veter_m <= 0:
        raise ValueError(f"Veter uzunlugu sifirdan buyuk olmali: {veter_m}")
    return float((rho * hiz_ms * veter_m) / mu)


def _re_degerini_normallestir(ham_re: int) -> int:
    if ham_re == 4000000:
        return 400000
    if ham_re == 4500000:
        return 450000
    if ham_re == 50000:
        return 500000
    return ham_re


def _norm_profil_anahtari(metin: str) -> str:
    return re.sub(r"[^a-z0-9]", "", metin.lower())


def polar_oku(dosya: Path) -> tuple[np.ndarray, np.ndarray]:
    veri: dict[float, float] = {}
    basladi = False
    for satir in dosya.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "---" in satir:
            basladi = True
            continue
        if not basladi:
            continue
        parca = satir.split()
        if len(parca) >= 2:
            try:
                alfa_val = float(parca[0])
                cl_val = float(parca[1])
                veri[alfa_val] = cl_val
            except ValueError:
                continue

    if len(veri) < 3:
        raise ValueError(f"{dosya.name}: yeterli polar verisi yok ({len(veri)} nokta).")

    alfa = np.array(sorted(veri.keys()), dtype=float)
    cl = np.array([veri[a] for a in alfa], dtype=float)
    return alfa, cl


def profil_ozellikleri_cikar(
    alfa: np.ndarray,
    cl: np.ndarray,
) -> dict:
    maske = (alfa >= LINEER_ALPHA_MIN) & (alfa <= LINEER_ALPHA_MAX)
    if int(maske.sum()) < 3:
        maske = np.ones_like(alfa, dtype=bool)

    a0_rad, b_kesisim = np.polyfit(np.deg2rad(alfa[maske]), cl[maske], 1)
    if a0_rad <= 0:
        alt_maske = alfa <= 4.0
        if int(alt_maske.sum()) >= 3:
            a0_rad, b_kesisim = np.polyfit(np.deg2rad(alfa[alt_maske]), cl[alt_maske], 1)
    if a0_rad <= 0:
        raise ValueError("2B tasima egimi (a0) pozitif degil.")

    i_max = int(np.argmax(cl))
    return {
        "a0_rad": float(a0_rad),
        "alpha_L0_rad": float(-b_kesisim / a0_rad),
        "alpha_L0_deg": float(np.rad2deg(-b_kesisim / a0_rad)),
        "cl_max_2d": float(cl[i_max]),
        "cl_min_2d": float(cl.min()),
        "alpha_stall_deg": float(alfa[i_max]),
        "alfa": alfa,
        "cl": cl,
    }


_polar_indeks_onbellek: dict[Path, dict[str, dict[int, dict]]] = {}

def polar_klasorunu_indeksle(polar_klasoru: Path = POLAR_KLASORU) -> dict[str, dict[int, dict]]:
    polar_klasoru = Path(polar_klasoru).resolve()
    if polar_klasoru in _polar_indeks_onbellek:
        return _polar_indeks_onbellek[polar_klasoru]

    if not polar_klasoru.exists():
        raise FileNotFoundError(f"Polar klasoru bulunamadi: {polar_klasoru}")

    indeks: dict[str, dict[int, dict]] = {}
    desen = re.compile(r"^(.*)_re(\d+)(?:\.pol)?$", re.IGNORECASE)

    for pol_dosyasi in sorted(polar_klasoru.iterdir()):
        if pol_dosyasi.is_dir():
            continue
        eslesme = desen.match(pol_dosyasi.name)
        if not eslesme:
            continue
        profil_adi = eslesme.group(1).strip()
        ham_re = int(eslesme.group(2))
        re_degeri = _re_degerini_normallestir(ham_re)

        try:
            alfa, cl = polar_oku(pol_dosyasi)
            ozellik = profil_ozellikleri_cikar(alfa, cl)
            ozellik["dosya_yolu"] = pol_dosyasi
            ozellik["profil_adi"] = profil_adi
            ozellik["re_polar"] = re_degeri

            anahtar_tam = profil_adi.strip().lower()
            indeks.setdefault(anahtar_tam, {})[re_degeri] = ozellik

            anahtar_norm = _norm_profil_anahtari(profil_adi)
            indeks.setdefault(anahtar_norm, {})[re_degeri] = ozellik
        except Exception as exc:
            print(f"  [UYARI] Polar okunamadi ve atlandi ({pol_dosyasi.name}): {exc}")

    # Yeni '<kanat>_flap' anahtarları ile eski polar adlarını çift yönlü bağla
    for yeni_ad, eski_ad in ESKI_POLAR_TAKMA_ADLARI.items():
        yeni_k = yeni_ad.lower()
        eski_k = eski_ad.lower()
        yeni_norm = _norm_profil_anahtari(yeni_ad)
        eski_norm = _norm_profil_anahtari(eski_ad)
        if eski_k in indeks and yeni_k not in indeks:
            indeks[yeni_k] = indeks[eski_k]
            indeks[yeni_norm] = indeks[eski_k]
        elif yeni_k in indeks and eski_k not in indeks:
            indeks[eski_k] = indeks[yeni_k]
            indeks[eski_norm] = indeks[yeni_k]

    if not indeks:
        raise RuntimeError(f"{polar_klasoru} klasorunde gecerli hic polar dosyasi bulunamadi.")

    _polar_indeks_onbellek[polar_klasoru] = indeks
    return indeks


def profil_re_haritasi_al(
    polar_indeksi: dict[str, dict[int, dict]],
    profil_adi: str,
) -> dict[int, dict]:
    anahtar_tam = Path(profil_adi.strip()).stem.lower()
    anahtar_norm = _norm_profil_anahtari(anahtar_tam)

    if anahtar_tam in polar_indeksi:
        return polar_indeksi[anahtar_tam]
    if anahtar_norm in polar_indeksi:
        return polar_indeksi[anahtar_norm]

    if anahtar_tam in ESKI_POLAR_TAKMA_ADLARI:
        eski = ESKI_POLAR_TAKMA_ADLARI[anahtar_tam].lower()
        if eski in polar_indeksi:
            return polar_indeksi[eski]
        if _norm_profil_anahtari(eski) in polar_indeksi:
            return polar_indeksi[_norm_profil_anahtari(eski)]

    for k, v in polar_indeksi.items():
        if anahtar_norm in k or k in anahtar_norm:
            return v

    raise KeyError(f"'{profil_adi}' profili icin polar klasorunde hic polar bulunamadi.")


def en_yakin_polar_sec(
    re_haritasi: dict[int, dict],
    re_hedef: float,
) -> tuple[int, dict]:
    en_yakin_re = min(re_haritasi.keys(), key=lambda r: abs(r - re_hedef))
    return en_yakin_re, re_haritasi[en_yakin_re]


# ---------------------------------------------------------------------------------
# 3) LLT MATRİS ÇÖZÜCÜ VE REYNOLDS İNTERPOLASYONU
# ---------------------------------------------------------------------------------
_llt_onbellek: dict[tuple[float, float, int], tuple[float, float]] = {}
_yuzey_cozum_onbellek: dict[tuple[str, int, float], dict] = {}


def llt_dikdortgen_kanat(
    ar: float,
    a0_rad: float,
    n_terim: int = LLT_TERIM_SAYISI,
) -> tuple[float, float]:
    anahtar = (round(float(ar), 6), round(float(a0_rad), 6), int(n_terim))
    if anahtar in _llt_onbellek:
        return _llt_onbellek[anahtar]

    i = np.arange(1, n_terim + 1, dtype=float)
    theta = i * np.pi / (2.0 * n_terim)
    n = 2.0 * i - 1.0

    matris = np.sin(np.outer(theta, n)) * (
        (4.0 * ar) / a0_rad + n[np.newaxis, :] / np.sin(theta)[:, np.newaxis]
    )
    A = np.linalg.solve(matris, np.ones(n_terim, dtype=float))

    cl_alpha_3d = float(np.pi * ar * A[0])
    e_span = 1.0 / (1.0 + float(np.sum(n[1:] * (A[1:] / A[0]) ** 2)))
    k_fus = 0.995   # Gövde blokajı düzeltmesi (d_fus=0.14m silindirik orta gövde, b=2.70-3.00m için 1 - 2*(d_fus/b)^2)
    k_visc = 0.007  # Taşımaya bağlı viskoz profil sürükleme artış katsayısı (Kroo/Raymer)
    e_oswald = float(1.0 / (1.0 / (e_span * k_fus) + np.pi * ar * k_visc))

    _llt_onbellek[anahtar] = (cl_alpha_3d, e_oswald)
    return cl_alpha_3d, e_oswald


def yuzey_coklu_aci_coz(
    ar: float,
    prof: dict,
    re_polar: int,
    alfa_dizisi_deg: np.ndarray = ALFA_DIZISI_DEG,
) -> dict:
    onbellek_anahtari = (prof["profil_adi"].lower(), int(re_polar), round(float(ar), 6))
    if onbellek_anahtari in _yuzey_cozum_onbellek:
        return _yuzey_cozum_onbellek[onbellek_anahtari]

    cl_a, e = llt_dikdortgen_kanat(ar, prof["a0_rad"])
    cl_a_deg = float(np.deg2rad(cl_a))

    alfa_0_rad = np.deg2rad(SEYIR_ALPHA_DEG)
    cl_0deg = float(cl_a * (alfa_0_rad - prof["alpha_L0_rad"]))

    cl_max_3d = float(cl_a * (np.deg2rad(prof["alpha_stall_deg"]) - prof["alpha_L0_rad"]))
    cl_max_3d = min(cl_max_3d, prof["cl_max_2d"])

    alfa_rad_dizisi = np.deg2rad(alfa_dizisi_deg)
    cl_dizisi = np.clip(
        cl_a * (alfa_rad_dizisi - prof["alpha_L0_rad"]),
        prof["cl_min_2d"],
        cl_max_3d,
    )

    sonuc = {
        "Re_polar": int(re_polar),
        "a0_2d_per_deg": float(np.deg2rad(prof["a0_rad"])),
        "alpha_L0_deg": float(prof["alpha_L0_deg"]),
        "CL_max_2d": float(prof["cl_max_2d"]),
        "alpha_stall_2d_deg": float(prof["alpha_stall_deg"]),
        "CL_alpha_3d_per_deg": cl_a_deg,
        "e_oswald": float(e),
        "CL_seyir": cl_0deg,
        "CL_max_3d_tahmini": float(cl_max_3d),
        "alfa_dizisi_deg": alfa_dizisi_deg,
        "CL_dizisi": cl_dizisi,
    }
    _yuzey_cozum_onbellek[onbellek_anahtari] = sonuc
    return sonuc


def kuyruk_yuzey_coz(ar: float, prof: dict) -> dict:
    cl_a, e = llt_dikdortgen_kanat(ar, prof["a0_rad"])
    cl_a_deg = float(np.deg2rad(cl_a))
    cl_max_3d = min(
        float(cl_a * (np.deg2rad(prof["alpha_stall_deg"]) - prof["alpha_L0_rad"])),
        float(prof["cl_max_2d"]),
    )
    return {
        "CL_alpha_per_deg": cl_a_deg,
        "e": float(e),
        "alpha_stall_deg": float(prof["alpha_stall_deg"]),
        "CL_max_3d": cl_max_3d,
        "alpha_min_deg": float(prof["alfa"].min()),
        "alpha_max_deg": float(prof["alfa"].max()),
    }


def re_interpolasyonu(
    re_sonuclari: dict[int, dict],
    re_hedef: float,
) -> tuple[dict, str]:
    sirali_re = sorted(re_sonuclari.keys())
    if not sirali_re:
        raise ValueError("Interpolasyon icin sonuc yok.")

    ornek_sozluk = re_sonuclari[sirali_re[0]]
    skaler_anahtarlar = [
        k for k, v in ornek_sozluk.items() if isinstance(v, (int, float))
    ]
    dizi_anahtarlar = [
        k for k, v in ornek_sozluk.items()
        if isinstance(v, np.ndarray) and k != "alfa_dizisi_deg"
    ]

    if re_hedef <= sirali_re[0]:
        durum = f"ekstrapolasyon (< {sirali_re[0]})"
        secilen = re_sonuclari[sirali_re[0]]
        sonuc = {k: float(secilen[k]) for k in skaler_anahtarlar}
        for k in dizi_anahtarlar:
            sonuc[k] = np.array(secilen[k], dtype=float)
        return sonuc, durum

    if re_hedef >= sirali_re[-1]:
        durum = f"ekstrapolasyon (> {sirali_re[-1]})"
        secilen = re_sonuclari[sirali_re[-1]]
        sonuc = {k: float(secilen[k]) for k in skaler_anahtarlar}
        for k in dizi_anahtarlar:
            sonuc[k] = np.array(secilen[k], dtype=float)
        return sonuc, durum

    idx = 0
    for j in range(len(sirali_re) - 1):
        if sirali_re[j] <= re_hedef <= sirali_re[j + 1]:
            idx = j
            break

    r0, r1 = sirali_re[idx], sirali_re[idx + 1]
    t = (re_hedef - r0) / (r1 - r0)
    durum = f"ara deger {r0}-{r1}"

    s0, s1 = re_sonuclari[r0], re_sonuclari[r1]
    sonuc = {}
    for k in skaler_anahtarlar:
        sonuc[k] = float(s0[k] + t * (s1[k] - s0[k]))
    for k in dizi_anahtarlar:
        sonuc[k] = s0[k] + t * (s1[k] - s0[k])

    return sonuc, durum


# ---------------------------------------------------------------------------------
# 4) BELLEK İÇİ (IN-MEMORY) DOĞRUDAN ÇÖZÜCÜ API FONKSİYONLARI
# ---------------------------------------------------------------------------------
def flapsiz_kanat_coz_re(
    profil_adi: str,
    ar_w: float,
    re_hedef: float,
    flapsiz_indeks: dict[str, dict[int, dict]] | None = None,
) -> dict:
    if flapsiz_indeks is None:
        flapsiz_indeks = polar_klasorunu_indeksle(POLAR_KLASORU)
    kanat_re_haritasi = profil_re_haritasi_al(flapsiz_indeks, profil_adi)
    sonuclar_re = {
        r_pol: yuzey_coklu_aci_coz(ar_w, p_ozel, r_pol)
        for r_pol, p_ozel in sorted(kanat_re_haritasi.items())
    }
    ara, durum_re = re_interpolasyonu(sonuclar_re, re_hedef)
    cl_dizisi = ara.pop("CL_dizisi")
    curve = {float(ALFA_DIZISI_DEG[k]): float(cl_dizisi[k]) for k in range(len(ALFA_DIZISI_DEG))}
    return {
        **ara,
        "Re_aday": float(re_hedef),
        "Re_durumu": durum_re,
        "CL_dizisi": cl_dizisi,
        "curve": curve,
        "e": ara["e_oswald"],
        "cl_max": ara["CL_max_3d_tahmini"],
        "stall": ara["alpha_stall_2d_deg"],
    }


def flapli_kanat_coz_re(
    profil_adi: str,
    flap_profil_adi: str,
    ar_w: float,
    oran_flap: float,
    re_hedef: float,
    flapsiz_indeks: dict[str, dict[int, dict]] | None = None,
    flapli_indeks: dict[str, dict[int, dict]] | None = None,
) -> dict:
    if flapsiz_indeks is None:
        flapsiz_indeks = polar_klasorunu_indeksle(POLAR_KLASORU)
    if flapli_indeks is None:
        flapli_indeks = polar_klasorunu_indeksle(FLAP_POLAR_KLASORU)

    oran_flap = max(0.0, min(1.0, float(oran_flap)))
    oran_flapsiz = 1.0 - oran_flap

    flapsiz_re_haritasi = profil_re_haritasi_al(flapsiz_indeks, profil_adi)
    flapli_re_haritasi = profil_re_haritasi_al(flapli_indeks, flap_profil_adi)

    ortak_re_listesi = sorted(set(flapsiz_re_haritasi.keys()) & set(flapli_re_haritasi.keys()))
    if not ortak_re_listesi:
        ortak_re_listesi = sorted(flapsiz_re_haritasi.keys())

    sonuclar_birlesik_re: dict[int, dict] = {}
    for r_pol in ortak_re_listesi:
        p_flapsiz = flapsiz_re_haritasi.get(r_pol) or en_yakin_polar_sec(flapsiz_re_haritasi, r_pol)[1]
        p_flapli = flapli_re_haritasi.get(r_pol) or en_yakin_polar_sec(flapli_re_haritasi, r_pol)[1]

        sol_flapsiz = yuzey_coklu_aci_coz(ar_w, p_flapsiz, r_pol)
        sol_flapli = yuzey_coklu_aci_coz(ar_w, p_flapli, r_pol)

        cl_0deg_birl = (
            sol_flapli["CL_seyir"] * oran_flap
            + sol_flapsiz["CL_seyir"] * oran_flapsiz
        )
        cl_a_birl = (
            sol_flapli["CL_alpha_3d_per_deg"] * oran_flap
            + sol_flapsiz["CL_alpha_3d_per_deg"] * oran_flapsiz
        )
        e_birl = (
            sol_flapli["e_oswald"] * oran_flap
            + sol_flapsiz["e_oswald"] * oran_flapsiz
        )
        alpha_l0_birl = -cl_0deg_birl / cl_a_birl if cl_a_birl > 0 else 0.0

        alpha_stall_kritik = min(
            sol_flapli["alpha_stall_2d_deg"],
            sol_flapsiz["alpha_stall_2d_deg"],
        )

        cl_dizisi_flapsiz = sol_flapsiz["CL_dizisi"]
        cl_dizisi_flapli = sol_flapli["CL_dizisi"]
        cl_dizisi_birl = (
            cl_dizisi_flapli * oran_flap
            + cl_dizisi_flapsiz * oran_flapsiz
        )

        stall_maske = ALFA_DIZISI_DEG <= (alpha_stall_kritik + 1e-6)
        if np.any(stall_maske):
            cl_max_birl = float(np.max(cl_dizisi_birl[stall_maske]))
        else:
            cl_max_birl = float(np.max(cl_dizisi_birl))
        cl_dizisi_birl = np.minimum(cl_dizisi_birl, cl_max_birl)

        sonuclar_birlesik_re[r_pol] = {
            "Re_polar": r_pol,
            "a0_2d_flapsiz_per_deg": sol_flapsiz["a0_2d_per_deg"],
            "a0_2d_flapli_per_deg": sol_flapli["a0_2d_per_deg"],
            "alpha_L0_flapsiz_deg": sol_flapsiz["alpha_L0_deg"],
            "alpha_L0_flapli_deg": sol_flapli["alpha_L0_deg"],
            "alpha_L0_deg": alpha_l0_birl,
            "CL_max_2d_flapsiz": sol_flapsiz["CL_max_2d"],
            "CL_max_2d_flapli": sol_flapli["CL_max_2d"],
            "alpha_stall_flapsiz_deg": sol_flapsiz["alpha_stall_2d_deg"],
            "alpha_stall_flapli_deg": sol_flapli["alpha_stall_2d_deg"],
            "alpha_stall_2d_deg": alpha_stall_kritik,
            "CL_alpha_3d_kanat_per_deg": cl_a_birl,
            "CL_alpha_3d_per_deg": cl_a_birl,
            "e_oswald_kanat": e_birl,
            "e_oswald": e_birl,
            "CL_seyir": cl_0deg_birl,
            "CL_max_3d_tahmini": cl_max_birl,
            "CL_dizisi_flapsiz": cl_dizisi_flapsiz,
            "CL_dizisi_flapli": cl_dizisi_flapli,
            "CL_dizisi": cl_dizisi_birl,
        }

    ara, durum_re = re_interpolasyonu(sonuclar_birlesik_re, re_hedef)
    cl_dizisi = ara.pop("CL_dizisi")
    cl_dizisi_flapsiz = ara.pop("CL_dizisi_flapsiz")
    cl_dizisi_flapli = ara.pop("CL_dizisi_flapli")
    curve = {float(ALFA_DIZISI_DEG[k]): float(cl_dizisi[k]) for k in range(len(ALFA_DIZISI_DEG))}
    return {
        **ara,
        "flap_airfoil": flap_profil_adi,
        "flap_aciklik_orani": oran_flap,
        "Re_aday": float(re_hedef),
        "Re_durumu": durum_re,
        "CL_dizisi": cl_dizisi,
        "CL_dizisi_flapsiz": cl_dizisi_flapsiz,
        "CL_dizisi_flapli": cl_dizisi_flapli,
        "curve": curve,
        "e": ara["e_oswald"],
        "cl_max": ara["CL_max_3d_tahmini"],
        "stall": ara["alpha_stall_2d_deg"],
    }


def kuyruk_coz_re(
    kuyruk_profil_adi: str,
    ar_kuyruk: float,
    re_hedef: float,
    flapsiz_indeks: dict[str, dict[int, dict]] | None = None,
) -> dict:
    if flapsiz_indeks is None:
        flapsiz_indeks = polar_klasorunu_indeksle(POLAR_KLASORU)
    try:
        kuyruk_prof_haritasi = profil_re_haritasi_al(flapsiz_indeks, kuyruk_profil_adi)
    except KeyError:
        kuyruk_prof_haritasi = profil_re_haritasi_al(flapsiz_indeks, KUYRUK_AIRFOIL_ADI)

    sk = {r: kuyruk_yuzey_coz(ar_kuyruk, pr) for r, pr in kuyruk_prof_haritasi.items()}
    ara, durum = re_interpolasyonu(sk, re_hedef)
    return {
        "slope": ara["CL_alpha_per_deg"],
        "CL_alpha_per_deg": ara["CL_alpha_per_deg"],
        "e": ara["e"],
        "stall": ara["alpha_stall_deg"],
        "cl_max": ara["CL_max_3d"],
        "alpha_min": ara["alpha_min_deg"],
        "alpha_max": ara["alpha_max_deg"],
        "Re_durumu": durum,
    }


def csv_yaz(satirlar: list[dict], dosya_yolu: Path) -> None:
    if not satirlar:
        return
    dosya_yolu.parent.mkdir(parents=True, exist_ok=True)
    with dosya_yolu.open("w", newline="", encoding="utf-8-sig") as f:
        yazici = csv.DictWriter(f, fieldnames=list(satirlar[0].keys()))
        yazici.writeheader()
        yazici.writerows(satirlar)


def adaylari_yukle(csv_adi: str, flapli_mi: bool = False) -> list[dict]:
    olasi_yollar = [
        PROJE_KLASORU / "kanat_kuyruk_govde_kutleli.csv",
        SCRIPT_DIR / "kanat_kuyruk_govde_kutleli.csv",
        PROJE_KLASORU / csv_adi,
        SCRIPT_DIR / csv_adi,
    ]
    for yol in olasi_yollar:
        if yol.exists() and yol.stat().st_size > 0:
            with yol.open("r", encoding="utf-8-sig", newline="") as f:
                return list(csv.DictReader(f))

    if kb is not None:
        if flapli_mi and hasattr(kb, "flapli_aday_kombinasyonlari_uretir"):
            return kb.flapli_aday_kombinasyonlari_uretir()
        if hasattr(kb, "aday_kombinasyonlari_uretir"):
            return kb.aday_kombinasyonlari_uretir()
    return []


def flapsiz_adaylari_coz_ve_yaz(
    adaylar: list[dict],
    polar_indeksi: dict[str, dict[int, dict]],
    yol_tum_sonuclar: Path | None,
    yol_aday_re: Path,
) -> tuple[int, int]:
    aday_re_satirlari: list[dict] = []
    toplam_tum_satir = 0

    for idx, aday in enumerate(adaylar, start=1):
        profil_adi = Path(str(aday.get("kanat_airfoil") or aday.get("profil"))).stem.strip()
        kuyruk_profil_adi = Path(str(aday.get("kuyruk_airfoil") or aday.get("ht_airfoil") or KUYRUK_AIRFOIL_ADI)).stem.strip()

        b_w = aday_degeri_al(aday, "b_w_m")
        ar_w = aday_degeri_al(aday, "AR_w")
        s_w = aday_degeri_al(aday, "S_w_m2")
        c_w = aday_degeri_al(aday, "c_root_w_m", "c_w_m")
        re_aday = reynolds_hesapla(c_w, hiz_ms=SEYIR_HIZI_MS)

        c_ht = aday_degeri_al(aday, "c_root_HT_m", "c_root_h_m", "c_HT_m")
        s_ht = aday_degeri_al(aday, "S_HT_m2", "S_h_m2")
        ar_ht = aday_degeri_al(aday, "AR_HT", "AR_h", varsayilan=4.0)
        re_ht = reynolds_hesapla(c_ht, hiz_ms=SEYIR_HIZI_MS)

        c_vt = aday_degeri_al(aday, "c_root_VT_m", "c_root_v_m", "c_VT_m")
        s_vt = aday_degeri_al(aday, "S_VT_m2", "S_v_m2")
        ar_vt = aday_degeri_al(aday, "AR_VT", "AR_v", varsayilan=4.0)
        re_vt = reynolds_hesapla(c_vt, hiz_ms=SEYIR_HIZI_MS)

        kanat_cozum = flapsiz_kanat_coz_re(profil_adi, ar_w, re_aday, polar_indeksi)
        ht_cozum = kuyruk_coz_re(kuyruk_profil_adi, ar_ht, re_ht, polar_indeksi)
        vt_cozum = kuyruk_coz_re(kuyruk_profil_adi, ar_vt, re_vt, polar_indeksi)

        cl_dizisi = kanat_cozum["CL_dizisi"]
        for k, alfa_deg in enumerate(ALFA_DIZISI_DEG):
            aday_re_satirlari.append({
                "aday_no": idx,
                "profil": profil_adi,
                "kanat_airfoil": profil_adi,
                "b_w_m": b_w,
                "AR_w": ar_w,
                "S_w_m2": s_w,
                "c_w_m": c_w,
                "Re_aday": re_aday,
                "Re_durumu": kanat_cozum["Re_durumu"],
                "alpha_deg": float(alfa_deg),
                "a0_2d_per_deg": kanat_cozum["a0_2d_per_deg"],
                "alpha_L0_deg": kanat_cozum["alpha_L0_deg"],
                "CL_max_2d": kanat_cozum["CL_max_2d"],
                "alpha_stall_2d_deg": kanat_cozum["alpha_stall_2d_deg"],
                "CL_alpha_3d_per_deg": kanat_cozum["CL_alpha_3d_per_deg"],
                "e_oswald": kanat_cozum["e_oswald"],
                "CL": float(cl_dizisi[k]),
                "CL_seyir": kanat_cozum["CL_seyir"],
                "CL_max_3d_tahmini": kanat_cozum["CL_max_3d_tahmini"],
                "c_HT_m": c_ht,
                "S_HT_m2": s_ht,
                "Re_HT": re_ht,
                "CL_alpha_HT_per_deg": ht_cozum["CL_alpha_per_deg"],
                "e_HT": ht_cozum["e"],
                "alpha_stall_HT_deg": ht_cozum["stall"],
                "CL_max_HT": ht_cozum["cl_max"],
                "c_VT_m": c_vt,
                "S_VT_m2": s_vt,
                "Re_VT": re_vt,
                "CL_alpha_VT_per_deg": vt_cozum["CL_alpha_per_deg"],
                "e_VT": vt_cozum["e"],
            })

    csv_yaz(aday_re_satirlari, yol_aday_re)
    return toplam_tum_satir, len(aday_re_satirlari)


def flapli_adaylari_coz_ve_yaz(
    adaylar: list[dict],
    flapsiz_polar_indeksi: dict[str, dict[int, dict]],
    flapli_polar_indeksi: dict[str, dict[int, dict]],
    yol_tum_sonuclar: Path | None,
    yol_aday_re: Path,
) -> tuple[int, int]:
    aday_re_satirlari: list[dict] = []
    toplam_tum_satir = 0

    for idx, aday in enumerate(adaylar, start=1):
        profil_adi = Path(str(aday.get("kanat_airfoil") or aday.get("profil"))).stem.strip()
        flap_profil_adi = flap_airfoil_bul(aday)
        kuyruk_profil_adi = Path(str(aday.get("kuyruk_airfoil") or aday.get("ht_airfoil") or KUYRUK_AIRFOIL_ADI)).stem.strip()

        b_w = aday_degeri_al(aday, "b_w_m")
        ar_w = aday_degeri_al(aday, "AR_w")
        s_w = aday_degeri_al(aday, "S_w_m2")
        c_w = aday_degeri_al(aday, "c_root_w_m", "c_w_m")

        b_flap = aday_degeri_al(aday, "b_flap_m", varsayilan=min(1.0, b_w))
        b_flapsiz = b_w - b_flap
        s_flap = b_flap * c_w
        s_flapsiz = s_w - s_flap
        oran_flap = b_flap / b_w

        re_flapsiz = reynolds_hesapla(c_w, hiz_ms=KALKIS_HIZI_MS)

        c_ht = aday_degeri_al(aday, "c_root_HT_m", "c_root_h_m", "c_HT_m")
        s_ht = aday_degeri_al(aday, "S_HT_m2", "S_h_m2")
        ar_ht = aday_degeri_al(aday, "AR_HT", "AR_h", varsayilan=4.0)
        re_ht = reynolds_hesapla(c_ht, hiz_ms=KALKIS_HIZI_MS)

        c_vt = aday_degeri_al(aday, "c_root_VT_m", "c_root_v_m", "c_VT_m")
        s_vt = aday_degeri_al(aday, "S_VT_m2", "S_v_m2")
        ar_vt = aday_degeri_al(aday, "AR_VT", "AR_v", varsayilan=4.0)
        re_vt = reynolds_hesapla(c_vt, hiz_ms=KALKIS_HIZI_MS)

        kanat_cozum = flapli_kanat_coz_re(
            profil_adi, flap_profil_adi, ar_w, oran_flap, re_flapsiz,
            flapsiz_polar_indeksi, flapli_polar_indeksi,
        )
        ht_cozum = kuyruk_coz_re(kuyruk_profil_adi, ar_ht, re_ht, flapsiz_polar_indeksi)
        vt_cozum = kuyruk_coz_re(kuyruk_profil_adi, ar_vt, re_vt, flapsiz_polar_indeksi)

        cl_birl = kanat_cozum["CL_dizisi"]
        cl_flapsiz = kanat_cozum["CL_dizisi_flapsiz"]
        cl_flapli = kanat_cozum["CL_dizisi_flapli"]

        for k, alfa_deg in enumerate(ALFA_DIZISI_DEG):
            aday_re_satirlari.append({
                "aday_no": idx,
                "kanat_airfoil": profil_adi,
                "flap_airfoil": flap_profil_adi,
                "b_w_m": b_w,
                "AR_w": ar_w,
                "S_w_m2": s_w,
                "c_w_m": c_w,
                "b_flap_m": b_flap,
                "b_flapsiz_toplam_m": b_flapsiz,
                "S_flap_m2": s_flap,
                "S_flapsiz_m2": s_flapsiz,
                "flap_aciklik_orani": oran_flap,
                "Re_aday": re_flapsiz,
                "Re_flapsiz": re_flapsiz,
                "Re_durumu": kanat_cozum["Re_durumu"],
                "alpha_deg": float(alfa_deg),
                "alpha_L0_deg": kanat_cozum["alpha_L0_deg"],
                "alpha_stall_2d_deg": kanat_cozum["alpha_stall_2d_deg"],
                "CL_alpha_3d_kanat_per_deg": kanat_cozum["CL_alpha_3d_per_deg"],
                "CL_alpha_3d_per_deg": kanat_cozum["CL_alpha_3d_per_deg"],
                "e_oswald_kanat": kanat_cozum["e_oswald"],
                "e_oswald": kanat_cozum["e_oswald"],
                "CL_flapsiz_kisim": float(cl_flapsiz[k]),
                "CL_flapli_kisim": float(cl_flapli[k]),
                "CL_kanat": float(cl_birl[k]),
                "CL": float(cl_birl[k]),
                "CL_seyir": kanat_cozum["CL_seyir"],
                "CL_max_3d_tahmini": kanat_cozum["CL_max_3d_tahmini"],
                "c_HT_m": c_ht,
                "S_HT_m2": s_ht,
                "Re_HT": re_ht,
                "CL_alpha_HT_per_deg": ht_cozum["CL_alpha_per_deg"],
                "e_HT": ht_cozum["e"],
                "alpha_stall_HT_deg": ht_cozum["stall"],
                "CL_max_HT": ht_cozum["cl_max"],
                "c_VT_m": c_vt,
                "S_VT_m2": s_vt,
                "Re_VT": re_vt,
                "CL_alpha_VT_per_deg": vt_cozum["CL_alpha_per_deg"],
                "e_VT": vt_cozum["e"],
            })

    csv_yaz(aday_re_satirlari, yol_aday_re)
    return toplam_tum_satir, len(aday_re_satirlari)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args(argv)

    print("(1) 2B polar klasorleri indeksleniyor (sadece CL verileri)...")
    flapsiz_indeks = polar_klasorunu_indeksle(POLAR_KLASORU)
    flapli_indeks = polar_klasorunu_indeksle(FLAP_POLAR_KLASORU)
    print(f"  - {POLAR_KLASORU.name}: {len(flapsiz_indeks)} anahtar indekslendi.")
    print(f"  - {FLAP_POLAR_KLASORU.name}: {len(flapli_indeks)} flapli anahtar indekslendi.")

    print("\n2) Flapsiz kanat + kuyruk adaylari LLT ile cozuluyor (-5..15° aday Re)...")
    flapsiz_adaylar = adaylari_yukle(FLAPSIZ_ADAY_CSV_ADI, flapli_mi=False)
    yol_flapsiz_aday = PROJE_KLASORU / "llt_flapsiz_aday_re.csv"
    _, n_aday_flapsiz = flapsiz_adaylari_coz_ve_yaz(
        flapsiz_adaylar, flapsiz_indeks, None, yol_flapsiz_aday
    )
    print(f"  - Kaydedildi: {yol_flapsiz_aday} ({n_aday_flapsiz} satir)")

    print("\n3) Flapli kanat + kuyruk adaylari LLT ile cozuluyor (-5..15° aday Re)...")
    flapli_adaylar = adaylari_yukle(FLAPLI_ADAY_CSV_ADI, flapli_mi=True)
    yol_flapli_aday = PROJE_KLASORU / "llt_flapli_aday_re.csv"
    _, n_aday_flapli = flapli_adaylari_coz_ve_yaz(
        flapli_adaylar, flapsiz_indeks, flapli_indeks, None, yol_flapli_aday
    )
    print(f"  - Kaydedildi: {yol_flapli_aday} ({n_aday_flapli} satir)")
    print("\nIslem basariyla tamamlandi!")


if __name__ == "__main__":
    main()