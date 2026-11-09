"""
LLT ÇÖZÜCÜ - FLAPSIZ VE FLAPLI KANAT + KUYRUK TAŞIMA (CL) ÇIKARICI
==================================================================
Bu betik sürükleme (CD) hesabı yapmaz (CD hesabı drag_hesabi.py içinde yapılır).
Yalnızca Lifting Line Teorisi (LLT) ile 3 boyutlu taşıma (CL) ve aerodinamik
eğim/verim parametrelerini çözer:

1) Flapsız uçak adayları için:
   - llt_flapsiz_tum_sonuclar.csv : Tüm uçak adayları için farklı Reynolds sayılarında
                                    -5° ile +15° arasındaki hücum açısı sonuçları.
   - llt_flapsiz_aday_re.csv      : Her uçağın kendine ait Reynolds değerinde 0° hücum
                                    açısındaki kanat + kuyruk sonuçları.

2) Flaplı uçak adayları için (ortada 1 m simetrik flaplı + dışta flapsız kısımlar ayrı
   çözülüp alan oranlarıyla birleştirilerek):
   - llt_flapli_tum_sonuclar.csv  : Tüm flaplı uçak adayları için farklı Reynolds sayılarında
                                    -5° ile +15° arasındaki hücum açısı sonuçları.
   - llt_flapli_aday_re.csv       : Her flaplı uçağın kendine ait Reynolds değerinde 0° hücum
                                    açısındaki birleşik kanat + kuyruk sonuçları.
"""
from __future__ import annotations

import contextlib
import csv
import io
import re
import sys
from pathlib import Path

import numpy as np

# =============================== AYARLAR ===============================
SEYIR_HIZI_MS = 18.0        # Flapsız uçaklar için referans seyir hızı (m/s)
KALKIS_HIZI_MS = 12.0       # Flaplı uçaklar için referans kalkış hızı (m/s)
RHO = 1.175                 # Hava yoğunluğu (kg/m^3), sabitler.txt değeri
MU = 1.789e-5               # Havanın dinamik viskozitesi (Pa.s)
SEYIR_ALPHA_DEG = 0.0       # aday_re.csv dosyaları için referans hücum açısı (0 derece)

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
FLAPLI_ADAY_CSV_ADI = "kanat_kuyruk_adayları_flapli.csv"
# =======================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

with contextlib.redirect_stdout(io.StringIO()):
    import kanat_boyutlandirma as kb

PROJE_KLASORU = kb.AIRFOIL_KLASORU.parent
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


# ------------------------------------------------------------------
# 1) YARDIMCI VE POLAR İNDEKSLEME FONKSİYONLARI (CD İÇERMEZ)
# ------------------------------------------------------------------
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
    """XFOIL .pol dosyasından sıralı ve tekrarsız (alpha, Cl) dizilerini okur."""
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
    """2B polar verisinden yalnızca LLT taşıma hesabı için gerekli parametreleri çıkarır."""
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


def polar_klasorunu_indeksle(polar_klasoru: Path) -> dict[str, dict[int, dict]]:
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

    if not indeks:
        raise RuntimeError(f"{polar_klasoru} klasorunde gecerli hic polar dosyasi bulunamadi.")

    return indeks


def profil_re_haritasi_al(
    polar_indeksi: dict[str, dict[int, dict]],
    profil_adi: str,
) -> dict[int, dict]:
    anahtar_tam = profil_adi.strip().lower()
    anahtar_norm = _norm_profil_anahtari(profil_adi)

    if anahtar_tam in polar_indeksi:
        return polar_indeksi[anahtar_tam]
    if anahtar_norm in polar_indeksi:
        return polar_indeksi[anahtar_norm]

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


# ------------------------------------------------------------------
# 2) LLT MATRİS ÇÖZÜCÜ (SADECE TAŞIMA VE OSWALD VERİMİ)
# ------------------------------------------------------------------
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
    e_oswald = 1.0 / (1.0 + float(np.sum(n[1:] * (A[1:] / A[0]) ** 2)))

    _llt_onbellek[anahtar] = (cl_alpha_3d, e_oswald)
    return cl_alpha_3d, e_oswald


def yuzey_coklu_aci_coz(
    ar: float,
    prof: dict,
    re_polar: int,
    alfa_dizisi_deg: np.ndarray = ALFA_DIZISI_DEG,
) -> dict:
    """
    Belirli bir Reynolds poları ve açıklık oranı (AR) için:
    - 0° hücum açısındaki (CL_seyir) değeri
    - -5°..+15° aralığındaki tüm hücum açıları için 3B CL dizisini
    hesaplar ve önbelleğe alır (CD hesabı içermez).
    """
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
        prof["cl_max_2d"],
    )

    sonuc = {
        "Re_polar": int(re_polar),
        "a0_2d_per_deg": float(np.deg2rad(prof["a0_rad"])),
        "alpha_L0_deg": float(prof["alpha_L0_deg"]),
        "Cl_max_2d": float(prof["cl_max_2d"]),
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
    return {
        "CL_alpha_per_deg": float(np.deg2rad(cl_a)),
        "e": float(e),
    }


# ------------------------------------------------------------------
# 3) REYNOLDS İNTERPOLASYONU
# ------------------------------------------------------------------
def re_interpolasyonu(
    re_sonuclari: dict[int, dict],
    re_hedef: float,
) -> tuple[dict, str]:
    sirali_re = sorted(re_sonuclari.keys())
    if not sirali_re:
        raise ValueError("Interpolasyon icin sonuc yok.")

    ornek_sozluk = re_sonuclari[sirali_re[0]]
    anahtarlar = [
        k for k, v in ornek_sozluk.items() if isinstance(v, (int, float))
    ]

    if re_hedef <= sirali_re[0]:
        durum = f"ekstrapolasyon (< {sirali_re[0]})"
        return {k: float(re_sonuclari[sirali_re[0]][k]) for k in anahtarlar}, durum

    if re_hedef >= sirali_re[-1]:
        durum = f"ekstrapolasyon (> {sirali_re[-1]})"
        return {k: float(re_sonuclari[sirali_re[-1]][k]) for k in anahtarlar}, durum

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
    for k in anahtarlar:
        sonuc[k] = float(s0[k] + t * (s1[k] - s0[k]))

    return sonuc, durum


# ------------------------------------------------------------------
# 4) FLAPSIZ UÇAK ÇÖZÜMÜ (-5..15° TÜM SONUÇLAR VE 0° ADAY RE)
# ------------------------------------------------------------------
def flapsiz_adaylari_coz_ve_yaz(
    adaylar: list[dict],
    polar_indeksi: dict[str, dict[int, dict]],
    yol_tum_sonuclar: Path,
    yol_aday_re: Path,
) -> tuple[int, int]:
    kuyruk_prof_haritasi = profil_re_haritasi_al(polar_indeksi, KUYRUK_AIRFOIL_ADI)

    tum_alan_adlari = [
        "aday_no",
        "profil",
        "b_w_m",
        "AR_w",
        "S_w_m2",
        "c_w_m",
        "Re_aday",
        "Re_polar",
        "alpha_deg",
        "a0_2d_per_deg",
        "alpha_L0_deg",
        "Cl_max_2d",
        "alpha_stall_2d_deg",
        "CL_alpha_3d_per_deg",
        "e_oswald",
        "CL",
        "CL_seyir",
        "CL_max_3d_tahmini",
    ]

    aday_re_satirlari: list[dict] = []
    toplam_tum_satir = 0

    yol_tum_sonuclar.parent.mkdir(parents=True, exist_ok=True)
    with yol_tum_sonuclar.open("w", newline="", encoding="utf-8-sig") as f_tum:
        yazici_tum = csv.DictWriter(f_tum, fieldnames=tum_alan_adlari)
        yazici_tum.writeheader()

        for idx, aday in enumerate(adaylar, start=1):
            profil_adi = str(aday["kanat_airfoil"])
            b_w = float(aday["b_w_m"])
            ar_w = float(aday["AR_w"])
            s_w = float(aday["S_w_m2"])
            c_w = float(aday["c_root_w_m"])

            re_aday = reynolds_hesapla(c_w, hiz_ms=SEYIR_HIZI_MS)

            c_ht = float(aday["c_root_HT_m"])
            s_ht = float(aday["S_HT_m2"])
            ar_ht = float(aday["AR_HT"])
            re_ht = reynolds_hesapla(c_ht, hiz_ms=SEYIR_HIZI_MS)

            c_vt = float(aday["c_root_VT_m"])
            s_vt = float(aday["S_VT_m2"])
            ar_vt = float(aday["AR_VT"])
            re_vt = reynolds_hesapla(c_vt, hiz_ms=SEYIR_HIZI_MS)

            kanat_re_haritasi = profil_re_haritasi_al(polar_indeksi, profil_adi)

            temel = {
                "aday_no": idx,
                "profil": profil_adi,
                "b_w_m": b_w,
                "AR_w": ar_w,
                "S_w_m2": s_w,
                "c_w_m": c_w,
                "Re_aday": re_aday,
            }

            # 1) Tüm Reynolds sayılarında -5°..+15° açı taraması
            sonuclar_kanat_re: dict[int, dict] = {}
            for r_pol, p_ozel in sorted(kanat_re_haritasi.items()):
                sol = yuzey_coklu_aci_coz(ar_w, p_ozel, r_pol)
                sonuclar_kanat_re[r_pol] = sol

                ortak_bilgi = {
                    **temel,
                    "Re_polar": r_pol,
                    "a0_2d_per_deg": sol["a0_2d_per_deg"],
                    "alpha_L0_deg": sol["alpha_L0_deg"],
                    "Cl_max_2d": sol["Cl_max_2d"],
                    "alpha_stall_2d_deg": sol["alpha_stall_2d_deg"],
                    "CL_alpha_3d_per_deg": sol["CL_alpha_3d_per_deg"],
                    "e_oswald": sol["e_oswald"],
                    "CL_seyir": sol["CL_seyir"],
                    "CL_max_3d_tahmini": sol["CL_max_3d_tahmini"],
                }

                alfa_list = sol["alfa_dizisi_deg"]
                cl_list = sol["CL_dizisi"]
                satir_grubu = [
                    {
                        **ortak_bilgi,
                        "alpha_deg": float(alfa_list[k]),
                        "CL": float(cl_list[k]),
                    }
                    for k in range(len(alfa_list))
                ]
                yazici_tum.writerows(satir_grubu)
                toplam_tum_satir += len(satir_grubu)

            # 2) Adayın kendine ait Reynolds sayısında 0° hücum açısı sonuçları
            ara_kanat, durum_re = re_interpolasyonu(sonuclar_kanat_re, re_aday)
            ara_kanat.pop("Re_polar", None)

            # 3) Kuyruk çözümü (HT ve VT - CD olmadan)
            sk_ht = {r: kuyruk_yuzey_coz(ar_ht, pr) for r, pr in kuyruk_prof_haritasi.items()}
            ara_ht, durum_ht = re_interpolasyonu(sk_ht, re_ht)

            sk_vt = {r: kuyruk_yuzey_coz(ar_vt, pr) for r, pr in kuyruk_prof_haritasi.items()}
            ara_vt, durum_vt = re_interpolasyonu(sk_vt, re_vt)

            kuyruk_bilgi = {
                "c_HT_m": c_ht,
                "S_HT_m2": s_ht,
                "Re_HT": re_ht,
                "c_VT_m": c_vt,
                "S_VT_m2": s_vt,
                "Re_VT": re_vt,
                "Re_HT_durumu": durum_ht,
                "CL_alpha_HT_per_deg": ara_ht["CL_alpha_per_deg"],
                "e_HT": ara_ht["e"],
                "Re_VT_durumu": durum_vt,
                "CL_alpha_VT_per_deg": ara_vt["CL_alpha_per_deg"],
                "e_VT": ara_vt["e"],
            }

            satir_aday = {
                **temel,
                "Re_durumu": durum_re,
                "alpha_deg": SEYIR_ALPHA_DEG,
                **ara_kanat,
                **kuyruk_bilgi,
            }
            aday_re_satirlari.append(satir_aday)

    csv_yaz(aday_re_satirlari, yol_aday_re)
    return toplam_tum_satir, len(aday_re_satirlari)


# ------------------------------------------------------------------
# 5) FLAPLI UÇAK ÇÖZÜMÜ (-5..15° TÜM SONUÇLAR VE 0° ADAY RE)
# ------------------------------------------------------------------
def flapli_adaylari_coz_ve_yaz(
    adaylar: list[dict],
    flapsiz_polar_indeksi: dict[str, dict[int, dict]],
    flapli_polar_indeksi: dict[str, dict[int, dict]],
    yol_tum_sonuclar: Path,
    yol_aday_re: Path,
) -> tuple[int, int]:
    kuyruk_prof_haritasi = profil_re_haritasi_al(flapsiz_polar_indeksi, KUYRUK_AIRFOIL_ADI)

    tum_alan_adlari = [
        "aday_no",
        "kanat_airfoil",
        "flap_airfoil",
        "b_w_m",
        "AR_w",
        "S_w_m2",
        "c_w_m",
        "b_flap_m",
        "b_flapsiz_toplam_m",
        "S_flap_m2",
        "S_flapsiz_m2",
        "flap_aciklik_orani",
        "Re_flapsiz",
        "Re_flap",
        "Re_polar",
        "alpha_deg",
        "a0_2d_flapsiz_per_deg",
        "a0_2d_flapli_per_deg",
        "alpha_L0_flapsiz_deg",
        "alpha_L0_flapli_deg",
        "Cl_max_2d_flapsiz",
        "Cl_max_2d_flapli",
        "CL_alpha_3d_kanat_per_deg",
        "e_oswald_kanat",
        "CL_flapsiz_kisim",
        "CL_flapli_kisim",
        "CL",
        "CL_seyir",
        "CL_max_3d_tahmini",
    ]

    aday_re_satirlari: list[dict] = []
    toplam_tum_satir = 0

    yol_tum_sonuclar.parent.mkdir(parents=True, exist_ok=True)
    with yol_tum_sonuclar.open("w", newline="", encoding="utf-8-sig") as f_tum:
        yazici_tum = csv.DictWriter(f_tum, fieldnames=tum_alan_adlari)
        yazici_tum.writeheader()

        for idx, aday in enumerate(adaylar, start=1):
            profil_adi = str(aday["kanat_airfoil"])
            flap_profil_adi = str(aday["flap_airfoil"])

            b_w = float(aday["b_w_m"])
            ar_w = float(aday["AR_w"])
            s_w = float(aday["S_w_m2"])
            c_w = float(aday["c_root_w_m"])

            b_flap = float(aday.get("b_flap_m", 1.0))
            b_flapsiz = float(aday.get("b_flapsiz_toplam_m", b_w - b_flap))
            c_flap = float(aday.get("c_flap_w_m", c_w))
            s_flap = float(aday.get("S_flap_m2", b_flap * c_flap))
            s_flapsiz = float(aday.get("S_flapsiz_m2", s_w - s_flap))

            oran_flap = s_flap / s_w
            oran_flapsiz = s_flapsiz / s_w

            re_flapsiz = reynolds_hesapla(c_w, hiz_ms=KALKIS_HIZI_MS)
            re_flap = reynolds_hesapla(c_flap, hiz_ms=KALKIS_HIZI_MS)

            c_ht = float(aday["c_root_HT_m"])
            s_ht = float(aday["S_HT_m2"])
            ar_ht = float(aday["AR_HT"])
            re_ht = reynolds_hesapla(c_ht, hiz_ms=KALKIS_HIZI_MS)

            c_vt = float(aday["c_root_VT_m"])
            s_vt = float(aday["S_VT_m2"])
            ar_vt = float(aday["AR_VT"])
            re_vt = reynolds_hesapla(c_vt, hiz_ms=KALKIS_HIZI_MS)

            flapsiz_re_haritasi = profil_re_haritasi_al(flapsiz_polar_indeksi, profil_adi)
            flapli_re_haritasi = profil_re_haritasi_al(flapli_polar_indeksi, flap_profil_adi)

            temel = {
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
                "Re_flapsiz": re_flapsiz,
                "Re_flap": re_flap,
            }

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
                cl_max_birl = (
                    sol_flapli["CL_max_3d_tahmini"] * oran_flap
                    + sol_flapsiz["CL_max_3d_tahmini"] * oran_flapsiz
                )
                cl_a_birl = (
                    sol_flapli["CL_alpha_3d_per_deg"] * oran_flap
                    + sol_flapsiz["CL_alpha_3d_per_deg"] * oran_flapsiz
                )
                e_birl = (
                    sol_flapli["e_oswald"] * oran_flap
                    + sol_flapsiz["e_oswald"] * oran_flapsiz
                )

                cl_dizisi_flapsiz = sol_flapsiz["CL_dizisi"]
                cl_dizisi_flapli = sol_flapli["CL_dizisi"]
                cl_dizisi_birl = (
                    cl_dizisi_flapli * oran_flap
                    + cl_dizisi_flapsiz * oran_flapsiz
                )

                birlesik_0deg = {
                    "Re_polar": r_pol,
                    "a0_2d_flapsiz_per_deg": sol_flapsiz["a0_2d_per_deg"],
                    "a0_2d_flapli_per_deg": sol_flapli["a0_2d_per_deg"],
                    "alpha_L0_flapsiz_deg": sol_flapsiz["alpha_L0_deg"],
                    "alpha_L0_flapli_deg": sol_flapli["alpha_L0_deg"],
                    "Cl_max_2d_flapsiz": sol_flapsiz["Cl_max_2d"],
                    "Cl_max_2d_flapli": sol_flapli["Cl_max_2d"],
                    "CL_alpha_3d_kanat_per_deg": cl_a_birl,
                    "e_oswald_kanat": e_birl,
                    "CL_flapsiz_0deg": sol_flapsiz["CL_seyir"],
                    "CL_flapli_0deg": sol_flapli["CL_seyir"],
                    "CL_seyir": cl_0deg_birl,
                    "CL_max_3d_tahmini": cl_max_birl,
                }
                sonuclar_birlesik_re[r_pol] = birlesik_0deg

                ortak_bilgi = {
                    **temel,
                    "Re_polar": r_pol,
                    "a0_2d_flapsiz_per_deg": sol_flapsiz["a0_2d_per_deg"],
                    "a0_2d_flapli_per_deg": sol_flapli["a0_2d_per_deg"],
                    "alpha_L0_flapsiz_deg": sol_flapsiz["alpha_L0_deg"],
                    "alpha_L0_flapli_deg": sol_flapli["alpha_L0_deg"],
                    "Cl_max_2d_flapsiz": sol_flapsiz["Cl_max_2d"],
                    "Cl_max_2d_flapli": sol_flapli["Cl_max_2d"],
                    "CL_alpha_3d_kanat_per_deg": cl_a_birl,
                    "e_oswald_kanat": e_birl,
                    "CL_seyir": cl_0deg_birl,
                    "CL_max_3d_tahmini": cl_max_birl,
                }

                alfa_list = sol_flapsiz["alfa_dizisi_deg"]
                satir_grubu = [
                    {
                        **ortak_bilgi,
                        "alpha_deg": float(alfa_list[k]),
                        "CL_flapsiz_kisim": float(cl_dizisi_flapsiz[k]),
                        "CL_flapli_kisim": float(cl_dizisi_flapli[k]),
                        "CL": float(cl_dizisi_birl[k]),
                    }
                    for k in range(len(alfa_list))
                ]
                yazici_tum.writerows(satir_grubu)
                toplam_tum_satir += len(satir_grubu)

            # 2) Adayın kendi Reynolds değerinde 0° hücum açısı sonuçları
            ara_kanat, durum_re = re_interpolasyonu(sonuclar_birlesik_re, re_flapsiz)
            ara_kanat.pop("Re_polar", None)

            # 3) Kuyruk çözümü (HT ve VT - CD olmadan)
            sk_ht = {r: kuyruk_yuzey_coz(ar_ht, pr) for r, pr in kuyruk_prof_haritasi.items()}
            ara_ht, durum_ht = re_interpolasyonu(sk_ht, re_ht)

            sk_vt = {r: kuyruk_yuzey_coz(ar_vt, pr) for r, pr in kuyruk_prof_haritasi.items()}
            ara_vt, durum_vt = re_interpolasyonu(sk_vt, re_vt)

            kuyruk_bilgi = {
                "c_HT_m": c_ht,
                "S_HT_m2": s_ht,
                "Re_HT": re_ht,
                "c_VT_m": c_vt,
                "S_VT_m2": s_vt,
                "Re_VT": re_vt,
                "Re_HT_durumu": durum_ht,
                "CL_alpha_HT_per_deg": ara_ht["CL_alpha_per_deg"],
                "e_HT": ara_ht["e"],
                "Re_VT_durumu": durum_vt,
                "CL_alpha_VT_per_deg": ara_vt["CL_alpha_per_deg"],
                "e_VT": ara_vt["e"],
            }

            satir_aday = {
                **temel,
                "Re_durumu": durum_re,
                "alpha_deg": SEYIR_ALPHA_DEG,
                **ara_kanat,
                **kuyruk_bilgi,
            }
            aday_re_satirlari.append(satir_aday)

    csv_yaz(aday_re_satirlari, yol_aday_re)
    return toplam_tum_satir, len(aday_re_satirlari)


# ------------------------------------------------------------------
# 6) CSV YAZMA VE ANA ÇALIŞTIRMA
# ------------------------------------------------------------------
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
        PROJE_KLASORU / csv_adi,
        SCRIPT_DIR / csv_adi,
        Path(csv_adi),
    ]
    for yol in olasi_yollar:
        if yol.exists() and yol.stat().st_size > 0:
            with yol.open("r", encoding="utf-8-sig", newline="") as f:
                return list(csv.DictReader(f))

    if flapli_mi:
        return kb.tum_flapli_adaylari_uret(ar_ht=4.0, ar_vt=4.0)
    return kb.tum_adaylari_uret(ar_ht=4.0, ar_vt=4.0)


def main() -> None:
    print("1) 2B polar klasorleri indeksleniyor (sadece CL verileri)...")
    flapsiz_indeks = polar_klasorunu_indeksle(POLAR_KLASORU)
    flapli_indeks = polar_klasorunu_indeksle(FLAP_POLAR_KLASORU)
    print(f"   - {POLAR_KLASORU.name}: {len(flapsiz_indeks)} anahtar indekslendi.")
    print(f"   - {FLAP_POLAR_KLASORU.name}: {len(flapli_indeks)} flapli anahtar indekslendi.")

    # ------------------ FLAPSIZ ÇÖZÜM ------------------
    print("\n2) Flapsiz kanat + kuyruk adaylari LLT ile cozuluyor (-5..15° ve 0° aday Re)...")
    flapsiz_adaylar = adaylari_yukle(FLAPSIZ_ADAY_CSV_ADI, flapli_mi=False)
    yol_flapsiz_tum = PROJE_KLASORU / "llt_flapsiz_tum_sonuclar.csv"
    yol_flapsiz_aday = PROJE_KLASORU / "llt_flapsiz_aday_re.csv"

    n_tum_flapsiz, n_aday_flapsiz = flapsiz_adaylari_coz_ve_yaz(
        flapsiz_adaylar,
        flapsiz_indeks,
        yol_flapsiz_tum,
        yol_flapsiz_aday,
    )
    print(f"   - Kaydedildi: {yol_flapsiz_tum} ({n_tum_flapsiz} satir)")
    print(f"   - Kaydedildi: {yol_flapsiz_aday} ({n_aday_flapsiz} aday)")

    # ------------------ FLAPLI ÇÖZÜM ------------------
    print("\n3) Flapli kanat + kuyruk adaylari LLT ile cozuluyor (-5..15° ve 0° aday Re)...")
    flapli_adaylar = adaylari_yukle(FLAPLI_ADAY_CSV_ADI, flapli_mi=True)
    yol_flapli_tum = PROJE_KLASORU / "llt_flapli_tum_sonuclar.csv"
    yol_flapli_aday = PROJE_KLASORU / "llt_flapli_aday_re.csv"

    n_tum_flapli, n_aday_flapli = flapli_adaylari_coz_ve_yaz(
        flapli_adaylar,
        flapsiz_indeks,
        flapli_indeks,
        yol_flapli_tum,
        yol_flapli_aday,
    )
    print(f"   - Kaydedildi: {yol_flapli_tum} ({n_tum_flapli} satir)")
    print(f"   - Kaydedildi: {yol_flapli_aday} ({n_aday_flapli} aday)")

    print("\nIslem basariyla tamamlandi!")


if __name__ == "__main__":
    main()