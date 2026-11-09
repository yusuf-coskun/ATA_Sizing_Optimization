from __future__ import annotations

import csv
import re
import sys
from collections.abc import Iterable
from math import sqrt
from pathlib import Path

_SCRIPT_KLASORU = Path(__file__).resolve().parent
if str(_SCRIPT_KLASORU) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_KLASORU))

from govde_boyutlandirma import L_total as GOVDE_L_TOTAL

V_HT = 0.50
V_VT = 0.05
KUYRUK_AIRFOIL_DOSYA_ADI = "naca0010.dat"
FLAP_TOPLAM_ACIKLIK_M = 1.0  # Kanadin tam ortasinda simetrik toplam 1.0 m flap

# airfoils ve airfoils_flap klasörleri depo ana dizininde (../) veya script dizininde (./) olabilir
AIRFOIL_KLASORU = (
    _SCRIPT_KLASORU.parent / "airfoils"
    if (_SCRIPT_KLASORU.parent / "airfoils").exists()
    else _SCRIPT_KLASORU / "airfoils"
)

AIRFOIL_FLAP_KLASORU = (
    _SCRIPT_KLASORU.parent / "airfoils_flap"
    if (_SCRIPT_KLASORU.parent / "airfoils_flap").exists()
    else _SCRIPT_KLASORU / "airfoils_flap"
)

# airfoils klasöründeki flapsız profil dosya kök adları ile airfoils_flap klasöründeki
# karşılık gelen flaplı profil dosya kök adlarının birebir eşleşmesi
FLAPSIZ_FLAPLI_AIRFOIL_ESLESMESI: dict[str, str] = {
    "clarkysm-il": "CLARK-Y 11.7%_flap smoothed",
    "cr001sm-il": "cr001sm_flap",
    "dae51-il": "DAE-51 AIRFOIL_flap",
    "e174-il": "E174 (Dicke 8.92%)_flap",
    "e212-il": "E212 (10.55%)_flap",
    "e387-il": "E387_flap",
    "e392-il": "E392 (10.15%)_flap",
    "hq2590sm-il": "HQ 2.5 9.0 smoothed by Eppler_flap",
    "hq3010-il": "HQ 3.0 10 AIRFOIL_flap",
    "mh70-il": "MH 70 11.08%_flap",
    "n9-il": "N-9_flap",
    "naca2410-il": "NACA 2410_flap",
    "naca4412-il": "NACA 4412_flap",
    "psu94097-il": "PSU 94-097 (AIAA 2001-2478)_flap",
    "s2046-il": "S2046_flap",
    "s2091-il": "S2091-101-83_flap",
    "s3010-il": "S3010-103-84_flap",
    "s4062-il": "S4062-095-87_flap",
    "s4110-il": "S4110_flap",
    "s4310-il": "S4310_flap",
    "s7075-il": "S7075 (9%)_flap",
    "sd6080-il": "SD6080 (9.2%)_flap",
    "sd7032-il": "SD7032-099-88_flap",
    "sd7037-il": "SD7037-092-88_flap",
    "sd7043-il": "SD7043 (9.1%)_flap",
    "sd8040-il": "SD8040 (10%)_flap",
    "sg6041-il": "SG6041_flap",
    "sg6042-il": "SG6042_flap",
    "waspsm-il": "WASP (smoothed)_flap",
}


def _pozitif(deger: float, isim: str) -> None:
    if deger <= 0:
        raise ValueError(f"{isim} sifirdan buyuk olmali (gelen: {deger}).")


def _norm_airfoil_adi(dosya_yolu: Path) -> str:
    return (
        dosya_yolu.stem.upper()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def _airfoil_cekirdek_kodu(dosya_yolu: Path) -> str:
    """
    Flapsiz (orn. 'clarkysm-il.dat') ve flapli (orn. 'CLARK-Y 11.7%_flap smoothed.dat')
    airfoil dosyalarini ayni temel profil koduna indirger.
    """
    ad = dosya_yolu.stem.lower()
    ad = re.sub(r"(?:sm)?(?:-il)?$", "", ad)
    ad = ad.replace("_flap", "").replace("smoothed", "").replace("airfoil", "")
    ad = re.sub(r"\(.*?\)", "", ad)
    return re.sub(r"[^a-z0-9]", "", ad)


def airfoilleri_yukle(
    klasor: str | Path = AIRFOIL_KLASORU,
) -> tuple[Path, list[Path]]:
    """
    airfoils klasöründeki naca0010.dat dosyasını kuyruk için,
    NACA0010 hariç geri kalan tüm .dat dosyalarını ise kanat adayları için döndürür.
    """
    klasor_yolu = Path(klasor)
    if not klasor_yolu.exists():
        raise FileNotFoundError(f"Airfoil klasoru bulunamadi: {klasor_yolu}")

    dat_dosyalari = sorted(klasor_yolu.glob("*.dat"))
    if not dat_dosyalari:
        raise FileNotFoundError(
            f"{klasor_yolu} altinda hic .dat dosyasi bulunamadi."
        )

    kuyruk_airfoilleri = [
        p
        for p in dat_dosyalari
        if p.name.lower() == KUYRUK_AIRFOIL_DOSYA_ADI.lower()
        or _norm_airfoil_adi(p) == "NACA0010"
    ]
    if not kuyruk_airfoilleri:
        raise FileNotFoundError(
            f"{klasor_yolu} altinda {KUYRUK_AIRFOIL_DOSYA_ADI} dosyasi bulunamadi."
        )
    kuyruk_airfoil = kuyruk_airfoilleri[0]

    kanat_airfoilleri = [
        p
        for p in dat_dosyalari
        if p.name.lower() != KUYRUK_AIRFOIL_DOSYA_ADI.lower()
        and _norm_airfoil_adi(p) != "NACA0010"
    ]
    if not kanat_airfoilleri:
        raise ValueError(
            f"{klasor_yolu} altinda NACA0010 haric kanat airfoili bulunamadi."
        )

    return kuyruk_airfoil, kanat_airfoilleri


def flapli_airfoilleri_yukle(
    airfoil_klasoru: str | Path = AIRFOIL_KLASORU,
    airfoil_flap_klasoru: str | Path = AIRFOIL_FLAP_KLASORU,
) -> tuple[Path, list[tuple[Path, Path]]]:
    """
    airfoils ve airfoils_flap klasörlerindeki profilleri okur; her kanat için
    flapsız ve flaplı airfoilin aynı temel profile ait olmasını garanti ederek
    (flapsiz_airfoil_yolu, flapli_airfoil_yolu) çiftlerini döndürür.
    """
    kuyruk_airfoil, kanat_airfoilleri = airfoilleri_yukle(airfoil_klasoru)

    flap_klasor_yolu = Path(airfoil_flap_klasoru)
    if not flap_klasor_yolu.exists():
        raise FileNotFoundError(
            f"Flapli airfoil klasoru bulunamadi: {flap_klasor_yolu}"
        )

    flap_dat_dosyalari = sorted(flap_klasor_yolu.glob("*.dat"))
    if not flap_dat_dosyalari:
        raise FileNotFoundError(
            f"{flap_klasor_yolu} altinda hic .dat dosyasi bulunamadi."
        )

    flap_stem_haritasi = {p.stem.lower(): p for p in flap_dat_dosyalari}
    eslesen_airfoiller: list[tuple[Path, Path]] = []

    for flapsiz_yol in kanat_airfoilleri:
        hedef_flap_stem = FLAPSIZ_FLAPLI_AIRFOIL_ESLESMESI.get(
            flapsiz_yol.stem.lower()
        )
        flapli_yol: Path | None = None

        if hedef_flap_stem is not None:
            flapli_yol = flap_stem_haritasi.get(hedef_flap_stem.lower())

        if flapli_yol is None:
            cekirdek = _airfoil_cekirdek_kodu(flapsiz_yol)
            aday_eslesmeler = [
                p
                for p in flap_dat_dosyalari
                if _airfoil_cekirdek_kodu(p).startswith(cekirdek)
            ]
            if len(aday_eslesmeler) == 1:
                flapli_yol = aday_eslesmeler[0]

        if flapli_yol is None:
            raise ValueError(
                f"'{flapsiz_yol.name}' icin '{flap_klasor_yolu}' klasorunde "
                "eslesen ayni flapli airfoil bulunamadi."
            )

        eslesen_airfoiller.append((flapsiz_yol, flapli_yol))

    return kuyruk_airfoil, eslesen_airfoiller


def kanat_boyutlandir(
    b_w_m: float,
    ar_w: float,
    kanat_airfoil_dat: str | Path,
) -> dict:
    _pozitif(b_w_m, "b_w_m")
    _pozitif(ar_w, "ar_w")

    airfoil_yolu = Path(kanat_airfoil_dat)
    s_w = b_w_m**2 / ar_w
    b_w = b_w_m
    c_w = s_w / b_w

    return {
        "kanat_airfoil": airfoil_yolu.stem,
        "kanat_airfoil_dat": airfoil_yolu.as_posix(),
        "AR_w": ar_w,
        "S_w_m2": s_w,
        "b_w_m": b_w,
        "c_root_w_m": c_w,
        "c_tip_w_m": c_w,
        "MAC_w_m": c_w,
    }


def kanat_boyutlandir_flapli(
    b_w_m: float,
    ar_w: float,
    kanat_airfoil_dat: str | Path,
    flap_airfoil_dat: str | Path,
    *,
    b_flap_m: float = FLAP_TOPLAM_ACIKLIK_M,
) -> dict:
    """
    Kanadın tam ortasına (y = -b_flap_m/2 ile +b_flap_m/2 arasına) simetrik,
    toplamda b_flap_m (varsayılan 1.0 m) açıklığa sahip flap yerleştirerek
    kanat boyutlandırmasını yapar. Merkezdeki 1 m'lik kısım flaplı airfoili,
    sol ve sağ dış kısımlar ise aynı profilin flapsız airfoilini kullanır.
    """
    _pozitif(b_w_m, "b_w_m")
    _pozitif(ar_w, "ar_w")
    _pozitif(b_flap_m, "b_flap_m")

    if b_flap_m >= b_w_m:
        raise ValueError(
            f"Toplam flap acikligi ({b_flap_m} m), kanat acikligindan ({b_w_m} m) kucuk olmali."
        )

    airfoil_yolu = Path(kanat_airfoil_dat)
    flap_airfoil_yolu = Path(flap_airfoil_dat)

    s_w = b_w_m**2 / ar_w
    b_w = b_w_m
    c_w = s_w / b_w

    # Kanadin tam ortasinda simetrik flap yerlesimi: [-b_flap/2, +b_flap/2]
    b_flap_yari = b_flap_m / 2.0
    y_flap_baslangic = -b_flap_yari
    y_flap_bitis = b_flap_yari

    # Simetrik sol ve sag dis (flapsiz) kanat bolgeleri
    b_flapsiz_toplam = b_w - b_flap_m
    b_flapsiz_sol = b_flapsiz_toplam / 2.0
    b_flapsiz_sag = b_flapsiz_toplam / 2.0

    s_flap = b_flap_m * c_w
    s_flapsiz = b_flapsiz_toplam * c_w
    flap_aciklik_orani = b_flap_m / b_w

    return {
        "kanat_airfoil": airfoil_yolu.stem,
        "kanat_airfoil_dat": airfoil_yolu.as_posix(),
        "flap_airfoil": flap_airfoil_yolu.stem,
        "flap_airfoil_dat": flap_airfoil_yolu.as_posix(),
        "AR_w": ar_w,
        "S_w_m2": s_w,
        "b_w_m": b_w,
        "c_root_w_m": c_w,
        "c_tip_w_m": c_w,
        "MAC_w_m": c_w,
        "b_flap_m": b_flap_m,
        "b_flap_yari_m": b_flap_yari,
        "y_flap_baslangic_m": y_flap_baslangic,
        "y_flap_bitis_m": y_flap_bitis,
        "b_flapsiz_toplam_m": b_flapsiz_toplam,
        "b_flapsiz_sol_m": b_flapsiz_sol,
        "b_flapsiz_sag_m": b_flapsiz_sag,
        "c_flap_w_m": c_w,
        "S_flap_m2": s_flap,
        "S_flapsiz_m2": s_flapsiz,
        "flap_aciklik_orani": flap_aciklik_orani,
    }


def kuyruk_boyutlandir(
    kanat: dict,
    govde_uzunlugu_m: float,
    *,
    kuyruk_airfoil_dat: str | Path,
    tail_arm_m: float | None = None,
    ar_ht: float = 4.0,
    ar_vt: float = 4.0,
) -> dict:
    _pozitif(govde_uzunlugu_m, "govde_uzunlugu_m")
    _pozitif(ar_ht, "ar_ht")
    _pozitif(ar_vt, "ar_vt")

    airfoil_yolu = Path(kuyruk_airfoil_dat)
    l_t = 0.55 * govde_uzunlugu_m if tail_arm_m is None else tail_arm_m
    _pozitif(l_t, "tail_arm_m")

    s_w = kanat["S_w_m2"]
    c_mac = kanat["MAC_w_m"]
    b_w = kanat["b_w_m"]

    s_ht = V_HT * s_w * c_mac / l_t
    s_vt = V_VT * s_w * b_w / l_t

    b_ht = sqrt(ar_ht * s_ht)
    c_ht = s_ht / b_ht
    h_vt = sqrt(ar_vt * s_vt)
    c_vt = s_vt / h_vt

    return {
        "govde_uzunlugu_m": govde_uzunlugu_m,
        "tail_arm_m": l_t,
        "kuyruk_airfoil": airfoil_yolu.stem,
        "kuyruk_airfoil_dat": airfoil_yolu.as_posix(),
        "V_HT": V_HT,
        "V_VT": V_VT,
        "AR_HT": ar_ht,
        "AR_VT": ar_vt,
        "S_HT_m2": s_ht,
        "b_HT_m": b_ht,
        "c_root_HT_m": c_ht,
        "S_VT_m2": s_vt,
        "h_VT_m": h_vt,
        "c_root_VT_m": c_vt,
    }


def aday_boyutlandir(
    govde_uzunlugu_m: float,
    b_w_m: float,
    ar_w: float,
    *,
    kanat_airfoil_dat: str | Path,
    kuyruk_airfoil_dat: str | Path,
    tail_arm_m: float | None = None,
    ar_ht: float = 4.0,
    ar_vt: float = 4.0,
) -> dict:
    kanat = kanat_boyutlandir(b_w_m, ar_w, kanat_airfoil_dat=kanat_airfoil_dat)
    kuyruk = kuyruk_boyutlandir(
        kanat,
        govde_uzunlugu_m,
        kuyruk_airfoil_dat=kuyruk_airfoil_dat,
        tail_arm_m=tail_arm_m,
        ar_ht=ar_ht,
        ar_vt=ar_vt,
    )
    return {**kanat, **kuyruk}


def aday_boyutlandir_flapli(
    govde_uzunlugu_m: float,
    b_w_m: float,
    ar_w: float,
    *,
    kanat_airfoil_dat: str | Path,
    flap_airfoil_dat: str | Path,
    kuyruk_airfoil_dat: str | Path,
    b_flap_m: float = FLAP_TOPLAM_ACIKLIK_M,
    tail_arm_m: float | None = None,
    ar_ht: float = 4.0,
    ar_vt: float = 4.0,
) -> dict:
    kanat = kanat_boyutlandir_flapli(
        b_w_m,
        ar_w,
        kanat_airfoil_dat=kanat_airfoil_dat,
        flap_airfoil_dat=flap_airfoil_dat,
        b_flap_m=b_flap_m,
    )
    kuyruk = kuyruk_boyutlandir(
        kanat,
        govde_uzunlugu_m,
        kuyruk_airfoil_dat=kuyruk_airfoil_dat,
        tail_arm_m=tail_arm_m,
        ar_ht=ar_ht,
        ar_vt=ar_vt,
    )
    return {**kanat, **kuyruk}


def tum_adaylari_uret(
    govde_uzunluklari_m: float | Iterable[float] = GOVDE_L_TOTAL,
    *,
    airfoil_klasoru: str | Path = AIRFOIL_KLASORU,
    tail_arm_m: float | None = None,
    ar_ht: float = 4.0,
    ar_vt: float = 4.0,
) -> list[dict]:
    kuyruk_airfoil, kanat_airfoilleri = airfoilleri_yukle(airfoil_klasoru)

    if isinstance(govde_uzunluklari_m, Iterable):
        govde_listesi = [float(uzunluk) for uzunluk in govde_uzunluklari_m]
    else:
        govde_listesi = [float(govde_uzunluklari_m)]

    b_degerleri = [round(2.0 + 0.02 * i, 2) for i in range(3)] #range 61 olacak
    ar_degerleri = range(6, 7) #7, 11 olacak

    return [
        aday_boyutlandir(
            govde_uzunlugu_m,
            b_w_m,
            ar_w,
            kanat_airfoil_dat=kanat_airfoil,
            kuyruk_airfoil_dat=kuyruk_airfoil,
            tail_arm_m=tail_arm_m,
            ar_ht=ar_ht,
            ar_vt=ar_vt,
        )
        for govde_uzunlugu_m in govde_listesi
        for kanat_airfoil in kanat_airfoilleri
        for b_w_m in b_degerleri
        for ar_w in ar_degerleri
        if b_w_m**2 / ar_w >= 0.6
    ]


def tum_flapli_adaylari_uret(
    govde_uzunluklari_m: float | Iterable[float] = GOVDE_L_TOTAL,
    *,
    airfoil_klasoru: str | Path = AIRFOIL_KLASORU,
    airfoil_flap_klasoru: str | Path = AIRFOIL_FLAP_KLASORU,
    b_flap_m: float = FLAP_TOPLAM_ACIKLIK_M,
    tail_arm_m: float | None = None,
    ar_ht: float = 4.0,
    ar_vt: float = 4.0,
) -> list[dict]:
    """
    Tüm gövde uzunluğu, kanat açıklığı, AR ve eşleşen (flapsız, flaplı) airfoil
    kombinasyonları için ortasında simetrik 1 m flap bulunan kanat+kuyruk
    adaylarını üretir.
    """
    kuyruk_airfoil, eslesen_airfoiller = flapli_airfoilleri_yukle(
        airfoil_klasoru=airfoil_klasoru,
        airfoil_flap_klasoru=airfoil_flap_klasoru,
    )

    if isinstance(govde_uzunluklari_m, Iterable):
        govde_listesi = [float(uzunluk) for uzunluk in govde_uzunluklari_m]
    else:
        govde_listesi = [float(govde_uzunluklari_m)]

    b_degerleri = [round(2.0 + 0.02 * i, 2) for i in range(3)] #range 61 olacak
    ar_degerleri = range(6, 7) #7, 11 olacak

    return [
        aday_boyutlandir_flapli(
            govde_uzunlugu_m,
            b_w_m,
            ar_w,
            kanat_airfoil_dat=kanat_airfoil,
            flap_airfoil_dat=flap_airfoil,
            kuyruk_airfoil_dat=kuyruk_airfoil,
            b_flap_m=b_flap_m,
            tail_arm_m=tail_arm_m,
            ar_ht=ar_ht,
            ar_vt=ar_vt,
        )
        for govde_uzunlugu_m in govde_listesi
        for kanat_airfoil, flap_airfoil in eslesen_airfoiller
        for b_w_m in b_degerleri
        for ar_w in ar_degerleri
        if b_w_m**2 / ar_w >= 0.6 and b_w_m > b_flap_m
    ]


def csv_kaydet(adaylar: list[dict], dosya_yolu: str | Path) -> None:
    if not adaylar:
        raise ValueError("Kaydedilecek aday bulunamadi.")
    with Path(dosya_yolu).open("w", newline="", encoding="utf-8-sig") as f:
        yazici = csv.DictWriter(f, fieldnames=list(adaylar[0].keys()))
        yazici.writeheader()
        yazici.writerows(adaylar)


if __name__ == "__main__":
    adaylar = tum_adaylari_uret(
        ar_ht=4.0,
        ar_vt=4.0,
    )
    print(f"Toplam flapsiz aday sayisi: {len(adaylar)}")
    print("Ilk flapsiz aday:", adaylar[0])
    csv_kaydet(adaylar, "kanat_kuyruk_adaylari.csv")

    flapli_adaylar = tum_flapli_adaylari_uret(
        ar_ht=4.0,
        ar_vt=4.0,
        b_flap_m=FLAP_TOPLAM_ACIKLIK_M,
    )
    print(f"Toplam flapli aday sayisi: {len(flapli_adaylar)}")
    print("Ilk flapli aday:", flapli_adaylar[0])
    csv_kaydet(flapli_adaylar, "kanat_kuyruk_adayları_flapli.csv")