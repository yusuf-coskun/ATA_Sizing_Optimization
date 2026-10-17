from __future__ import annotations

import csv
from collections.abc import Iterable
from math import sqrt
from pathlib import Path

from govde_boyutlandirma import L_total as GOVDE_L_TOTAL

V_HT = 0.20
V_VT = 0.05
AIRFOIL_KLASORU = Path(__file__).resolve().parent / "airfoils"


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


def airfoilleri_yukle(
    klasor: str | Path = AIRFOIL_KLASORU,
) -> tuple[Path, list[Path]]:
    """
    airfoils klasöründeki NACA0010.dat dosyasını kuyruk için,
    geri kalan tüm .dat dosyalarını ise kanat adayları için döndürür.
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
        p for p in dat_dosyalari if _norm_airfoil_adi(p) == "NACA0010.dat"
    ]
    if not kuyruk_airfoilleri:
        raise FileNotFoundError(
            f"{klasor_yolu} altinda NACA0010.dat dosyasi bulunamadi."
        )
    kuyruk_airfoil = kuyruk_airfoilleri[0]

    kanat_airfoilleri = [
        p for p in dat_dosyalari if _norm_airfoil_adi(p) != "NACA0010.dat"
    ]
    if not kanat_airfoilleri:
        raise ValueError(
            f"{klasor_yolu} altinda NACA0010 haric kanat airfoili bulunamadi."
        )

    return kuyruk_airfoil, kanat_airfoilleri


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


def kuyruk_boyutlandir(
    kanat: dict,
    govde_uzunlugu_m: float,
    *,
    kuyruk_airfoil_dat: str | Path,
    tail_arm_m: float | None = None,
    ar_ht: float = 4.0,
    ar_vt: float = 1.5,
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
    ar_vt: float = 1.5,
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


def tum_adaylari_uret(
    govde_uzunluklari_m: float | Iterable[float] = GOVDE_L_TOTAL,
    *,
    airfoil_klasoru: str | Path = AIRFOIL_KLASORU,
    tail_arm_m: float | None = None,
    ar_ht: float = 4.0,
    ar_vt: float = 1.5,
) -> list[dict]:
    kuyruk_airfoil, kanat_airfoilleri = airfoilleri_yukle(airfoils)

    if isinstance(govde_uzunluklari_m, Iterable):
        govde_listesi = [float(uzunluk) for uzunluk in govde_uzunluklari_m]
    else:
        govde_listesi = [float(govde_uzunluklari_m)]

    b_degerleri = [round(2.0 + 0.02 * i, 2) for i in range(61)]
    ar_degerleri = range(6, 11)

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
        ar_vt=1.5,
    )
    print(f"Toplam aday sayisi: {len(adaylar)}")
    print("Ilk aday:", adaylar[0])
    csv_kaydet(adaylar, "kanat_kuyruk_adaylari.csv")