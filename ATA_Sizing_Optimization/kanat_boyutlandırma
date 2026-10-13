"""Dikdortgen kanat + konvansiyonel kuyruk ilk boyutlandirmasi.

Kanat acikligi taramasi: 2.00...3.20 m, 0.05 m artis (25 deger)
Kanat AR taramasi: 6...10, 1 artis (5 deger)
Toplam: 125 aday.

Notlar:
- W/S taranmaz; sonuc tablosunda hesaplanan W/S icin birimi ('kg/m2' ya da 'N/m2') secin.
- Kuyruk hacim katsayilari V_HT=0.20 ve V_VT=0.05.
- Tail arm verilmezse govde uzunlugunun %55'i kullanilir.
- AR_HT=4 ve AR_VT=1.5 SADECE ornek baslangic varsayimlaridir.
- Yatay/dikey kuyruklar da ilk yaklasimda dikdortgen kabul edilmistir.
- Aerodinamik analiz ve yapisal agirlik hesaplari bu modulun disindadir.
"""

from __future__ import annotations

import csv
from math import sqrt
from pathlib import Path
from typing import Literal

G = 9.80665  # m/s^2
V_HT = 0.20
V_VT = 0.05
WSUnit = Literal["kg/m2", "N/m2"]


def _pozitif(deger: float, isim: str) -> None:
    if deger <= 0:
        raise ValueError(f"{isim} sifirdan buyuk olmali (gelen: {deger}).")


def kanat_boyutlandir(
    kutle_kg: float,
    b_w_m: float,
    ar_w: float,
    *,
    ws_birimi: WSUnit,
) -> dict:
    """Dikdortgen kanat icin alan, aciklik ve kok veter hesaplar.

    Kutle kg cinsindedir. ws_birimi, taranmayan W/S ciktisinin birimidir:
      - 'kg/m2': ws = kutle/S (kutlesel kanat yuklemesi)
      - 'N/m2': ws = agirlik/S (kuvvet bazli W/S)
    """
    _pozitif(kutle_kg, "kutle_kg")
    _pozitif(b_w_m, "b_w_m")
    _pozitif(ar_w, "ar_w")

    s_w = b_w_m**2 / ar_w
    if ws_birimi == "kg/m2":
        ws = kutle_kg / s_w
    elif ws_birimi == "N/m2":
        ws = (kutle_kg * G) / s_w
    else:
        raise ValueError("ws_birimi 'kg/m2' veya 'N/m2' olmali.")

    b_w = b_w_m
    c_w = s_w / b_w

    return {
        "WS": ws,
        "WS_birimi": ws_birimi,
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
    tail_arm_m: float | None = None,
    ar_ht: float = 4.0,
    ar_vt: float = 1.5,
) -> dict:
    """Hacim katsayilari ile konvansiyonel kuyruk alanlarini hesaplar.

    Tail arm, kanat ve kuyruk aerodinamik merkezlerinin mesafesidir.
    ar_ht ve ar_vt kuyruk planformu icin ORNEK varsayimlardir;
    projenin gereksinimlerine gore bunlari degistirin.

    Dikey kuyruk AR tanimi: AR_VT = h_VT**2 / S_VT.
    """
    _pozitif(govde_uzunlugu_m, "govde_uzunlugu_m")
    _pozitif(ar_ht, "ar_ht")
    _pozitif(ar_vt, "ar_vt")

    l_t = 0.55 * govde_uzunlugu_m if tail_arm_m is None else tail_arm_m
    _pozitif(l_t, "tail_arm_m")

    s_w = kanat["S_w_m2"]
    c_mac = kanat["MAC_w_m"]
    b_w = kanat["b_w_m"]

    # V_HT = S_HT * l_t / (S_w * MAC_w)
    # V_VT = S_VT * l_t / (S_w * b_w)
    s_ht = V_HT * s_w * c_mac / l_t
    s_vt = V_VT * s_w * b_w / l_t

    # Dikdortgen kuyruk geometrisi (ilk yaklasim).
    b_ht = sqrt(ar_ht * s_ht)
    c_ht = s_ht / b_ht
    h_vt = sqrt(ar_vt * s_vt)
    c_vt = s_vt / h_vt

    return {
        "tail_arm_m": l_t,
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
    kutle_kg: float,
    govde_uzunlugu_m: float,
    b_w_m: float,
    ar_w: float,
    *,
    ws_birimi: WSUnit,
    tail_arm_m: float | None = None,
    ar_ht: float = 4.0,
    ar_vt: float = 1.5,
) -> dict:
    """Tek bir tasarim adayinin kanat ve kuyruk boyutlarini return eder."""
    kanat = kanat_boyutlandir(kutle_kg, b_w_m, ar_w, ws_birimi=ws_birimi)
    kuyruk = kuyruk_boyutlandir(
        kanat,
        govde_uzunlugu_m,
        tail_arm_m=tail_arm_m,
        ar_ht=ar_ht,
        ar_vt=ar_vt,
    )
    return {**kanat, **kuyruk}


def tum_adaylari_uret(
    kutle_kg: float,
    govde_uzunlugu_m: float,
    *,
    ws_birimi: WSUnit,
    tail_arm_m: float | None = None,
    ar_ht: float = 4.0,
    ar_vt: float = 1.5,
) -> list[dict]:
    """125 kombinasyondan kanat alani en az 0.6 m2 olan adaylari dondurur."""
    b_degerleri = [round(2.0 + 0.02 * i, 2) for i in range(61)]
    ar_degerleri = range(6, 11)

    return [
        aday_boyutlandir(
            kutle_kg,
            govde_uzunlugu_m,
            b_w_m,
            ar_w,
            ws_birimi=ws_birimi,
            tail_arm_m=tail_arm_m,
            ar_ht=ar_ht,
            ar_vt=ar_vt,
        )
        for b_w_m in b_degerleri
        for ar_w in ar_degerleri
        if b_w_m**2 / ar_w >= 0.6
    ]


def csv_kaydet(adaylar: list[dict], dosya_yolu: str | Path) -> None:
    """Diger optimizasyon modullerine aktarim icin tum adaylari CSV'ye yazar."""
    if not adaylar:
        raise ValueError("Kaydedilecek aday bulunamadi.")
    with Path(dosya_yolu).open("w", newline="", encoding="utf-8-sig") as f:
        yazici = csv.DictWriter(f, fieldnames=list(adaylar[0].keys()))
        yazici.writeheader()
        yazici.writerows(adaylar)


if __name__ == "__main__":
    # ASAGIDAKI KUTLE/GOVDE UZUNLUGU SADECE ORNEKTIR.
    # Gercek degerler toplam agirlik ve govde boyutlandirma modullerinden gelmeli.
    adaylar = tum_adaylari_uret(
        kutle_kg=6.0,
        govde_uzunlugu_m=1.5,
        ws_birimi="kg/m2",
        ar_ht=4.0,
        ar_vt=1.5,
    )
    print(f"Toplam aday sayisi: {len(adaylar)}")
    print("Ilk aday:", adaylar[0])
    csv_kaydet(adaylar, "kanat_kuyruk_adaylari.csv")

print(len(adaylar))  # 155

for aday in adaylar:
    print(
        aday["S_w_m2"],
        aday["c_root_w_m"],
        aday["S_HT_m2"],
        aday["S_VT_m2"]
    )
