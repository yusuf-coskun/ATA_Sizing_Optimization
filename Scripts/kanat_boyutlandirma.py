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
    _pozitif(govde_uzunlugu_m, "govde_uzunlugu_m")
    _pozitif(ar_ht, "ar_ht")
    _pozitif(ar_vt, "ar_vt")

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
    if not adaylar:
        raise ValueError("Kaydedilecek aday bulunamadi.")
    with Path(dosya_yolu).open("w", newline="", encoding="utf-8-sig") as f:
        yazici = csv.DictWriter(f, fieldnames=list(adaylar[0].keys()))
        yazici.writeheader()
        yazici.writerows(adaylar)


if __name__ == "__main__":
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

    for aday in adaylar:
        print(
            aday["S_w_m2"],
            aday["c_root_w_m"],
            aday["S_HT_m2"],
            aday["S_VT_m2"],
        )