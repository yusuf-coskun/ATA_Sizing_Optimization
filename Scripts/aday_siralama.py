#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATA Sizing Optimization - Ana Aday Değerlendirme ve Sıralama Modülü

Tüm performans modüllerinin (kalkış, tırmanış, seyir, yük bırakma, iniş)
ürettiği CSV çıktılarını ortak aday/konfigürasyon kimliği üzerinden birleştirir,
modüllerden gelen toplam puanları toplar, batarya ve süre kısıtlarını denetler
ve VLM/VSPAERO aşaması için en iyi ilk N (varsayılan: 500) adayı sıralar.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from aday_pervane import (
        GERILIM_V,
        csv_oku as _ap_csv_oku,
        csv_yaz as _ap_csv_yaz,
        kimlik_alanlari,
        performans_adaylarini_hazirla,
        temel_kimlik,
    )
except ImportError:
    GERILIM_V = 22.5
    _ap_csv_oku = None
    _ap_csv_yaz = None
    kimlik_alanlari = None
    performans_adaylarini_hazirla = None
    temel_kimlik = None


# Sabit Görev ve Batarya Sınırları
BATARYA_KAPASITE_MAH = 12000.0
INIS_REZERV_ORANI = 0.20
KULLANILABILIR_MAH = BATARYA_KAPASITE_MAH * (1.0 - INIS_REZERV_ORANI)  # 9600.0 mAh
MAX_GOREV_SURESI_SN = 600.0  # 10 dakika


def dosya_yolu_coz(dosya_adi: str, ek_dizinler: Optional[Iterable[Path]] = None) -> Path:
    """Verilen dosya adını çalışma dizini, Scripts/ ve depo kök dizininde arar."""
    aday_yol = Path(dosya_adi)
    if aday_yol.is_file():
        return aday_yol

    arama_dizinleri: List[Path] = [Path.cwd(), SCRIPT_DIR, REPO_ROOT, REPO_ROOT / "Scripts"]
    if ek_dizinler:
        arama_dizinleri.extend(ek_dizinler)

    for dizin in arama_dizinleri:
        test_yol = dizin / aday_yol.name
        if test_yol.is_file():
            return test_yol
        test_bagil = dizin / aday_yol
        if test_bagil.is_file():
            return test_bagil

    return SCRIPT_DIR / aday_yol.name


def csv_oku(yol: Path) -> List[Dict[str, str]]:
    """UTF-8 / UTF-8-SIG destekli güvenli CSV okuyucu."""
    if _ap_csv_oku is not None:
        try:
            return _ap_csv_oku(yol)
        except Exception:
            pass

    with yol.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def csv_yaz(satirlar: List[Dict[str, Any]], yol: Path) -> None:
    """UTF-8-SIG kodlamasıyla CSV dosyası yazar."""
    yol.parent.mkdir(parents=True, exist_ok=True)
    if not satirlar:
        yol.write_text("", encoding="utf-8-sig")
        return

    alanlar: List[str] = []
    gorulen = set()
    for satir in satirlar:
        for anahtar in satir.keys():
            if anahtar not in gorulen:
                gorulen.add(anahtar)
                alanlar.append(anahtar)

    with yol.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=alanlar, extrasaction="ignore")
        writer.writeheader()
        for satir in satirlar:
            temiz = {
                k: ("" if v is None else v)
                for k, v in satir.items()
            }
            writer.writerow(temiz)


def sayi_al(row: Optional[Dict[str, Any]], *anahtarlar: str, varsayilan: Optional[float] = None) -> Optional[float]:
    """Sözlükten verilen anahtarlardan ilk geçerli float değeri döndürür."""
    if not row:
        return varsayilan
    for k in anahtarlar:
        val = row.get(k)
        if val is None:
            continue
        if isinstance(val, (int, float)):
            if math.isfinite(val):
                return float(val)
            continue
        metin = str(val).strip()
        if not metin or metin.lower() in ("none", "nan", "null", "-"):
            continue
        try:
            sayi = float(metin)
            if math.isfinite(sayi):
                return sayi
        except ValueError:
            continue
    return varsayilan


def bool_al(row: Optional[Dict[str, Any]], *anahtarlar: str, varsayilan: bool = False) -> bool:
    """Sözlükten boolean değer çözümler."""
    if not row:
        return varsayilan
    for k in anahtarlar:
        val = row.get(k)
        if val is None:
            continue
        if isinstance(val, bool):
            return val
        metin = str(val).strip().lower()
        if metin in ("true", "1", "evet", "yes", "uygun", "basarili"):
            return True
        if metin in ("false", "0", "hayir", "no"):
            return False
    return varsayilan


def metin_al(row: Optional[Dict[str, Any]], *anahtarlar: str, varsayilan: str = "") -> str:
    """Sözlükten ilk dolu metin değerini döndürür."""
    if not row:
        return varsayilan
    for k in anahtarlar:
        val = row.get(k)
        if val is not None and str(val).strip() != "":
            return str(val).strip()
    return varsayilan


def _normalize_pervane(pervane: str) -> str:
    return pervane.strip().lower()


def kimlikleri_hazirla(satirlar: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Satırlarda temel_aday_id ve konfigurasyon_id alanlarının bulunmasını garanti eder."""
    if not satirlar:
        return []
    ornek = satirlar[0]
    if "temel_aday_id" in ornek and "konfigurasyon_id" in ornek:
        return satirlar
    if performans_adaylarini_hazirla is not None:
        try:
            return performans_adaylarini_hazirla(satirlar)
        except Exception:
            pass
    return satirlar


def indeks_olustur(
    satirlar: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, str], Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Satırları üç seviyeli arama tablosuna indeksler:
    1) konfigurasyon_id -> satır
    2) (temel_aday_id, pervane_norm) -> satır
    3) temel_aday_id -> satır (pervaneden bağımsız modüller için, örn. iniş)
    """
    hazir = kimlikleri_hazirla(satirlar)
    konfig_haritasi: Dict[str, Dict[str, Any]] = {}
    aday_pervane_haritasi: Dict[Tuple[str, str], Dict[str, Any]] = {}
    temel_aday_haritasi: Dict[str, Dict[str, Any]] = {}

    for row in hazir:
        konf_id = metin_al(row, "konfigurasyon_id")
        temel_id = metin_al(row, "temel_aday_id")
        if not temel_id and temel_kimlik is not None:
            try:
                temel_id = temel_kimlik(row)
            except Exception:
                temel_id = ""
        pervane = _normalize_pervane(metin_al(row, "pervane"))

        if konf_id:
            konfig_haritasi[konf_id] = row
        if temel_id and pervane:
            aday_pervane_haritasi[(temel_id, pervane)] = row
        if temel_id and temel_id not in temel_aday_haritasi:
            temel_aday_haritasi[temel_id] = row

    return konfig_haritasi, aday_pervane_haritasi, temel_aday_haritasi


def eslesen_satir_bul(
    konf_id: str,
    temel_id: str,
    pervane: str,
    indeks: Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, str], Dict[str, Any]], Dict[str, Dict[str, Any]]],
    pervaneden_bagimsiz_izin_ver: bool = False,
) -> Optional[Dict[str, Any]]:
    """Konfigürasyon veya (temel_aday_id, pervane) üzerinden eşleşen satırı döndürür."""
    konfig_haritasi, aday_pervane_haritasi, temel_aday_haritasi = indeks
    if konf_id and konf_id in konfig_haritasi:
        return konfig_haritasi[konf_id]
    perv_norm = _normalize_pervane(pervane)
    if temel_id and perv_norm and (temel_id, perv_norm) in aday_pervane_haritasi:
        return aday_pervane_haritasi[(temel_id, perv_norm)]
    if pervaneden_bagimsiz_izin_ver and temel_id and temel_id in temel_aday_haritasi:
        return temel_aday_haritasi[temel_id]
    return None


def adaylari_degerlendir_ve_sirala(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """5 performans modülünün CSV çıktılarını birleştirir, puanlar ve sıralar."""
    seyir_yolu = dosya_yolu_coz(args.seyir)
    kalkis_yolu = dosya_yolu_coz(args.kalkis)
    tirmanis_yolu = dosya_yolu_coz(args.tirmanis)
    yuk_birakma_yolu = dosya_yolu_coz(args.yuk_birakma)
    inis_yolu = dosya_yolu_coz(args.inis)

    eksik_dosyalar = [
        str(p) for p in (seyir_yolu, kalkis_yolu, tirmanis_yolu, yuk_birakma_yolu, inis_yolu)
        if not p.is_file()
    ]
    if eksik_dosyalar:
        raise FileNotFoundError(
            "Aşağıdaki performans çıktı dosyaları bulunamadı:\n  - " + "\n  - ".join(eksik_dosyalar)
        )

    seyir_satirlari = kimlikleri_hazirla(csv_oku(seyir_yolu))
    kalkis_indeks = indeks_olustur(csv_oku(kalkis_yolu))
    tirmanis_indeks = indeks_olustur(csv_oku(tirmanis_yolu))
    yuk_indeks = indeks_olustur(csv_oku(yuk_birakma_yolu))
    inis_indeks = indeks_olustur(csv_oku(inis_yolu))

    # İsteğe bağlı temel kütle tablosu (ek geometri sütunları için)
    kutle_indeks = None
    kutle_yolu = dosya_yolu_coz(args.kutle)
    if kutle_yolu.is_file():
        kutle_indeks = indeks_olustur(csv_oku(kutle_yolu))

    birlestirilmis: List[Dict[str, Any]] = []

    for seyir in seyir_satirlari:
        konf_id = metin_al(seyir, "konfigurasyon_id")
        temel_id = metin_al(seyir, "temel_aday_id")
        if not temel_id and temel_kimlik is not None:
            try:
                temel_id = temel_kimlik(seyir)
            except Exception:
                temel_id = ""
        pervane = metin_al(seyir, "pervane")

        kalkis = eslesen_satir_bul(konf_id, temel_id, pervane, kalkis_indeks)
        tirmanis = eslesen_satir_bul(konf_id, temel_id, pervane, tirmanis_indeks)
        yuk = eslesen_satir_bul(konf_id, temel_id, pervane, yuk_indeks)
        inis = eslesen_satir_bul(konf_id, temel_id, pervane, inis_indeks, pervaneden_bagimsiz_izin_ver=True)
        kutle = (
            eslesen_satir_bul(konf_id, temel_id, pervane, kutle_indeks, pervaneden_bagimsiz_izin_ver=True)
            if kutle_indeks is not None
            else None
        )

        # 1. Geometri ve Kütle Bilgileri
        su_kg = sayi_al(seyir, "su_kutlesi_kg")
        if su_kg is None:
            su_kg = sayi_al(kutle, "su_kutlesi_kg", "govde_m_water_kg", varsayilan=0.0)

        toplam_kutle_kg = sayi_al(seyir, "toplam_kutle_kg")
        if toplam_kutle_kg is None:
            toplam_kutle_kg = sayi_al(kutle, "total_kutle_kg", "toplam_kutle_kg", varsayilan=0.0)

        bos_kutle_kg = sayi_al(seyir, "bos_kutle_kg")
        if bos_kutle_kg is None:
            bos_kutle_kg = sayi_al(kutle, "total_susuz_kutle_kg", "bos_kutle_kg", varsayilan=0.0)

        # 2. Performans Modüllerinden Çıkan Toplam Puanlar
        puan_kalkis_60 = sayi_al(
            kalkis,
            "resmi_kalkis_puani_60",
            varsayilan=sayi_al(seyir, "resmi_kalkis_puani_60", varsayilan=0.0),
        )
        puan_kalkis_m2_20 = sayi_al(kalkis, "puan_M2_yuklu_kalkis_20", "kural_puani_20", varsayilan=0.0)
        tasarim_kalkis_100 = sayi_al(kalkis, "tasarim_kalkis_puani_100", varsayilan=0.0)

        puan_yuklu_2tur = sayi_al(seyir, "puan_yuklu_2tur", varsayilan=0.0)
        puan_bos_1tur = sayi_al(seyir, "puan_bos_1tur", varsayilan=0.0)
        puan_navigasyon_140 = sayi_al(
            seyir,
            "puan_navigasyon",
            varsayilan=(puan_yuklu_2tur or 0.0) + (puan_bos_1tur or 0.0),
        )
        puan_enerji_60 = sayi_al(seyir, "puan_enerji_60", varsayilan=0.0)

        # Yük bırakma puanı (yuk_birakma_sonuclari.csv'deki 'puan' sütunu)
        puan_yuk_birakma_120 = sayi_al(yuk, "puan", varsayilan=0.0)
        ham_puan_yuk_birakma = sayi_al(yuk, "ham_puan", varsayilan=0.0)

        # İniş puanı (inis_sonuclari.csv'deki 'resmi_inis_puani_140' sütunu)
        puan_inis_140 = sayi_al(inis, "resmi_inis_puani_140", varsayilan=0.0)
        puan_inis_m2_60 = sayi_al(inis, "puan_M2_inis_60", varsayilan=0.0)

        # Modüllerden gelen toplam puanların doğrudan toplamı (60 + 140 + 60 + 120 + 140 = 520)
        toplam_performans_puani = round(
            (puan_kalkis_60 or 0.0)
            + (puan_navigasyon_140 or 0.0)
            + (puan_enerji_60 or 0.0)
            + (puan_yuk_birakma_120 or 0.0)
            + (puan_inis_140 or 0.0),
            4,
        )

        # Bilgi amaçlı: Sadece Mission 2 (400 puan üzerinden) karşılığı
        mission2_400_puani = round(
            (puan_kalkis_m2_20 or 0.0)
            + (puan_navigasyon_140 or 0.0)
            + (puan_enerji_60 or 0.0)
            + (puan_yuk_birakma_120 or 0.0)
            + (puan_inis_m2_60 or 0.0),
            4,
        )

        # 3. Süre, Enerji (Wh) ve Batarya (mAh) Bütçesi Birleştirmesi
        kalkis_sure_s = sayi_al(kalkis, "sure_M2_kriter_s")
        kalkis_mah = sayi_al(kalkis, "tuketim_M2_kriter_mAh")
        kalkis_wh = sayi_al(kalkis, "enerji_M2_kriter_Wh")
        kalkis_mesafe_m = sayi_al(kalkis, "S_G_M2_kriter_m")
        kalkis_olcumleri = {
            "sure_M2_kriter_s": kalkis_sure_s,
            "tuketim_M2_kriter_mAh": kalkis_mah,
            "enerji_M2_kriter_Wh": kalkis_wh,
            "S_G_M2_kriter_m": kalkis_mesafe_m,
        }
        eksik_kalkis_alanlari = [ad for ad, deger in kalkis_olcumleri.items() if deger is None]

        tirmanis_sure_s = sayi_al(tirmanis, "sure_s", varsayilan=sayi_al(seyir, "tirmanis_sure_sn", varsayilan=0.0))
        tirmanis_mah = sayi_al(tirmanis, "tuketim_mAh", varsayilan=sayi_al(seyir, "tirmanis_tuketim_mah", varsayilan=0.0))
        tirmanis_wh = sayi_al(tirmanis, "enerji_Wh", varsayilan=sayi_al(seyir, "tirmanis_enerji_Wh", varsayilan=0.0))

        seyir_nav_sure_s = sayi_al(seyir, "navigasyon_suresi_sn", varsayilan=0.0)
        seyir_mah = sayi_al(seyir, "seyir_tuketim_mah", varsayilan=0.0)
        seyir_wh = sayi_al(seyir, "seyir_enerji_Wh", varsayilan=0.0)

        dalis_sure_s = sayi_al(yuk, "dalis_sure_s", varsayilan=0.0)
        birakma_sure_s = sayi_al(yuk, "birakma_sure_s", varsayilan=0.0)
        yuk_tirmanis_sure_s = sayi_al(yuk, "tirmanis_sure_s", varsayilan=0.0)
        yuk_toplam_sure_s = sayi_al(
            yuk,
            "sure_s",
            varsayilan=(dalis_sure_s or 0.0) + (birakma_sure_s or 0.0) + (yuk_tirmanis_sure_s or 0.0),
        )
        yuk_durum = metin_al(yuk, "durum", varsayilan="eksik")
        yuk_durum_duz = yuk_durum.lower()
        yuk_durum_kabul = yuk_durum_duz in ("uygun", "model_kapsaminda_hesaplandi", "kosullu_tahmin")
        yuk_kosullu = yuk_durum_duz == "kosullu_tahmin"
        yuk_mah = sayi_al(yuk, "tuketim_mAh")
        yuk_wh = sayi_al(yuk, "enerji_Wh")
        # Eski "uygun" kayıtlarında tamlık bayrağı yoksa iki toplamın varlığına bakılır.
        eski_tamlik = yuk_durum_duz == "uygun" and yuk is not None and "tuketim_tam" not in yuk
        yuk_enerji_tam = (
            yuk_durum_kabul
            and bool_al(yuk, "tuketim_tam", varsayilan=eski_tamlik)
            and yuk_mah is not None and yuk_mah >= 0.0
            and yuk_wh is not None and yuk_wh >= 0.0
        )
        if not yuk_enerji_tam:
            yuk_mah = yuk_wh = None

        inis_sure_s = sayi_al(inis, "sure_inis_kriter_s", "sure_inis_ort_ivme_s", varsayilan=0.0)
        inis_mesafe_m = sayi_al(inis, "S_G_inis_kriter_m", "S_G_inis_ort_ivme_m")

        gercek_toplam_sure_sn = None
        if kalkis_sure_s is not None:
            gercek_toplam_sure_sn = round(
                kalkis_sure_s + tirmanis_sure_s + seyir_nav_sure_s + yuk_toplam_sure_s + inis_sure_s,
                3,
            )

        gercek_toplam_tuketim_mah = None
        kalan_batarya_yuzde = None
        if kalkis_mah is not None and yuk_mah is not None:
            gercek_toplam_tuketim_mah = round(
                kalkis_mah + tirmanis_mah + seyir_mah + yuk_mah + args.inis_ek_butce_mah,
                3,
            )
            kalan_batarya_yuzde = round(
                100.0 * (BATARYA_KAPASITE_MAH - gercek_toplam_tuketim_mah) / BATARYA_KAPASITE_MAH,
                2,
            )

        gercek_toplam_enerji_wh = None
        if kalkis_wh is not None and yuk_wh is not None:
            gercek_toplam_enerji_wh = round(
                kalkis_wh + tirmanis_wh + seyir_wh + yuk_wh
                + args.inis_ek_butce_mah * GERILIM_V / 1000.0,
                4,
            )

        # 4. Faz Bazlı Fizibilite (Uygunluk) Kontrolleri
        uygunsuzluklar: List[str] = []

        kalkis_durum = metin_al(kalkis, "durum", varsayilan="uygun" if puan_kalkis_60 and puan_kalkis_60 > 0 else "eksik")
        kalkis_20m = bool_al(kalkis, "kalkis_20m_basarili", "kutu_20m_icinde")
        kalkis_uygun = (kalkis is not None) and (kalkis_durum.lower() in ("uygun", "ok", "basarili") or (puan_kalkis_60 or 0.0) > 0.0)
        if eksik_kalkis_alanlari:
            kalkis_uygun = False
            uygunsuzluklar.append("kalkis_M2_verisi_eksik:" + ",".join(eksik_kalkis_alanlari))
        elif not kalkis_uygun:
            uygunsuzluklar.append(f"kalkis:{kalkis_durum}")
        elif args.kalkis_20m_zorunlu and not kalkis_20m:
            kalkis_uygun = False
            uygunsuzluklar.append("kalkis_20m_asimi")

        tirmanis_durum = metin_al(tirmanis, "durum", varsayilan="uygun" if bool_al(seyir, "tirmanis_uygun") else "eksik")
        tirmanis_uygun = (tirmanis is not None and tirmanis_durum.lower() == "uygun") or bool_al(seyir, "tirmanis_uygun", varsayilan=False)
        if not tirmanis_uygun:
            uygunsuzluklar.append(f"tirmanis:{tirmanis_durum}")

        seyir_feasible = bool_al(seyir, "feasible", varsayilan=False)
        seyir_neden = metin_al(seyir, "uygunsuzluk_nedenleri")
        if not seyir_feasible:
            uygunsuzluklar.append(f"seyir:{seyir_neden or 'uygunsuz'}")

        yuk_uygun = yuk_durum_kabul and yuk_enerji_tam
        if not yuk_durum_kabul:
            uygunsuzluklar.append(f"yuk_birakma:{yuk_durum}")
        elif not yuk_enerji_tam:
            uygunsuzluklar.append("yuk_birakma:enerji_verisi_eksik")

        inis_durum = metin_al(inis, "durum", varsayilan="eksik")
        inis_20m = bool_al(inis, "inis_20m_basarili", varsayilan=False)
        if args.inis_20m_zorunlu:
            inis_uygun = (inis is not None) and (inis_durum.lower() == "uygun") and inis_20m
        else:
            # 20m aşımı olan uçaklar sert kuralları (10kg, 4m, 40kts) ihlal etmiyorsa 40 puan alır
            inis_uygun = (inis is not None) and (inis_durum.lower() in ("uygun", "mesafe_asimi_20m"))
        if not inis_uygun:
            uygunsuzluklar.append(f"inis:{inis_durum}")

        enerji_butcesi_durum = "eksik"
        if gercek_toplam_tuketim_mah is not None and gercek_toplam_enerji_wh is not None:
            enerji_butcesi_durum = "kosullu_tahmin" if yuk_kosullu else "hesaplandi"

        batarya_uygun = False
        if gercek_toplam_tuketim_mah is None:
            batarya_durum = "belirsiz"
        elif gercek_toplam_tuketim_mah > KULLANILABILIR_MAH:
            batarya_durum = "uygun_degil"
            uygunsuzluklar.append("batarya_9600mAh_asimi")
        else:
            batarya_durum = "kosullu_uygun" if yuk_kosullu else "uygun"
            batarya_uygun = True

        sure_uygun = gercek_toplam_sure_sn is not None and gercek_toplam_sure_sn <= MAX_GOREV_SURESI_SN
        if gercek_toplam_sure_sn is not None and not sure_uygun:
            uygunsuzluklar.append("gorev_suresi_600s_asimi")

        tum_fazlar_uygun = (
            kalkis_uygun
            and tirmanis_uygun
            and seyir_feasible
            and yuk_uygun
            and inis_uygun
            and batarya_uygun
            and sure_uygun
        )

        kayit: Dict[str, Any] = {
            "sira": 0,
            "temel_aday_id": temel_id,
            "konfigurasyon_id": konf_id,
            "pervane": pervane,
            "kombinasyon_no": metin_al(seyir, "kombinasyon_no"),
            "kanat_aday_no": metin_al(seyir, "kanat_aday_no", varsayilan=metin_al(kutle, "aday_no")),
            "govde_aday_no": metin_al(seyir, "govde_aday_no"),
            "kanat_airfoil": metin_al(seyir, "kanat_airfoil", varsayilan=metin_al(kutle, "kanat_airfoil")),
            "AR_w": sayi_al(seyir, "AR_w", varsayilan=sayi_al(kutle, "AR")),
            "b_w_m": sayi_al(seyir, "b_w_m", varsayilan=sayi_al(kutle, "b_w_m", "aciklik_m")),
            "S_w_m2": sayi_al(seyir, "S_w_m2", varsayilan=sayi_al(kutle, "S_w_m2", "alan_m2")),
            "c_w_m": sayi_al(seyir, "c_w_m", varsayilan=sayi_al(kutle, "c_w_m", "veter_m")),
            "govde_D_m": sayi_al(seyir, "govde_D_m", varsayilan=sayi_al(kutle, "govde_D_m")),
            "govde_L_m": sayi_al(seyir, "govde_L_m", varsayilan=sayi_al(kutle, "govde_L_total_m")),
            "su_kutlesi_kg": su_kg,
            "bos_kutle_kg": bos_kutle_kg,
            "toplam_kutle_kg": toplam_kutle_kg,
            "su_kutle_orani": sayi_al(seyir, "su_kutle_orani"),
            # Fizibilite Özeti
            "tum_fazlar_uygun": tum_fazlar_uygun,
            "kalkis_uygun": kalkis_uygun,
            "tirmanis_uygun": tirmanis_uygun,
            "seyir_uygun": seyir_feasible,
            "yuk_birakma_uygun": yuk_uygun,
            "yuk_birakma_durum": yuk_durum,
            "inis_uygun": inis_uygun,
            "batarya_uygun": batarya_uygun,
            "batarya_durum": batarya_durum,
            "enerji_butcesi_durum": enerji_butcesi_durum,
            "sure_uygun": sure_uygun,
            "uygunsuzluk_ozeti": "; ".join(uygunsuzluklar),
            # Puan Sütunları (Modüllerden Çıkan Toplam Puanlar)
            "toplam_performans_puani": toplam_performans_puani,
            "resmi_kalkis_puani_60": puan_kalkis_60,
            "puan_navigasyon_140": puan_navigasyon_140,
            "puan_yuklu_2tur_120": puan_yuklu_2tur,
            "puan_bos_1tur_20": puan_bos_1tur,
            "puan_enerji_60": puan_enerji_60,
            "puan_yuk_birakma_120": puan_yuk_birakma_120,
            "ham_puan_yuk_birakma": ham_puan_yuk_birakma,
            "resmi_inis_puani_140": puan_inis_140,
            # Ek Puan Karşılaştırmaları
            "mission2_400_puani": mission2_400_puani,
            "puan_M2_yuklu_kalkis_20": puan_kalkis_m2_20,
            "puan_M2_inis_60": puan_inis_m2_60,
            "tasarim_kalkis_puani_100": tasarim_kalkis_100,
            # Performans, Süre ve Enerji Metrikleri
            "kalkis_mesafe_m": kalkis_mesafe_m,
            "kalkis_20m_icinde": kalkis_20m,
            "inis_mesafe_m": inis_mesafe_m,
            "inis_20m_icinde": inis_20m,
            "kalkis_sure_s": kalkis_sure_s,
            "tirmanis_sure_s": tirmanis_sure_s,
            "seyir_nav_sure_s": seyir_nav_sure_s,
            "yuk_birakma_sure_s": yuk_toplam_sure_s,
            "inis_sure_s": inis_sure_s,
            "gercek_toplam_sure_sn": gercek_toplam_sure_sn,
            "kalkis_tuketim_mAh": kalkis_mah,
            "tirmanis_tuketim_mAh": tirmanis_mah,
            "seyir_tuketim_mAh": seyir_mah,
            "yuk_birakma_tuketim_mAh": yuk_mah,
            "gercek_toplam_tuketim_mAh": gercek_toplam_tuketim_mah,
            "kalan_batarya_yuzde": kalan_batarya_yuzde,
            "gercek_toplam_enerji_Wh": gercek_toplam_enerji_wh,
            "CD_0": sayi_al(seyir, "CD_0"),
            "e_oswald": sayi_al(seyir, "e_oswald"),
            "LD_yuklu_duz": sayi_al(seyir, "LD_yuklu_duz"),
            "LD_bos_duz": sayi_al(seyir, "LD_bos_duz"),
        }
        birlestirilmis.append(kayit)

    # Sıralama:
    # 1) Tüm fazları uygun olanlar önce (tum_fazlar_uygun == True)
    # 2) Seçilen sıralama ölçütüne göre toplam puan (azalan)
    # 3) Eşitlik bozucu: tasarım kalkış puanı (azalan), toplam enerji Wh (artan), toplam kütle kg (artan)
    puan_anahtari = "mission2_400_puani" if args.siralama_puani == "mission2" else "toplam_performans_puani"

    birlestirilmis.sort(
        key=lambda r: (
            0 if r["tum_fazlar_uygun"] else 1,
            -(r[puan_anahtari] or 0.0),
            -(r["tasarim_kalkis_puani_100"] or 0.0),
            r["gercek_toplam_enerji_Wh"] if r["gercek_toplam_enerji_Wh"] is not None else float("inf"),
            r["toplam_kutle_kg"] if r["toplam_kutle_kg"] is not None else float("inf"),
        )
    )

    for idx, satir in enumerate(birlestirilmis, start=1):
        satir["sira"] = idx

    # VLM için İlk N Adayın Seçimi
    aday_havuzu = (
        [r for r in birlestirilmis if r["tum_fazlar_uygun"]]
        if args.sadece_uygun
        else birlestirilmis
    )

    ilk_n_listesi: List[Dict[str, Any]] = []
    gorulen_geometriler = set()

    for satir in aday_havuzu:
        if args.vlm_tekil_geometri:
            geom_id = satir["temel_aday_id"]
            if geom_id and geom_id in gorulen_geometriler:
                continue
            if geom_id:
                gorulen_geometriler.add(geom_id)

        kopya = dict(satir)
        kopya["vlm_sira"] = len(ilk_n_listesi) + 1
        ilk_n_listesi.append(kopya)
        if len(ilk_n_listesi) >= args.ilk_n:
            break

    return birlestirilmis, ilk_n_listesi


def argumanlari_coz(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tüm performans modüllerinin çıktılarını birleştirip aday uçakları puanlar ve sıralar."
    )
    parser.add_argument(
        "--seyir",
        default="seyir_optimizasyon_sonuclari.csv",
        help="Seyir optimizasyon sonuçları CSV dosyası (varsayılan: seyir_optimizasyon_sonuclari.csv)",
    )
    parser.add_argument(
        "--kalkis",
        default="kalkis_sonuclari.csv",
        help="Kalkış sonuçları CSV dosyası (varsayılan: kalkis_sonuclari.csv)",
    )
    parser.add_argument(
        "--tirmanis",
        default="tirmanis_sonuclari.csv",
        help="Tırmanış sonuçları CSV dosyası (varsayılan: tirmanis_sonuclari.csv)",
    )
    parser.add_argument(
        "--yuk-birakma",
        dest="yuk_birakma",
        default="yuk_birakma_sonuclari.csv",
        help="Yük bırakma sonuçları CSV dosyası (varsayılan: yuk_birakma_sonuclari.csv)",
    )
    parser.add_argument(
        "--inis",
        default="inis_sonuclari.csv",
        help="İniş sonuçları CSV dosyası (varsayılan: inis_sonuclari.csv)",
    )
    parser.add_argument(
        "--kutle",
        default="kanat_kuyruk_govde_kutleli.csv",
        help="Temel kütle ve geometri CSV dosyası (varsayılan: kanat_kuyruk_govde_kutleli.csv)",
    )
    parser.add_argument(
        "--cikti",
        default="tum_adaylar_siralama.csv",
        help="Tüm sıralanmış adayların yazılacağı CSV dosyası (varsayılan: tum_adaylar_siralama.csv)",
    )
    parser.add_argument(
        "--vlm-cikti",
        dest="vlm_cikti",
        default="vlm_ilk_500_aday.csv",
        help="VLM aşamasına girecek ilk N adayın yazılacağı CSV dosyası (varsayılan: vlm_ilk_500_aday.csv)",
    )
    parser.add_argument(
        "--ilk-n",
        dest="ilk_n",
        type=int,
        default=500,
        help="VLM için seçilecek aday sayısı (varsayılan: 500)",
    )
    parser.add_argument(
        "--siralama-puani",
        dest="siralama_puani",
        choices=("modul_toplamlari", "mission2"),
        default="modul_toplamlari",
        help="Sıralamada baz alınacak toplam puan: 'modul_toplamlari' (520p) veya 'mission2' (400p)",
    )
    parser.add_argument(
        "--vlm-tekil-geometri",
        dest="vlm_tekil_geometri",
        action="store_true",
        default=False,
        help="Etkinleştirilirse VLM listesinde her temel_aday_id (uçak geometrisi) yalnızca en iyi pervanesiyle 1 kez yer alır.",
    )
    parser.add_argument(
        "--sadece-uygun",
        dest="sadece_uygun",
        action="store_true",
        default=False,
        help="Etkinleştirilirse VLM ilk N listesine yalnızca tüm fazları uygun (feasible) olan adaylar alınır.",
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
        help="Etkinleştirilirse 20m kutu içi duruşu zorunlu fizibilite kriteri yapar (varsayılan: False, 20m aşımı 40p alır).",
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
    tum_sirali, vlm_ilk_n = adaylari_degerlendir_ve_sirala(args)

    cikti_yolu = Path(args.cikti)
    if not cikti_yolu.is_absolute():
        cikti_yolu = SCRIPT_DIR / cikti_yolu

    vlm_cikti_yolu = Path(args.vlm_cikti)
    if not vlm_cikti_yolu.is_absolute():
        vlm_cikti_yolu = SCRIPT_DIR / vlm_cikti_yolu

    csv_yaz(tum_sirali, cikti_yolu)
    csv_yaz(vlm_ilk_n, vlm_cikti_yolu)

    uygun_sayisi = sum(1 for r in tum_sirali if r["tum_fazlar_uygun"])
    print(f"Toplam değerlendirilen konfigürasyon : {len(tum_sirali)}")
    print(f"Tüm fazları uygun (feasible) aday    : {uygun_sayisi}")
    print(f"Genel sıralama çıktısı               : {cikti_yolu}")
    print(f"VLM ilk {len(vlm_ilk_n)} aday çıktısı              : {vlm_cikti_yolu}")

    if tum_sirali:
        en_iyi = tum_sirali[0]
        print(
            "\nEn İyi 1. Aday Özeti:\n"
            f"  konfigurasyon_id        : {en_iyi['konfigurasyon_id']}\n"
            f"  kanat_airfoil / pervane : {en_iyi['kanat_airfoil']} / {en_iyi['pervane']}\n"
            f"  toplam_performans_puani : {en_iyi['toplam_performans_puani']} "
            f"(Kalkış: {en_iyi['resmi_kalkis_puani_60']}, Nav: {en_iyi['puan_navigasyon_140']}, "
            f"Enerji: {en_iyi['puan_enerji_60']}, Yük: {en_iyi['puan_yuk_birakma_120']}, "
            f"İniş: {en_iyi['resmi_inis_puani_140']})\n"
            f"  mission2_400_puani      : {en_iyi['mission2_400_puani']}\n"
            f"  toplam_tuketim / süre   : {en_iyi['gercek_toplam_tuketim_mAh']} mAh "
            f"(Kalan: %{en_iyi['kalan_batarya_yuzde']}) / {en_iyi['gercek_toplam_sure_sn']} s" +
            f"\n  batarya_durum / enerji  : {en_iyi['batarya_durum']} / {en_iyi['enerji_butcesi_durum']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
