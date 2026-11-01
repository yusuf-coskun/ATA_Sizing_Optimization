"""Pervane olcumlerinden 0..20 m/s itki/akim tablosu ve SQLite veritabani.

Yalnizca Python standart kutuphanesini kullanir. Kalkis hesabi yapmaz.
Veritabaninda itki N, hiz m/s, akim A, gaz yuzde olarak saklanir.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
import json
from math import isfinite
from pathlib import Path
import sqlite3

KLASOR = Path(__file__).resolve().parent
VARSAYILAN_DB = KLASOR / "itki.sqlite3"
ITKI_CARPANLARI = {"N": 1.0, "gf": 0.00980665, "kgf": 9.80665}
ALANLAR = (
    "pervane", "gaz_yuzde", "hiz_m_s", "itki_n", "akim_a",
    "itki_yontemi", "akim_yontemi",
)


def _sayi(deger, ad, *, minimum=None, maksimum=None):
    if isinstance(deger, bool):
        raise ValueError(f"{ad}: sayi gerekli.")
    try:
        sonuc = float(deger)
    except (TypeError, ValueError):
        raise ValueError(f"{ad}: sayisal deger gerekli; bos alanlari doldurun.") from None
    if not isfinite(sonuc):
        raise ValueError(f"{ad}: sonlu sayi gerekli.")
    if minimum is not None and sonuc < minimum:
        raise ValueError(f"{ad}: en az {minimum} olmali.")
    if maksimum is not None and sonuc > maksimum:
        raise ValueError(f"{ad}: en fazla {maksimum} olmali.")
    return sonuc


def _hazirla(veri):
    """Girdiyi dogrula; olcum araliginda, 0..20 icinde 1 m/s adimli satirlar uret."""
    gruplar, anahtarlar, model_istekleri = [], set(), []
    for pervane in veri["pervaneler"]:
        ad = pervane["ad"]
        if not isinstance(ad, str) or not ad.strip():
            raise ValueError("Pervane adi bos olamaz.")
        ad = ad.strip()
        birim = pervane["itki_birimi"]
        if birim not in ITKI_CARPANLARI:
            raise ValueError("itki_birimi N, gf veya kgf olmali.")
        if not pervane["gaz_seviyeleri"]:
            raise ValueError(f"{ad}: gaz seviyesi bulunamadi.")
        for seviye in pervane["gaz_seviyeleri"]:
            gaz = _sayi(seviye["gaz_yuzde"], "gaz_yuzde", minimum=0, maksimum=100)
            anahtar = (ad, gaz)
            if anahtar in anahtarlar:
                raise ValueError(f"Tekrarlanan pervane/gaz: {anahtar}")
            anahtarlar.add(anahtar)
            if "model" in seviye:
                if "olcumler" in seviye:
                    raise ValueError("Bir gaz seviyesinde model ve olcumler birlikte kullanilamaz.")
                model_istekleri.append((ad, gaz, birim, seviye))
                continue
            sabit = "sabit_akim_a" in seviye
            sabit_akim = (
                _sayi(seviye["sabit_akim_a"], "sabit_akim_a", minimum=0)
                if sabit else None
            )
            noktalar = []
            for olcum in seviye["olcumler"]:
                hiz = _sayi(olcum["hiz_m_s"], "hiz_m_s", minimum=0)
                itki = _sayi(olcum["itki"], "itki") * ITKI_CARPANLARI[birim]
                if sabit and "akim_a" in olcum:
                    raise ValueError("Sabit akim ile nokta bazli akimi birlikte girmeyin.")
                akim = sabit_akim if sabit else _sayi(olcum["akim_a"], "akim_a", minimum=0)
                noktalar.append((hiz, itki, akim))
            noktalar.sort()
            hizlar = [p[0] for p in noktalar]
            if len(noktalar) < 2 or len(set(hizlar)) != len(hizlar):
                raise ValueError(f"{anahtar}: en az iki farkli hiz gerekli; hizlar tekrarlanamaz.")
            if hizlar[0] > 0 or hizlar[-1] < 1:
                raise ValueError(f"{anahtar}: olcumler 0 m/s'den baslamali ve en az 1 m/s'ye ulasmali.")
            satirlar = []
            # Eksik yuksek hiz olcumlerinden ekstrapolasyon yapilmaz.
            for hiz in range(min(20, int(hizlar[-1])) + 1):
                i = bisect_left(hizlar, hiz)
                olculmus = i < len(hizlar) and hizlar[i] == hiz
                if olculmus:
                    _, itki, akim = noktalar[i]
                else:
                    v0, t0, a0 = noktalar[i - 1]
                    v1, t1, a1 = noktalar[i]
                    oran = (hiz - v0) / (v1 - v0)
                    itki = t0 + oran * (t1 - t0)
                    akim = a0 + oran * (a1 - a0)
                itki_yontemi = "olcum" if olculmus else "dogrusal_interpolasyon"
                akim_yontemi = "sabit_varsayim" if sabit else itki_yontemi
                satirlar.append((ad, gaz, hiz, itki, akim, itki_yontemi, akim_yontemi))
            gruplar.append((anahtar, noktalar, sabit, satirlar, None))
    # Referanslar yalniz bu JSON'daki olcum gruplarindan secilir.
    # Modelden modele zincirleme tahmin yapilmaz; JSON sirasi onemsizdir.
    olcum_gruplari = {grup[0]: grup for grup in gruplar}
    for ad, gaz, birim, seviye in model_istekleri:
        model = seviye["model"]
        if model["tur"] != "statik_oran":
            raise ValueError(f"Bilinmeyen model: {model['tur']}")
        referans_gaz = _sayi(model["referans_gaz_yuzde"], "referans_gaz_yuzde", minimum=0, maksimum=100)
        referans = olcum_gruplari.get((ad, referans_gaz))
        if referans is None or len(referans[3]) != 21:
            raise ValueError("Model referansi ayni JSON'da 0..20 m/s'yi kapsayan bir olcum grubu olmali.")
        referans_statik = referans[3][0][3]
        if referans_statik <= 0:
            raise ValueError("Referans statik itki pozitif olmali.")
        statik = _sayi(seviye["statik_itki"], "statik_itki", minimum=0) * ITKI_CARPANLARI[birim]
        akim = _sayi(seviye["sabit_akim_a"], "sabit_akim_a", minimum=0)
        satirlar = [
            (ad, gaz, s[2], statik if s[2] == 0 else statik * s[3] / referans_statik,
             akim, "olcum" if s[2] == 0 else "statik_oran_modeli", "sabit_varsayim")
            for s in referans[3]
        ]
        bilgi = {
            "tur": "statik_oran", "referans_gaz_yuzde": referans_gaz,
            "statik_itki_n": statik, "referans_statik_itki_n": referans_statik,
            "referans_egri": [{"hiz_m_s": s[2], "itki_n": s[3]} for s in referans[3]],
        }
        # Kaynak olcum tablosuna yalniz verilen statik nokta girer.
        gruplar.append(((ad, gaz), [(0, statik, akim)], True, satirlar, bilgi))
    if not gruplar:
        raise ValueError("En az bir pervane/gaz grubu gerekli.")
    return gruplar


def veritabani_olustur(girdi_json, veritabani=VARSAYILAN_DB):
    """Girilen pervane/gaz gruplarini ekler veya yeniler; digerlerini korur."""
    with Path(girdi_json).open(encoding="utf-8-sig") as dosya:
        gruplar = _hazirla(json.load(dosya))
    yol = Path(veritabani).resolve()
    yol.parent.mkdir(parents=True, exist_ok=True)
    baglanti = sqlite3.connect(yol)
    try:
        with baglanti:
            baglanti.execute("""CREATE TABLE IF NOT EXISTS olcumler (
                pervane TEXT NOT NULL, gaz_yuzde REAL NOT NULL,
                hiz_m_s REAL NOT NULL, itki_n REAL NOT NULL, akim_a REAL NOT NULL,
                akim_yontemi TEXT NOT NULL,
                PRIMARY KEY (pervane, gaz_yuzde, hiz_m_s))""")
            baglanti.execute("""CREATE TABLE IF NOT EXISTS itki_verileri (
                pervane TEXT NOT NULL, gaz_yuzde REAL NOT NULL,
                hiz_m_s INTEGER NOT NULL CHECK (hiz_m_s BETWEEN 0 AND 20),
                itki_n REAL NOT NULL, akim_a REAL NOT NULL,
                itki_yontemi TEXT NOT NULL, akim_yontemi TEXT NOT NULL,
                PRIMARY KEY (pervane, gaz_yuzde, hiz_m_s))""")
            baglanti.execute("""CREATE TABLE IF NOT EXISTS modeller (
                pervane TEXT NOT NULL, gaz_yuzde REAL NOT NULL,
                model_json TEXT NOT NULL,
                PRIMARY KEY (pervane, gaz_yuzde))""")
            for (ad, gaz), noktalar, sabit, satirlar, model in gruplar:
                baglanti.execute("DELETE FROM olcumler WHERE pervane=? AND gaz_yuzde=?", (ad, gaz))
                baglanti.execute("DELETE FROM itki_verileri WHERE pervane=? AND gaz_yuzde=?", (ad, gaz))
                baglanti.execute("DELETE FROM modeller WHERE pervane=? AND gaz_yuzde=?", (ad, gaz))
                if model is not None:
                    baglanti.execute("INSERT INTO modeller VALUES (?, ?, ?)",
                                     (ad, gaz, json.dumps(model, ensure_ascii=False)))
                baglanti.executemany(
                    "INSERT INTO olcumler VALUES (?, ?, ?, ?, ?, ?)",
                    [(ad, gaz, v, t, a, "sabit_varsayim" if sabit else "olcum")
                     for v, t, a in noktalar],
                )
                baglanti.executemany("INSERT INTO itki_verileri VALUES (?, ?, ?, ?, ?, ?, ?)", satirlar)
    finally:
        baglanti.close()
    return sum(len(grup[3]) for grup in gruplar)


def _baglan(veritabani):
    # Okuma sirasinda yanlis bir yolda bos veritabani olusturulmaz.
    yol = Path(veritabani).resolve()
    if not yol.is_file():
        raise FileNotFoundError(f"Veritabani bulunamadi: {yol}")
    baglanti = sqlite3.connect(yol.as_uri() + "?mode=ro", uri=True)
    baglanti.row_factory = sqlite3.Row
    return baglanti


def itki_getir(pervane, gaz_yuzde, hiz_m_s, *, veritabani=VARSAYILAN_DB):
    """Kayitli gaz seviyesinde, 0..20 arasindaki tam sayi hiz icin dict verir.

    Ornek: itki_getir('14x7', 100, 12)['itki_n']
    Gaz seviyeleri arasinda interpolasyon veya aralik disina ekstrapolasyon yoktur.
    """
    gaz = _sayi(gaz_yuzde, "gaz_yuzde", minimum=0, maksimum=100)
    hiz = _sayi(hiz_m_s, "hiz_m_s", minimum=0, maksimum=20)
    if not hiz.is_integer():
        raise ValueError("Bu tablo 1 m/s adimlidir; hiz tam sayi olmali.")
    baglanti = _baglan(veritabani)
    try:
        satir = baglanti.execute(
            "SELECT * FROM itki_verileri WHERE pervane=? AND gaz_yuzde=? AND hiz_m_s=?",
            (pervane, gaz, int(hiz)),
        ).fetchone()
    finally:
        baglanti.close()
    if satir is None:
        raise LookupError(
            f"Kayit yok: {pervane}, %{gaz:g} gaz, {hiz:g} m/s. "
            "Bu pervane/gaz/hiz icin kayitli olcum veya model sonucu bulunamadi."
        )
    return dict(satir)


def tablo_getir(*, veritabani=VARSAYILAN_DB):
    baglanti = _baglan(veritabani)
    try:
        return [dict(satir) for satir in baglanti.execute(
            "SELECT * FROM itki_verileri ORDER BY pervane, gaz_yuzde DESC, hiz_m_s"
        )]
    finally:
        baglanti.close()


def csv_kaydet(yol, *, veritabani=VARSAYILAN_DB, ondalik_ayirici="."):
    """Noktali virgul sutun, varsayilan olarak nokta ondalik ayirici.

    Yalniz CSV gosterimi en fazla 6 ondaliga yuvarlanir; SQLite degismez.
    Turkce Excel icin ondalik_ayirici=',' kullanilabilir.
    """
    if ondalik_ayirici not in (",", "."):
        raise ValueError("Ondalik ayirici virgul veya nokta olmali.")
    satirlar = tablo_getir(veritabani=veritabani)
    def bicimle(deger):
        if isinstance(deger, float):
            metin = f"{deger:.6f}".rstrip("0").rstrip(".")
            return metin.replace(".", ondalik_ayirici)
        return deger

    with Path(yol).open("w", encoding="utf-8-sig", newline="") as dosya:
        yazici = csv.DictWriter(dosya, fieldnames=ALANLAR, delimiter=";")
        yazici.writeheader()
        yazici.writerows({k: bicimle(v) for k, v in s.items()} for s in satirlar)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "girdi_json", type=Path, nargs="?",
        default=KLASOR / "itki_olcumleri.json",
        help="Olcum JSON'u; verilmezse bu Python dosyasinin yanindaki itki_olcumleri.json okunur",
    )
    parser.add_argument("--db", type=Path, default=VARSAYILAN_DB)
    parser.add_argument("--csv", type=Path, default=KLASOR / "itki_tablosu.csv")
    parser.add_argument("--ondalik", choices=(",", "."), default=".",
                        help="CSV ondalik ayirici; varsayilan nokta")
    args = parser.parse_args()
    # Girdi veya veritabani yanlislikla CSV ile ezilmesin.
    yollar = [p.resolve() for p in (args.girdi_json, args.db, args.csv)]
    if len(set(yollar)) != 3:
        parser.error("JSON, veritabani ve CSV icin farkli dosya yollari kullanin.")
    try:
        adet = veritabani_olustur(args.girdi_json, args.db)
        csv_kaydet(args.csv, veritabani=args.db, ondalik_ayirici=args.ondalik)
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error) as hata:
        parser.exit(1, f"Hata: {hata}\n")
    print(f"{adet} satir eklendi/guncellendi.")
    print("Pervane   Gaz (%)   Hiz (m/s)   Itki (N)   Akim (A)")
    for s in tablo_getir(veritabani=args.db):
        print(f"{s['pervane']:<10} {s['gaz_yuzde']:>6g} {s['hiz_m_s']:>11} "
              f"{s['itki_n']:>10.3f} {s['akim_a']:>10.3f}")
    print(f"Veritabani: {args.db.resolve()}")
    print(f"CSV: {args.csv.resolve()}")


if __name__ == "__main__":
    main()
