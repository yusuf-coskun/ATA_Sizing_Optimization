"""Ortak aday kimlikleri, pervane atamasi ve elektriksel guc birimleri."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path

GERILIM_V = 22.5
KIMLIK_ALANLARI = ("temel_aday_id", "konfigurasyon_id", "pervane")


def metin(value):
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in ("nan", "none", "<na>") else text


def csv_oku(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        first = stream.readline()
        stream.seek(0)
        rows = list(csv.DictReader(stream, delimiter=";" if first.count(";") > first.count(",") else ","))
    if not rows:
        raise ValueError(f"Bos CSV: {path}")
    return rows


def csv_yaz(rows, path):
    if not rows:
        raise ValueError("Kaydedilecek aday yok.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def temel_kimlik(aday):
    """Tam aday verisinden, satir sirasindan ve .dat yollarindan bagimsiz kimlik.

    CSV/pandas donusum farklari icin yalniz kimlikte 12 anlamli basamak
    kullanilir; fiziksel hesap degerleri yuvarlanmaz. Geometri ve kutleler
    dahil edilir, sira numaralari dahil edilmez.
    """
    canonical = {}
    for key, value in aday.items():
        if key in KIMLIK_ALANLARI or key.endswith(("_no", "_dat")):
            continue
        text = metin(value)
        if not text:
            continue
        if text.lower() in ("true", "false"):
            canonical[key] = text.lower()
            continue
        try:
            number = float(text)
        except ValueError:
            canonical[key] = text
        else:
            if not math.isfinite(number):
                raise ValueError(f"Sonlu olmayan aday verisi: {key}")
            canonical[key] = format(number, ".12g") if number else "0"
    if not canonical:
        raise ValueError("Temel aday verisi bos.")
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "A_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def kimlik_ekle(aday, pervane):
    prop = metin(pervane)
    if not prop:
        raise ValueError("Adayda pervane eksik. Once ana.py ile pervaneli adaylari uretin.")
    base = temel_kimlik(aday)
    previous = metin(aday.get("temel_aday_id"))
    if previous and previous != base:
        raise ValueError("Aday verisi kimligiyle uyusmuyor; adaylari yeniden uretin.")
    config = f"{base}__{prop}"
    previous = metin(aday.get("konfigurasyon_id"))
    if previous and previous != config:
        raise ValueError("Konfigurasyon kimligi ile pervane uyusmuyor.")
    return {**aday, "temel_aday_id": base, "konfigurasyon_id": config, "pervane": prop}


def pervaneleri_oku(rows):
    props = sorted({metin(row.get("pervane")) for row in rows})
    if not props or "" in props:
        raise ValueError("Itki tablosunda pervane adi eksik.")
    return props


def adaylari_pervanelerle_esle(adaylar, pervaneler):
    """Ham adaylari caprazlar; zaten atanmis girdiyi ikinci kez cogaltmaz."""
    props = sorted(set(map(metin, pervaneler)))
    if not props or "" in props:
        raise ValueError("En az bir gecerli pervane gerekli.")
    assigned = [bool(metin(row.get("pervane"))) for row in adaylar]
    if any(assigned) and not all(assigned):
        raise ValueError("Ham ve pervane atanmis adaylar ayni tabloda karistirilamaz.")
    output, seen = [], set()
    for row in adaylar:
        row_props = [metin(row.get("pervane"))] if all(assigned) else props
        for prop in row_props:
            if prop not in props:
                raise ValueError(f"Itki tablosunda pervane bulunamadi: {prop}")
            result = kimlik_ekle(row, prop)
            key = result["konfigurasyon_id"]
            if key in seen:
                raise ValueError(f"Tekrarlanan aday-pervane: {key}")
            seen.add(key)
            output.append(result)
    if not output:
        raise ValueError("Aday tablosu bos.")
    return output


def performans_adaylarini_hazirla(rows, varsayilan_pervane=None):
    """Fazlar caprazlamaz. Ham tablo ancak acik --pervane ile kullanilabilir."""
    output, seen = [], set()
    for row in rows:
        result = kimlik_ekle(row, metin(row.get("pervane")) or varsayilan_pervane)
        if result["konfigurasyon_id"] in seen:
            raise ValueError(f"Tekrarlanan konfigurasyon: {result['konfigurasyon_id']}")
        seen.add(result["konfigurasyon_id"])
        output.append(result)
    return output


def kimlik_alanlari(row):
    return {key: row[key] for key in KIMLIK_ALANLARI}


def faz_anahtari(row):
    """Sonuc satirinda yeniden hash hesaplanmaz; kayitli kimlik kullanilir."""
    base, config, prop = (metin(row.get(key)) for key in KIMLIK_ALANLARI)
    if not base or not prop or config != f"{base}__{prop}":
        raise ValueError("Faz kimligi eksik/tutarsiz. Kalkis ve tirmanis sonuclarini yeniden uretin.")
    return config


def elektrik_gucu(row, voltage, mode="auto"):
    """guc_w/power_w watt; akim_a amper. Watt degeri gerilimle carpilmaz."""
    if not math.isfinite(voltage) or voltage <= 0:
        raise ValueError("Gerilim sonlu ve pozitif olmali.")
    if mode not in ("auto", "guc_w", "akim_a"):
        raise ValueError(f"Gecersiz tablo degeri modu: {mode}")
    columns = ("guc_w", "power_w", "akim_a") if mode == "auto" else (
        ("guc_w", "power_w") if mode == "guc_w" else ("akim_a",))
    for column in columns:
        if metin(row.get(column)):
            power = float(row[column]) * (voltage if column == "akim_a" else 1.0)
            if not math.isfinite(power) or power < 0:
                raise ValueError(f"Gecersiz elektriksel guc: {column}")
            return power, "akim_a_x_gerilim" if column == "akim_a" else column
    raise ValueError(f"Guc verisi yok; beklenen sutunlar: {', '.join(columns)}")
