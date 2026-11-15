"""Tam ucak adaylarini uret, kutleleri hesapla ve pervanelerle bir kez esle.

Normal kullanim: python ana.py
Hazir kutle tablosuna atama:
    python ana.py --mevcut-kutle kanat_kuyruk_govde_kutleli.csv
Her iki durumda kalkis, tirmanis ve seyir ayni pervaneli CSV'yi kullanir.
"""
import argparse
from pathlib import Path

from aday_pervane import adaylari_pervanelerle_esle, csv_oku, csv_yaz, pervaneleri_oku

ROOT = Path(__file__).resolve().parent


def temel_adaylari_uret():
    import govde_boyutlandirma as govde_geometrisi
    from agirlik_tahmini import agirlik_ekle
    from govde_agirlik_tahmini import geometriden_agirlik_hesapla
    from kanat_boyutlandirma import tum_adaylari_uret

    toplam_adaylar = []
    for govde in geometriden_agirlik_hesapla(govde_geometrisi):
        adaylar = tum_adaylari_uret(
            govde_uzunluklari_m=[govde["L_toplam_m"]], ar_ht=4.0, ar_vt=4.0,
        )
        for aday in adaylar:
            sonuc = agirlik_ekle(aday)
            sonuc.update({f"govde_{ad}": deger for ad, deger in govde.items() if ad != "iskeletler"})
            sonuc["total_kutle_g"] = sonuc["agirlik_toplam_g"] + govde["toplam_g"]
            sonuc["total_kutle_kg"] = sonuc["total_kutle_g"] / 1000.0
            sonuc["total_susuz_kutle_g"] = sonuc["agirlik_toplam_g"] + govde["susuz_toplam_g"]
            sonuc["total_susuz_kutle_kg"] = sonuc["total_susuz_kutle_g"] / 1000.0
            toplam_adaylar.append(sonuc)
    return toplam_adaylar


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mevcut-kutle", type=Path, help="Geometri/kutleyi yeniden hesaplamadan bu CSV'ye pervane ata.")
    parser.add_argument("--itki", type=Path, default=ROOT / "itki_tablosu.csv")
    parser.add_argument("--cikti", type=Path, default=ROOT / "kanat_kuyruk_govde_kutleli.csv")
    args = parser.parse_args(argv)
    pervaneler = pervaneleri_oku(csv_oku(args.itki))
    temel_adaylar = csv_oku(args.mevcut_kutle) if args.mevcut_kutle else temel_adaylari_uret()
    adaylar = adaylari_pervanelerle_esle(temel_adaylar, pervaneler)
    csv_yaz(adaylar, args.cikti)
    kutleler = [float(aday["total_kutle_kg"]) for aday in adaylar]
    print(f"{len({a['temel_aday_id'] for a in adaylar})} temel ucak, {len(adaylar)} ucak-pervane konfigurasyonu kaydedildi.")
    print(f"Pervaneler: {', '.join(pervaneler)}")
    print(f"Su dahil toplam kutle araligi: {min(kutleler):.3f} .. {max(kutleler):.3f} kg")
    print(f"Dosya: {args.cikti.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())