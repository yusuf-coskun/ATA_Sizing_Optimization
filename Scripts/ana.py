"""Kanat, kuyruk ve govde dahil kutle hesabi; kutleler gram ve kilogram.

Bu dosyayi kanat_boyutlandirma.py, agirlik_tahmini.py,
govde_boyutlandirma.py ve govde_agirlik_tahmini.py ile ayni klasore koyun.
Mevcut airfoils klasorunu koruyun.

Varsayim: govde ve kanat/kuyruk modullerindeki servo ve uretim sarfi
kalemleri ayri bilesenlere aittir; iki modulun toplamlari aynen eklenir.
"""

from pathlib import Path

import govde_boyutlandirma as govde_geometrisi
from agirlik_tahmini import agirlik_ekle
from govde_agirlik_tahmini import geometriden_agirlik_hesapla
from kanat_boyutlandirma import csv_kaydet, tum_adaylari_uret


def main():
    govdeler = geometriden_agirlik_hesapla(govde_geometrisi)
    toplam_adaylar = []

    for govde in govdeler:
        # Her govde icin ayri uretim: ayni uzunluktaki farkli govdeler
        # birbirine karismaz; her satir govde_aday_no ile izlenebilir.
        adaylar = tum_adaylari_uret(
            govde_uzunluklari_m=[govde["L_toplam_m"]],
            ar_ht=4.0,
            ar_vt=2.0,
        )

        for aday in adaylar:
            sonuc = agirlik_ekle(aday)
            # Govde alanlarini ayri adlarla ekle; iskelet listesi CSV'ye girmez.
            sonuc.update({
                f"govde_{ad}": deger
                for ad, deger in govde.items()
                if ad != "iskeletler"
            })

            # agirlik_toplam_g: mevcut kanat/kuyruk hesabi ve sabitleri.
            # govde toplam_g: su, batarya ve aviyonik dahil govde hesabi.
            sonuc["total_kutle_g"] = (
                sonuc["agirlik_toplam_g"] + govde["toplam_g"]
            )
            sonuc["total_kutle_kg"] = sonuc["total_kutle_g"] / 1000.0
            sonuc["total_susuz_kutle_g"] = (
                sonuc["agirlik_toplam_g"] + govde["susuz_toplam_g"]
            )
            sonuc["total_susuz_kutle_kg"] = (
                sonuc["total_susuz_kutle_g"] / 1000.0
            )
            toplam_adaylar.append(sonuc)

    cikti = Path(__file__).resolve().parent / "kanat_kuyruk_govde_kutleli.csv"
    csv_kaydet(toplam_adaylar, cikti)

    kutleler = [aday["total_kutle_kg"] for aday in toplam_adaylar]
    print(f"{len(toplam_adaylar)} aday kaydedildi.")
    print(f"Su dahil en dusuk toplam kutle: {min(kutleler):.3f} kg")
    print(f"Su dahil en yuksek toplam kutle: {max(kutleler):.3f} kg")
    print(f"Dosya: {cikti}")

if __name__ == "__main__":
    main()
