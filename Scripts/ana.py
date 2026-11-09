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
from tirmanis_hesabi import tirmanis_enerjisi_hesapla


def main():
    govdeler = geometriden_agirlik_hesapla(govde_geometrisi)
    toplam_adaylar = []
    tirmanis_listesi = []

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

            enerji_wh, enerji_j, sure, guc_w, v_toplam = tirmanis_enerjisi_hesapla(
                m=sonuc["total_kutle_kg"], 
                v_yatay=15.0, 
                v_dikey=4.0, 
                h_hedef=50.0, 
                CD0=0.025, 
                K=0.04, 
                S=sonuc["S_w_m2"],
                rho=1.225, 
                eta_prop=0.70, 
                eta_motor=0.85,
                kaydet=False
            )
            
            tirmanis_listesi.append({
                "kanat_airfoil": sonuc.get("kanat_airfoil", ""),
                "kanat_acikligi_b_m": sonuc.get("b_w_m", 0),
                "total_kutle_kg": round(sonuc["total_kutle_kg"], 3),
                "tirmanis_suresi_s": round(sure, 2),
                "tirmanis_guc_w": round(guc_w, 2),
                "tirmanis_enerjisi_wh": round(enerji_wh, 2),
                "ucus_hizi_toplam_m_s": round(v_toplam, 2)
            })

            toplam_adaylar.append(sonuc)

    cikti = Path(__file__).resolve().parent / "kanat_kuyruk_govde_kutleli.csv"
    csv_kaydet(toplam_adaylar, cikti)

    cikti_tirmanis = Path(__file__).resolve().parent / "tirmanis_sonuclari.csv"
    csv_kaydet(tirmanis_listesi, cikti_tirmanis)

    kutleler = [aday["total_kutle_kg"] for aday in toplam_adaylar]
    print(f"{len(toplam_adaylar)} aday kaydedildi.")
    print(f"Su dahil en dusuk toplam kutle: {min(kutleler):.3f} kg")
    print(f"Su dahil en yuksek toplam kutle: {max(kutleler):.3f} kg")
    print(f"Dosya: {cikti}")
    print(f"Tirmanis Dosyasi: {cikti_tirmanis}")

if __name__ == "__main__":
    main()
