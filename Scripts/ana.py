from pathlib import Path

from kanat_boyutlandirma import tum_adaylari_uret, csv_kaydet
from agirlik_tahmini import agirlik_ekle


if __name__ == "__main__":
    adaylar = tum_adaylari_uret(
        ar_ht=4.0,
        ar_vt=2.0,
    )

    kutleli_adaylar = [agirlik_ekle(aday) for aday in adaylar]

    cikti = Path(__file__).resolve().parent / "kanat_kuyruk_kutleli.csv"
    csv_kaydet(kutleli_adaylar, cikti)

    print(f"{len(kutleli_adaylar)} aday kaydedildi.")
    print(f"Dosya: {cikti}")