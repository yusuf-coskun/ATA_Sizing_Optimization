"""Gövde kütle hesabı: uzunluk m, alan m², kütle g.

Bu dosyayı govde_boyutlandirma.py ile aynı klasöre koyup çalıştırın:
    python govde_agirlik_tahmini.py

Mevcut birleşik kütle CSV’sini yerinde güncellemek için:
    python govde_agirlik_tahmini.py --kutle-csv kanat_kuyruk_govde_kutleli.csv
Ayrı çıktı için bu komuta --csv yeni_kutle.csv ekleyin.

Kaynak boyutlandırma dosyası ve parametreleri değiştirilmez.
Entegrasyon: sonuclar = geometriden_agirlik_hesapla(geometri_modulu)
Çıktılar gram cinsindedir; bataryalar dahil sabit aviyonik kütlesi eklenir.
"""
from __future__ import annotations

import argparse
import csv
import runpy
from collections.abc import Mapping
from math import floor, isfinite, pi
from pathlib import Path
from tempfile import NamedTemporaryFile

from aday_pervane import kimlik_ekle, temel_kimlik

DIS_KABUK_GSM = 395.9  # Kürlenmiş karbon/cam/karbon: 176.7 + 42.5 + 176.7.
ISKELET_GSM = 802.8
ISKELET_ARALIGI_M = 0.15
ILK_DISK_CAPI_M = 0.080
HALKA_RADYAL_GENISLIGI_M = 0.005
KUYRUK_UC_CAPI_M = 0.080  # Kaynak geometrinin D_min değeri.
BORU_DIS_CAPI_M = 0.003
BORU_IC_CAPI_M = 0.002
BORU_G_M = 6.3  # Kullanıcının verdiği ölçülmüş doğrusal kütle.
SU_YOGUNLUGU_KG_M3 = 1000.0

PERVANE_G = 84.0
ARKA_INIS_TAKIMI_G = 30.0
MOTOR_G = 460.0
SERVOLAR_G = 126.6
YAPISTIRICI_VE_DIGER_G = 50.0

BATARYA_ADEDI = 2
TEK_BATARYA_G = 885.0
IC_AVIYONIK_G = 650.0
KABLO_G = 75.0
SIGORTA_G = 10.0


def _sayi(deger, ad, *, sifir_olabilir=False):
    deger = float(deger)
    if not isfinite(deger) or deger < 0 or (deger == 0 and not sifir_olabilir):
        raise ValueError(f"{ad}: sonlu, {'negatif olmayan' if sifir_olabilir else 'pozitif'} sayı gerekli.")
    return deger


def govde_agirligi(*, D_m, V_su_m3, L_kuyruk_m, L_toplam_m,
                   L_aviyonik_m, L_su_m, S_toplam_m2):
    """Tek adayın geometrisinden ayrı bir sonuç sözlüğü üretir.

    İskeletler x=0, 0.15, 0.30, ... <= L_toplam konumlarında bulunur.
    Uç, 15 cm ızgarasına denk gelmiyorsa ilave uç halkası eklenmez.
    İlk eleman dolu 80 mm disk; diğerleri yerel dış çapta 5 mm geniş halkadır.
    Tek borunun boyu ikinci ve son iskelet arasındaki eksenel mesafedir.
    Verilen toplam dış alan (kaynak koddaki kapaklar dahil) aynen kullanılır.
    İskelet GSM değeri tamamlanmış serim içindir; alan bir kez çarpılır.
    """
    D = _sayi(D_m, "D_m")
    V = _sayi(V_su_m3, "V_su_m3", sifir_olabilir=True)
    Lt = _sayi(L_kuyruk_m, "L_kuyruk_m")
    L = _sayi(L_toplam_m, "L_toplam_m")
    La = _sayi(L_aviyonik_m, "L_aviyonik_m", sifir_olabilir=True)
    Ls = _sayi(L_su_m, "L_su_m", sifir_olabilir=True)
    S = _sayi(S_toplam_m2, "S_toplam_m2")
    if abs(L - (La + Ls + Lt)) > 1e-9 * max(1.0, L):
        raise ValueError("Toplam uzunluk bölüm uzunluklarıyla uyuşmuyor.")
    if min(D, KUYRUK_UC_CAPI_M) <= 2 * HALKA_RADYAL_GENISLIGI_M:
        raise ValueError("Gövde çapı 10 mm'den büyük olmalı.")

    son_indeks = floor((L + 1e-12) / ISKELET_ARALIGI_M)
    konumlar = [min(i * ISKELET_ARALIGI_M, L) for i in range(son_indeks + 1)]
    silindir_sonu = La + Ls
    elemanlar = []
    for i, x in enumerate(konumlar):
        oran = max(0.0, min(1.0, (x - silindir_sonu) / Lt))
        yerel_cap = D + (KUYRUK_UC_CAPI_M - D) * oran
        dis_cap = ILK_DISK_CAPI_M if i == 0 else yerel_cap
        ic_cap = 0.0 if i == 0 else dis_cap - 2 * HALKA_RADYAL_GENISLIGI_M
        alan = pi / 4 * (dis_cap**2 - ic_cap**2)
        elemanlar.append({
            "no": i + 1, "konum_m": x,
            "tur": "dolu_disk" if i == 0 else "halka",
            "govde_dis_capi_m": yerel_cap,
            "eleman_dis_capi_m": dis_cap, "eleman_ic_capi_m": ic_cap,
            "alan_m2": alan, "kutle_g": alan * ISKELET_GSM,
        })

    boru_boyu = konumlar[-1] - konumlar[1] if len(konumlar) >= 2 else 0.0
    kabuk = S * DIS_KABUK_GSM
    iskelet = sum(e["kutle_g"] for e in elemanlar)
    boru = boru_boyu * BORU_G_M
    su = V * SU_YOGUNLUGU_KG_M3 * 1000.0
    bataryalar = BATARYA_ADEDI * TEK_BATARYA_G
    aviyonik = bataryalar + IC_AVIYONIK_G + KABLO_G + SIGORTA_G
    # Aviyonik alt kalemleri bu toplam üzerinden yalnızca bir kez eklenir.
    sabit = (PERVANE_G + ARKA_INIS_TAKIMI_G + MOTOR_G + SERVOLAR_G
             + YAPISTIRICI_VE_DIGER_G + aviyonik)
    susuz = kabuk + iskelet + boru + sabit
    return {
        "D_m": D, "V_su_m3": V, "L_toplam_m": L, "S_toplam_m2": S,
        "iskelet_sayisi": len(elemanlar), "iskeletler": elemanlar,
        "iskelet_alani_m2": sum(e["alan_m2"] for e in elemanlar),
        "boru_uzunlugu_m": boru_boyu,
        "dis_kabuk_g": kabuk, "iskelet_g": iskelet, "karbon_boru_g": boru,
        "su_g": su, "pervane_g": PERVANE_G,
        "arka_inis_takimi_g": ARKA_INIS_TAKIMI_G, "motor_g": MOTOR_G,
        "servolar_g": SERVOLAR_G, "yapistirici_ve_diger_g": YAPISTIRICI_VE_DIGER_G,
        "bataryalar_g": bataryalar, "ic_aviyonik_g": IC_AVIYONIK_G,
        "kablo_g": KABLO_G, "sigorta_g": SIGORTA_G,
        "aviyonik_toplam_g": aviyonik,
        "sabit_toplam_g": sabit, "aviyonik_dahil": True,
        "susuz_toplam_g": susuz, "toplam_g": susuz + su,
        "toplam_kg": (susuz + su) / 1000.0,
    }


def geometriden_agirlik_hesapla(geometri):
    """Orijinal dosyanın modülünü veya runpy sözlüğünü alır; girdileri değiştirmez."""
    def oku(ad):
        return geometri[ad] if isinstance(geometri, Mapping) else getattr(geometri, ad)

    alanlar = {
        "D_m": "D_arr", "V_su_m3": "V_su_arr", "L_kuyruk_m": "L_kuyruk_arr",
        "L_toplam_m": "L_total", "L_su_m": "L_su_arr", "S_toplam_m2": "S_toplam",
    }
    diziler = {hedef: oku(kaynak) for hedef, kaynak in alanlar.items()}
    n = len(diziler["D_m"])
    if any(len(dizi) != n for dizi in diziler.values()):
        raise ValueError("Geometri çıktı dizilerinin uzunlukları eşit olmalı.")
    # Kaynak dosyada L_aviyonik_arr ortak kullanılan tek bir sayıdır.
    La = oku("L_aviyonik_arr")
    sonuclar = []
    for i in range(n):
        veri = {ad: dizi[i] for ad, dizi in diziler.items()}
        sonuc = govde_agirligi(**veri, L_aviyonik_m=La)
        sonuclar.append({"aday_no": i + 1, **sonuc})
    return sonuclar


def csv_kaydet(sonuclar, yol):
    """Liste içeren iskelet ayrıntıları hariç, her aday için bir CSV satırı."""
    if not sonuclar:
        raise ValueError("Kaydedilecek aday yok.")
    alanlar = [k for k in sonuclar[0] if k != "iskeletler"]
    with Path(yol).open("w", encoding="utf-8-sig", newline="") as dosya:
        yazici = csv.DictWriter(dosya, fieldnames=alanlar, delimiter=";", extrasaction="ignore")
        yazici.writeheader()
        yazici.writerows(sonuclar)


def kutle_csv_guncelle(kaynak, hedef=None):
    """Birleşik uçak CSV'sini güncel motor/pervane kütleleriyle yeniler.

    Geometri, kanat/kuyruk, aviyonik ve diğer bileşenler korunur.
    Toplamlar bileşenlerden hesaplanır; tekrar çalıştırmak ek fark yaratmaz.
    Hedef verilmezse kaynak dosya, tüm satırlar doğrulandıktan sonra değiştirilir.
    """
    kaynak = Path(kaynak)
    hedef = Path(hedef) if hedef is not None else kaynak
    sabit_alanlar = (
        "govde_pervane_g", "govde_arka_inis_takimi_g", "govde_motor_g",
        "govde_servolar_g", "govde_yapistirici_ve_diger_g",
        "govde_aviyonik_toplam_g",
    )
    yapisal_alanlar = ("govde_dis_kabuk_g", "govde_iskelet_g", "govde_karbon_boru_g")
    toplam_alanlar = (
        "govde_sabit_toplam_g", "govde_susuz_toplam_g", "govde_toplam_g",
        "govde_toplam_kg", "total_kutle_g", "total_kutle_kg",
        "total_susuz_kutle_g", "total_susuz_kutle_kg",
    )
    gerekli = set(sabit_alanlar + yapisal_alanlar + toplam_alanlar
                  + ("govde_su_g", "agirlik_toplam_g"))
    with kaynak.open("r", encoding="utf-8-sig", newline="") as dosya:
        baslik = dosya.readline()
        dosya.seek(0)
        ayirici = ";" if baslik.count(";") > baslik.count(",") else ","
        okuyucu = csv.DictReader(dosya, delimiter=ayirici)
        alanlar = okuyucu.fieldnames
        if not alanlar or len(set(alanlar)) != len(alanlar):
            raise ValueError("CSV başlıkları boş veya yinelenmiş.")
        eksik = gerekli - set(alanlar)
        if eksik:
            raise ValueError("CSV sütunları eksik: " + ", ".join(sorted(eksik)))
        kimlik_sutunları = {"temel_aday_id","konfigurasyon_id"}
        bulunan_kimlikler = kimlik_sutunları & set(alanlar)
        if bulunan_kimlikler and bulunan_kimlikler != kimlik_sutunları:
            raise ValueError("CSV kimlikleri sütunları eksik ikisi birlikte bulunmalı")
        kimlikli = bool(bulunan_kimlikler)
        if kimlikli and "pervane" not in alanlar:
            raise ValueError("kimlikli csv'de pervane sütunu gerekli")
        satirlar = list(okuyucu)

    if not satirlar:
        raise ValueError("CSV içinde güncellenecek aday yok.")

    pervane = _sayi(PERVANE_G, "PERVANE_G")
    motor = _sayi(MOTOR_G, "MOTOR_G")
    for no, satir in enumerate(satirlar, start=2):
        if None in satir or any(v is None for v in satir.values()):
            raise ValueError(f"CSV satır {no}: sütun sayısı başlıkla uyuşmuyor.")
        if kimlikli:
            if not all(satir[ad].strip() for ad in (*kimlik_sutunları,"pervane")):
                raise ValueError(f"CSV satir {no} : kimlik veya pervane boş")
            try:
                kimlik_ekle(satir,satir["pervane"])
            except ValueError as hata:
                raise ValueError(f"CSV satır {no} : {hata}") from hata

        def sayi(ad):
            return _sayi(satir[ad], f"CSV satır {no}, {ad}", sifir_olabilir=True)

        satir["govde_pervane_g"] = pervane
        satir["govde_motor_g"] = motor
        sabit = sum(sayi(ad) for ad in sabit_alanlar)
        susuz = sum(sayi(ad) for ad in yapisal_alanlar) + sabit
        toplam = susuz + sayi("govde_su_g")
        kanat_kuyruk = sayi("agirlik_toplam_g")
        satir.update({
            "govde_sabit_toplam_g": sabit,
            "govde_susuz_toplam_g": susuz,
            "govde_toplam_g": toplam,
            "govde_toplam_kg": toplam / 1000.0,
            "total_kutle_g": kanat_kuyruk + toplam,
            "total_kutle_kg": (kanat_kuyruk + toplam) / 1000.0,
            "total_susuz_kutle_g": kanat_kuyruk + susuz,
            "total_susuz_kutle_kg": (kanat_kuyruk + susuz) / 1000.0,
        })
        for ad in toplam_alanlar:
            _sayi(satir[ad], f"CSV satır {no}, {ad}", sifir_olabilir=True)
        if kimlikli:
            yeni_id = temel_kimlik(satir)
            satir["temel_aday_id"] = yeni_id
            satir["konfigurasyon_id"] = f"{yeni_id}__{satir['pervane'].strip()}"


    # Hata halinde mevcut dosyanın kısmen yazılmış durumda kalmasını önler.
    gecici = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="",
                                dir=hedef.resolve().parent, delete=False) as dosya:
            gecici = Path(dosya.name)
            yazici = csv.DictWriter(dosya, fieldnames=alanlar, delimiter=ayirici)
            yazici.writeheader()
            yazici.writerows(satirlar)
        gecici.replace(hedef)
    finally:
        if gecici is not None:
            gecici.unlink(missing_ok=True)
    return len(satirlar)


def main():
    parser = argparse.ArgumentParser(description="Gövde geometrisinden kütle hesabı; bataryalar ve aviyonik dahil.")
    parser.add_argument("--geometri", type=Path,
                        default=Path(__file__).resolve().with_name("govde_boyutlandirma.py"))
    parser.add_argument("--csv", type=Path, help="İstenirse sonuç CSV dosyasının yolu.")
    parser.add_argument("--kutle-csv", type=Path,
                        help="Mevcut birleşik kütle CSV’sini yenile. --csv yoksa aynı dosyaya yazılır.")
    args = parser.parse_args()
    if args.kutle_csv is not None:
        hedef = args.csv if args.csv is not None else args.kutle_csv
        try:
            adet = kutle_csv_guncelle(args.kutle_csv, hedef)
        except (OSError, ValueError, csv.Error) as hata:
            parser.error(str(hata))
        print(f"{adet} satır güncellendi: {hedef.resolve()}")
        print(f"Pervane: {PERVANE_G:g} g | Motor: {MOTOR_G:g} g")
        print("Kütle değiştiyse eski performans CSV'lerini yeniden üretin; "
              "seyir ve sıralama eski kimlikli sonuçları reddeder.")
        return
    if not args.geometri.is_file():
        parser.error(f"Geometri dosyası bulunamadı: {args.geometri}. İki dosyayı aynı klasöre koyun.")
    try:
        # Orijinal parametreler ve orijinal fonksiyon aynen çalıştırılır.
        geometri = runpy.run_path(str(args.geometri.resolve()))
    except ModuleNotFoundError as hata:
        if hata.name == "numpy":
            parser.error("Boyutlandırma kodu NumPy gerektiriyor. Kurulum: python -m pip install numpy")
        raise
    sonuclar = geometriden_agirlik_hesapla(geometri)
    print("\nGÖVDE KÜTLE DÖKÜMÜ (bataryalar ve aviyonik dahil)")
    print(f"Kabuk: {DIS_KABUK_GSM:g} g/m² | İskelet: {ISKELET_GSM:g} g/m² | Tek 3/2 mm boru: {BORU_G_M:g} g/m")
    kalemler = (
        ("dis_kabuk_g", "Dış kabuk"), ("iskelet_g", "İskelet"),
        ("karbon_boru_g", "Karbon boru"), ("pervane_g", "Pervane"),
        ("arka_inis_takimi_g", "Arka iniş takımı"), ("motor_g", "Motor"),
        ("servolar_g", "Servolar"), ("yapistirici_ve_diger_g", "Yapıştırıcı ve diğer"),
        ("bataryalar_g", "Bataryalar (2 adet)"), ("ic_aviyonik_g", "İç aviyonik"),
        ("kablo_g", "Kablo"), ("sigorta_g", "Sigorta"),
        ("aviyonik_toplam_g", "Aviyonik ara toplamı"),
        ("susuz_toplam_g", "Susuz toplam"), ("su_g", "Su"), ("toplam_g", "SU DAHİL TOPLAM"),
    )
    for s in sonuclar:
        print(f"\nAday {s['aday_no']} | {s['iskelet_sayisi']} iskelet | boru {s['boru_uzunlugu_m']:.3f} m")
        print("İskelet konumları (m): " + ", ".join(f"{e['konum_m']:.2f}" for e in s["iskeletler"]))
        for anahtar, etiket in kalemler:
            print(f"  {etiket:<23}: {s[anahtar]:>10.3f} g")
    if args.csv:
        csv_kaydet(sonuclar, args.csv)
        print(f"\nCSV: {args.csv.resolve()}")


if __name__ == "__main__":
    main()
