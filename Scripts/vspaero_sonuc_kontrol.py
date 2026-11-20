# -*- coding: utf-8 -*-
"""
VSPAERO TAM AKIŞ SONUÇ KONTROLÜ  [DEĞİŞİKLİK 2026-11-19 v9 | YENİ DOSYA: sonuç kontrol kodu]

Ne yapar?  vlm_entegrasyonu.py tam akışı bittikten sonra çalıştırılır. Hiçbir dosyayı DEĞİŞTİRMEZ,
sadece okur ve vlm_sonuclari/VSPAERO_KONTROL_RAPORU.txt dosyasını yazar.

Kontroller:
  1) Program düzgün çalıştı mı?  (çözücü adı, gövde dahil mi, kalite bayrakları)
  2) Sayılar mantıklı aralıkta mı? (CLα, e, α0, Cm_α)
  3) Elle hesap: CLα'yı ders kitabı formülüyle (Helmbold) karşılaştır
  4) Sıralama ne kadar değişti? (LLT sıralaması vs VSPAERO sıralaması)
  5) Görev B için elle kontrol edilecek 3 aday (.vsp3 dosya yollarıyla)

Kullanım (Scripts klasöründe):
  py -3.13 vspaero_sonuc_kontrol.py
  py -3.13 vspaero_sonuc_kontrol.py --dosya baska_dosya.csv
"""
import argparse, csv, math
from collections import Counter
from pathlib import Path

# Beklenen aralıklar (değiştirmek istersen burayı düzenle)
ARALIKLAR = {
    "CL_alpha_3d_vlm":        (0.060, 0.085, "Flapsız CLα [1/°] — AR=6 dikdörtgen kanat için ders kitabı aralığı"),
    "CL_alpha_3d_flapli_vlm": (0.055, 0.085, "Flaplı CLα [1/°]"),
    "e_oswald_vlm":           (0.80, 1.00,  "Flapsız e (viskoz düzeltmeli) — 1'i geçemez"),
    "e_oswald_flapli_vlm":    (0.55, 1.00,  "Flaplı e"),
    "alpha_L0_vlm":           (-8.0, 0.0,   "Flapsız sıfır-taşıma açısı [°]"),
    "alpha_L0_flapli_vlm":    (-20.0, -5.0, "Flaplı sıfır-taşıma açısı [°] (VLM flabı abartır, LLT ≈ −10°)"),
    "vlm_Cm_alpha_per_deg":   (-0.01, 0.015, "Cm_α [1/°] (kuyruksuz; küçük pozitif beklenir)"),
}


def sayi(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def sirala(degerler):
    """Sıra numarası ver (eşitlerde ortalama sıra)."""
    sirali = sorted(range(len(degerler)), key=lambda i: degerler[i])
    r = [0.0] * len(degerler); i = 0
    while i < len(sirali):
        j = i
        while j + 1 < len(sirali) and degerler[sirali[j + 1]] == degerler[sirali[i]]:
            j += 1
        for k in range(i, j + 1):
            r[sirali[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = sirala(a), sirala(b)
    n = len(a); ma, mb = sum(ra) / n, sum(rb) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in ra)); sb = math.sqrt(sum((x - mb) ** 2 for x in rb))
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / (sa * sb) if sa and sb else float("nan")


def helmbold_cla_deg(ar):
    """Helmbold formülü (ince profil, 2B eğim 2π): CLα = 2π·AR / (2 + √(AR²+4))  [1/rad -> 1/°]"""
    return 2 * math.pi * ar / (2 + math.sqrt(ar * ar + 4)) * math.pi / 180


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dosya", default="vlm_nihai_ucak_siralamasi.csv")
    ap.add_argument("--rapor", default="vlm_sonuclari/VSPAERO_KONTROL_RAPORU.txt")
    a = ap.parse_args()
    yol = Path(a.dosya)
    if not yol.exists():
        print(f"[HATA] {yol} bulunamadı. Önce tam akışı çalıştır."); return 1
    rows = list(csv.DictReader(open(yol, encoding="utf-8-sig")))
    n = len(rows)
    L = []; p = L.append
    sorun = 0

    p("VSPAERO TAM AKIŞ SONUÇ KONTROL RAPORU")
    p("=" * 70)
    p(f"Dosya: {yol.resolve()}")
    p(f"Satır (uçak x pervane): {n}")
    p("İşaretler: [TAMAM] sorun yok | [DİKKAT] bakılmalı | [BİLGİ] bilinen durum")
    p("")

    # 1) Program düzgün çalıştı mı
    p("1) PROGRAM DÜZGÜN ÇALIŞTI MI?")
    p("-" * 70)
    for sut, ad in (("vlm_cozucu", "Flapsız çözücü"), ("vlm_cozucu_flapli", "Flaplı çözücü")):
        c = Counter(r.get(sut, "") for r in rows)
        kotu = sum(v for k, v in c.items() if k != "vspaero_api")
        if kotu:
            sorun += 1
            p(f"[DİKKAT] {ad}: {kotu} satır VSPAERO DEĞİL -> {dict(c)}")
        else:
            p(f"[TAMAM] {ad}: {n}/{n} satır vspaero_api")
    g = Counter(r.get("vlm_govde_dahil", "") for r in rows)
    if g.get("True", 0) == n:
        p(f"[TAMAM] Gövde: {n}/{n} satırda gövde dahil")
    else:
        p(f"[BİLGİ] Gövde dahil dağılımı: {dict(g)} (gövdesiz çalıştırdıysan normal)")
    bayrak = Counter()
    for r in rows:
        for b in (r.get("vlm_kalite") or "").split(";"):
            if b.strip():
                bayrak[b.strip()] += 1
    bilinen = {"ok", "e_kirpildi", "e_aralik_disi", "yuzey_iz_CL_farki_buyuk"}
    p("Kalite bayrakları (kaç satırda):")
    for b, k in bayrak.most_common():
        etiket = "[BİLGİ]" if b in bilinen else "[DİKKAT]"
        if b not in bilinen:
            sorun += 1
        aciklama = {
            "ok": "sorun yok",
            "e_kirpildi": "ağ hatası nedeniyle e 1'i geçti, 1'e kırpıldı (beklenen)",
            "e_aralik_disi": "ham e aralık dışında (genelde e>1, kırpma ile aynı sebep)",
            "yuzey_iz_CL_farki_buyuk": "flaplıda iki CL hesabı ayrışıyor (beklenen; e'ye temkinli bak)",
        }.get(b, "BEKLENMEYEN — bu satırlara ayrıca bak")
        p(f"  {etiket} {b}: {k}  -> {aciklama}")
    p("")

    # 2) Aralık kontrolü
    p("2) SAYILAR MANTIKLI ARALIKTA MI?")
    p("-" * 70)
    for sut, (lo, hi, acik) in ARALIKLAR.items():
        v = [(r, sayi(r.get(sut))) for r in rows]
        v = [(r, x) for r, x in v if x is not None]
        if not v:
            p(f"[BİLGİ] {sut}: değer yok"); continue
        xs = [x for _, x in v]
        disari = [(r, x) for r, x in v if not (lo <= x <= hi)]
        ozet = f"min {min(xs):.4f} | ort {sum(xs)/len(xs):.4f} | maks {max(xs):.4f}"
        if disari:
            sorun += 1
            p(f"[DİKKAT] {acik}\n         beklenen {lo}..{hi} | {ozet} | aralık dışı {len(disari)} satır:")
            for r, x in disari[:8]:
                p(f"           sıra {r.get('vlm_nihai_sira') or r.get('sira')}: {r.get('kanat_airfoil')} {r.get('pervane')} -> {x:.4f}")
        else:
            p(f"[TAMAM] {acik}\n         beklenen {lo}..{hi} | {ozet}")
    p("")

    # 3) Elle hesap
    p("3) ELLE HESAP: CLα (Helmbold formülü)")
    p("-" * 70)
    p("Formül: CLα = 2π·AR / (2 + √(AR²+4)). İnce profil varsayar (2B eğim 2π/rad).")
    p("Gerçek profiller biraz daha düşük eğim verir, bu yüzden VSPAERO/Helmbold ≈ 0.85–1.02 beklenir.")
    oranlar = []
    for r in rows:
        ar, cla = sayi(r.get("AR_w")), sayi(r.get("CL_alpha_3d_vlm"))
        if ar and cla:
            oranlar.append(cla / helmbold_cla_deg(ar))
    if oranlar:
        ar0 = sayi(rows[0].get("AR_w")) or 6.0
        p(f"AR={ar0:g} için Helmbold CLα = {helmbold_cla_deg(ar0):.4f} /°")
        ok = all(0.85 <= o <= 1.02 for o in oranlar)
        if not ok:
            sorun += 1
        p(f"{'[TAMAM]' if ok else '[DİKKAT]'} VSPAERO/Helmbold oranı: min {min(oranlar):.3f} | maks {max(oranlar):.3f}")
    llt = [sayi(r.get("CL_alpha_3d_vlm")) for r in rows]
    p("Not: Eski LLT tablosunda CLα 0.090–0.099 /° idi; Helmbold'dan yüksek -> LLT ekibi kontrol etmeli.")
    p("Gövdenin Cm_α etkisi için elle hesap (Raymer gövde formülü) ayrıca yapılacak.")
    p("")

    # 4) Sıralama
    p("4) SIRALAMA NE KADAR DEĞİŞTİ? (LLT öncesi sıra -> VSPAERO sonrası sıra)")
    p("-" * 70)
    ciftler = [(sayi(r.get("llt_onc_sira")), sayi(r.get("vlm_nihai_sira")), r) for r in rows]
    ciftler = [c for c in ciftler if c[0] is not None and c[1] is not None]
    if ciftler:
        rho = spearman([c[0] for c in ciftler], [c[1] for c in ciftler])
        p(f"Sıra benzerliği (Spearman, 1 = birebir aynı, 0 = ilgisiz): {rho:.3f}")
        if rho >= 0.9:
            p("[TAMAM] Sıralama büyük ölçüde korunmuş: seçim modelin küçük hatalarına duyarlı değil.")
        else:
            sorun += 1
            p("[DİKKAT] Sıralama belirgin değişmiş: hangi değerin değiştirdiğine bakılmalı.")
        farklar = [sayi(c[2].get("puan_farki_vlm_eksi_llt")) for c in ciftler]
        farklar = [f for f in farklar if f is not None]
        if farklar:
            p(f"Puan farkı (VSPAERO − LLT): min {min(farklar):+.2f} | maks {max(farklar):+.2f} puan")
            p("  Not: Üst sıralardaki adaylar arasında puan farkı 1 puandan azsa sıra değişimi küçük")
            p("  farklardan gelir; asıl önemli olan büyük puan düşüşleri (aşağıdaki liste).")
        for k in (5, 10):
            eski = {id(c[2]) for c in ciftler if c[0] <= k}
            yeni = {id(c[2]) for c in ciftler if c[1] <= k}
            p(f"İlk {k} içinde ortak aday: {len(eski & yeni)}/{k}")
        p("VSPAERO'ya göre ilk 5:")
        for c in sorted(ciftler, key=lambda c: c[1])[:5]:
            r = c[2]
            p(f"  {int(c[1]):>3}. {r.get('kanat_airfoil'):<14} {r.get('pervane'):<8} (LLT sırası {int(c[0])}) "
              f"puan {sayi(r.get('vlm_sonrasi_toplam_puan')) or 0:.2f} (LLT {sayi(r.get('llt_onc_toplam_puan')) or 0:.2f})")
        buyuk = sorted(ciftler, key=lambda c: -abs(c[1] - c[0]))[:5]
        p("En çok yer değiştiren 5 aday:")
        for c in buyuk:
            r = c[2]
            p(f"  {r.get('kanat_airfoil'):<14} {r.get('pervane'):<8} LLT {int(c[0])} -> VSPAERO {int(c[1])}")
    else:
        p("[BİLGİ] Sıra sütunları bulunamadı.")
    p("")

    # 5) Görev B
    p("5) GÖREV B İÇİN ELLE KONTROL EDİLECEK 3 ADAY")
    p("-" * 70)
    p("Bu .vsp3 dosyalarını OpenVSP arayüzünde açıp VSPAERO'yu çalıştırın; CLα ve e bu tablodakiyle")
    p("aynı çıkmalı (kanat tek başınayken %0.1 uyum bulunmuştu).")
    sirali = sorted(rows, key=lambda r: sayi(r.get("vlm_nihai_sira")) or 1e9)
    if sirali:
        secim = [("1. sıra", sirali[0]), ("orta", sirali[len(sirali) // 2]), ("son", sirali[-1])]
        for etiket, r in secim:
            p(f"  [{etiket}] {r.get('kanat_airfoil')} {r.get('pervane')} | CLα {r.get('CL_alpha_3d_vlm')} | "
              f"e {r.get('e_oswald_vlm')} | Cm_α {r.get('vlm_Cm_alpha_per_deg')}")
            p(f"           flapsız: {r.get('vsp3_flapsiz')}")
            p(f"           flaplı : {r.get('vsp3_flapli')}")
    p("")

    p("=" * 70)
    p(f"SONUÇ: {'Bakılması gereken madde yok.' if sorun == 0 else f'{sorun} maddeye bakılmalı ([DİKKAT] satırları).'}")
    p("Bu rapor sonuçların KESİN doğru olduğunu kanıtlamaz; kodun düzgün çalıştığını ve sayıların")
    p("mantıklı olduğunu gösterir. Kesin doğrulama: Görev B arayüz kontrolü + elle hesap + ölçüm.")

    rp = Path(a.rapor); rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[RAPOR] Yazıldı: {rp.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
