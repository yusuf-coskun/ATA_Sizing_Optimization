"""
[YENİ DOSYA 2026-11-19 v10 | flaplı α0 kontrolü]
Flaplı profilin sıfır-taşıma açısını (α0) 2B olarak üç yoldan bulur ve VSPAERO'nun
3B flaplı α0'ını "flap verimi" ile düzeltmek için gereken oranı hesaplar.
Hiçbir dosyayı DEĞİŞTİRMEZ; yalnızca rapor (CSV + ekran) üretir.

Sade anlatım:
  - İnce profil teorisi (sürtünmesiz): profilin orta (kamburluk) çizgisinden α0'ı hesaplar.
    VSPAERO da sürtünmesiz olduğu için flabı bu teoriye yakın "görür".
  - XFOIL (sürtünmeli): gerçek havada 30° flabın arkasında akış kısmen ayrılır, flap
    teorideki kadar etkili olmaz. Bu yüzden XFOIL'deki α0 kayması daha küçüktür.
  - Flap verimi  eta = (XFOIL α0 kayması) / (ince profil α0 kayması)
    3B düzeltme:  α0_3B_flapli = α0_3B_flapsiz + eta * (α0_VSPAERO_flapli - α0_VSPAERO_flapsiz)

Çalıştırma (Scripts klasöründe):  py -3.13 flap_alpha0_kontrol.py [--re 250000]
Çıktı: vlm_sonuclari/flap_alpha0_kontrol.csv
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parent.parent  # ana proje klasörü (airfoils* klasörleri burada)


def dat_oku(yol: Path) -> np.ndarray:
    satirlar = []
    for s in yol.read_text(errors="ignore").splitlines()[1:]:
        p = s.split()
        if len(p) >= 2:
            try:
                satirlar.append((float(p[0]), float(p[1])))
            except ValueError:
                pass
    return np.array(satirlar)


def ince_profil_alpha0(yol: Path) -> float:
    """İnce profil teorisi: α0 = -(1/π)∫ dz/dx (cosθ - 1) dθ   (açı: koordinatların x eksenine göre, XFOIL ile aynı)."""
    xy = dat_oku(yol)
    i = int(np.argmin(xy[:, 0]))
    ust = xy[: i + 1][::-1]   # burundan firar kenarına
    alt = xy[i:]
    x0 = xy[i, 0]
    x1 = min(ust[:, 0].max(), alt[:, 0].max())
    c = x1 - x0
    th = np.linspace(1e-4, math.pi - 1e-4, 4001)
    x = x0 + 0.5 * c * (1 - np.cos(th))
    zu = np.interp(x, ust[:, 0], ust[:, 1])
    zl = np.interp(x, alt[:, 0], alt[:, 1])
    z = 0.5 * (zu + zl)
    dz = np.gradient(z, x)
    integral = (np.trapezoid if hasattr(np, 'trapezoid') else np.trapz)(dz * (np.cos(th) - 1.0), th)
    return math.degrees(-integral / math.pi)


def polar_oku(yol: Path):
    a, cl = [], []
    basladi = False
    for s in yol.read_text(errors="ignore").splitlines():
        if s.strip().startswith("------"):
            basladi = True
            continue
        if basladi:
            p = s.split()
            if len(p) >= 2:
                a.append(float(p[0])); cl.append(float(p[1]))
    return np.array(a), np.array(cl)


def xfoil_alpha0(yol: Path):
    """XFOIL polarından α0. Polar CL=0'ı kapsıyorsa ara değer; kapsamıyorsa en düşük açılardaki
    doğrusal bölgeden geriye uzatma (dış değer) — 'yontem' sütununda belirtilir."""
    a, cl = polar_oku(yol)
    if len(a) < 3:
        return float("nan"), "veri_yok", float("nan")
    s = np.argsort(a); a, cl = a[s], cl[s]
    for k in range(len(a) - 1):
        if cl[k] <= 0.0 <= cl[k + 1]:
            t = -cl[k] / (cl[k + 1] - cl[k])
            egim = (cl[k + 1] - cl[k]) / (a[k + 1] - a[k])
            return float(a[k] + t * (a[k + 1] - a[k])), "ara_deger", float(egim)
    sec = a <= a[0] + 3.0 + 1e-9
    if sec.sum() < 3:
        sec = np.arange(len(a)) < 4
    egim, kes = np.polyfit(a[sec], cl[sec], 1)
    return float(-kes / egim), f"dis_deger({a[0]:.1f}..{a[sec][-1]:.1f})", float(egim)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--re", type=int, default=250000,  # [DEĞİŞİKLİK 2026-11-19 v11 | flap ve k diğer kodlardan] kalkış Re'si: lifting_line_cozucu KALKIS_HIZI_MS=12 m/s, c≈0.333 m -> Re≈260 bin
                    help="Kullanılacak XFOIL Reynolds sayısı (varsayılan 250000 = kalkış hızı 12 m/s).")
    args = ap.parse_args()
    satirlar = []
    for dat in sorted((KOK / "airfoils").glob("*.dat")):
        ad = dat.stem
        dat_f = KOK / "airfoils_flap" / f"{ad}_flap.dat"
        pol = KOK / "airfoils_polar" / f"{ad}_re{args.re}.pol"
        pol_f = KOK / "airfoils_polar_flap" / f"{ad}_flap_re{args.re}.pol"
        if not (dat_f.exists() and pol.exists() and pol_f.exists()):
            continue
        ti0, ti1 = ince_profil_alpha0(dat), ince_profil_alpha0(dat_f)
        xf0, y0, _ = xfoil_alpha0(pol)
        xf1, y1, egim1 = xfoil_alpha0(pol_f)
        d_ti, d_xf = ti1 - ti0, xf1 - xf0
        eta = d_xf / d_ti if abs(d_ti) > 1e-6 else float("nan")
        satirlar.append(dict(profil=ad, Re=args.re,
                             a0_ince_flapsiz=round(ti0, 3), a0_ince_flapli=round(ti1, 3), d_a0_ince=round(d_ti, 3),
                             a0_xfoil_flapsiz=round(xf0, 3), a0_xfoil_flapli=round(xf1, 3), d_a0_xfoil=round(d_xf, 3),
                             xfoil_flapli_yontem=y1, xfoil_flapli_egim_per_deg=round(egim1, 4),
                             flap_verimi_eta=round(eta, 3)))
    if not satirlar:
        print("Profil bulunamadı."); return 1
    cikti = Path(__file__).resolve().parent / "vlm_sonuclari" / "flap_alpha0_kontrol.csv"
    cikti.parent.mkdir(exist_ok=True)
    with cikti.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0])); w.writeheader(); w.writerows(satirlar)
    print(f"{'profil':<14}{'ince Δα0':>9}{'XFOIL Δα0':>10}{'eta':>7}  XFOIL flaplı yöntem")
    for r in satirlar:
        print(f"{r['profil']:<14}{r['d_a0_ince']:>9.2f}{r['d_a0_xfoil']:>10.2f}{r['flap_verimi_eta']:>7.2f}  {r['xfoil_flapli_yontem']}")
    etalar = np.array([r["flap_verimi_eta"] for r in satirlar], dtype=float)
    etalar = etalar[np.isfinite(etalar)]
    print(f"\neta: ortanca {np.median(etalar):.3f}, aralık {etalar.min():.3f} .. {etalar.max():.3f} ({len(etalar)} profil)")
    print(f"Rapor: {cikti}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
