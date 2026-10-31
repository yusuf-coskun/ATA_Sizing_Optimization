"""Kanat/kuyruk kutle tahmini. Tum geometri metre, alan m2, kutle gram.

Boyutlandirma kodunu ve aday sozluklerini degistirmez. Excel baglantisi yoktur.
DAT: Selig kapali kontur veya Lednicer ust/alt yuzey bloklari (sayimli/sayimsiz).
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from math import hypot, isfinite, sqrt
from pathlib import Path
import re

SKIN_GSM = 491.9
ISKELET_GSM = 802.8
ON_BORU_G_M = 96.0
ARKA_BORU_G_M = 84.38
ON_INIS_TAKIMI_G = 300.0
# Tum flap, aileron, elevator ve rudder servolarinin toplam kutlesi.
SERVOLAR_TOPLAM_G = 261.0
SPAR_KONUM_XC = (0.25, 0.70)
BORU_UZUNLUK_ORANI = 0.10


def _pozitif(x: float, ad: str) -> float:
    x = float(x)
    if not isfinite(x) or x <= 0:
        raise ValueError(f"{ad} sonlu ve pozitif olmali: {x}")
    return x


def kanat_rib_sayisi(aciklik_m: float) -> int:
    b = _pozitif(aciklik_m, "aciklik_m")
    if not 2.0 <= b <= 3.2:
        raise ValueError("Kanat rib kurali yalnizca 2.0 <= b <= 3.2 m icin tanimli.")
    return 4 if b < 2.4 else (5 if b < 2.8 else 6)


def _dal(noktalar):
    # Tek yuzey x yonunde monoton olmali; tekrarli ayni noktalar silinir.
    temiz = []
    for p in noktalar:
        if not temiz or p != temiz[-1]:
            temiz.append(p)
    if len(temiz) < 2:
        raise ValueError("DAT yuzeyinde en az iki farkli nokta gerekli.")
    if temiz[0][0] > temiz[-1][0]:
        temiz.reverse()
    if any(b[0] <= a[0] for a, b in zip(temiz, temiz[1:])):
        raise ValueError("DAT ust/alt yuzeyinde x degerleri tekil ve monoton olmali.")
    return tuple(temiz)


def _y(dal, x):
    xs = [p[0] for p in dal]
    i = max(0, min(len(dal) - 2, bisect_right(xs, x) - 1))
    (x0, y0), (x1, y1) = dal[i:i + 2]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


@dataclass(frozen=True)
class Profil:
    ust: tuple
    alt: tuple
    kaynak: str = ""

    @property
    def alan_katsayisi(self):
        xs = sorted({p[0] for p in self.ust + self.alt})
        return sum((self.kalinlik(a) + self.kalinlik(b)) * (b - a) / 2
                   for a, b in zip(xs, xs[1:]))

    @property
    def cevre_katsayisi(self):
        # Yalniz ust+alt; kut firar kenari kapamasi dahil degil.
        return sum(hypot(b[0] - a[0], b[1] - a[1])
                   for dal in (self.ust, self.alt) for a, b in zip(dal, dal[1:]))

    def kalinlik(self, xc):
        return _y(self.ust, xc) - _y(self.alt, xc)


@lru_cache(maxsize=256)
def _profil_oku_cache(yol: str, mtime_ns: int, boyut: int) -> Profil:
    del mtime_ns, boyut  # Dosya degisince cache anahtari degisir.
    bloklar, blok = [], []
    for satir in Path(yol).read_text(encoding="utf-8-sig").splitlines():
        satir = re.split(r"[#!]", satir, maxsplit=1)[0].strip()
        parcalar = satir.split()
        p = None
        if len(parcalar) == 2:
            try:
                p = tuple(float(v.replace("D", "E").replace("d", "e")) for v in parcalar)
            except ValueError:
                pass
        if p is not None:
            if not all(isfinite(v) for v in p):
                raise ValueError(f"DAT sonlu olmayan koordinat iceriyor: {yol}")
            blok.append(p)
        elif blok:
            bloklar.append(blok)
            blok = []
    if blok:
        bloklar.append(blok)
    pts = [p for b in bloklar for p in b]
    if len(pts) < 4:
        raise ValueError(f"Yetersiz DAT koordinati: {yol}")

    sayim = pts[0]
    if all(v >= 2 and v.is_integer() for v in sayim):
        n, m = map(int, sayim)
        pts = pts[1:]
        if len(pts) != n + m:
            raise ValueError("Lednicer nokta sayilari dosya ile uyusmuyor.")
        d1, d2 = _dal(pts[:n]), _dal(pts[n:])
    elif len(bloklar) == 2 and all(len(b) >= 2 for b in bloklar):
        d1, d2 = map(_dal, bloklar)
    else:
        # TE -> LE -> TE veya LE -> TE -> LE siralamasi.
        k = min(range(len(pts)), key=lambda i: pts[i][0])
        if k in (0, len(pts) - 1):
            k = max(range(len(pts)), key=lambda i: pts[i][0])
        if k in (0, len(pts) - 1):
            raise ValueError("DAT konturunun iki yuzeyi ayristirilamadi.")
        d1, d2 = _dal(pts[:k + 1]), _dal(pts[k:])

        xmin = min(d1[0][0], d2[0][0])
    xmax = max(d1[-1][0], d2[-1][0])
    chord = xmax - xmin

    if chord <= 0:
        raise ValueError(f"DAT veter uzunlugu pozitif olmali: {yol}")

    tolerans = 2e-5 * chord

    if any(
        abs(d[0][0] - xmin) > tolerans
        or abs(d[-1][0] - xmax) > tolerans
        for d in (d1, d2)
    ):
        raise ValueError(
            f"DAT veter sinirlari uyusmuyor: {yol}\n"
            f"Yuzey 1: {d1[0][0]:.10f} -> {d1[-1][0]:.10f}\n"
            f"Yuzey 2: {d2[0][0]:.10f} -> {d2[-1][0]:.10f}\n"
            f"Tolerans: {tolerans:.10f}"
        )

    # Yalniz tolerans icindeki uc farklarini ortak sinirlara esitle.
    def uclari_esitle(d):
        return (
            (xmin, d[0][1]),
            *d[1:-1],
            (xmax, d[-1][1]),
        )

    d1 = uclari_esitle(d1)
    d2 = uclari_esitle(d2)

    d1, d2 = (
        tuple(((x - xmin) / chord, y / chord) for x, y in d)
        for d in (d1, d2)
    )
    if _y(d1, 0.5) < _y(d2, 0.5):
        d1, d2 = d2, d1
    profil = Profil(d1, d2, yol)
    xs = {p[0] for p in d1 + d2}
    if any(profil.kalinlik(x) < -1e-10 for x in xs) or profil.alan_katsayisi <= 0:
        raise ValueError("DAT yuzeyleri kesisiyor veya kesit alani sifir.")
    return profil


def profil_oku(dosya: str | Path) -> Profil:
    p = Path(dosya).resolve()
    s = p.stat()  # Eksik dosyada tahmini profil uydurulmaz.
    return _profil_oku_cache(str(p), s.st_mtime_ns, s.st_size)


def _skin_alan(profil, boy, kok, uc, simetrik):
    # Dogrusal taper, sifir sweep/twist. DAT polyline segmentlerinin gercek
    # 3B yuzey alani; dikdortgende tam olarak cevre_katsayisi * c * b.
    parca = boy / 2 if simetrik else boy
    dc = uc - kok
    alan = 0.0
    for dal in (profil.ust, profil.alt):
        for (x0, z0), (x1, z1) in zip(dal, dal[1:]):
            dx, dz = x1 - x0, z1 - z0
            carpi = sqrt((parca * dz)**2 + (dc * (dz*x0 - dx*z0))**2 + (parca * dx)**2)
            alan += (kok + uc) / 2 * carpi
    return alan * (2 if simetrik else 1)


def yuzey_agirligi(*, aciklik_m: float, kok_veter_m: float,
                   profil_dat: str | Path, tur: str,
                   uc_veter_m: float | None = None, simetrik: bool | None = None) -> dict:
    """tur: kanat/yatay/dikey/hibrit. Rib sayisi tum yuzey icin toplamdir.

    Hibritte aciklik, iki panelin toplam GERCEK gelistirilmis acikligidir;
    yatay izdususumu degildir. V kuyruk butunu icin 5 rib kullanilir.
    """
    b = _pozitif(aciklik_m, "aciklik_m")
    cr = _pozitif(kok_veter_m, "kok_veter_m")
    ct = cr if uc_veter_m is None else _pozitif(uc_veter_m, "uc_veter_m")
    if tur not in ("kanat", "yatay", "dikey", "hibrit"):
        raise ValueError(f"Bilinmeyen yuzey turu: {tur}")
    n = kanat_rib_sayisi(b) if tur == "kanat" else {"yatay": 4, "dikey": 5, "hibrit": 5}[tur]
    simetrik = (tur != "dikey") if simetrik is None else simetrik
    p = profil_oku(profil_dat)
    konum = [-b/2 + i*b/(n-1) if simetrik else i*b/(n-1) for i in range(n)]
    veter = [cr + (ct-cr)*(abs(y)/(b/2) if simetrik else y/b) for y in konum]
    rib_alan = [p.alan_katsayisi*c*c for c in veter]
    # Iki yari kanat trapezinin toplami veya tek dikey kuyruk trapezi.
    spar_alan = [p.kalinlik(xc)*(cr+ct)/2*b for xc in SPAR_KONUM_XC]
    skin = _skin_alan(p, b, cr, ct, simetrik)
    on_boru = b*BORU_UZUNLUK_ORANI*ON_BORU_G_M
    arka_boru = b*BORU_UZUNLUK_ORANI*ARKA_BORU_G_M
    rib_g, spar_g, skin_g = sum(rib_alan)*ISKELET_GSM, sum(spar_alan)*ISKELET_GSM, skin*SKIN_GSM
    return {
        "rib_sayisi": n, "rib_konumlari_m": konum, "rib_veterleri_m": veter,
        "rib_alanlari_m2": rib_alan, "rib_alani_m2": sum(rib_alan),
        "spar_xc": list(SPAR_KONUM_XC), "spar_alanlari_m2": spar_alan,
        "spar_alani_m2": sum(spar_alan), "skin_alani_m2": skin,
        "planform_alani_m2": b*(cr+ct)/2,
        "profil_kesit_katsayisi": p.alan_katsayisi,
        "profil_cevre_katsayisi": p.cevre_katsayisi,
        "rib_g": rib_g, "spar_g": spar_g, "skin_g": skin_g,
        "her_boru_uzunlugu_m": b*BORU_UZUNLUK_ORANI,
        "on_boru_20mm_g": on_boru, "arka_boru_18mm_g": arka_boru,
        "borular_g": on_boru+arka_boru,
        "toplam_g": rib_g+spar_g+skin_g+on_boru+arka_boru,
    }


def aday_agirligi(aday: dict, *, hibrit_kuyruk: dict | None = None) -> dict:
    """Orijinal adaydan okur. Konvansiyonel HT+VT; hibritte ikisi yerine tek kuyruk.

    hibrit_kuyruk anahtarlari: aciklik_m, kok_veter_m, profil_dat;
    opsiyonel uc_veter_m, simetrik. Mevcut kod hibrit geometri uretmez.
    """
    kanat = yuzey_agirligi(aciklik_m=aday["b_w_m"], kok_veter_m=aday["c_root_w_m"],
                          uc_veter_m=aday["c_tip_w_m"], profil_dat=aday["kanat_airfoil_dat"], tur="kanat")
    yuzeyler = {"kanat": kanat}
    if hibrit_kuyruk is not None:
        yuzeyler["hibrit_kuyruk"] = yuzey_agirligi(tur="hibrit", **hibrit_kuyruk)
    else:
        for ad, kisaltma, boy, tur in (("yatay_kuyruk", "HT", "b_HT_m", "yatay"),
                                      ("dikey_kuyruk", "VT", "h_VT_m", "dikey")):
            yuzeyler[ad] = yuzey_agirligi(
                aciklik_m=aday[boy], kok_veter_m=aday[f"c_root_{kisaltma}_m"],
                uc_veter_m=aday.get(f"c_tip_{kisaltma}_m", aday[f"c_root_{kisaltma}_m"]),
                profil_dat=aday["kuyruk_airfoil_dat"], tur=tur)
    kuyruk = sum(v["toplam_g"] for k, v in yuzeyler.items() if k != "kanat")
    # Sabit ekipman kutleleri genel toplama yalnizca bir kez eklenir.
    toplam = kanat["toplam_g"] + kuyruk + ON_INIS_TAKIMI_G + SERVOLAR_TOPLAM_G
    return {"yuzeyler": yuzeyler, "kanat_toplam_g": kanat["toplam_g"],
            "kuyruk_toplam_g": kuyruk, "on_inis_takimi_g": ON_INIS_TAKIMI_G,
            "servolar_toplam_g": SERVOLAR_TOPLAM_G,
            "toplam_g": toplam, "toplam_kg": toplam/1000}


def agirlik_ekle(aday: dict, *, hibrit_kuyruk: dict | None = None) -> dict:
    """Adayin YENI kopyasina CSV uyumlu agirlik_... sutunlari ekler."""
    sonuc = aday_agirligi(aday, hibrit_kuyruk=hibrit_kuyruk)
    ek = {f"agirlik_{k}": v for k, v in sonuc.items() if k != "yuzeyler"}
    for ad, detay in sonuc["yuzeyler"].items():
        for k, v in detay.items():
            if not isinstance(v, list):
                ek[f"agirlik_{ad}_{k}"] = v
    if set(ek) & set(aday):
        raise ValueError("Aday zaten agirlik alanlari iceriyor; uzerine yazilmadi.")
    return {**aday, **ek}