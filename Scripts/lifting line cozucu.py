"""
LLT ÇÖZÜCÜ - ÇOKLU REYNOLDS SAYISI
==================================
Mevcut dosyaların HİÇBİRİNİ değiştirmez. Sadece kanat_boyutlandirma.py'den
aday listesini okur (import eder).

Ne yapar?
  1) Her kanat airfoili için, REYNOLDS_LISTESI'ndeki her Reynolds sayısında
     XFOIL poları üretir (zaten varsa tekrar üretmez).
  2) Her kanat adayını (b, AR, airfoil) HER Reynolds sayısında LLT ile çözer.
  3) Her adayın kendi veterine (chord) ve seyir hızına göre GERÇEK Reynolds
     sayısını hesaplar ve sonuçları o Re'ye göre ara değerle (interpolasyon) verir.

Çıktılar (proje ana klasörüne yazılır):
  - llt_coklu_re_tum_sonuclar.csv : her aday x her Reynolds için bir satır
  - llt_coklu_re_aday_re.csv      : her aday için, kendi Reynolds'unda bir satır
  - re_etkisi_grafikleri/         : her airfoil için Cl-alpha ve Cl-Cd eğrileri (Re'ye göre)

Kısa sözlük:
  Reynolds sayısı (Re) : Re = V * c / nu. Havanın kanat etrafında ne kadar
                         "akıcı" davrandığını anlatan sayı. Küçük uçakta küçük
                         olur; küçük Re = daha çok sürtünme, daha erken stall.
  Polar                : Bir airfoilin açıya göre Cl (taşıma) ve Cd (sürükleme)
                         tablosu. XFOIL tek bir Re için üretir.
  LLT                  : Lifting Line Teorisi; 2B airfoil verisinden 3B kanadın
                         taşımasını hesaplar.
"""
from __future__ import annotations

import contextlib
import csv
import io
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ============================== AYARLAR ==============================
REYNOLDS_LISTESI = [200000, 250000, 300000, 350000, 400000, 450000, 500000, 550000, 600000]

# TEST AYARI: Sadece ilk N airfoil ile çalış (hızlı deneme için).
# Tüm airfoiller için None yapın.
TEST_AIRFOIL_SAYISI = None

# KUYRUK (yatay + dikey) için Reynolds listesi. Kuyruk veterleri kanattan
# kısa olduğu için Re'leri daha düşüktür. Kuyruk airfoili: naca0010.dat
KUYRUK_REYNOLDS_LISTESI = [100000, 150000, 200000, 250000, 300000]

SEYIR_HIZI_MS = 18.0        # Seyir hızı (m/s) -> adayın gerçek Re'sini bulmak için
RHO = 1.175                 # Hava yoğunluğu (kg/m^3), takımın sabitler.txt değeri
MU = 1.789e-5               # Havanın dinamik viskozitesi (Pa.s), 15 °C
SEYIR_ALPHA_DEG = 0.0       # L/D tahmini yapılacak kanat hücum açısı (derece)
GOVDE_UZUNLUGU_M = 1.5      # Kuyruk boyutlandırması için (kanat aerodinamiğini etkilemez)

XFOIL_EXE = str(Path(__file__).resolve().parent.parent / "xfoil.exe")   # proje ana klasöründeki xfoil.exe
POLAR_URET = True           # Eksik polarları XFOIL ile üret
POLAR_KLASORU_ADI = "airfoils_polar"            # Yeni polarlar takımın polar klasörüne yazılır
ESKI_POLAR_KLASORU_ADI = "airfoils_polar_coklu_re"  # Eski ayrı klasör (yedek kaynak olarak okunur)
ALPHA_MIN, ALPHA_MAX, ALPHA_STEP = -5.0, 15.0, 0.5
ITERATIONS = 300
XFOIL_TIMEOUT = 60        # saniye; bir airfoil-Re çözümü için en fazla bekleme

LINEER_ALPHA_MIN, LINEER_ALPHA_MAX = -4.0, 6.0  # Taşıma eğiminin ölçüldüğü düz bölge

# YARIM POLAR KONTROLÜ: Bir polar aşağıdaki şartları sağlamazsa "yarım" sayılır,
# hesaba KATILMAZ ve ekrana yazılır. (XFOIL zaman aşımına uğrarsa ya da
# birçok açıda çözüme ulaşamazsa polar yarım kalır.)
YARIM_EN_KUCUK_ALFA = -1.0   # polar en az bu açıya kadar inmeli (eksi açılar var mı?)
YARIM_EN_BUYUK_ALFA = 10.0   # polar en az bu açıya kadar çıkmalı (stall bölgesi var mı?)
YARIM_EN_AZ_NOKTA = 20       # polarda en az bu kadar açı noktası olmalı
GRAFIK_CIZ = True
CANLI_GRAFIK = True         # Çalışırken ekranda canlı grafik penceresi göster
# =====================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# govde_boyutlandirma import edilince tablo basıyor; ekranı kirletmesin diye susturuyoruz
with contextlib.redirect_stdout(io.StringIO()):
    import kanat_boyutlandirma as kb

PROJE_KLASORU = kb.AIRFOIL_KLASORU.parent
POLAR_KLASORU = PROJE_KLASORU / POLAR_KLASORU_ADI
# Eski polarlar için olası klasör adları (takımda iki yazım da kullanılmış)
ESKI_POLAR_KLASORLERI = [PROJE_KLASORU / ESKI_POLAR_KLASORU_ADI,
                         PROJE_KLASORU / "airfoils_polars"]
_xfoil_var = True   # XFOIL bir kez bulunamazsa tekrar denemeyiz
_bekle_fn = None    # canlı grafik açıksa: XFOIL beklerken pencereyi canlı tutan fonksiyon


# ---------------------------------------------------------------------
# 1) POLAR ÜRETME / BULMA
# ---------------------------------------------------------------------
def _xfoil_komutlari(dat_yolu: Path, polar_yolu: Path, reynolds: int) -> str:
    """0'dan yukarı tarama, sonra INIT ve eksi taraf için 0'dan aşağı tarama.
    Eksi tarafa doğrudan -0.5° ile değil, önce kolay açı olan 0°'yi yeniden çözerek
    başlıyoruz; böylece "ısınmadan zor açıyla başlama" kaynaklı yarım polarlar azalır.
    (0° iki kez kaydedilir; okuyucu tekrarları zaten tek satıra indiriyor.)"""
    k = [
        "PLOP", "G", "",   # grafik penceresini kapat (hızlı ve sessiz çalışsın)
        f"LOAD {dat_yolu}", "PANE", "OPER", "ALFA 0",
        f"ITER {ITERATIONS}", f"VISC {reynolds}", "PACC", f"{polar_yolu}", "",
    ]
    a = 0.0
    while a <= ALPHA_MAX + 1e-5:
        k.append(f"ALFA {a:.2f}")
        a += ALPHA_STEP
    k.append("INIT")
    k.append("ALFA 0.00")      # eksi tarafa kolay açıdan başla
    a = -ALPHA_STEP
    while a >= ALPHA_MIN - 1e-5:
        k.append(f"ALFA {a:.2f}")
        a -= ALPHA_STEP
    k += ["PACC", "QUIT"]
    return "\n".join(k) + "\n"


def polar_dosyasi_bul_veya_uret(dat_yolu: Path, reynolds: int) -> Path | None:
    global _xfoil_var
    ad = dat_yolu.stem
    for klasor in (POLAR_KLASORU, *ESKI_POLAR_KLASORLERI):
        p = klasor / f"{ad}_re{reynolds}.pol"
        if p.exists() and p.stat().st_size > 0:
            return p
    if not POLAR_URET or not _xfoil_var:
        return None

    POLAR_KLASORU.mkdir(parents=True, exist_ok=True)
    hedef = POLAR_KLASORU / f"{ad}_re{reynolds}.pol"
    # XFOIL önce GEÇİCİ dosyaya yazar; iş bitince adı .pol yapılır.
    # Böylece yazılmakta olan bir dosya asla "hazır polar" sanılmaz.
    gecici = POLAR_KLASORU / f"{ad}_re{reynolds}.yaziliyor"
    for eski in (hedef, gecici):
        if eski.exists():
            eski.unlink()
    try:
        # XFOIL uzun yolları sevmez; göreli yol veriyoruz
        komutlar = _xfoil_komutlari(Path(os.path.relpath(dat_yolu, PROJE_KLASORU)),
                                    Path(os.path.relpath(gecici, PROJE_KLASORU)), reynolds)
        # XFOIL'i başlatıp bitmesini "aktif" bekliyoruz: beklerken canlı grafik
        # penceresi de yenilenir, böylece pencere donmaz ("Yanıt vermiyor" olmaz).
        proc = subprocess.Popen([XFOIL_EXE], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, text=True, cwd=PROJE_KLASORU)
        try:
            proc.stdin.write(komutlar)
            proc.stdin.close()
        except OSError:
            pass
        baslangic = time.time()
        while proc.poll() is None:
            gecen = time.time() - baslangic
            if gecen > XFOIL_TIMEOUT:
                proc.kill()
                proc.wait()
                print(f"  [ZAMAN AŞIMI] {ad} Re={reynolds} (polar yarım kalmış olabilir, kontrol edilecek)")
                break
            if _bekle_fn:
                _bekle_fn(f"XFOIL çalışıyor: {ad}   Re={reynolds:,}   ({gecen:.0f} sn)", 0.1)
            else:
                time.sleep(0.1)
    except FileNotFoundError:
        _xfoil_var = False
        print(f"\n  [UYARI] XFOIL bulunamadı ('{XFOIL_EXE}'). Yeni polar üretilmeyecek;\n"
              f"          sadece klasörde HAZIR olan polarlarla devam ediliyor.\n"
              f"          XFOIL kuruluysa dosyanın başındaki XFOIL_EXE'ye tam yolunu yazın.\n")
        return None
    if gecici.exists() and gecici.stat().st_size > 0:
        for _ in range(10):   # Windows, durdurulan XFOIL'in dosyayı bırakmasını bekleyebilir
            try:
                os.replace(gecici, hedef)
                return hedef
            except PermissionError:
                time.sleep(0.5)
        print(f"  [UYARI] {gecici.name} adı değiştirilemedi, atlandı.")
        return None
    print(f"  [YAKINSAMADI] {ad} Re={reynolds}")
    return None


def polar_oku(dosya: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """XFOIL .pol dosyasından alpha, Cl, Cd sütunlarını okur (sıralı, tekrarsız)."""
    veri = {}
    basladi = False
    for satir in dosya.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "----" in satir:
            basladi = True
            continue
        if not basladi:
            continue
        parca = satir.split()
        if len(parca) >= 3:
            try:
                veri[float(parca[0])] = (float(parca[1]), float(parca[2]))
            except ValueError:
                continue
    if len(veri) < 5:
        raise ValueError(f"{dosya.name}: yeterli veri yok ({len(veri)} nokta)")
    alfa = np.array(sorted(veri))
    cl = np.array([veri[a][0] for a in alfa])
    cd = np.array([veri[a][1] for a in alfa])
    return alfa, cl, cd


def profil_ozellikleri(alfa, cl, cd) -> dict:
    """Polardan LLT'nin ihtiyaç duyduğu sayıları çıkarır."""
    maske = (alfa >= LINEER_ALPHA_MIN) & (alfa <= LINEER_ALPHA_MAX)
    if maske.sum() < 3:
        maske = np.ones_like(alfa, dtype=bool)
    a0, b = np.polyfit(np.deg2rad(alfa[maske]), cl[maske], 1)
    if a0 <= 0:
        raise ValueError("taşıma eğimi pozitif değil")
    i_max = int(np.argmax(cl))
    return {
        "a0_rad": float(a0),                       # 2B taşıma eğimi (1/rad)
        "alpha_l0_rad": float(-b / a0),            # sıfır taşıma açısı
        "cl_max_2d": float(cl[i_max]),
        "alpha_stall_deg": float(alfa[i_max]),
        "stall_veride_mi": i_max < len(alfa) - 1,  # en son açıda max ise gerçek stall görülmemiş
        "cd_min_2d": float(cd.min()),
        "alfa": alfa, "cl": cl, "cd": cd,
    }


# ---------------------------------------------------------------------
# 2) LLT (orijinal lifting_line_cozucu.py ile aynı matematik)
# ---------------------------------------------------------------------
_llt_onbellek: dict = {}


def llt_dikdortgen_kanat(ar_w: float, a0_rad: float, n_terim: int = 25) -> tuple[float, float]:
    """Dikdörtgen kanat için 3B taşıma eğimi (1/rad) ve Oswald verimi e döndürür."""
    anahtar = (round(ar_w, 6), round(a0_rad, 6), n_terim)
    if anahtar in _llt_onbellek:
        return _llt_onbellek[anahtar]
    i = np.arange(1, n_terim + 1)
    theta = i * np.pi / (2.0 * n_terim)
    n = 2 * i - 1
    m = np.sin(np.outer(theta, n)) * ((4.0 * ar_w) / a0_rad + n[np.newaxis, :] / np.sin(theta)[:, np.newaxis])
    A = np.linalg.solve(m, np.ones(n_terim))
    cl_a = float(np.pi * ar_w * A[0])
    e = 1.0 / (1.0 + float(np.sum(n[1:] * (A[1:] / A[0]) ** 2)))
    _llt_onbellek[anahtar] = (cl_a, e)
    return cl_a, e


def aday_coz(aday: dict, prof: dict, reynolds_polar: int) -> dict:
    ar = aday["AR_w"]
    cl_a, e = llt_dikdortgen_kanat(ar, prof["a0_rad"])
    alfa_s = np.deg2rad(SEYIR_ALPHA_DEG)
    CL = cl_a * (alfa_s - prof["alpha_l0_rad"])
    # Kanadın profil sürüklemesi: kanadın seyirdeki CL'sinde airfoilin Cd'si (yaklaşık)
    cl_pol, cd_pol = prof["cl"], prof["cd"]
    lin = (prof["alfa"] >= ALPHA_MIN) & (prof["alfa"] <= prof["alpha_stall_deg"])
    sira = np.argsort(cl_pol[lin])
    CDp = float(np.interp(CL, cl_pol[lin][sira], cd_pol[lin][sira]))
    CDi = CL**2 / (np.pi * ar * e)
    CD = CDp + CDi
    return {
        "Re_polar": reynolds_polar,
        "a0_2d_per_deg": np.deg2rad(prof["a0_rad"]),
        "alpha_L0_deg": np.rad2deg(prof["alpha_l0_rad"]),
        "Cl_max_2d": prof["cl_max_2d"],
        "alpha_stall_2d_deg": prof["alpha_stall_deg"],
        "Cd_min_2d": prof["cd_min_2d"],
        "CL_alpha_3d_per_deg": np.deg2rad(cl_a),
        "e_oswald": e,
        "CL_seyir": CL,
        "CDp_seyir": CDp,
        "CDi_seyir": CDi,
        "CD_seyir": CD,
        "L_D_seyir": CL / CD if CD > 0 else np.nan,
        "CL_max_3d_tahmini": 0.9 * prof["cl_max_2d"],   # kaba kural: 3B ≈ 0.9 x 2B
    }


SAYISAL_SONUCLAR = [
    "a0_2d_per_deg", "alpha_L0_deg", "Cl_max_2d", "alpha_stall_2d_deg", "Cd_min_2d",
    "CL_alpha_3d_per_deg", "e_oswald", "CL_seyir", "CDp_seyir", "CDi_seyir", "CD_seyir",
    "L_D_seyir", "CL_max_3d_tahmini",
]


def kuyruk_coz(ar: float, prof: dict) -> dict:
    """Kuyruk yüzeyi (yatay ya da dikey) için LLT.
    Seyirde kuyruğun taşıma üretmediği (CL=0) kabul edilir; bu yüzden
    sürüklemesi airfoilin Cl=0'daki Cd'sidir (Cd0)."""
    cl_a, e = llt_dikdortgen_kanat(ar, prof["a0_rad"])
    lin = (prof["alfa"] >= ALPHA_MIN) & (prof["alfa"] <= prof["alpha_stall_deg"])
    sira = np.argsort(prof["cl"][lin])
    cd0 = float(np.interp(0.0, prof["cl"][lin][sira], prof["cd"][lin][sira]))
    return {"CL_alpha_per_deg": np.deg2rad(cl_a), "e": e, "Cd0": cd0}


KUYRUK_SONUCLARI = ["CL_alpha_per_deg", "e", "Cd0"]


def re_interpolasyonu(sonuclar_re: dict[int, dict], re_hedef: float,
                      anahtarlar: list[str] | None = None) -> tuple[dict, str]:
    """Farklı Re sonuçları arasında log(Re)'ye göre doğrusal ara değer alır."""
    anahtarlar = anahtarlar or SAYISAL_SONUCLAR
    re_ler = sorted(sonuclar_re)
    if re_hedef <= re_ler[0]:
        return dict(sonuclar_re[re_ler[0]]), f"ARALIK_ALTI (en yakın Re={re_ler[0]})"
    if re_hedef >= re_ler[-1]:
        return dict(sonuclar_re[re_ler[-1]]), f"ARALIK_USTU (en yakın Re={re_ler[-1]})"
    for r1, r2 in zip(re_ler, re_ler[1:]):
        if r1 <= re_hedef <= r2:
            w = (np.log(re_hedef) - np.log(r1)) / (np.log(r2) - np.log(r1))
            s1, s2 = sonuclar_re[r1], sonuclar_re[r2]
            return {k: (1 - w) * s1[k] + w * s2[k] for k in anahtarlar}, f"ara deger {r1}-{r2}"
    raise RuntimeError("beklenmeyen durum")


# ---------------------------------------------------------------------
# 3) GRAFİK: Reynolds'un airfoile etkisi
# ---------------------------------------------------------------------
def re_grafigi_ciz(profil_adi: str, profiller: dict[int, dict], klasor: Path) -> None:
    # Ekran penceresinden bağımsız çizim (canlı grafikle çakışmasın diye pyplot kullanmıyoruz)
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12, 5), dpi=110)
    ax1, ax2 = fig.subplots(1, 2)
    for re_, p in sorted(profiller.items()):
        ax1.plot(p["alfa"], p["cl"], label=f"Re={re_:,}")
        ax2.plot(p["cd"], p["cl"], label=f"Re={re_:,}")
    ax1.set(xlabel="Hücum açısı α (°)", ylabel="Cl", title=f"{profil_adi}: Cl-α")
    ax2.set(xlabel="Cd", ylabel="Cl", title=f"{profil_adi}: Cl-Cd (sürükleme poları)")
    for ax in (ax1, ax2):
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(klasor / f"{profil_adi}_re_etkisi.png")


class CanliGrafik:
    """Program çalışırken ekranda güncellenen pencere.
    Pencere kapatılırsa program çalışmaya devam eder; grafik hatası hesabı asla durdurmaz."""

    def __init__(self, re_listesi: list[int]):
        self.acik = False
        self.re_listesi = re_listesi
        self.tam = {r: 0 for r in re_listesi}
        self.yarim = {r: 0 for r in re_listesi}
        try:
            import matplotlib as mpl
            # Güncellenince pencere öne fırlamasın: küçültülmüşse küçük, arkadaysa arkada kalsın
            mpl.rcParams["figure.raise_window"] = False
            import matplotlib.pyplot as plt
            self.plt = plt
            plt.ion()
            self.fig, self.ax = plt.subplots(1, 3, figsize=(16, 5.2),
                                             gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
            try:
                self.fig.canvas.manager.set_window_title("LLT Çoklu Reynolds - canlı izleme")
            except Exception:
                pass
            self.durum = self.fig.text(0.01, 0.005, "", fontsize=9, color="dimgray")
            self._son_mesaj = ""
            plt.show(block=False)          # ilk açılış (sadece bir kez)
            self.fig.canvas.flush_events()
            self.acik = True
        except Exception as e:
            print(f"  [BİLGİ] Canlı grafik açılamadı ({e}); program grafiksiz devam ediyor.")

    def guncelle(self, sira: str, ad: str, tamlar: dict[int, dict], yarimlar: list[tuple],
                 sayaca_ekle: bool = True, bitti: bool = True) -> None:
        if sayaca_ekle:
            for r in tamlar:
                if r in self.tam:
                    self.tam[r] += 1
            for r, *_ in yarimlar:
                if r in self.yarim:
                    self.yarim[r] += 1
        if not self.acik:
            return
        try:
            if not self.plt.fignum_exists(self.fig.number):
                self.acik = False   # kullanıcı pencereyi kapattı
                return
            ax1, ax2, ax3 = self.ax
            for a in self.ax:
                a.clear()
            renkler = self.plt.cm.viridis(np.linspace(0, 0.9, max(len(self.re_listesi), 2)))
            renk = {r: renkler[i] for i, r in enumerate(sorted(self.re_listesi))}
            for r, pr in sorted(tamlar.items()):
                ax1.plot(pr["alfa"], pr["cl"], color=renk.get(r), label=f"Re={r:,}")
                ax2.plot(pr["cd"], pr["cl"], color=renk.get(r), label=f"Re={r:,}")
            for r, alfa, cl, cd in yarimlar:
                ax1.plot(alfa, cl, "r--", lw=1.5, label=f"Re={r:,} YARIM")
                ax2.plot(cd, cl, "r--", lw=1.5, label=f"Re={r:,} YARIM")
            ax1.set(xlabel="Hücum açısı α (°)", ylabel="Cl", title="Cl – α", xlim=(-6, 16))
            ax2.set(xlabel="Cd", ylabel="Cl", title="Cl – Cd")
            for a in (ax1, ax2):
                a.grid(True, ls="--", alpha=0.5)
                if a.lines:
                    a.legend(fontsize=7, loc="lower right")
            x = np.arange(len(self.re_listesi))
            t = [self.tam[r] for r in self.re_listesi]
            y = [self.yarim[r] for r in self.re_listesi]
            ax3.bar(x, t, color="seagreen", label="tam")
            ax3.bar(x, y, bottom=t, color="indianred", label="yarım (kullanılmadı)")
            ax3.set_xticks(x, [f"{r // 1000}k" for r in self.re_listesi])
            ax3.set(title="Şimdiye kadar: Re'ye göre polar sayısı", ylabel="airfoil sayısı")
            ax3.legend(fontsize=8)
            durum = "bitti" if bitti else "işleniyor..."
            self.fig.suptitle(f"{sira}  {ad}  ({durum})   |   bu airfoil: {len(tamlar)} tam, "
                              f"{len(yarimlar)} yarım", fontsize=12, fontweight="bold")
            self.fig.tight_layout()
            # "Boş vakit bulunca çiz" yerine ANINDA çiz (gecikme olmasın)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception as e:
            print(f"  [BİLGİ] Canlı grafik güncellenemedi ({e}); grafik kapatıldı.")
            self.acik = False

    def bekle(self, mesaj: str, sure: float) -> None:
        """XFOIL çalışırken çağrılır: pencereyi canlı tutar, altta durumu yazar."""
        if not self.acik:
            time.sleep(sure)
            return
        try:
            if not self.plt.fignum_exists(self.fig.number):
                self.acik = False
                time.sleep(sure)
                return
            if mesaj != self._son_mesaj:      # saniyede bir değişir; gereksiz çizim yok
                self.durum.set_text(mesaj)
                self._son_mesaj = mesaj
                self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self.fig.canvas.start_event_loop(sure)
        except Exception as e:
            print(f"  [BİLGİ] Canlı grafik beklerken hata verdi ({e}); grafik kapatıldı, hesap devam ediyor.")
            self.acik = False
            time.sleep(sure)

    def sonda_acik_tut(self) -> None:
        if self.acik and self.plt.fignum_exists(self.fig.number):
            self.durum.set_text("BİTTİ - sonuçlar kaydedildi. Pencereyi kapatabilirsiniz.")
            self.fig.canvas.draw_idle()
            print("\n(Canlı grafik penceresi açık. Kapattığınızda program tamamen biter.)")
            self.plt.ioff()
            self.plt.show()


# ---------------------------------------------------------------------
# 4) ANA AKIŞ
# ---------------------------------------------------------------------
def polarlari_hazirla(dat: Path, re_listesi: list[int], yarim_polarlar: list,
                      yarim_veri: list | None = None, ilerleme=None) -> dict[int, dict]:
    """Bir airfoil için listedeki her Re'nin polarını bulur/üretir, yarımları ayıklar.
    yarim_veri verilirse yarım polarların (Re, alfa, cl, cd) verisi buna eklenir (grafik için)."""
    sonuc: dict[int, dict] = {}
    if yarim_veri is None:
        yarim_veri = []
    for re_ in re_listesi:
        if ilerleme:
            ilerleme(sonuc, yarim_veri)   # her Re'den önce grafiği güncelle (yeni eğri anında görünsün)
        pol = polar_dosyasi_bul_veya_uret(dat, re_)
        if pol is None:
            continue
        try:
            alfa, cl, cd = polar_oku(pol)
        except ValueError as e:
            print(f"  [ATLANDI] {pol.name}: {e}")
            continue
        # Yarım polar kontrolü
        eksikler = []
        if alfa.min() > YARIM_EN_KUCUK_ALFA:
            eksikler.append(f"en küçük açı {alfa.min():g}°")
        if alfa.max() < YARIM_EN_BUYUK_ALFA:
            eksikler.append(f"en büyük açı {alfa.max():g}°")
        if len(alfa) < YARIM_EN_AZ_NOKTA:
            eksikler.append(f"{len(alfa)} nokta")
        if eksikler:
            print(f"  [YARIM POLAR - KULLANILMADI] {dat.stem} Re={re_}: {', '.join(eksikler)}")
            yarim_polarlar.append((dat.stem, re_, ", ".join(eksikler), pol))
            yarim_veri.append((re_, alfa, cl, cd))
            continue
        try:
            sonuc[re_] = profil_ozellikleri(alfa, cl, cd)
        except ValueError as e:
            print(f"  [ATLANDI] {pol.name}: {e}")
    return sonuc


_kilit_dosyasi = None


def tek_kopya_kilidi() -> None:
    """Program aynı anda iki kez çalışmasın diye kilit alır.
    Kilit, program kapanınca (çökse bile) işletim sistemi tarafından bırakılır."""
    global _kilit_dosyasi
    yol = PROJE_KLASORU / ".llt_coklu_reynolds.lock"
    f = open(yol, "a+")
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        sys.exit("\n[DURDU] Program zaten başka bir pencerede/terminalde çalışıyor!\n"
                 "        Aynı anda iki kopya polarları birbirine karıştırır.\n"
                 "        Önce diğerini kapatın (Ctrl+C), sonra tekrar çalıştırın.")
    _kilit_dosyasi = f
    # Önceki çökmüş bir çalıştırmadan kalan geçici dosyaları temizle
    if POLAR_KLASORU.exists():
        for g in POLAR_KLASORU.glob("*.yaziliyor"):
            try:
                g.unlink()
            except OSError:
                pass


def csv_yaz(satirlar: list[dict], yol: Path) -> None:
    with yol.open("w", newline="", encoding="utf-8-sig") as f:
        y = csv.DictWriter(f, fieldnames=list(satirlar[0].keys()))
        y.writeheader()
        for s in satirlar:
            y.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in s.items()})


def main() -> None:
    tek_kopya_kilidi()
    nu = MU / RHO
    adaylar = kb.tum_adaylari_uret(GOVDE_UZUNLUGU_M)
    kuyruk_dat, kanat_datlari = kb.airfoilleri_yukle()
    if TEST_AIRFOIL_SAYISI:
        kanat_datlari = kanat_datlari[:TEST_AIRFOIL_SAYISI]
        secilen = {d.stem for d in kanat_datlari}
        adaylar = [a for a in adaylar if a["kanat_airfoil"] in secilen]
        print(f"*** TEST MODU: sadece {len(kanat_datlari)} airfoil kullanılıyor "
              f"(TEST_AIRFOIL_SAYISI = None yapınca hepsi) ***")
    print(f"Kanat airfoili: {len(kanat_datlari)} | Aday: {len(adaylar)} | "
          f"Reynolds listesi: {REYNOLDS_LISTESI}")

    re_adaylar = [SEYIR_HIZI_MS * a["c_root_w_m"] / nu for a in adaylar]
    print(f"Adayların gerçek Re aralığı (V={SEYIR_HIZI_MS} m/s): "
          f"{min(re_adaylar):,.0f} - {max(re_adaylar):,.0f}")
    if min(re_adaylar) < min(REYNOLDS_LISTESI) or max(re_adaylar) > max(REYNOLDS_LISTESI):
        print("  [UYARI] Bazı adaylar REYNOLDS_LISTESI aralığının dışında; "
              "listeyi genişletmeniz önerilir.")

    re_ht = [SEYIR_HIZI_MS * a["c_root_HT_m"] / nu for a in adaylar]
    re_vt = [SEYIR_HIZI_MS * a["c_root_VT_m"] / nu for a in adaylar]
    print(f"Yatay kuyruk Re aralığı: {min(re_ht):,.0f} - {max(re_ht):,.0f}")
    print(f"Dikey kuyruk Re aralığı: {min(re_vt):,.0f} - {max(re_vt):,.0f}")
    if (min(min(re_ht), min(re_vt)) < min(KUYRUK_REYNOLDS_LISTESI)
            or max(max(re_ht), max(re_vt)) > max(KUYRUK_REYNOLDS_LISTESI)):
        print("  [UYARI] Bazı kuyruklar KUYRUK_REYNOLDS_LISTESI aralığının dışında.")

    # Her airfoil x her Re için polar özelliklerini hazırla
    profiller: dict[str, dict[int, dict]] = {}
    yarim_polarlar: list[tuple] = []
    global _bekle_fn
    canli = CanliGrafik(REYNOLDS_LISTESI) if CANLI_GRAFIK else None
    _bekle_fn = canli.bekle if canli else None
    for i, dat in enumerate(kanat_datlari, 1):
        print(f"[{i}/{len(kanat_datlari)}] Polarlar: {dat.stem}")
        yarim_veri: list = []
        sira = f"[{i}/{len(kanat_datlari)}]"
        ilerleme = None
        if canli:
            ilerleme = (lambda tam, yar, _s=sira, _a=dat.stem:
                        canli.guncelle(_s, _a, tam, yar, sayaca_ekle=False, bitti=False))
        p = polarlari_hazirla(dat, REYNOLDS_LISTESI, yarim_polarlar, yarim_veri, ilerleme)
        if p:
            profiller[dat.stem] = p
        if canli:
            canli.guncelle(sira, dat.stem, p, yarim_veri)

    print("\nKullanılan TAM polar sayısı (Reynolds'a göre):")
    for re_ in REYNOLDS_LISTESI:
        adet = sum(1 for p in profiller.values() if re_ in p)
        yarim = sum(1 for y in yarim_polarlar if y[1] == re_)
        print(f"  Re={re_:>7}: {adet}/{len(kanat_datlari)} airfoil"
              + (f"   (+{yarim} yarım, kullanılmadı)" if yarim else ""))
    if yarim_polarlar:
        print("\nYARIM POLARLAR (hesaba katılmadı):")
        for ad_, re_, neden, pol in yarim_polarlar:
            print(f"  {ad_:<16} Re={re_:>7}  -> {neden}")
        print("  Not: Tekrar denemek için bu .pol dosyalarını klasörden çıkarıp programı yeniden çalıştırın.")
    if not profiller:
        print("\n[HATA] Hiç polar bulunamadı. Aranan klasörler:")
        for k in (POLAR_KLASORU, *ESKI_POLAR_KLASORLERI):
            print(f"  {k}  -> {'VAR' if k.exists() else 'YOK'}")
        sys.exit(1)

    print(f"\nKuyruk polarları: {kuyruk_dat.stem}")
    k_yarim_veri: list = []
    k_ilerleme = None
    if canli:
        k_ilerleme = (lambda tam, yar: canli.guncelle("[KUYRUK]", kuyruk_dat.stem, tam, yar,
                                                       sayaca_ekle=False, bitti=False))
    kuyruk_prof = polarlari_hazirla(kuyruk_dat, KUYRUK_REYNOLDS_LISTESI, yarim_polarlar,
                                    k_yarim_veri, k_ilerleme)
    if canli:
        canli.guncelle("[KUYRUK]", kuyruk_dat.stem, kuyruk_prof, k_yarim_veri, sayaca_ekle=False)
    print(f"  Kuyruk için kullanılan Re'ler: {sorted(kuyruk_prof) or 'YOK'}")
    if not kuyruk_prof:
        print("  [UYARI] Kuyruk poları yok; kuyruk sütunları boş kalacak.")

    if GRAFIK_CIZ:
        if kuyruk_prof:
            profiller_ve_kuyruk = {**profiller, kuyruk_dat.stem: kuyruk_prof}
        else:
            profiller_ve_kuyruk = profiller
        gk = PROJE_KLASORU / "re_etkisi_grafikleri"
        gk.mkdir(exist_ok=True)
        for ad, p in profiller_ve_kuyruk.items():
            re_grafigi_ciz(ad, p, gk)

    tum, aday_re = [], []
    for no, (aday, re_gercek, r_ht, r_vt) in enumerate(zip(adaylar, re_adaylar, re_ht, re_vt), start=1):
        ad = aday["kanat_airfoil"]
        if ad not in profiller:
            continue
        temel = {
            "aday_no": no, "profil": ad, "b_w_m": aday["b_w_m"], "AR_w": aday["AR_w"],
            "S_w_m2": aday["S_w_m2"], "c_w_m": aday["c_root_w_m"], "Re_aday": re_gercek,
        }
        sonuclar_re = {re_: aday_coz(aday, prof, re_) for re_, prof in profiller[ad].items()}
        for s in sonuclar_re.values():
            tum.append({**temel, **s})
        ara, durum = re_interpolasyonu(sonuclar_re, re_gercek)
        ara.pop("Re_polar", None)
        satir = {**temel, "Re_durumu": durum, **ara}

        # --- KUYRUK: her yüzey kendi veterinin Re'sinde ---
        kuyruk = {"c_HT_m": aday["c_root_HT_m"], "S_HT_m2": aday["S_HT_m2"], "Re_HT": r_ht,
                  "c_VT_m": aday["c_root_VT_m"], "S_VT_m2": aday["S_VT_m2"], "Re_VT": r_vt}
        if kuyruk_prof:
            for ek, r_k, ar_k in (("HT", r_ht, aday["AR_HT"]), ("VT", r_vt, aday["AR_VT"])):
                sk = {re_: kuyruk_coz(ar_k, pr) for re_, pr in kuyruk_prof.items()}
                k_ara, k_durum = re_interpolasyonu(sk, r_k, KUYRUK_SONUCLARI)
                kuyruk[f"Re_{ek}_durumu"] = k_durum
                kuyruk[f"CL_alpha_{ek}_per_deg"] = k_ara["CL_alpha_per_deg"]
                kuyruk[f"e_{ek}"] = k_ara["e"]
                kuyruk[f"Cd0_{ek}"] = k_ara["Cd0"]
            # Kuyruk sürüklemesi, kanat alanına göre (kanat CD'sine eklenebilsin diye)
            cd_kuyruk = (kuyruk["Cd0_HT"] * aday["S_HT_m2"]
                         + kuyruk["Cd0_VT"] * aday["S_VT_m2"]) / aday["S_w_m2"]
            kuyruk["CD_kuyruk"] = cd_kuyruk
            kuyruk["CD_kanat_kuyruk"] = ara["CD_seyir"] + cd_kuyruk
            kuyruk["L_D_kanat_kuyruk"] = ara["CL_seyir"] / kuyruk["CD_kanat_kuyruk"]
        aday_re.append({**satir, **kuyruk})

    if not tum:
        sys.exit("Hiç sonuç üretilemedi (polar bulunamadı?).")
    csv_yaz(tum, PROJE_KLASORU / "llt_coklu_re_tum_sonuclar.csv")
    csv_yaz(aday_re, PROJE_KLASORU / "llt_coklu_re_aday_re.csv")

    anahtar = "L_D_kanat_kuyruk" if kuyruk_prof else "L_D_seyir"
    en_iyi = sorted(aday_re, key=lambda s: -s[anahtar])[:5]
    print(f"\nKendi Re'sinde en yüksek L/D'ye sahip 5 aday ({'kanat + kuyruk' if kuyruk_prof else 'sadece kanat'}):")
    for s in en_iyi:
        satir = (f"  #{s['aday_no']:<5} {s['profil']:<14} b={s['b_w_m']:.2f} AR={s['AR_w']} "
                 f"c={s['c_w_m']:.3f} Re={s['Re_aday']:,.0f}  CL={s['CL_seyir']:.3f}  "
                 f"L/D(kanat)={s['L_D_seyir']:.1f}")
        if kuyruk_prof:
            satir += (f"  | Re_HT={s['Re_HT']:,.0f} Re_VT={s['Re_VT']:,.0f}"
                      f"  L/D(kanat+kuyruk)={s['L_D_kanat_kuyruk']:.1f}")
        print(satir)
    print(f"\nKaydedildi: {PROJE_KLASORU / 'llt_coklu_re_tum_sonuclar.csv'}")
    print(f"Kaydedildi: {PROJE_KLASORU / 'llt_coklu_re_aday_re.csv'}")
    if canli:
        canli.sonda_acik_tut()


if __name__ == "__main__":
    main()
