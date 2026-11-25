#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[YENİ DOSYA - 2026-11-18 | Görev A: VSPAERO'nun koda entegrasyonu]

OpenVSP Python API'si ile aday kanadı (isteğe bağlı yatay kuyrukla) kurar,
VSPAERO'yu VLM (Vortex Lattice) modunda bir hücum açısı taramasıyla çalıştırır
ve sonuçları `vlm_entegrasyonu.py`'nin beklediği sözlük formatında döndürür.

Neden ayrı dosya?
- `vlm_entegrasyonu.py` içindeki eski `vspaero_cli_coz` .vspaero girdi dosyasını
  elle yazıyordu, .polar dosyasında yanlış sütunları okuyordu (AoA yerine Mach),
  kuyruk/profil eklemiyordu ve flaba gerçek açı yerine Δα0 veriyordu. Eski hali
  `yedekler/vlm_entegrasyonu_YEDEK_2026-11-18_gorevA_oncesi.py` içinde duruyor.

Modelleme kararları (Görev B / arayüz tarafı AYNI şekilde kurmalı):
- Kanat: dikdörtgen, sweep=0, dihedral=0, twist=0, kök hücum kenarı x=0.
- Profil: `airfoils/<profil>.dat` dosyası (VSPAERO VLM profilin kamburluğunu kullanır).
- Flap: kontrol yüzeyi yerine iç açıklıkta (|y| <= b_flap/2) flaplı profil dosyası
  `airfoils_flap/<profil>_flap.dat` kullanılır. Bu dosyalarda flap zaten çizili:
  menteşe ~%70 veter (flap veteri %30), sapma ~30 derece. Flaplı ve temiz kesit
  arasında 5 mm'lik bir geçiş kesiti vardır.
- Varsayılan olarak SADECE KANAT çözülür (LLT tablolarındaki kanat katsayılarıyla
  birebir karşılaştırılabilsin diye; kuyruk katsayıları LLT tablosunda ayrıca var).
  `kuyruk_dahil=True` ile yatay kuyruk da eklenir (kuyruk 0 derece, z=0,
  kuyruk 1/4 veteri kanat 1/4 veterinden `tail_arm_m` geride).
- Referanslar: Sref = S_w, cref = c_w, bref = b_w, Xcg = 0.25*c_w.
- Sweep: alfa = -2 ... 10 derece (7 nokta), Mach 0.06, sürtünmesiz (CL, CDi).
- CL_max VSPAERO'dan GELMEZ (sürtünmesiz çözücü stall bulamaz); eski kodla aynı
  2B ağırlıklı tahmin kullanılır ve `CL_max_kaynak` sütununda bu belirtilir.

Ortam değişkenleri (Windows'ta gerekirse):
- OPENVSP_PYTHON_YOLU : `openvsp` paketinin bulunduğu klasör (ör. ...\\OpenVSP-3.xx\\python)
- VSPAERO_YOLU        : vspaero.exe'nin bulunduğu klasör (ör. ...\\OpenVSP-3.xx)
"""

from __future__ import annotations

import glob
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

ALFA_TARAMASI_DERECE: Tuple[float, ...] = (-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0)
MACH_SAYISI = 0.06
GECIS_ACIKLIGI_M = 0.005          # flaplı -> temiz profil geçiş kesiti
FLAP_ACISI_BILGI_DERECE = 30.0    # sadece bilgi: açı flaplı .dat dosyasına gömülü
VARSAYILAN_MODEL_KLASORU = SCRIPT_DIR / "vlm_sonuclari" / "vspaero_modelleri"

_VSP_MODUL: Any = None


# ---------------------------------------------------------------------------
# Kurulum kontrolü
# ---------------------------------------------------------------------------

def openvsp_yukle() -> Any:
    """`openvsp` Python paketini yükler (gerekirse OPENVSP_PYTHON_YOLU'nu sys.path'e ekler)."""
    global _VSP_MODUL
    if _VSP_MODUL is not None:
        return _VSP_MODUL
    ek = os.environ.get("OPENVSP_PYTHON_YOLU", "").strip()
    if ek:
        for p in ek.split(os.pathsep):
            if p and p not in sys.path:
                sys.path.insert(0, p)
    try:
        import openvsp as vsp  # type: ignore
    except Exception as exc:  # ImportError veya DLL hatası
        raise ImportError(
            f"'openvsp' Python paketi yüklenemedi ({exc}). OpenVSP kurulum klasöründeki "
            "python paketini kurun veya OPENVSP_PYTHON_YOLU ortam değişkenini ayarlayın."
        ) from exc
    _VSP_MODUL = vsp
    return vsp


def vspaero_klasoru_bul(vsp: Any) -> Optional[Path]:
    """vspaero(.exe) dosyasının bulunduğu klasörü arar."""
    adaylar: List[Path] = []
    env = os.environ.get("VSPAERO_YOLU", "").strip()
    if env:
        adaylar.append(Path(env))
    for ad in ("vspaero", "vspaero.exe"):
        w = shutil.which(ad)
        if w:
            adaylar.append(Path(w))
    try:
        adaylar.append(Path(vsp.GetVSPExePath()))
    except Exception:
        pass
    for a in adaylar:
        klasor = a if a.is_dir() else a.parent
        for ad in ("vspaero.exe", "vspaero"):
            if (klasor / ad).is_file():
                return klasor
    return None


def vspaero_hazir_mi() -> Tuple[bool, str]:
    """(hazır_mı, açıklama) döndürür. Hiçbir zaman sessizce başarısız olmaz."""
    try:
        vsp = openvsp_yukle()
    except ImportError as exc:
        return False, str(exc)
    klasor = vspaero_klasoru_bul(vsp)
    if klasor is None:
        return False, (
            "vspaero(.exe) bulunamadı. PATH'e ekleyin veya VSPAERO_YOLU ortam "
            "değişkenine OpenVSP klasörünü yazın."
        )
    try:
        vsp.SetVSPAEROPath(str(klasor))
    except Exception:
        pass
    try:
        surum = vsp.GetVSPVersion()
    except Exception:
        surum = "?"
    return True, f"{surum} | VSPAERO klasörü: {klasor}"


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def _ascii_mi(yol: Path) -> bool:
    try:
        str(yol).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def ascii_calisma_klasoru(hedef: Path) -> Path:
    """
    [DEĞİŞİKLİK 2026-11-18 v3 | Türkçe karakterli yol] OpenVSP/VSPAERO (C++) Türkçe karakterli yolları (ör. 'boyutlandırma')
    okuyamıyor/yazamıyor: ReadFileAirfoil 'Error Code: 9', VSPAERO 'Unable to create
    case file' hatası veriyor. Hedef klasör ASCII değilse VSPAERO İngilizce karakterli
    bir geçici klasörde çalıştırılır, sonuç dosyaları sonra hedef klasöre kopyalanır.
    Öncelik: hedef (ASCII ise) -> %TEMP%\\ata_vspaero -> C:\\ata_vspaero_calisma (Windows).
    """
    if _ascii_mi(hedef):
        return hedef
    adaylar = [Path(tempfile.gettempdir()) / "ata_vspaero"]
    if os.name == "nt":
        adaylar.append(Path("C:/ata_vspaero_calisma"))
    else:
        adaylar.append(Path("/tmp/ata_vspaero_calisma"))
    for a in adaylar:
        if _ascii_mi(a):
            a.mkdir(parents=True, exist_ok=True)
            return a
    raise RuntimeError(
        "Türkçe karaktersiz bir çalışma klasörü bulunamadı. VSPAERO_CALISMA_KLASORU "
        "ortam değişkenine C:\\vspaero gibi bir klasör yazın."
    )

def profil_dosyasi_bul(profil: str, flapli: bool = False) -> Optional[Path]:
    profil = Path(str(profil)).stem
    ad = f"{profil}_flap.dat" if flapli else f"{profil}.dat"
    klasor_adi = "airfoils_flap" if flapli else "airfoils"
    for k in (REPO_ROOT / klasor_adi, SCRIPT_DIR / klasor_adi):
        if (k / ad).is_file():
            return k / ad
    return None


# [DEĞİŞİKLİK 2026-11-18 v6 | inceleme-2 düzeltmeleri] Bu bölüm (kesit / kanat / gövde kurucuları)
# YENİDEN YAZILDI. Önceki hali: yedekler/vspaero_cozucu_YEDEK_2026-11-18_v5_duzeltme_oncesi.py
# Gövde incelemesine (GOVDE_INCELEME_RAPORU.txt) göre değişenler:
#  - [K1] Gövde kesitlerinin teğet (eğim) açıları açıkça ayarlanıyor: silindirde 0°, konide doğrusal.
#         Eski halde OpenVSP varsayılanı ±45° kalıyor, gövde 0.14 m yerine 0.257 m genişliğe şişiyordu.
#         Ayrıca gövde genişliği ve kesit konumları kurulumdan sonra KONTROL ediliyor (yanlışsa HATA).
#  - [K2] Panel sayısı artık yarım kanadın TOPLAM şerit sayısı; bölümlere boylarıyla orantılı dağıtılır
#         (flaplı/flapsız kanat aynı ağ yoğunluğunu alır). Uç sıklaştırma (OutCluster) parametresi eklendi.
#  - [3. adım] Kanat istenen açıklık konumlarında ek bölüm sınırı alabilir (gövdeli modelde y = R otomatik).
#  - [O8] Flap iç kenarı parametre (merkez = 0 veya gövde kenarı = R).
#  - [K3] Kanadın dikey yeri profil kalınlığından hesaplanır: ust / ust_gomulu / orta / alt / alt_gomulu / sayı.

_SPAN_PANEL: Optional[int] = None  # son çağrıdaki yarım-kanat panel sayısı (model adı için)


def _kesit_ayarla(vsp: Any, gid: str, idx: int, span: float, chord: float,
                  tess_u: Optional[int] = None, out_cluster: Optional[float] = None) -> None:
    """Kanat bölümünü (XSec_idx) dikdörtgen, sıfır sweep/dihedral/twist olarak ayarlar."""
    vsp.SetDriverGroup(gid, idx, vsp.SPAN_WSECT_DRIVER, vsp.ROOTC_WSECT_DRIVER, vsp.TIPC_WSECT_DRIVER)
    grup = f"XSec_{idx}"
    vsp.SetParmVal(gid, "Span", grup, float(span))
    vsp.SetParmVal(gid, "Root_Chord", grup, float(chord))
    vsp.SetParmVal(gid, "Tip_Chord", grup, float(chord))
    for parm in ("Sweep", "Dihedral", "Twist"):
        try:
            vsp.SetParmVal(gid, parm, grup, 0.0)
        except Exception:
            pass
    if tess_u is not None:
        try:
            vsp.SetParmVal(gid, "SectTess_U", grup, float(tess_u))
        except Exception:
            pass
    if out_cluster is not None:  # uç sıklaştırma (<1 = uca doğru sıklaşır)
        try:
            vsp.SetParmVal(gid, "OutCluster", grup, float(out_cluster))
        except Exception:
            print(f"[VSPAERO] Uyarı: OutCluster ayarlanamadı ({grup})")
    vsp.Update()


def _profil_ata(vsp: Any, gid: str, xsec_idx: int, dosya: Path) -> None:
    xs_surf = vsp.GetXSecSurf(gid, 0)
    vsp.ChangeXSecShape(xs_surf, xsec_idx, vsp.XS_FILE_AIRFOIL)
    xs = vsp.GetXSec(xs_surf, xsec_idx)
    vsp.ReadFileAirfoil(xs, str(dosya))
    vsp.Update()


def kanat_sinirlarini_hesapla(
    yari_aciklik: float,
    flap_ic: float = 0.0,
    flap_dis: float = 0.0,
    ek_sinirlar: Sequence[float] = (),
    gecis: float = GECIS_ACIKLIGI_M,
) -> Tuple[List[float], bool, float, float]:
    """
    Yarım kanadın bölüm sınırlarını (y, m) döndürür: (sınırlar, flap_var, flap_ic, flap_dis).
    Flaplı bölge [flap_ic, flap_dis]; profilin değiştiği her kenara `gecis` boyunda geçiş bölümü konur.
    """
    pts = {0.0, float(yari_aciklik)}
    flap_var = (flap_dis - flap_ic) > 1e-3
    if flap_var:
        if flap_dis >= yari_aciklik - gecis - 1e-3:  # uca kadar flap: dış geçiş yok
            flap_dis = float(yari_aciklik)
        pts |= {float(flap_ic), float(flap_dis)}
        if flap_dis < yari_aciklik - 1e-6:
            pts.add(float(flap_dis) + gecis)
        if flap_ic > gecis + 1e-3:
            pts.add(float(flap_ic) - gecis)
    for y in ek_sinirlar:
        if 1e-3 < float(y) < yari_aciklik - 1e-3:
            pts.add(float(y))
    sirali: List[float] = []
    for y in sorted(pts):
        if not sirali or y - sirali[-1] > 1e-4:
            sirali.append(y)
    return sirali, flap_var, float(flap_ic), float(flap_dis)


def _kanat_kur(
    vsp: Any,
    ad: str,
    yari_aciklik: float,
    veter: float,
    temiz_profil: Path,
    flap_ic: float = 0.0,
    flap_dis: float = 0.0,
    flapli_profil: Optional[Path] = None,
    x_konum: float = 0.0,
    z_konum: float = 0.0,
    ek_sinirlar: Sequence[float] = (),
    span_panel: Optional[int] = None,
    uc_sik: Optional[float] = None,
    veter_tess: Optional[int] = None,  # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson] veter yönü panel (OpenVSP "Tess_W", grup "Shape"; varsayılan 33)
) -> str:
    """
    Simetrik dikdörtgen kanat. Bölümler kanat_sinirlarini_hesapla() ile belirlenir; flaplı bölgedeki
    kesitlere flaplı profil, diğerlerine temiz profil atanır. Panel ayarı verilmezse (span_panel=None)
    yalnızca 5 mm geçiş bölümlerine 2 şerit verilir, diğerleri OpenVSP varsayılanında kalır
    (gövdesiz eski modellerle BİREBİR aynı geometri/ağ).
    """
    gid = vsp.AddGeom("WING")
    vsp.SetGeomName(gid, ad)
    if x_konum:
        vsp.SetParmVal(gid, "X_Rel_Location", "XForm", float(x_konum))
    if z_konum:
        vsp.SetParmVal(gid, "Z_Rel_Location", "XForm", float(z_konum))
    if veter_tess:  # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson]
        vsp.SetParmVal(gid, "Tess_W", "Shape", float(veter_tess))
        vsp.Update()

    if flapli_profil is None:
        flap_dis = flap_ic  # flap yok
    pts, flap_var, flap_ic, flap_dis = kanat_sinirlarini_hesapla(yari_aciklik, flap_ic, flap_dis, ek_sinirlar)
    n_bolum = len(pts) - 1

    # Varsayılan kanatta XSec_0 + XSec_1 (1 bölüm) var; eksik bölümleri ekle
    for k in range(n_bolum - 1):
        vsp.InsertXSec(gid, k + 1, vsp.XS_FILE_AIRFOIL)
    vsp.Update()

    for i in range(1, n_bolum + 1):
        span_i = pts[i] - pts[i - 1]
        gecis_bolumu = abs(span_i - GECIS_ACIKLIGI_M) < 1e-6
        if gecis_bolumu:
            tess = 2
        elif span_panel:
            tess = max(2, int(round(float(span_panel) * span_i / yari_aciklik)))
        else:
            tess = None
        _kesit_ayarla(vsp, gid, i, span_i, veter, tess_u=tess,
                      out_cluster=(uc_sik if (uc_sik and i == n_bolum) else None))

    def _profil_y(y: float) -> Path:
        if flap_var and flapli_profil is not None and (flap_ic - 1e-6) <= y <= (flap_dis + 1e-6):
            return flapli_profil
        return temiz_profil

    for i, y in enumerate(pts):
        _profil_ata(vsp, gid, i, _profil_y(y))
    return gid


# --- Gövde -------------------------------------------------------------------
# Gövde, govde_boyutlandirma.py'deki tanımla birebir:
#   [0, L_av]            aviyonik silindiri, çap D
#   [L_av, L-L_kuy]      su haznesi silindiri, çap D   (L_su = L - L_av - L_kuy)
#   [L-L_kuy, L]         kuyruk konisi, çap D -> D_min (doğrusal)
#   Burun ve kuyruk ucu: DÜZ KAPAK (ekip kararı: koddaki gibi)
GOVDE_VARSAYILAN = {"L_aviyonik": 0.30, "L_kuyruk": 1.00, "D_min": 0.08}
KANAT_Z_SECENEKLERI = ("ust", "ust_gomulu", "orta", "alt", "alt_gomulu")
VARSAYILAN_GOMME_M = 0.005  # kanadın gövdeye gömülme payı (VSPAERO kesişimi bulabilsin)


def govde_olculeri(D: float, L_toplam: float, L_aviyonik: float = 0.30, L_kuyruk: float = 1.00,
                   D_min: float = 0.08) -> Dict[str, Any]:
    """Gövde kesitlerinin x konumlarını (m) ve çaplarını döndürür; su haznesi ortasını da verir."""
    # [v6 | inceleme küçük] giriş kontrolü
    if D <= 0 or L_toplam <= 0 or D_min <= 0 or D_min > D:
        raise ValueError(f"Geçersiz gövde ölçüsü: D={D}, L={L_toplam}, D_min={D_min}")
    if L_toplam <= L_aviyonik + L_kuyruk + 1e-3:
        raise ValueError(
            f"Gövde boyu ({L_toplam:.3f} m) aviyonik + kuyruk konisinden ({L_aviyonik + L_kuyruk:.3f} m) "
            "kısa: su haznesi bölümü kalmıyor."
        )
    L_su = L_toplam - L_aviyonik - L_kuyruk
    x = [0.0, L_aviyonik, L_aviyonik + L_su, L_aviyonik + L_su + 0.5 * L_kuyruk, L_toplam]
    d = [D, D, D, 0.5 * (D + D_min), D_min]
    return {"x": x, "d": d, "L_su": L_su, "su_haznesi_orta_x": L_aviyonik + 0.5 * L_su}


def _govde_kur(vsp: Any, D: float, L_toplam: float, L_aviyonik: float = 0.30,
               L_kuyruk: float = 1.00, D_min: float = 0.08) -> str:
    """OpenVSP FUSELAGE: 5 dairesel kesit, doğrusal koni, düz uç kapakları, burun x=0. Kurulum kontrol edilir."""
    olc = govde_olculeri(D, L_toplam, L_aviyonik, L_kuyruk, D_min)
    fid = vsp.AddGeom("FUSELAGE")
    vsp.SetGeomName(fid, "Govde")
    vsp.SetParmVal(fid, "Length", "Design", float(L_toplam))
    vsp.Update()
    surf = vsp.GetXSecSurf(fid, 0)
    n = vsp.GetNumXSec(surf)
    while n < 5:
        vsp.InsertXSec(fid, n - 2, vsp.XS_CIRCLE)
        vsp.Update()
        n = vsp.GetNumXSec(surf)
    while n > 5:
        vsp.CutXSec(fid, 1)
        vsp.Update()
        n = vsp.GetNumXSec(surf)
    for i in range(5):
        vsp.ChangeXSecShape(surf, i, vsp.XS_CIRCLE)
    vsp.Update()
    for i in range(5):
        vsp.SetParmVal(fid, "Circle_Diameter", f"XSecCurve_{i}", float(olc["d"][i]))
        for parm in ("YLocPercent", "ZLocPercent"):
            try:
                vsp.SetParmVal(fid, parm, f"XSec_{i}", 0.0)
            except Exception:
                pass
    for _ in range(2):
        for i in (1, 2, 3):
            vsp.SetParmVal(fid, "XLocPercent", f"XSec_{i}", float(olc["x"][i] / L_toplam))
        vsp.Update()

    # [K1] Teğet açıları: OpenVSP varsayılan gövdesinde uçlar ±45° (sivri burun için). Silindirde 0°,
    # konide doğrusal iniş açısı (θ < 0 = yüzey eksene doğru iner). (sol, sağ) = kesitin önü, arkası.
    theta = -math.degrees(math.atan((D - D_min) / (2.0 * L_kuyruk))) if L_kuyruk > 0 else 0.0
    acilar = {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (0.0, theta), 3: (theta, theta), 4: (theta, theta)}
    for i, (sol, sag) in acilar.items():
        xs = vsp.GetXSec(surf, i)
        for parm in ("TopLRAngleEq", "RightLRAngleEq", "BottomLRAngleEq", "LeftLRAngleEq"):
            try:
                vsp.SetParmVal(vsp.GetXSecParm(xs, parm), 0.0)  # sol/sağ açı eşitliğini kapat
            except Exception:
                pass
        vsp.SetXSecTanAngles(xs, vsp.XSEC_LEFT_SIDE, sol, sol, sol, sol)
        vsp.SetXSecTanAngles(xs, vsp.XSEC_RIGHT_SIDE, sag, sag, sag, sag)
    vsp.Update()

    for parm in ("CapUMinOption", "CapUMaxOption"):
        try:
            vsp.SetParmVal(fid, parm, "EndCap", float(vsp.FLAT_END_CAP))
        except Exception as exc:
            print(f"[VSPAERO] Uyarı: gövde uç kapağı ayarlanamadı ({parm}: {exc})")
    vsp.Update()

    # [K1] Kurulum kontrolü: kesit konumları ve gövde genişliği
    for i in (1, 2, 3):
        try:
            oku = float(vsp.GetParmVal(fid, "XLocPercent", f"XSec_{i}")) * L_toplam
        except Exception:
            continue
        if abs(oku - olc["x"][i]) > 1e-3:
            raise RuntimeError(f"Gövde kesiti XSec_{i} yanlış konumda: {oku:.4f} m (beklenen {olc['x'][i]:.4f} m)")
    try:
        genislik = float(vsp.GetParmVal(fid, "Y_Len", "BBox"))
    except Exception:
        genislik = 0.0
    if genislik > 1e-6:
        if abs(genislik - D) > 0.001:
            raise RuntimeError(
                f"Gövde genişliği {genislik:.4f} m, beklenen {D:.4f} m: gövde şekli yanlış kuruldu "
                "(teğet açıları uygulanamamış olabilir)."
            )
    else:
        print("[VSPAERO] Uyarı: gövde genişliği (BBox Y_Len) okunamadı; şekli .vsp3 dosyasında gözle kontrol edin.")
    return fid


def profil_kalinlik_sinirlari(dosya: Path) -> Tuple[float, float]:
    """Profil dosyasından (veter=1) üst yüzeyin en yüksek ve alt yüzeyin en alçak y değeri."""
    ys: List[float] = []
    for satir in Path(dosya).read_text(encoding="utf-8", errors="ignore").splitlines()[1:]:
        parca = satir.split()
        if len(parca) >= 2:
            try:
                ys.append(float(parca[1]))
            except ValueError:
                continue
    if not ys:
        raise ValueError(f"Profil dosyası okunamadı: {dosya}")
    return max(ys), min(ys)


def kanat_z_hesapla(secim: Any, R: float, veter: float, profil: Path, gomme: float = VARSAYILAN_GOMME_M) -> float:
    """
    [K3] Kanat veter düzleminin z konumu (gövde ekseni z=0):
      ust        : gerçek yüksek kanat — kanadın EN ALÇAK noktası gövde tepesinin `gomme` kadar içinde
      ust_gomulu : kanat tamamen gövde içinde, EN YÜKSEK noktası tepenin `gomme` kadar altında
      orta       : z = 0 (VLM'de gövdenin yatay plakasıyla aynı düzlem → VLM'de kullanmayın)
      alt        : gerçek alçak kanat — kanadın EN YÜKSEK noktası gövde tabanının `gomme` kadar içinde
      alt_gomulu : kanat tamamen gövde içinde, EN ALÇAK noktası tabanın `gomme` kadar üstünde
      sayı       : doğrudan z [m]
    """
    if not isinstance(secim, str):
        return float(secim)
    y_ust, y_alt = profil_kalinlik_sinirlari(profil)
    if secim == "ust":
        return R - gomme - y_alt * veter
    if secim == "ust_gomulu":
        return R - gomme - y_ust * veter
    if secim == "orta":
        return 0.0
    if secim == "alt":
        return -R + gomme - y_ust * veter
    if secim == "alt_gomulu":
        return -R + gomme - y_alt * veter
    raise ValueError(f"Bilinmeyen kanat_z seçimi: {secim} (geçerli: {', '.join(KANAT_Z_SECENEKLERI)} veya sayı)")


def _girdi_ayarla(vsp: Any, analiz: str, ad: str, deger: Sequence[Any], tip: str, atlananlar: List[str]) -> None:
    """Analiz girdisini, bu OpenVSP sürümünde varsa ayarlar (sürüm farklarına dayanıklı)."""
    try:
        isimler = set(vsp.GetAnalysisInputNames(analiz))
    except Exception:
        isimler = set()
    if isimler and ad not in isimler:
        atlananlar.append(f"{analiz}.{ad}")
        return
    if tip == "int":
        vsp.SetIntAnalysisInput(analiz, ad, [int(v) for v in deger], 0)
    else:
        vsp.SetDoubleAnalysisInput(analiz, ad, [float(v) for v in deger], 0)


def _geometri_setini_ayarla(vsp: Any, analiz: str, atlananlar: List[str], yontem: str = "vlm") -> None:
    try:
        isimler = set(vsp.GetAnalysisInputNames(analiz))
    except Exception:
        isimler = set()
    if yontem == "panel":  # [DEĞİŞİKLİK 2026-11-18 v5 | gövde] Panel yöntemi: tüm geometri kalın (thick) set
        _girdi_ayarla(vsp, analiz, "GeomSet", [vsp.SET_ALL], "int", atlananlar)
        if "ThinGeomSet" in isimler:
            _girdi_ayarla(vsp, analiz, "ThinGeomSet", [vsp.SET_NONE], "int", atlananlar)
        return
    if "ThinGeomSet" in isimler:  # yeni OpenVSP sürümleri: VLM için ince (thin) set
        _girdi_ayarla(vsp, analiz, "ThinGeomSet", [vsp.SET_ALL], "int", atlananlar)
        _girdi_ayarla(vsp, analiz, "GeomSet", [vsp.SET_NONE], "int", atlananlar)
    else:
        _girdi_ayarla(vsp, analiz, "GeomSet", [vsp.SET_ALL], "int", atlananlar)
        _girdi_ayarla(vsp, analiz, "AnalysisMethod", [vsp.VORTEX_LATTICE], "int", atlananlar)


# [DEĞİŞİKLİK 2026-11-18 v2 | OpenVSP 3.56 uyumu] Sonuç adları OpenVSP 3.56 ile gelen resmi örneklerden
# (vspaero_ex/Swept_Wing_API/TR1208.vspscript, scripts/Master_VSP_VV_Script.vspscript) alındı:
# ExecAnalysis("VSPAEROSweep") -> "ResultsVec" (her hücum açısı için bir VSPAERO_History kaydı)
# -> her kayıtta "Alpha", "CLtot", ... dizileri; dizinin SON elemanı son iz (wake) iterasyonudur.
# CDi / E adları örneklerde geçmediği için birkaç olası ad sırayla denenir; bulunamazsa
# mevcut adlar ekrana yazılır ve .polar dosyası okunmaya çalışılır.
_ALFA_ADLARI = ("Alpha", "AoA")
_CL_ADLARI = ("CLtot", "CL")
_CDI_ADLARI = ("CDi", "CDitot", "CDi_tot", "CDtot_i")
_E_ADLARI = ("E", "Oswald", "e")
_CLW_ADLARI = ("CLiw", "CLwtot")  # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson] önce CLiw (sürtünmesiz; CLwtot viskoz CLo terimi içerir, e'de ~%1.4 fark)  # [DEĞİŞİKLİK 2026-11-18 v6 | inceleme-2 düzeltmeleri] iz (wake/Trefftz) tabanlı taşıma
_CDIW_ADLARI = ("CDiw",)    # [DEĞİŞİKLİK 2026-11-18 v6 | inceleme-2 düzeltmeleri] iz tabanlı indüklenmiş sürükleme (VSPAERO'nun Ew sütunu KULLANILMAZ:
                            #     yüzey CL'si ile iz CDi'sini karıştırıyor — gövde incelemesi)
_CM_ADLARI = ("CMytot", "CMy")  # [DEĞİŞİKLİK 2026-11-18 v5 | gövde] yunuslama momenti (gövde kararlılığa etki eder)


def _son_deger(vsp: Any, rid: str, adaylar: Sequence[str], mevcut: set) -> Optional[float]:
    for a in adaylar:
        if mevcut and a not in mevcut:
            continue
        try:
            v = list(vsp.GetDoubleResults(rid, a, 0))
        except Exception:
            continue
        if v:
            return float(v[-1])
    return None


def _sonuclari_api_ile_oku(vsp: Any, sweep_rid: Optional[str] = None) -> Optional[Dict[str, np.ndarray]]:
    """[DEĞİŞİKLİK 2026-11-18 v2 | OpenVSP 3.56 uyumu] Önce ResultsVec (3.56 resmi yöntemi), olmazsa VSPAERO_Polar sonucu denenir."""
    # 1) ResultsVec -> her alfa için History kaydı
    if sweep_rid:
        try:
            gecmisler = list(vsp.GetStringResults(sweep_rid, "ResultsVec", 0))
        except Exception:
            gecmisler = []
        if gecmisler:
            try:
                mevcut = set(vsp.GetAllDataNames(gecmisler[0]))
            except Exception:
                mevcut = set()
            alfa, cl, cdi, e = [], [], [], []
            for g in gecmisler:
                a = _son_deger(vsp, g, _ALFA_ADLARI, mevcut)
                c = _son_deger(vsp, g, _CL_ADLARI, mevcut)
                d = _son_deger(vsp, g, _CDI_ADLARI, mevcut)
                if a is None or c is None:
                    continue  # ResultsVec içinde History dışı kayıtlar (ör. Load) olabilir
                alfa.append(a); cl.append(c)
                if d is not None:
                    cdi.append(d)
                ee = _son_deger(vsp, g, _E_ADLARI, mevcut)
                if ee is not None:
                    e.append(ee)
            if alfa and len(cdi) == len(alfa):
                sonuc = {"alfa": np.asarray(alfa), "CL": np.asarray(cl), "CDi": np.asarray(cdi)}
                if len(e) == len(alfa):
                    sonuc["E"] = np.asarray(e)
                return sonuc
            print(f"[VSPAERO] ResultsVec okunurken CDi bulunamadı. Mevcut adlar: {sorted(mevcut)}")

    # 2) Eski yöntem: VSPAERO_Polar sonuç kaydı
    try:
        rid = vsp.FindLatestResultsID("VSPAERO_Polar")
    except Exception:
        return None
    if not rid:
        return None
    try:
        adlar = set(vsp.GetAllDataNames(rid))
    except Exception:
        adlar = set()

    def al(adaylar: Sequence[str]) -> Optional[np.ndarray]:
        for a in adaylar:
            if not adlar or a in adlar:
                try:
                    v = np.asarray(vsp.GetDoubleResults(rid, a, 0), dtype=float)
                    if v.size:
                        return v
                except Exception:
                    continue
        return None

    alfa, cl, cdi = al(_ALFA_ADLARI), al(_CL_ADLARI), al(_CDI_ADLARI)
    if alfa is None or cl is None or cdi is None:
        print(f"[VSPAERO] VSPAERO_Polar sonucunda beklenen sütunlar yok. Mevcut adlar: {sorted(adlar)}")
        return None
    sonuc = {"alfa": alfa, "CL": cl, "CDi": cdi}
    e = al(_E_ADLARI)
    if e is not None:
        sonuc["E"] = e
    return sonuc


def polar_dosyasi_oku(polar_yolu: Path) -> Dict[str, np.ndarray]:
    """VSPAERO .polar dosyasını BAŞLIK ADLARINA göre okur (sütun sırasına güvenmez)."""
    basliklar: Optional[List[str]] = None
    satirlar: List[List[float]] = []
    for ham in polar_yolu.read_text(encoding="utf-8", errors="ignore").splitlines():
        parcalar = ham.split()
        if not parcalar:
            continue
        if basliklar is None and any(p in ("AoA", "Alpha") for p in parcalar):
            basliklar = parcalar
            continue
        if basliklar is not None:
            try:
                degerler = [float(p) for p in parcalar]
            except ValueError:
                continue
            if len(degerler) == len(basliklar):
                satirlar.append(degerler)
    if basliklar is None or not satirlar:
        raise ValueError(f"Polar dosyası okunamadı: {polar_yolu}")
    tablo = np.asarray(satirlar, dtype=float)

    def kol(adaylar: Sequence[str]) -> Optional[np.ndarray]:
        for a in adaylar:
            if a in basliklar:  # type: ignore[operator]
                return tablo[:, basliklar.index(a)]  # type: ignore[union-attr]
        return None

    alfa, cl, cdi = kol(_ALFA_ADLARI), kol(_CL_ADLARI), kol(_CDI_ADLARI)
    if alfa is None or cl is None or cdi is None:
        raise ValueError(f"Polar dosyasında AoA/CL/CDi sütunları bulunamadı. Başlıklar: {basliklar}")
    sonuc = {"alfa": alfa, "CL": cl, "CDi": cdi}
    e = kol(_E_ADLARI)
    if e is not None:
        sonuc["E"] = e
    cm = kol(_CM_ADLARI)  # [DEĞİŞİKLİK 2026-11-18 v5 | gövde]
    if cm is not None:
        sonuc["CMy"] = cm
    clw, cdiw = kol(_CLW_ADLARI), kol(_CDIW_ADLARI)  # [DEĞİŞİKLİK 2026-11-18 v6 | inceleme-2 düzeltmeleri]
    if clw is not None and cdiw is not None:
        sonuc["CLw"] = clw
        sonuc["CDiw"] = cdiw
    return sonuc


def _firar_kenari_noktalari(vsp: Any, gid: str) -> np.ndarray:
    """Kurulan simetrik kanadın sağ yarısındaki fiziksel firar kenarını örnekle."""
    u = np.linspace(0.0, 1.0, 257).tolist()
    noktalar = vsp.CompVecPnt01(gid, 0, u, [0.0] * len(u))
    xyz = np.array([(p.x(), p.y(), p.z()) for p in noktalar], dtype=float)
    if xyz.shape != (len(u), 3) or not np.isfinite(xyz).all():
        raise ValueError(f"VSPAERO firar kenarı geometrisi okunamadı: {gid}.")
    xyz[:, 1] = abs(xyz[:, 1])
    aciklik, indisler = np.unique(xyz[:, 1], return_index=True)
    if len(aciklik) < 2:
        raise ValueError(f"VSPAERO firar kenarının açıklığı sıfır: {gid}.")
    return xyz[indisler]


def _kanat_yuk_polarini_oku(
    lod_yolu: Path, toplam: Dict[str, np.ndarray], kenarlar: Dict[str, np.ndarray],
    s_ref: float, govde_dahil: bool,
) -> Dict[str, np.ndarray]:
    """Kanat şeritlerinin Cl ve Cdi kuvvetlerini .lod dosyasından Sref'e göre topla."""
    alanlar = ("VortexSheet", "Xavg", "Yavg", "Zavg", "dSpan", "Chord", "Cl", "Cdi")
    durumlar: Dict[float, List[Tuple[float, ...]]] = {}
    alfa: Optional[float] = None
    sutunlar: Dict[str, int] = {}
    for satir in lod_yolu.read_text(encoding="utf-8", errors="ignore").splitlines():
        parcalar = satir.split()
        if parcalar[:1] == ["AoA_"]:
            alfa = float(parcalar[1])
            if alfa in durumlar:
                raise ValueError(f".lod dosyasında yinelenen hücum açısı: {alfa:g}°.")
            durumlar[alfa] = []
            sutunlar = {}
        elif all(ad in parcalar for ad in alanlar):
            sutunlar = {ad: i for i, ad in enumerate(parcalar)}
        elif alfa is not None and sutunlar and len(parcalar) == len(sutunlar):
            try:
                veriler = tuple(float(parcalar[sutunlar[ad]]) for ad in alanlar)
            except ValueError:
                continue
            if not all(math.isfinite(v) for v in veriler):
                raise ValueError(f".lod şerit verisi sonlu değil: alfa={alfa:g}°.")
            durumlar[alfa].append(veriler)

    beklenen_acilar = np.asarray(toplam["alfa"], dtype=float)
    if len(durumlar) != len(beklenen_acilar) or not np.allclose(
        sorted(durumlar), sorted(beklenen_acilar), rtol=0.0, atol=1e-5,
    ):
        raise ValueError(".lod ve .polar hücum açıları uyuşmuyor.")

    kanat_cl: List[float] = []
    kanat_cdi: List[float] = []
    for i, beklenen_alfa in enumerate(beklenen_acilar):
        alfa = min(durumlar, key=lambda a: abs(a - beklenen_alfa))
        yukler = np.asarray(durumlar[alfa], dtype=float)
        if not yukler.size:
            raise ValueError(f".lod dosyasında şerit yükü yok: alfa={alfa:g}°.")
        bilesenler: Dict[str, List[np.ndarray]] = {ad: [] for ad in kenarlar}
        for iz in np.unique(yukler[:, 0]):
            seritler = yukler[yukler[:, 0] == iz]
            x, y, z = seritler[:, 1], abs(seritler[:, 2]), seritler[:, 3]
            eslesenler = []
            for ad, kenar in kenarlar.items():
                # .lod konumları beş ondalıkla yazılır. Firar kenarı referansı doğrudan
                # OpenVSP geometrisinden alınır, iz numarasının sırasına güvenilmez.
                tolerans = 5e-4
                if (
                    np.all(y >= kenar[0, 1] - tolerans)
                    and np.all(y <= kenar[-1, 1] + tolerans)
                    and np.allclose(x, np.interp(y, kenar[:, 1], kenar[:, 0]), rtol=0.0, atol=tolerans)
                    and np.allclose(z, np.interp(y, kenar[:, 1], kenar[:, 2]), rtol=0.0, atol=tolerans)
                ):
                    eslesenler.append(ad)
            if len(eslesenler) == 1:
                bilesenler[eslesenler[0]].append(seritler)
            elif len(eslesenler) > 1 or not govde_dahil:
                raise ValueError(f"VSPAERO iz {iz:g}, alfa={alfa:g}° için bileşen eşleşmesi belirsiz.")

        if any(not seritler for seritler in bilesenler.values()):
            raise ValueError(f"VSPAERO kanat veya kuyruk yükü bulunamadı: alfa={alfa:g}°.")

        def kuvvet_katsayilari(yuzey: List[np.ndarray]) -> np.ndarray:
            seritler = np.concatenate(yuzey)
            agirlik = seritler[:, 4] * seritler[:, 5] / s_ref
            return agirlik @ seritler[:, 6:8]

        cl_w, cdi_w = kuvvet_katsayilari(bilesenler["kanat"])
        if not govde_dahil:
            cl_ht, cdi_ht = kuvvet_katsayilari(bilesenler["kuyruk"])
            if not np.allclose(
                (cl_w + cl_ht, cdi_w + cdi_ht),
                (toplam["CL"][i], toplam["CDi"][i]),
                atol=1e-3, rtol=2e-3,
            ):
                raise ValueError(f".lod kanat ve kuyruk kuvvetleri .polar toplamıyla uyuşmuyor: alfa={alfa:g}°.")
        kanat_cl.append(float(cl_w))
        kanat_cdi.append(float(cdi_w))
    return {"alfa": beklenen_acilar, "CL": np.asarray(kanat_cl), "CDi": np.asarray(kanat_cdi)}


def _polar_dosyasini_bul(vsp3_yolu: Path, baslangic_zamani: Optional[float] = None) -> Optional[Path]:
    """
    [DEĞİŞİKLİK 2026-11-18 v5 | inceleme düzeltmeleri] Bağımsız inceleme (kritik 1): eskiden '{ad}*.polar' deseniyle en yeni
    dosya alınıyordu; VSPAERO çökerse ÖNCEKİ çalıştırmanın dosyası 'başarılı' sanılabiliyordu
    ve desen başka adlı dosyaları da yakalayabiliyordu. Artık yalnızca TAM ad ({ad}.polar)
    kabul edilir ve dosya bu çalıştırma başladıktan SONRA yazılmış olmalıdır.
    """
    yol = vsp3_yolu.with_suffix(".polar")
    if not yol.is_file():
        return None
    if baslangic_zamani is not None and os.path.getmtime(yol) < baslangic_zamani - 1.0:
        raise RuntimeError(f"{yol.name} bu çalıştırmada üretilmemiş (eski dosya) - VSPAERO çalışmamış olabilir.")
    return yol


def polar_katsayilarini_cikar(polar: Dict[str, np.ndarray], ar_w: float) -> Dict[str, float]:
    """CL-alfa doğrusundan CLα ve α0; CDi = a*CL^2 + b*CL + c uydurmasından Oswald e."""
    alfa = np.asarray(polar["alfa"], dtype=float)
    cl = np.asarray(polar["CL"], dtype=float)
    cdi = np.asarray(polar["CDi"], dtype=float)
    dogrusal = alfa <= 6.0 + 1e-9
    if dogrusal.sum() < 2:
        dogrusal = np.ones_like(alfa, dtype=bool)
    egim, kesisim = np.polyfit(alfa[dogrusal], cl[dogrusal], 1)
    egim = float(max(egim, 1e-6))
    alfa0 = float(-kesisim / egim)
    a2 = float(np.polyfit(cl, cdi, 2)[0]) if len(cl) >= 3 else float("nan")
    e = 1.0 / (math.pi * ar_w * a2) if a2 > 1e-9 else float("nan")
    sonuc = {"CL_alpha_3d_per_deg": egim, "alpha_L0_deg": alfa0, "e_oswald_ham": e}
    idx4 = int(np.argmin(np.abs(alfa - 4.0)))
    if "E" in polar:
        sonuc["e_vspaero_alfa4"] = float(polar["E"][idx4])
    if "CLw" in polar and "CDiw" in polar:  # [DEĞİŞİKLİK 2026-11-18 v6 | inceleme-2 düzeltmeleri] iz tabanlı CLα, α0, e
        clw = np.asarray(polar["CLw"], dtype=float)
        cdiw = np.asarray(polar["CDiw"], dtype=float)
        egim_w, kes_w = np.polyfit(alfa[dogrusal], clw[dogrusal], 1)
        sonuc["CL_alpha_iz_per_deg"] = float(egim_w)
        sonuc["alpha_L0_iz_deg"] = float(-kes_w / egim_w) if abs(egim_w) > 1e-9 else float("nan")
        a2w = float(np.polyfit(clw, cdiw, 2)[0]) if len(clw) >= 3 else float("nan")
        sonuc["e_iz"] = 1.0 / (math.pi * ar_w * a2w) if a2w > 1e-9 else float("nan")
        sonuc["CL_iz_alfa4"] = float(clw[idx4])
    sonuc["CL_alfa4"] = float(cl[idx4])    # [DEĞİŞİKLİK 2026-11-18 v5 | gövde] karşılaştırma kolaylığı
    sonuc["CDi_alfa4"] = float(cdi[idx4])
    if "CMy" in polar:  # [DEĞİŞİKLİK 2026-11-18 v5 | gövde] dCm/dα (negatif = burun aşağı geri getirici, kararlı yönde)
        cm = np.asarray(polar["CMy"], dtype=float)
        sonuc["Cm_alpha_per_deg"] = float(np.polyfit(alfa[dogrusal], cm[dogrusal], 1)[0])
        sonuc["CMy_alfa4"] = float(cm[idx4])
    return sonuc


def yerel_e_iz(polar: Dict[str, np.ndarray], ar_w: float, cl_ref: float) -> float:
    """
    [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson] Belirli bir CL'deki YEREL e (iz tabanlı): e = CL² / (π·AR·CDiw(CL)).
    Flaplı kanatta kullanılır: parabol eğriliğinden çıkan e flabın getirdiği ek sürüklemeyi
    içermiyor (~%11 eksik CDi; bkz. PANEL_YAKINSAMA_RAPORU.txt). CDiw(CL), CDiw-CLiw verisine
    oturtulan 2. derece eğriden okunur.
    """
    if "CLw" not in polar or "CDiw" not in polar or not (cl_ref > 0):
        return float("nan")
    clw = np.asarray(polar["CLw"], dtype=float)
    cdiw = np.asarray(polar["CDiw"], dtype=float)
    katsayi = np.polyfit(clw, cdiw, 2)
    cdi_ref = float(np.polyval(katsayi, cl_ref))
    if cdi_ref <= 1e-9:
        return float("nan")
    return cl_ref ** 2 / (math.pi * ar_w * cdi_ref)


def kalite_bayraklari(polar: Dict[str, np.ndarray], kats: Dict[str, float]) -> List[str]:
    """[DEĞİŞİKLİK 2026-11-18 v6 | inceleme-2 düzeltmeleri] Sonucun güvenilirliği için otomatik uyarılar (boş liste = sorun görülmedi)."""
    b: List[str] = []
    if np.any(np.asarray(polar["CDi"]) < 0):
        b.append("CDi_negatif")
    if "CDiw" in polar and np.any(np.asarray(polar["CDiw"]) < 0):
        b.append("CDiw_negatif")
    if "CL_iz_alfa4" in kats and abs(kats["CL_alfa4"]) > 1e-6:
        if abs(kats["CL_iz_alfa4"] - kats["CL_alfa4"]) / abs(kats["CL_alfa4"]) > 0.03:
            b.append("yuzey_iz_CL_farki_buyuk")
    e = kats.get("e_iz", kats.get("e_oswald_ham", float("nan")))  # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson] aralık kontrolü iz e'sine
    if not math.isfinite(e):
        b.append("e_hesaplanamadi")
    elif not (0.70 <= e <= 1.00):
        b.append("e_aralik_disi")
    cla = kats.get("CL_alpha_3d_per_deg", float("nan"))
    if not (0.05 <= cla <= 0.11):
        b.append("CLa_aralik_disi")
    return b


# ---------------------------------------------------------------------------
# Ana çözücü
# ---------------------------------------------------------------------------

def vspaero_api_coz(
    profil: str,
    b_w: float,
    c_w: float,
    s_w: float,
    cl_max_2d: float,
    flap_aciklik_orani: float = 0.0,
    cl_max_flap_2d: Optional[float] = None,
    kuyruk_dahil: bool = False,
    b_ht: float = 0.0,
    c_ht: float = 0.0,
    tail_arm: float = 0.0,
    kuyruk_profil: str = "naca0010",
    model_klasoru: Optional[Path] = None,
    model_adi: Optional[str] = None,
    re_cref: float = 250000.0,
    govde: Optional[Dict[str, Any]] = None,
    yontem: str = "vlm",
    span_panel: Optional[int] = None,
    uc_sik: Optional[float] = None,
    zcg: Any = None,
    flap_ic_y: float = 0.0,
    ek_kesit_y: Sequence[float] = (),
    gecersize_izin: bool = False,
    e_kaynagi: str = "iz",               # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson] "iz" (önerilen) veya "yuzey" (eski)
    viskoz_e_carpani: float = 0.995,     # [DEĞİŞİKLİK 2026-11-19 v9 | tam akışta gövde] 1/e = 1/(k·e_span) + 0.007·π·AR formülündeki k (gövde modellenince 1.0 düşünülebilir)
    veter_tess: Optional[int] = None,    # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson] veter yönü panel (Tess_W)
    flap_profil: Optional[str] = None,
    geometri_etiketi: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tek bir geometriyi VSPAERO ile çözer. Hata olursa İSTİSNA FIRLATIR (sessiz yedek yok).

    [DEĞİŞİKLİK 2026-11-18 v6 | inceleme-2 düzeltmeleri] Fonksiyon gövdesi güncellendi
    (önceki hali: yedekler/vspaero_cozucu_YEDEK_2026-11-18_v5_duzeltme_oncesi.py). Yeni/değişen parametreler:
      span_panel   : YARIM KANAT toplam şerit sayısı; bölümlere boyla orantılı dağıtılır (eskiden bölüm başınaydı)
      uc_sik       : kanat ucuna doğru panel sıklaştırma (OutCluster, ör. 0.5); None = varsayılan
      zcg          : moment referans yüksekliği: None/0 = gövde ekseni (z=0, eskisiyle aynı), sayı [m] veya "kanat"
      flap_ic_y    : flabın başladığı açıklık konumu [m] (0 = merkez; gövde kenarı için R). Flap boyu değişmez
                     (toplam b·flap_aciklik_orani).
      ek_kesit_y   : kanada eklenecek ek bölüm sınırları [m]. Gövdeli modelde y = R otomatik eklenir.
      gecersize_izin: True ise e hesaplanamasa da sonuç döner (değerler NaN + kalite bayrağı); test modu için.
    [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson]
      e_kaynagi : "iz" (VARSAYILAN) -> flapsız: CDiw-CLiw parabol eğriliğinden e; flaplı: kalkış CL'sinde
                  (CL_ref = 0.83·CL_max_3d ≈ 1.1·V_stall) YEREL iz e'si. "yuzey" -> eski davranış (CDi-CLtot).
                  Gerekçe: yüzey e'si panel arttıkça ıraksıyor (1.03 -> 0.78), iz e'si yakınsıyor (~0.99).
      veter_tess: veter yönü panel sayısı (Tess_W). None = OpenVSP varsayılanı.
    govde sözlüğü:
      D, L_toplam (zorunlu); L_aviyonik, L_kuyruk, D_min (varsayılan 0.30 / 1.00 / 0.08 m)
      kanat_x : kanat 1/4 veterinin burundan uzaklığı [m]; None -> su haznesinin ortası
      kanat_z : ust | ust_gomulu | orta | alt | alt_gomulu | sayı [m]  (bkz. kanat_z_hesapla)
                [DEĞİŞİKLİK 2026-11-18 v7 | ekip kararı: kanat yeri] VARSAYILAN alt_gomulu: ekip kararı — kanat gövdenin içinden geçer,
                en alçak noktası gövde tabanının `gomme` (5 mm) üstünde.
      gomme   : kanadın gövdeye gömülme payı [m] (varsayılan 0.005)
    Gövdeli modelde Xcg = kanat 1/4 veteri.
    Diğer düzeltmeler: .polar yoksa artık API'ye DÜŞÜLMEZ (hata); .vsp3 analiz SONRASI tekrar yazılır
    (arayüzde açılınca ayarlar mümkün olduğunca dolu gelsin); model adına gövde D, L, x, z ve panel ayarları eklenir.
    """
    global _SPAN_PANEL
    _SPAN_PANEL = int(span_panel) if span_panel else None
    yontem = (yontem or "vlm").strip().lower()
    if yontem not in ("vlm", "panel"):
        raise ValueError(f"Bilinmeyen VSPAERO yöntemi: {yontem}")
    hazir, mesaj = vspaero_hazir_mi()
    if not hazir:
        raise RuntimeError(mesaj)
    vsp = openvsp_yukle()

    temiz = profil_dosyasi_bul(profil, flapli=False)
    if temiz is None:
        raise FileNotFoundError(f"Profil dosyası yok: airfoils/{profil}.dat")
    flap_orani = float(np.clip(flap_aciklik_orani, 0.0, 1.0))
    flapli_dosya: Optional[Path] = None
    if flap_orani > 1e-4:
        flap_adi = Path(flap_profil).stem if flap_profil else f"{profil}_flap"
        flapli_dosya = next(
            (yol for klasor in (REPO_ROOT, SCRIPT_DIR)
             if (yol := klasor / "airfoils_flap" / f"{flap_adi}.dat").is_file()),
            None,
        )
        if flapli_dosya is None:
            raise FileNotFoundError(f"Flaplı profil dosyası yok: airfoils_flap/{flap_adi}.dat")

    # Gövde ve kanat konumu (profil dosyası ASCII klasöre kopyalanmadan önce okunur)
    x_kanat_hk, z_kanat, xcg = 0.0, 0.0, 0.25 * c_w
    govde_bilgi: Dict[str, Any] = {}
    ek_sinirlar: List[float] = [float(y) for y in ek_kesit_y]
    g: Dict[str, Any] = {}
    if govde:
        g = {**GOVDE_VARSAYILAN, "gomme": VARSAYILAN_GOMME_M,
             **{k: v for k, v in govde.items() if v is not None}}
        olc = govde_olculeri(float(g["D"]), float(g["L_toplam"]), float(g["L_aviyonik"]),
                             float(g["L_kuyruk"]), float(g["D_min"]))
        R = 0.5 * float(g["D"])
        x_ceyrek = float(g["kanat_x"]) if g.get("kanat_x") is not None else olc["su_haznesi_orta_x"]
        kz = g.get("kanat_z", "alt_gomulu")  # [DEĞİŞİKLİK 2026-11-18 v7 | ekip kararı: kanat yeri] varsayılan (eskiden "ust")
        z_kanat = kanat_z_hesapla(kz, R, c_w, temiz, float(g["gomme"]))
        x_kanat_hk = x_ceyrek - 0.25 * c_w
        xcg = x_ceyrek
        ek_sinirlar.append(R)  # kanat-gövde birleşim çizgisi panel kenarına denk gelsin
        y_ust, y_alt = profil_kalinlik_sinirlari(temiz)
        govde_bilgi = {
            "govde_D_m": g["D"], "govde_L_m": g["L_toplam"], "kanat_x_ceyrek_m": round(x_ceyrek, 4),
            "kanat_z_m": round(z_kanat, 4), "kanat_z_secim": kz, "gomme_m": g["gomme"],
            "kanat_ust_z_m": round(z_kanat + y_ust * c_w, 4), "kanat_alt_z_m": round(z_kanat + y_alt * c_w, 4),
            "govde_tepe_z_m": round(R, 4),
        }
        if yontem == "vlm" and isinstance(kz, str) and kz == "orta":
            print("[VSPAERO] UYARI: VLM'de 'orta' kanat gövdenin yatay plakasıyla aynı düzlemde; "
                  "sonuç güvenilmez (CDi negatif çıkabilir).")
        # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson] Flap gövdenin içinden başlıyorsa, aşağı sapık flabın firar kenarı gövdeden taşabilir
        if flapli_dosya is not None and float(flap_ic_y) < R - 1e-6:
            _, y_alt_flap = profil_kalinlik_sinirlari(flapli_dosya)
            z_firar = z_kanat + y_alt_flap * c_w
            if z_firar < -R:
                print(f"[VSPAERO] UYARI: flap gövdenin içinden başlıyor ve firar kenarı gövdenin "
                      f"{(-R - z_firar) * 1000:.0f} mm altına taşıyor (z={z_firar:.3f} m). "
                      "Gerçekçi model için flap_ic_y = R (--flap-baslangic govde_kenari).")
                govde_bilgi["flap_govdeden_tasiyor_mm"] = round((-R - z_firar) * 1000, 1)

    # Moment referans yüksekliği
    if zcg is None:
        zcg_deger = 0.0
    elif isinstance(zcg, str):
        if zcg.strip().lower() != "kanat":
            raise ValueError(f"zcg için sayı veya 'kanat' bekleniyordu: {zcg}")
        zcg_deger = z_kanat
    else:
        zcg_deger = float(zcg)

    flap_ic = max(0.0, float(flap_ic_y))
    flap_dis = flap_ic + 0.5 * b_w * flap_orani if flap_orani > 1e-4 else flap_ic
    if flap_dis > 0.5 * b_w + 1e-6:
        raise ValueError(f"Flap kanat ucunu aşıyor: {flap_ic:.3f} + {flap_dis - flap_ic:.3f} > {0.5 * b_w:.3f} m")

    klasor = Path(model_klasoru) if model_klasoru else VARSAYILAN_MODEL_KLASORU
    klasor.mkdir(parents=True, exist_ok=True)
    ad = model_adi or f"{profil}_b{b_w:.4f}_S{s_w:.4f}_{'flapli' if flap_orani > 1e-4 else 'flapsiz'}"
    if model_adi is None:  # farklı modeller aynı dosya adını kullanmasın
        if kuyruk_dahil:
            ad += "_kuyruk"
        if govde:
            kz = g.get("kanat_z", "alt_gomulu")  # [DEĞİŞİKLİK 2026-11-18 v7 | ekip kararı: kanat yeri]
            ad += (f"_govde_D{float(g['D']):.3f}_L{float(g['L_toplam']):.3f}_x{xcg:.3f}"
                   f"_{kz if isinstance(kz, str) else f'z{float(kz):+.3f}'}")
            if abs(float(g["gomme"]) - VARSAYILAN_GOMME_M) > 1e-9:
                ad += f"_g{float(g['gomme']):.3f}"
        if flap_orani > 1e-4 and flap_ic > 1e-6:
            ad += f"_flapic{flap_ic:.3f}"
        if ek_kesit_y and not govde:
            ad += "_kok" + "_".join(f"{float(y):.3f}" for y in ek_kesit_y)
        if yontem != "vlm":
            ad += f"_{yontem}"
        if _SPAN_PANEL:
            ad += f"_span{_SPAN_PANEL}"
        if uc_sik:
            ad += f"_uc{float(uc_sik):.2f}"
        if veter_tess:  # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson]
            ad += f"_tw{int(veter_tess)}"
        if abs(zcg_deger) > 1e-9:
            ad += f"_zcg{zcg_deger:+.3f}"
    if geometri_etiketi:
        ad += f"_{geometri_etiketi}"

    # [v3] VSPAERO ve profil dosyaları ASCII (Türkçe karaktersiz) bir klasörde kullanılır.
    env_calisma = os.environ.get("VSPAERO_CALISMA_KLASORU", "").strip()
    calisma = Path(env_calisma).resolve() if env_calisma else ascii_calisma_klasoru(klasor.resolve())
    calisma.mkdir(parents=True, exist_ok=True)
    ayri_klasor = calisma != klasor.resolve()
    if ayri_klasor:
        prof_klasor = calisma / "profiller"
        prof_klasor.mkdir(parents=True, exist_ok=True)
        temiz = Path(shutil.copy2(temiz, prof_klasor / temiz.name))
        if flapli_dosya is not None:
            flapli_dosya = Path(shutil.copy2(flapli_dosya, prof_klasor / flapli_dosya.name))
    vsp3_yolu = (calisma / f"{ad}.vsp3").resolve()

    vsp.ClearVSPModel()
    vsp.Update()
    kanat_gid = _kanat_kur(
        vsp, "Kanat", 0.5 * b_w, c_w, temiz,
        flap_ic=flap_ic, flap_dis=flap_dis, flapli_profil=flapli_dosya,
        x_konum=x_kanat_hk, z_konum=z_kanat,
        ek_sinirlar=ek_sinirlar, span_panel=_SPAN_PANEL, uc_sik=uc_sik,
        veter_tess=veter_tess,  # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson]
    )
    if govde:
        _govde_kur(vsp, float(g["D"]), float(g["L_toplam"]), float(g["L_aviyonik"]),
                   float(g["L_kuyruk"]), float(g["D_min"]))
    firar_kenarlari: Dict[str, np.ndarray] = {}
    if kuyruk_dahil:
        if not (b_ht > 0.05 and c_ht > 0.02 and tail_arm > 0.05):
            raise ValueError("Kuyruklu VSPAERO için geçerli yatay kuyruk açıklığı, veteri ve kuyruk kolu gerekli.")
        kuyruk_dosya = profil_dosyasi_bul(kuyruk_profil, flapli=False)
        if kuyruk_dosya is None:
            raise FileNotFoundError(f"Yatay kuyruk profil dosyası yok: airfoils/{kuyruk_profil}.dat")
        if ayri_klasor and kuyruk_dosya != temiz:
            kuyruk_dosya = Path(shutil.copy2(kuyruk_dosya, calisma / "profiller" / kuyruk_dosya.name))
        x_kuyruk_hk = x_kanat_hk + 0.25 * c_w + tail_arm - 0.25 * c_ht
        if govde:
            print("[VSPAERO] UYARI: kuyruk + gövde birlikte: kuyruk z=0'da, VLM'de gövde plakasıyla çakışabilir.")
        kuyruk_gid = _kanat_kur(vsp, "YatayKuyruk", 0.5 * b_ht, c_ht, kuyruk_dosya, x_konum=x_kuyruk_hk,
                                span_panel=_SPAN_PANEL)
    vsp.Update()
    if kuyruk_dahil:
        firar_kenarlari = {
            "kanat": _firar_kenari_noktalari(vsp, kanat_gid),
            "kuyruk": _firar_kenari_noktalari(vsp, kuyruk_gid),
        }

    # [v5] Önceki çalıştırmadan kalan aynı adlı dosyaları sil (eski .polar okunmasın)
    import time as _time
    for eski in glob.glob(str(calisma / f"{ad}.*")):
        try:
            os.remove(eski)
        except OSError as exc:
            print(f"[VSPAERO] Uyarı: eski dosya silinemedi ({Path(eski).name}: {exc})")
    baslangic = _time.time()
    vsp.WriteVSPFile(str(vsp3_yolu), vsp.SET_ALL)
    vsp.SetVSP3FileName(str(vsp3_yolu))

    atlananlar: List[str] = []
    geo = "VSPAEROComputeGeometry"
    vsp.SetAnalysisInputDefaults(geo)
    _geometri_setini_ayarla(vsp, geo, atlananlar, yontem)
    vsp.ExecAnalysis(geo)

    swp = "VSPAEROSweep"
    vsp.SetAnalysisInputDefaults(swp)
    _geometri_setini_ayarla(vsp, swp, atlananlar, yontem)
    _girdi_ayarla(vsp, swp, "RefFlag", [vsp.MANUAL_REF], "int", atlananlar)
    _girdi_ayarla(vsp, swp, "Sref", [s_w], "double", atlananlar)
    _girdi_ayarla(vsp, swp, "bref", [b_w], "double", atlananlar)
    _girdi_ayarla(vsp, swp, "cref", [c_w], "double", atlananlar)
    _girdi_ayarla(vsp, swp, "Xcg", [xcg], "double", atlananlar)
    if abs(zcg_deger) > 1e-12:
        _girdi_ayarla(vsp, swp, "Zcg", [zcg_deger], "double", atlananlar)
    _girdi_ayarla(vsp, swp, "AlphaStart", [ALFA_TARAMASI_DERECE[0]], "double", atlananlar)
    _girdi_ayarla(vsp, swp, "AlphaEnd", [ALFA_TARAMASI_DERECE[-1]], "double", atlananlar)
    _girdi_ayarla(vsp, swp, "AlphaNpts", [len(ALFA_TARAMASI_DERECE)], "int", atlananlar)
    _girdi_ayarla(vsp, swp, "BetaStart", [0.0], "double", atlananlar)
    _girdi_ayarla(vsp, swp, "BetaNpts", [1], "int", atlananlar)
    _girdi_ayarla(vsp, swp, "MachStart", [MACH_SAYISI], "double", atlananlar)
    _girdi_ayarla(vsp, swp, "MachNpts", [1], "int", atlananlar)
    _girdi_ayarla(vsp, swp, "ReCref", [re_cref], "double", atlananlar)
    _girdi_ayarla(vsp, swp, "ReCrefNpts", [1], "int", atlananlar)
    _girdi_ayarla(vsp, swp, "WakeNumIter", [3], "int", atlananlar)
    _girdi_ayarla(vsp, swp, "NCPU", [4], "int", atlananlar)
    vsp.ExecAnalysis(swp)
    if atlananlar:
        print(f"[VSPAERO] Bu OpenVSP sürümünde bulunmayan girdiler atlandı: {', '.join(sorted(set(atlananlar)))}")

    # [O6] .vsp3'ü analiz ayarlarından SONRA tekrar yaz.
    # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson] İnceleme: analiz girdileri .vsp3'teki VSPAEROSettings'e geçmiyordu (Sref=100 kalıyordu).
    # Artık ayarlar "VSPAEROSettings" kapsayıcısının "VSPAERO" grubuna ayrıca yazılıyor; arayüzde açınca dolu gelir.
    try:
        cid = vsp.FindContainer("VSPAEROSettings", 0)
        ayarlar = {"Sref": s_w, "bref": b_w, "cref": c_w, "Xcg": xcg, "Zcg": zcg_deger,
                   "RefFlag": vsp.MANUAL_REF, "MachStart": MACH_SAYISI, "MachEnd": MACH_SAYISI, "MachNpts": 1,
                   "AlphaStart": ALFA_TARAMASI_DERECE[0], "AlphaEnd": ALFA_TARAMASI_DERECE[-1],
                   "AlphaNpts": len(ALFA_TARAMASI_DERECE), "ReCref": re_cref, "WakeNumIter": 3}
        yazilamayan = []
        for parm_ad, deger in ayarlar.items():
            pid = vsp.FindParm(cid, parm_ad, "VSPAERO") if cid else ""
            if pid:
                vsp.SetParmVal(pid, float(deger))
            else:
                yazilamayan.append(parm_ad)
        if yazilamayan:
            print(f"[VSPAERO] Uyarı: .vsp3 ayarlarına yazılamadı: {', '.join(yazilamayan)}")
    except Exception as exc:
        print(f"[VSPAERO] Uyarı: VSPAEROSettings yazılamadı ({exc})")
    try:
        vsp.WriteVSPFile(str(vsp3_yolu), vsp.SET_ALL)
    except Exception as exc:
        print(f"[VSPAERO] Uyarı: .vsp3 analiz sonrası yeniden yazılamadı ({exc})")

    # [v4/v5] Sonuç: yalnızca BU çalıştırmada yazılmış {ad}.polar okunur.
    # [O3] .polar yoksa artık API yoluna düşülmez (API 3.56'da yanlış değer vermişti) — hata.
    polar_yolu = _polar_dosyasini_bul(vsp3_yolu, baslangic)
    if polar_yolu is None:
        raise RuntimeError(f"VSPAERO sonuç vermedi (.polar dosyası yok): {vsp3_yolu.with_suffix('.polar')}")
    toplam_polar = polar_dosyasi_oku(polar_yolu)
    polar = toplam_polar
    kaynak = f"polar_dosyasi:{polar_yolu.name}"
    if kuyruk_dahil:
        lod_yolu = polar_yolu.with_suffix(".lod")
        if not lod_yolu.is_file() or lod_yolu.stat().st_mtime < baslangic - 2.0:
            raise RuntimeError(f"Bu VSPAERO çözümünün kanat ve kuyruk yükleri bulunamadı: {lod_yolu}")
        polar = _kanat_yuk_polarini_oku(lod_yolu, toplam_polar, firar_kenarlari, s_w, bool(govde))
        kaynak = f"kanat_serit_yuku:{lod_yolu.name}"

    if ayri_klasor:  # [v3] model/sonuç dosyalarını asıl klasöre kopyala
        for f in glob.glob(str(calisma / f"{ad}.*")):
            try:
                shutil.copy2(f, klasor / Path(f).name)
            except Exception as exc:
                print(f"[VSPAERO] Uyarı: {Path(f).name} asıl klasöre kopyalanamadı ({exc})")
        vsp3_yolu = (klasor / f"{ad}.vsp3").resolve()

    ar_w = (b_w ** 2) / max(s_w, 1e-9)
    kats = polar_katsayilarini_cikar(polar, ar_w)
    bayraklar = kalite_bayraklari(polar, kats)
    # Mevcut moment tanı alanları çözülen geometrinin tamamını raporlar.
    moment = polar_katsayilarini_cikar(toplam_polar, ar_w) if kuyruk_dahil else kats
    # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson] e kaynağı seçimi
    e_yuzey = kats["e_oswald_ham"]
    cl_max_flap_kesit_ = cl_max_flap_2d if (cl_max_flap_2d and cl_max_flap_2d > 0.1) else cl_max_2d
    cl_max_3d_ = (flap_orani * cl_max_flap_kesit_ + (1.0 - flap_orani) * float(cl_max_2d)) * (0.90 if flap_orani > 1e-4 else 0.92)
    cl_ref = 0.83 * cl_max_3d_ if flap_orani > 1e-4 else float("nan")
    e_iz_yerel = yerel_e_iz(polar, (b_w ** 2) / max(s_w, 1e-9), cl_ref) if flap_orani > 1e-4 else float("nan")
    ek_kaynak = (e_kaynagi or "iz").strip().lower()
    if kuyruk_dahil and ek_kaynak in ("iz", "yuzey"):
        e_ham = e_yuzey
        e_kaynak_etiketi = "kanat_yuzeyi_lod"
        if ek_kaynak == "iz":
            bayraklar.append("kanat_bileseni_iz_verisi_yok")
    elif ek_kaynak == "iz":
        e_ham = e_iz_yerel if flap_orani > 1e-4 else kats.get("e_iz", float("nan"))
        e_kaynak_etiketi = "iz_yerel_CLref" if flap_orani > 1e-4 else "iz_egrilik"
        if not math.isfinite(e_ham):  # iz verisi yoksa yüzeye düş ama işaretle
            e_ham = e_yuzey
            e_kaynak_etiketi = "yuzey(iz_yok)"
            bayraklar.append("iz_verisi_yok")
    elif ek_kaynak == "yuzey":
        e_ham = e_yuzey
        e_kaynak_etiketi = "yuzey"
    else:
        raise ValueError(f"Bilinmeyen e_kaynagi: {e_kaynagi} (iz | yuzey)")
    # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson] aralık bayrağını KULLANILAN e'ye göre yeniden belirle (flaplıda yerel e kullanılıyor)
    bayraklar = [b for b in bayraklar if b != "e_aralik_disi"]
    if math.isfinite(e_ham) and not (0.70 <= e_ham <= 1.00):
        bayraklar.append("e_aralik_disi")
    if not math.isfinite(e_ham):
        if not gecersize_izin:
            raise RuntimeError(f"VSPAERO CDi verisinden e hesaplanamadı (bayraklar: {', '.join(bayraklar)}).")
        e_span = float("nan")
        e_oswald_duzeltilmis = float("nan")
    else:
        # [v5] Düz kanatta e_span <= 1 (Munk); ağ testi netleşene kadar 1.0 ile sınırlı.
        # [v4] Tablolardaki LLT/NumPy e'si ampirik viskoz terim içerir -> aynı düzeltme:
        #      1/e = 1/(0.995·e_span) + 0.007·π·AR
        e_span = float(np.clip(e_ham, 0.30, 1.00))
        e_oswald_duzeltilmis = 1.0 / (1.0 / (float(viskoz_e_carpani) * e_span) + 0.007 * math.pi * ar_w)  # [DEĞİŞİKLİK 2026-11-19 v9 | tam akışta gövde] k parametre
    e_kirpildi = math.isfinite(e_ham) and (e_ham > 1.0 or e_ham < 0.30)
    if e_kirpildi:
        bayraklar.append("e_kirpildi")

    # CL_max: VSPAERO'dan gelmez -> eski kodla aynı 2B ağırlıklı tahmin
    cl_max_flap_kesit = cl_max_flap_2d if (cl_max_flap_2d and cl_max_flap_2d > 0.1) else cl_max_2d
    cl_max_eff_2d = flap_orani * cl_max_flap_kesit + (1.0 - flap_orani) * float(cl_max_2d)
    cl_max_3d = cl_max_eff_2d * (0.90 if flap_orani > 1e-4 else 0.92)

    def _y(v: Any, n: int = 6) -> Any:
        try:
            return round(float(v), n)
        except (TypeError, ValueError):
            return float("nan")

    return {
        "cozucu": "vspaero_api",
        "kuvvet_kapsami": "kanat" if kuyruk_dahil or not govde else "kanat_govde",
        "CL_alpha_3d_per_deg": _y(kats["CL_alpha_3d_per_deg"]),
        "alpha_L0_deg": _y(kats["alpha_L0_deg"], 4),
        "e_oswald": _y(e_oswald_duzeltilmis, 5),
        "e_span_vspaero_ham": _y(e_ham, 5),
        "e_kaynagi": e_kaynak_etiketi,                       # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson]
        "e_yuzey": _y(e_yuzey, 5),                           # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson] eski (yüzey) e, karşılaştırma için
        "e_iz_yerel": _y(e_iz_yerel, 5),                     # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson] flaplı: CL_ref'teki yerel iz e'si
        "CL_ref_flapli": _y(cl_ref, 4),                      # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson]
        "veter_tess": int(veter_tess) if veter_tess else "varsayilan",  # [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson]
        "e_oswald_ham": _y(e_ham, 5),
        "e_kirpildi": bool(e_kirpildi),
        "e_vspaero_alfa4": _y(kats.get("e_vspaero_alfa4", float("nan")), 5),
        "CL_alpha_iz_per_deg": _y(kats.get("CL_alpha_iz_per_deg", float("nan"))),
        "alpha_L0_iz_deg": _y(kats.get("alpha_L0_iz_deg", float("nan")), 4),
        "e_iz": _y(kats.get("e_iz", float("nan")), 5),
        "kalite": ";".join(bayraklar) if bayraklar else "ok",
        "CL_max_3d_tahmini": _y(cl_max_3d, 5),
        "CL_max_kaynak": "2B_agirlikli_tahmin (VSPAERO stall hesaplamaz)",
        "peak_cl_ratio": float("nan"),
        "flap_aciklik_orani": round(flap_orani, 4),
        "flap_ic_y_m": round(flap_ic, 4),
        "kuyruk_dahil": bool(kuyruk_dahil),
        "vsp3_dosyasi": str(vsp3_yolu),
        "sonuc_kaynagi": kaynak,
        "atlanan_girdiler": ";".join(sorted(set(atlananlar))),
        "polar_alfa": ";".join(f"{x:.2f}" for x in polar["alfa"]),
        "polar_CL": ";".join(f"{x:.5f}" for x in polar["CL"]),
        "polar_CDi": ";".join(f"{x:.6f}" for x in polar["CDi"]),
        "polar_CL_toplam": ";".join(f"{x:.5f}" for x in toplam_polar["CL"]),
        "polar_CDi_toplam": ";".join(f"{x:.6f}" for x in toplam_polar["CDi"]),
        "yontem": yontem,
        "span_panel": _SPAN_PANEL or "varsayilan",
        "uc_sik": uc_sik if uc_sik else "varsayilan",
        "zcg_m": round(zcg_deger, 4),
        "CL_alfa4": _y(kats["CL_alfa4"]),
        "CDi_alfa4": _y(kats["CDi_alfa4"]),
        "Cm_alpha_per_deg": _y(moment.get("Cm_alpha_per_deg", float("nan"))),
        "CMy_alfa4": _y(moment.get("CMy_alfa4", float("nan"))),
        "govde_dahil": bool(govde),
        **govde_bilgi,
    }


def vspaero_api_coz_richardson(n_kaba: int = 24, n_ince: int = 48, **kw: Any) -> Dict[str, Any]:
    """
    [DEĞİŞİKLİK 2026-11-19 v8 | iz e + Richardson] Aynı geometriyi iki ağla (n_kaba, n_ince yarım-kanat şeridi)
    çözer ve e için "sonsuz panel" tahmini yapar (Richardson, yakınsama mertebesi p=1 kabulü):
        e_∞ = e_ince + (e_ince - e_kaba) / (r - 1),   r = n_ince / n_kaba   (r=2 -> 2·e_ince - e_kaba)
    Yalnızca İZ e'sine uygulanmalı (yüzey e'si yakınsamadığı için Richardson onu bozar: ≈0.65).
    CLα, α0, Cm vb. ince ağdan alınır (CLα 24->48'de asimptotik bölgede değil; inceleme raporu).
    Dönen sözlük ince ağın sonucudur; e alanları Richardson değeriyle güncellenir, iki ağın ham e'leri eklenir.
    """
    if n_ince <= n_kaba:
        raise ValueError("Richardson için n_ince > n_kaba olmalı")
    kw = dict(kw)
    kw.pop("span_panel", None)
    if str(kw.get("e_kaynagi", "iz")).strip().lower() == "yuzey":
        print("[VSPAERO] UYARI: Richardson yüzey e'sine uygulanıyor; yüzey e'si yakınsamadığı için sonuç güvenilmez.")
    r_kaba = vspaero_api_coz(span_panel=n_kaba, **kw)
    r_ince = vspaero_api_coz(span_panel=n_ince, **kw)
    sonuc = dict(r_ince)
    e1, e2 = float(r_kaba["e_span_vspaero_ham"]), float(r_ince["e_span_vspaero_ham"])
    oran = float(n_ince) / float(n_kaba)
    kanat_yuzeyi = r_ince["e_kaynagi"] == "kanat_yuzeyi_lod"
    if not kanat_yuzeyi and math.isfinite(e1) and math.isfinite(e2):
        e_inf = e2 + (e2 - e1) / (oran - 1.0)
        e_span = float(np.clip(e_inf, 0.30, 1.00))
        b_w, s_w = float(kw["b_w"]), float(kw["s_w"])
        ar_w = b_w ** 2 / max(s_w, 1e-9)
        _k = float(kw.get("viskoz_e_carpani", 0.995))  # [DEĞİŞİKLİK 2026-11-19 v9 | tam akışta gövde]
        sonuc["e_oswald"] = round(1.0 / (1.0 / (_k * e_span) + 0.007 * math.pi * ar_w), 5)
        sonuc["e_span_vspaero_ham"] = round(e_inf, 5)
        sonuc["e_kirpildi"] = bool(e_inf > 1.0 or e_inf < 0.30)
        kalite = [k for k in str(sonuc.get("kalite", "ok")).split(";")
                  if k and k not in ("ok", "e_kirpildi", "e_aralik_disi")]
        if not (0.70 <= e_inf <= 1.00):
            kalite.append("e_aralik_disi")
        if sonuc["e_kirpildi"]:
            kalite.append("e_kirpildi")
        sonuc["kalite"] = ";".join(kalite) if kalite else "ok"
    sonuc["richardson"] = f"{n_kaba}/{n_ince} (e: ince ağ)" if kanat_yuzeyi else f"{n_kaba}/{n_ince}"
    sonuc["e_span_kaba"] = round(e1, 5)
    sonuc["e_span_ince"] = round(e2, 5)
    sonuc["CL_alpha_kaba_per_deg"] = r_kaba["CL_alpha_3d_per_deg"]
    sonuc["vsp3_dosyasi_kaba"] = r_kaba["vsp3_dosyasi"]
    return sonuc
