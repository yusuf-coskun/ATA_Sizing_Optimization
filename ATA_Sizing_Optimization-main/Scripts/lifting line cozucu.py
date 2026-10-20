from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Script ve proje ana dizinini tanimla (nereden calistirilirsa calistirilsin hata vermez)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kanat_boyutlandirma import tum_adaylari_uret


def xfoil_veya_csv_oku(dosya_yolu: Path) -> tuple[np.ndarray, np.ndarray]:
    """2B polar dosyasindan alpha ve cl sutunlarini ceker."""
    alphas, cls = [], []
    veri_basladi = False

    with dosya_yolu.open("r", encoding="utf-8", errors="ignore") as f:
        satirlar = f.readlines()

    # CSV formati kontrolu
    ilk = satirlar[0].strip().lower() if satirlar else ""
    if "," in ilk and ("alpha" in ilk or "aoa" in ilk):
        okuyucu = csv.DictReader(satirlar)
        alanlar = {k.strip().lower(): k for k in (okuyucu.fieldnames or [])}
        a_col = alanlar.get("alpha") or alanlar.get("aoa")
        c_col = alanlar.get("cl")
        if a_col and c_col:
            for row in okuyucu:
                try:
                    alphas.append(float(row[a_col]))
                    cls.append(float(row[c_col]))
                except (ValueError, TypeError):
                    continue
            if alphas:
                sirali = np.argsort(alphas)
                return np.array(alphas)[sirali], np.array(cls)[sirali]

    # XFOIL .txt / .pol veya boslukla ayrilmis format
    for satir in satirlar:
        temiz = satir.strip()
        if not temiz:
            continue
        if "----" in temiz:
            veri_basladi = True
            continue
        parcalar = re.split(r"[,\s]+", temiz)
        if len(parcalar) >= 2:
            try:
                a_val = float(parcalar[0])
                cl_val = float(parcalar[1])
                if veri_basladi or (-25.0 <= a_val <= 25.0 and -2.5 <= cl_val <= 3.0):
                    alphas.append(a_val)
                    cls.append(cl_val)
            except ValueError:
                continue

    if not alphas:
        raise ValueError(f"'{dosya_yolu.name}' icerisinde gecerli alpha-cl verisi bulunamadi.")

    sirali = np.argsort(alphas)
    return np.array(alphas)[sirali], np.array(cls)[sirali]


def _polar_yolunu_coz(kaynak_yolu: str | Path) -> Path:
    """Klasor adindaki 'airfoils_polar' / 'airfoils_polars' ve dizin farklarini cozer."""
    aday_yollar = [
        Path(kaynak_yolu),
        PROJECT_ROOT / kaynak_yolu,
        SCRIPT_DIR / kaynak_yolu,
        PROJECT_ROOT / "airfoils_polar",
        PROJECT_ROOT / "airfoils_polars",
    ]
    for p in aday_yollar:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Polar klasoru bulunamadi. Aranan dizin: {(PROJECT_ROOT / 'airfoils_polar').resolve()}"
    )


def polarlari_yukle(kaynak_yolu: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Polar klasorunu veya CSV dosyasini tarayarak profil polarlarini dondurur."""
    yol = _polar_yolunu_coz(kaynak_yolu)
    polarlar: dict[str, dict[str, np.ndarray]] = {}

    if yol.is_dir():
        for dosya in sorted(yol.glob("*")):
            if dosya.is_file() and dosya.suffix.lower() in {".csv", ".txt", ".pol", ".dat"}:
                try:
                    a_deg, c_2d = xfoil_veya_csv_oku(dosya)
                    polarlar[dosya.stem] = {"alpha_deg": a_deg, "cl_2d": c_2d}
                except ValueError as e:
                    print(f"[UYARI] {e} (Atlandi)")
    else:
        a_deg, c_2d = xfoil_veya_csv_oku(yol)
        polarlar[yol.stem] = {"alpha_deg": a_deg, "cl_2d": c_2d}

    if not polarlar:
        raise ValueError(f"'{yol}' icinde hicbir gecerli polar verisi okunamadi.")

    return polarlar


def lineer_parametreler(
    alpha_deg: np.ndarray,
    cl_2d: np.ndarray,
    a_min: float = -4.0,
    a_max: float = 6.0,
) -> tuple[float, float]:
    """Lineer bolgeden a0 (1/rad) ve alpha_L0 (rad) hesaplar."""
    maske = (alpha_deg >= a_min) & (alpha_deg <= a_max)
    if np.count_nonzero(maske) < 2:
        maske = np.ones_like(alpha_deg, dtype=bool)

    alpha_rad = np.deg2rad(alpha_deg[maske])
    cl_secim = cl_2d[maske]

    a0_rad, b = np.polyfit(alpha_rad, cl_secim, 1)
    if a0_rad <= 0:
        raise ValueError(f"Hesaplanan tasima egimi pozitif degil: a0={a0_rad:.3f}")

    alpha_l0_rad = -b / a0_rad
    return float(a0_rad), float(alpha_l0_rad)


def llt_dikdortgen_kanat(
    ar_w: float,
    a0_rad: float,
    alpha_l0_rad: float,
    alpha_deg_dizi: np.ndarray,
    n_terim: int = 25,
) -> tuple[np.ndarray, float, float]:
    """Prandtl Tasima Cizgisi matrisini cozer. Dondurur: (cl_3d, CL_alpha_rad, e_oswald)."""
    i_dizi = np.arange(1, n_terim + 1)
    theta = i_dizi * np.pi / (2.0 * n_terim)
    n_harmonik = 2 * i_dizi - 1

    geometrik_terim = (4.0 * ar_w) / a0_rad
    sin_n_theta = np.sin(np.outer(theta, n_harmonik))
    sin_theta = np.sin(theta)[:, np.newaxis]

    m_matrisi = sin_n_theta * (geometrik_terim + (n_harmonik[np.newaxis, :] / sin_theta))
    a_katsayilari = np.linalg.solve(m_matrisi, np.ones(n_terim))

    cl_alpha_3d_rad = float(np.pi * ar_w * a_katsayilari[0])
    delta = float(np.sum(n_harmonik[1:] * (a_katsayilari[1:] / a_katsayilari[0]) ** 2))
    e_oswald = 1.0 / (1.0 + delta)

    alpha_rad = np.deg2rad(alpha_deg_dizi)
    cl_3d = cl_alpha_3d_rad * (alpha_rad - alpha_l0_rad)

    return cl_3d, cl_alpha_3d_rad, e_oswald


def tum_adaylari_analiz_et_ve_ciz(
    polar_dizini: str = "airfoils_polar",
    cikti_grafik_klasoru: str = "cl_alpha_grafikleri",
    ozet_csv: str = "adaylar_3b_llt_sonuclari.csv",
    grafik_ciz: bool = True,
):
    # 1. Adaylari uret
    adaylar = tum_adaylari_uret()
    print(f"Toplam incelenecek aday sayisi: {len(adaylar)}")

    # 2. 2B polarlari yukle ve profil ozelliklerini cikar
    polarlar = polarlari_yukle(polar_dizini)
    profil_bilgileri = {}
    for profil_adi, veri in polarlar.items():
        try:
            a0, a_l0 = lineer_parametreler(veri["alpha_deg"], veri["cl_2d"])
            profil_bilgileri[profil_adi] = {
                "alpha_deg": veri["alpha_deg"],
                "cl_2d": veri["cl_2d"],
                "a0_rad": a0,
                "alpha_l0_rad": a_l0,
            }
        except ValueError as e:
            print(f"[UYARI] {profil_adi} profili atlandi: {e}")

    if not profil_bilgileri:
        raise RuntimeError("Analiz edilecek gecerli profil kalmadi.")

    # KUYRUK İÇİN NACA0012 PROFİLİNİ BUL
    # Klasörünüzdeki naca001264a08cli02 gibi isimleri otomatik yakalamak için 'naca0012' kelimesini arar.
    kuyruk_profil_anahtari = next((k for k in profil_bilgileri.keys() if "naca0012" in k.lower()), None)
    
    if not kuyruk_profil_anahtari:
        print("[UYARI] İsminde 'naca0012' geçen profil polar dizininde bulunamadi!")
        print("[UYARI] Kod gecici olarak teorik simetrik profil (a0=2pi, a_L0=0) kullanacak.")
        a0_rad_kuyruk = 2.0 * np.pi
        alpha_l0_rad_kuyruk = 0.0
        kuyruk_profil_adi = "Simetrik_Teorik_Fallback"
    else:
        a0_rad_kuyruk = profil_bilgileri[kuyruk_profil_anahtari]["a0_rad"]
        alpha_l0_rad_kuyruk = profil_bilgileri[kuyruk_profil_anahtari]["alpha_l0_rad"]
        kuyruk_profil_adi = kuyruk_profil_anahtari
        print(f"[BİLGİ] Kuyruk için gerçek profil verisi '{kuyruk_profil_adi}' başarıyla yüklendi.")

    grafik_yolu = PROJECT_ROOT / cikti_grafik_klasoru
    if grafik_ciz:
        grafik_yolu.mkdir(parents=True, exist_ok=True)

    ozet_tablo = []

    # 3. Her adayi tek tek donguye alip oku ve isle
    for idx, aday in enumerate(adaylar, start=1):
        
        # --- ANA KANAT GEOMETRİSİ ---
        b_w = aday["b_w_m"]
        ar_w = aday["AR_w"]
        s_w = aday["S_w_m2"]
        c_w = aday["c_root_w_m"]
        
        # --- KUYRUK GEOMETRİSİ ---
        ar_h = aday.get("AR_h", 4.0)          # Yatay kuyruk aciklik orani
        s_h = aday.get("S_h_m2", s_w * 0.2)   # Yatay kuyruk alani (kanadin %20'si kabul)
        i_t_deg = aday.get("i_t_deg", 0.0)    # Yatay kuyruk montaj acisi

        if grafik_ciz:
            fig, ax = plt.subplots(figsize=(9, 6), dpi=120)

        for profil_adi, profil in profil_bilgileri.items():
            alpha_dizi = profil["alpha_deg"]
            cl_2d = profil["cl_2d"]
            alpha_rad_dizi = np.deg2rad(alpha_dizi)

            # === 1. ANA KANAT ÇÖZÜMÜ ===
            cl_3d_kanat, cl_a_rad_kanat, e_osw_kanat = llt_dikdortgen_kanat(
                ar_w=ar_w,
                a0_rad=profil["a0_rad"],
                alpha_l0_rad=profil["alpha_l0_rad"],
                alpha_deg_dizi=alpha_dizi,
            )

            # === 2. KUYRUK ÇÖZÜMÜ (NACA 0012 gerçek polarından gelen değerler ile) ===
            _, cl_a_rad_kuyruk, e_osw_kuyruk = llt_dikdortgen_kanat(
                ar_w=ar_h,
                a0_rad=a0_rad_kuyruk, 
                alpha_l0_rad=alpha_l0_rad_kuyruk,
                alpha_deg_dizi=alpha_dizi,
            )
            
            # --- DOWNWASH (Aşağı Yıkama) VE EFEKTİF HÜCUM AÇISI ---
            epsilon_rad = (2.0 * cl_3d_kanat) / (np.pi * ar_w)
            i_t_rad = np.deg2rad(i_t_deg)
            
            # Kuyrugun uzerinden gecen net hucum acisi
            alpha_kuyruk_rad = alpha_rad_dizi - epsilon_rad + i_t_rad
            
            # Kuyruk 3B Tasima Katsayisi 
            cl_3d_kuyruk = cl_a_rad_kuyruk * (alpha_kuyruk_rad - alpha_l0_rad_kuyruk)
            
            # === 3. SÜPERPOZİSYON ===
            cl_toplam = cl_3d_kanat + cl_3d_kuyruk * (s_h / s_w)
            
            cl_a_deg_toplam = np.deg2rad(cl_a_rad_kanat + cl_a_rad_kuyruk * (1.0 - (2.0 * cl_a_rad_kanat / (np.pi * ar_w))) * (s_h / s_w))
            cl_toplam_0_deg = np.interp(0.0, alpha_dizi, cl_toplam)

            # Tablo kaydi
            ozet_tablo.append({
                "aday_no": idx,
                "b_w_m": b_w,
                "AR_w": ar_w,
                "S_w_m2": round(s_w, 4),
                "S_h_m2": round(s_h, 4),
                "profil_kanat": profil_adi,
                "profil_kuyruk": kuyruk_profil_adi,
                "a0_2d_per_deg": round(np.deg2rad(profil["a0_rad"]), 4),
                "CL_alpha_3d_kanat_deg": round(np.deg2rad(cl_a_rad_kanat), 4),
                "CL_alpha_3d_toplam_deg": round(cl_a_deg_toplam, 4),
                "e_oswald_kanat": round(e_osw_kanat, 4),
                "CL_3D_toplam_at_0deg": round(float(cl_toplam_0_deg), 4),
            })

            if grafik_ciz:
                # Grafigi cizelim
                cizgi = ax.plot(alpha_dizi, cl_toplam, lw=1.5, label=f"{profil_adi} Toplam (Uçak)")
                ax.plot(alpha_dizi, cl_3d_kanat, "--", lw=1.0, alpha=0.5, color=cizgi[0].get_color())

        if grafik_ciz:
            ax.axhline(0, color="gray", lw=0.6, ls=":")
            ax.axvline(0, color="gray", lw=0.6, ls=":")
            ax.set_title(
                f"Aday #{idx:03d} | $S_w$={s_w:.2f} m², $AR_w$={ar_w} | $S_h$={s_h:.2f} m², $AR_h$={ar_h}\n"
                f"Kanat + {kuyruk_profil_adi} Kuyruk Süperpozisyonu ($C_L - \\alpha$)",
                fontsize=11,
            )
            ax.set_xlabel("Hücum Açısı, $\\alpha$ (°)")
            ax.set_ylabel("Toplam Taşıma Katsayısı, $C_L$")
            ax.grid(True, ls="--", alpha=0.5)
            
            if len(profil_bilgileri) <= 15:
                ax.legend(loc="upper left", fontsize=7, ncol=2)
            fig.tight_layout()

            dosya_adi = f"aday_{idx:03d}_b{b_w:.2f}_AR{ar_w}.png"
            fig.savefig(grafik_yolu / dosya_adi)
            plt.close(fig)

        if idx % 25 == 0 or idx == len(adaylar):
            print(f"[{idx}/{len(adaylar)}] aday islendi.")

    # CSV kaydi
    ozet_csv_yolu = PROJECT_ROOT / ozet_csv
    if ozet_tablo:
        with ozet_csv_yolu.open("w", newline="", encoding="utf-8-sig") as f:
            yazici = csv.DictWriter(f, fieldnames=list(ozet_tablo[0].keys()))
            yazici.writeheader()
            yazici.writerows(ozet_tablo)
        print(f"\nAnaliz tamamlandi. Sayisal sonuclar kaydedildi: {ozet_csv_yolu}")
        if grafik_ciz:
            print(f"Grafikler kaydedildi: {grafik_yolu}")

if __name__ == "__main__":
    tum_adaylari_analiz_et_ve_ciz(
        polar_dizini="airfoils_polar",
        grafik_ciz=True,
    )