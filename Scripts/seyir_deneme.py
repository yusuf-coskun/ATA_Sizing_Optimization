import os
import numpy as np
import pandas as pd
import re
from pathlib import Path
from functools import lru_cache

# ==============================================================================
# 1. SABİTLER, BATARYA KISITLARI VE GÖREV PARAMETRELERİ
# ==============================================================================
V_SEYIR = 20.0                  # Seyir hızı (m/s)
RHO_HAVA = 1.175                # Hava yoğunluğu (kg/m^3)
MU_HAVA = 1.789e-5              # Dinamik viskozite (Pa.s); LLT ve CD0 aynı koşulda
PERVANE_SECIMI = "14x7"         # Farklı pervanelerin verileri karıştırılmaz
G_YERCEKIMI = 9.81              # Yerçekimi ivmesi (m/s^2)

N_WAYPOINT = 10                 # Mavi waypoint sayısı (WP1 - WP10)
L_TABAN_ROTA_M = 2800.0         # Waypoint düz birleştirme çevresi (m)
N_TUR_YUKLU = 2                 # Tam dolu (su yüklü) uçulacak tur sayısı
N_TUR_BOS = 1                   # Yüksüz (boş) uçulacak tur sayısı

BANK_ANGLE_DEG = 30.0           # Waypoint dışından dönüş yatış açısı (derece)
WP_CLEARANCE_M = 5.0            # Waypoint dış güvenlik mesafesi (m)

# --- Batarya ve Elektriksel Güç Modeli (6S 12000 mAh LiPo) ---
BATARYA_HUCRE_SAYISI = 6        # 6S LiPo (Nominal: 6 * 3.7V = 22.2V)
V_BAT_SABIT = 24.0             # Sabit uç gerilimi; P(W) = 24 * I(A)

BATARYA_TOPLAM_MAH = 12000.0    # Toplam batarya kapasitesi (mAh)
INIS_REZERV_ORANI = 0.20        # İnişte bataryada kalması zorunlu %20 rezerv (2400 mAh)
KULLANILABILIR_TOPLAM_MAH = BATARYA_TOPLAM_MAH * (1.0 - INIS_REZERV_ORANI)  # 9600 mAh

# Kalkış, tırmanış, su bırakma yaklaşması ve iniş için ayrılan kapasite rezervi (mAh)
SEYIR_DISI_KAPASITE_BUTCESI_MAH = 2000.0
# Sadece 3 turluk seyir (cruise) için kullanılabilecek maksimum kapasite sınırı (7600 mAh)
SEYIR_MAX_KULLANILABILIR_MAH = KULLANILABILIR_TOPLAM_MAH - SEYIR_DISI_KAPASITE_BUTCESI_MAH

MAX_ENERJI_PUANI = 60.0         # Enerji sıralamasında 1. olan adayın alacağı maksimum puan
MAX_UCUS_SURESI_SN = 600.0      # Kalkıştan inişe TÜM uçuşun süre sınırı (sn)
NAVIGASYON_DISI_SURE_SN = 100.0 # Kullanıcının geçici bütçesi: kalkış, tırmanış,
                               # TOAL/rota geçişleri, yük bırakma yaklaşması ve iniş dahil.
UCUS_SURE_EMNIYET_ORANI = 0.02 # 600 saniyenin %2'si = 12 sn; hedef toplam <= 588 sn.
STALL_EMNIYET_ORANI = 0.85      # Kanat CL_max emniyet payı; CSV kuyruk stall sınırı içermiyor
KUYRUK_OTURMA_DEG = 0.0        # Kanat referansına göre; downwash ve trim çözülmez
GEOMETRI_RTOL = 1e-4           # CSV yazım hassasiyeti için; şekil varsayımı eklenmez
GEOMETRI_ATOL = 6e-6
ROTA_SINIR_ORNEK_ADIM_M = 0.5   # Sınır kontrolü örnekleme adımı (sürekli ispat değildir)


# ==============================================================================
# 2. DOSYA YOLU ÇÖZÜCÜ VE VERİ YÜKLEME MOTORU
# ==============================================================================
def find_csv_file(filename):
    search_paths = [
        filename,
        os.path.join("Scripts", filename),
        os.path.join("..", filename),
        os.path.join("..", "Scripts", filename),
    ]
    for path in search_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"'{filename}' bulunamadı! Çalışma dizinini kontrol edin.")


def load_all_databases():
    itki_path = find_csv_file("itki_tablosu.csv")
    vsp_path = find_csv_file("openvsp_llt_flapsiz_polar_detay.csv")
    llt_path = find_csv_file("llt_flapsiz_tum_sonuclar.csv")
    kutle_path = find_csv_file("kanat_kuyruk_govde_kutleli.csv")

    itki_df = pd.read_csv(itki_path, sep=";")
    vsp_df = pd.read_csv(vsp_path)
    llt_df = pd.read_csv(llt_path)
    kutle_df = pd.read_csv(kutle_path)

    return itki_df, vsp_df, llt_df, kutle_df


# ==============================================================================
# 3. AERODİNAMİK VE İTKİ ENTERPOLASYON TABLOLARININ HAZIRLANMASI
# ==============================================================================
def geometry_key(profil, ar_w, b_w):
    if pd.isna(profil) or not str(profil).strip():
        raise ValueError("Kanat profil adı boş olamaz.")
    values = np.asarray([ar_w, b_w], dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("AR_w ve b_w_m sonlu, pozitif olmalı.")
    # Erken yuvarlama yapma: farklı geometriler aynı anahtarda birleşmesin.
    return (str(profil).strip(), float(ar_w), float(b_w))


def unique_value(group, column, context, numeric=True):
    values = group[column]
    if values.isna().any():
        raise ValueError(f"{context}: {column} içinde eksik değer var.")
    if numeric:
        values = values.to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.allclose(values, values[0], rtol=1e-9, atol=1e-12):
            raise ValueError(f"{context}: {column} değerleri çelişkili veya sonlu değil.")
        return float(values[0])
    values = values.astype(str).str.strip()
    if values.nunique() != 1:
        raise ValueError(f"{context}: {column} değerleri çelişkili.")
    return values.iloc[0]


class LookupTable(dict):
    def __init__(self):
        super().__init__()
        self.errors = []


def add_lookup_record(table, key, value):
    # Aynı kanadı kullanan farklı ıslak alanlı adaylar geçerlidir.
    # Önceki kaydı ezmeden bütün referans kayıtlarını sakla.
    table.setdefault(key, []).append(value)


# Drag CSV'si boyutları 5 ondalıkla yazar. Kuyruk ve gövdeyi birlikte
# doğrula; yalnız kanadı aynı olan başka bir uçağın CD0 değerlerini alma.
VSP_GEOMETRY_COLUMNS = {
    "S_w_m2": "S_w_m2", "c_w_m": "c_root_w_m",
    "S_HT_m2": "S_HT_m2", "b_HT_m": "b_HT_m",
    "c_HT_m": "c_root_HT_m", "S_VT_m2": "S_VT_m2",
    "h_VT_m": "h_VT_m", "c_VT_m": "c_root_VT_m",
    "tail_arm_m": "tail_arm_m", "L_govde_m": "govde_L_toplam_m",
    "D_govde_m": "govde_D_m",
}


def match_geometry(table, key, context, wet_area=None, candidate=None):
    # Yalnızca çok küçük dosya yazım hassasiyeti farklarına izin ver.
    matches = [k for k in table if k[0] == key[0]
               and np.allclose(k[1:], key[1:], rtol=1e-9, atol=5.1e-6)]
    if len(matches) != 1:
        raise ValueError(f"{context}: {key} için {len(matches)} eşleşme bulundu; tam bir eşleşme gerekli.")
    records = table[matches[0]]
    if candidate is not None:
        target = np.asarray([candidate[c] for c in VSP_GEOMETRY_COLUMNS.values()], dtype=float)
        if not np.isfinite(target).all() or (target <= 0).any():
            raise ValueError(f"{context}: kuyruk/gövde boyutları sonlu ve pozitif olmalı.")
        records = [r for r in records
                   if np.allclose([r[c] for c in VSP_GEOMETRY_COLUMNS], target,
                                  rtol=1e-9, atol=5.1e-6)
                   and r["kuyruk_airfoil"] == str(candidate["kuyruk_airfoil"]).strip()]
        if not records:
            raise ValueError(f"{context}: aynı kuyruk/gövde geometrisi bulunamadı. "
                             "Aday ve drag CSV dosyalarını aynı ayarlarla yeniden üretin.")
    if wet_area is not None:
        if not np.isfinite(wet_area) or wet_area <= 0:
            raise ValueError(f"{context}: gövde ıslak alanı sonlu ve pozitif olmalı.")
        areas = np.asarray([record["S_wet_govde_base"] for record in records])
        if not np.isfinite(areas).all() or (areas <= 0).any():
            raise ValueError(f"{context}: referans gövde ıslak alanları pozitif olmalı.")
        records = [record for record in records
                   if np.isclose(record["S_wet_govde_base"], wet_area, rtol=1e-9, atol=5.1e-6)]
        if not records:
            raise ValueError(f"{context}: aynı gövde ıslak alanı bulunamadı; drag CSV'sini yeniden üretin.")
    reference = records[0]
    # Aynı kanadın ortak LLT verisi tekrar edebilir; kimlik farkı hata değildir.
    # Aynı referansın gerçekten çelişkili aerodinamik verisini gizleme.
    fields = [field for field in reference if field not in ("kanat_aday_no", "kuyruk_airfoil")]
    for record in records[1:]:
        if not np.allclose([reference[field] for field in fields],
                           [record[field] for field in fields], rtol=1e-9, atol=1e-12):
            raise ValueError(f"{context}: aynı referans için çelişkili aerodinamik veri var: {key}.")
    return reference


def build_lookup_tables(itki_df, vsp_df, llt_df):
    from drag_hesabi import hesapla_openvsp_cd0
    from lifting_line_cozucu import LINEER_ALPHA_MIN, LINEER_ALPHA_MAX
    if llt_df.empty or vsp_df.empty:
        raise ValueError("LLT ve OpenVSP tabloları boş olamaz.")

    # --- A. LLT Çoklu Reynolds Enterpolasyonu (V = 20 m/s için) ---
    llt_lookup = LookupTable()
    for aday_no, grp in llt_df.groupby("aday_no", dropna=False):
        try:
            if not np.isfinite(float(aday_no)) or float(aday_no) != int(aday_no):
                raise ValueError("LLT aday kimliği eksik/geçersiz.")
            context = f"LLT aday {aday_no}"
            key = geometry_key(unique_value(grp, "profil", context, numeric=False),
                               unique_value(grp, "AR_w", context),
                               unique_value(grp, "b_w_m", context))
            chord = unique_value(grp, "c_w_m", context)
            if "S_w_m2" in grp:
                area = unique_value(grp, "S_w_m2", context)
                if not np.isclose(key[1], key[2] ** 2 / area, rtol=GEOMETRI_RTOL, atol=GEOMETRI_ATOL):
                    raise ValueError(f"{context}: AR, açıklık ve alan tutarsız.")
            if min(chord, V_SEYIR, RHO_HAVA, MU_HAVA) <= 0:
                raise ValueError(f"{context}: veter ve akış koşulları pozitif olmalı.")
            re_20 = RHO_HAVA * V_SEYIR * chord / MU_HAVA
            polar_columns = ["CL_alpha_3d_per_deg", "alpha_L0_deg", "e_oswald",
                             "CL_max_3d_tahmini"]
            if not np.isfinite(grp[["Re_polar"] + polar_columns].to_numpy(dtype=float)).all():
                raise ValueError(f"{context}: polar tablosunda eksik/sonlu olmayan değer var.")
            if (grp["Re_polar"] <= 0).any():
                raise ValueError(f"{context}: Re_polar pozitif olmalı.")
            polar_rows = []
            for reynolds, same_re in grp.groupby("Re_polar", sort=True):
                polar_rows.append([float(reynolds)] + [unique_value(same_re, c, context) for c in polar_columns])
            polars = np.asarray(polar_rows, dtype=float)
            re_polars = polars[:, 0]
            tolerance = 1e-9 * max(abs(re_20), abs(re_polars[-1]), 1.0)
            if re_20 < re_polars[0] - tolerance or re_20 > re_polars[-1] + tolerance:
                raise ValueError(f"{context}: hedef Re={re_20:.1f}, veri aralığı "
                                 f"[{re_polars[0]:.1f}, {re_polars[-1]:.1f}] dışında. "
                                 "Bu Reynolds aralığı için polar verisi ekleyin.")
            values = [float(np.interp(re_20, re_polars, polars[:, j + 1])) for j in range(len(polar_columns))]
            cl_alpha, alpha_l0, e_oswald, cl_max_3d = values
            if min(cl_alpha, e_oswald, cl_max_3d) <= 0:
                raise ValueError(f"{context}: taşıma eğimi, e ve CL_max pozitif olmalı.")
            add_lookup_record(llt_lookup, key, {
                "kanat_aday_no": int(aday_no),
                "Re_20": re_20,
                "c_w_m": chord,
                "S_w_m2": key[2] ** 2 / key[1],
                "CL_alpha_3d_per_deg": cl_alpha,
                "alpha_L0_deg": alpha_l0,
                "e_oswald": e_oswald,
                "CL_max_3d": cl_max_3d,
                "alpha_min_deg": float(LINEER_ALPHA_MIN),
                "alpha_max_deg": float(LINEER_ALPHA_MAX),
            })
        except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
            llt_lookup.errors.append(f"aday {aday_no}: {exc}")

    # --- B. OpenVSP CD0 Tablosu: tüm kimlikleri tara, çelişkili satırları reddet ---
    vsp_lookup = LookupTable()
    for (aday_no, _wet_area), sub in vsp_df.groupby(["aday_no", "S_wet_govde_m2"], sort=False, dropna=False):
        try:
            if not np.isfinite(float(aday_no)) or float(aday_no) != int(aday_no):
                raise ValueError("OpenVSP aday kimliği eksik/geçersiz.")
            context = f"OpenVSP aday {aday_no}"
            key = geometry_key(unique_value(sub, "kanat_airfoil", context, numeric=False),
                               unique_value(sub, "AR_w", context),
                               unique_value(sub, "b_w_m", context))
            # Mevcut drag fonksiyonunu yeniden kullan; 18 m/s katsayılarını
            # doğrudan 20 m/s'ye taşıma. Geometri ve t/c CSV'den korunur.
            get = lambda c: unique_value(sub, c, context)
            cd0 = hesapla_openvsp_cd0(
                S_w=get("S_w_m2"), Re_w=RHO_HAVA * V_SEYIR * get("c_w_m") / MU_HAVA,
                t_c_w=get("t_c_w"), S_HT=get("S_HT_m2"),
                Re_HT=RHO_HAVA * V_SEYIR * get("c_HT_m") / MU_HAVA,
                t_c_HT=get("t_c_HT"), S_VT=get("S_VT_m2"),
                Re_VT=RHO_HAVA * V_SEYIR * get("c_VT_m") / MU_HAVA,
                t_c_VT=get("t_c_VT"), L_govde=get("L_govde_m"),
                D_govde=get("D_govde_m"), S_wet_govde=get("S_wet_govde_m2"),
                V_seyir=V_SEYIR, rho=RHO_HAVA, mu=MU_HAVA)
            add_lookup_record(vsp_lookup, key, {
                **{c: unique_value(sub, c, context) for c in VSP_GEOMETRY_COLUMNS},
                "kuyruk_airfoil": unique_value(sub, "kuyruk_airfoil", context, numeric=False),
                "CD0_w": cd0["CD0_w"],
                "CD0_HT": cd0["CD0_HT"],
                "CD0_VT": cd0["CD0_VT"],
                "CD0_govde_base": cd0["CD0_govde"],
                "CD_0_base": cd0["CD_0"],
                "S_wet_govde_base": unique_value(sub, "S_wet_govde_m2", context),
            })
        except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
            vsp_lookup.errors.append(f"aday {aday_no}: {exc}")

    # --- C. İtki Tablosu (V = 20 m/s Hız Dilimi + Kısık Gaz Orijin Uzatması) ---
    itki_20 = itki_df[(itki_df["pervane"].astype(str).str.strip() == PERVANE_SECIMI)
                      & np.isclose(itki_df["hiz_m_s"], V_SEYIR)].sort_values("itki_n")
    if itki_20.empty:
        raise ValueError(f"İtki tablosunda {PERVANE_SECIMI}, {V_SEYIR} m/s verisi bulunamadı!")
    numeric = itki_20[["itki_n", "gaz_yuzde", "akim_a"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric[:, 0] <= 0).any() or (numeric[:, 2] < 0).any():
        raise ValueError("İtki pozitif, akım negatif olmayan ve tüm tablo değerleri sonlu olmalı.")
    if itki_20["itki_n"].duplicated().any():
        raise ValueError("Aynı itki için birden fazla kayıt var; akım enterpolasyonu belirsiz.")

    # CSV akim_a sütunu amperdir. Mevcut düşük gaz orijin varsayımı korunur.
    # Güç yalnız sabit 24 V ile hesaplanır; akım güçten geri türetilmez.
    itki_vals = np.insert(itki_20["itki_n"].values, 0, 0.0)
    gaz_vals = np.insert(itki_20["gaz_yuzde"].values, 0, 0.0)
    current_vals = np.insert(itki_20["akim_a"].to_numpy(dtype=float), 0, 0.0)

    thrust_table_info = {
        "pervane": str(itki_20["pervane"].iloc[0]),
        "itki_n": itki_vals,
        "gaz_yuzde": gaz_vals,
        "current_a": current_vals,
        "power_w": V_BAT_SABIT * current_vals,
        "max_itki_n": float(itki_20["itki_n"].max()),
        "min_tablo_itki_n": float(itki_20["itki_n"].min()),
    }

    return llt_lookup, vsp_lookup, thrust_table_info


# ==============================================================================
# 4. ROTA GEOMETRİSİ VE ELEKTRİKSEL GÜÇ / ENERJİ HESAPLAYICI
# ==============================================================================
def map_reference_geometry():
    """Örnek görselden yaklaşık merkezler; GPS/ölçülmüş arazi koordinatı değildir.

    Ölçek: WP1 -> ... -> WP10 -> WP1 düz bağlantılarının toplamı 2800 m.
    X görselde sağa, Y yukarı; başlangıç TOAL. Kuzey yönü varsayılmaz.
    """
    wp_pixels = np.array([
        [251, 556], [214, 480], [419, 333], [410, 225], [339, 284],
        [527, 248], [380, 99], [234, 336], [590, 335], [654, 415]
    ], dtype=float)
    boundary_pixels = np.array([
        [61, 676], [142, 421], [239, 203], [328, 47], [441, 18],
        [638, 247], [868, 416], [548, 550], [568, 600], [500, 624],
        [484, 577], [211, 676]
    ], dtype=float)
    origin = np.array([517.0, 545.0])
    perimeter_px = np.linalg.norm(np.roll(wp_pixels, -1, axis=0) - wp_pixels, axis=1).sum()
    scale = L_TABAN_ROTA_M / perimeter_px

    def convert(points):
        return (np.asarray(points, dtype=float) - origin) * np.array([scale, -scale])

    return {
        "wp_coords": convert(wp_pixels),
        "wp_names": [f"WP{i}" for i in range(1, 11)],
        "boundary_coords": convert(boundary_pixels),
        "toal_coord": np.zeros(2),
        "drop_coords": convert([[555, 473], [499, 495], [477, 505]]),
        "meters_per_pixel": scale,
    }


def _left_normal(heading):
    return np.array([-np.sin(heading), np.cos(heading)])


def _turn_sweep(h0, h1, direction):
    sweep = float((direction * (h1 - h0)) % (2.0 * np.pi))
    return 0.0 if abs(sweep - 2.0 * np.pi) < 1e-10 else sweep


def _shortest_csc(p0, h0, p1, h1, radius):
    options = _csc_candidates(p0, h0, p1, h1, radius)
    if not options:
        raise ValueError("Bu başlangıç/bitiş yönleri için CSC bağlantısı bulunamadı.")
    return options[0]


def _sample_segment(segment, step_m=0.5):
    count = max(2, int(np.ceil(segment["length"] / step_m)) + 1)
    if segment["kind"] == "straight":
        return np.linspace(segment["start"], segment["end"], count)
    angles = segment["angle0"] + segment["direction"] * np.linspace(0.0, segment["sweep"], count)
    return segment["center"] + segment["radius"] * np.column_stack([np.cos(angles), np.sin(angles)])


def _segment_point_distance(segment, point):
    """Doğru parçasına / dairesel yaya analitik en kısa mesafe."""
    if segment["kind"] == "straight":
        delta = segment["end"] - segment["start"]
        fraction = np.clip(np.dot(point - segment["start"], delta)
                           / max(np.dot(delta, delta), 1e-20), 0.0, 1.0)
        return float(np.linalg.norm(point - (segment["start"] + fraction * delta)))
    delta = point - segment["center"]
    angle = float(np.arctan2(delta[1], delta[0]))
    sweep = _turn_sweep(segment["angle0"], angle, segment["direction"])
    if sweep <= segment["sweep"] + 1e-10:
        return abs(float(np.linalg.norm(delta)) - segment["radius"])
    return min(float(np.linalg.norm(point - segment["start"])),
               float(np.linalg.norm(point - segment["end"])))


def _points_in_polygon(points, polygon):
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if abs(b[1] - a[1]) < 1e-12:
            continue
        inside ^= ((a[1] > y) != (b[1] > y)) & (
            x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0])
    return inside


def compute_mission_timing(route_info):
    """600 s tüm uçuş; %2 pay kullanılabilir süreyi 588 s'ye indirir."""
    if not 0 <= UCUS_SURE_EMNIYET_ORANI < 1 or NAVIGASYON_DISI_SURE_SN < 0:
        raise ValueError("Süre payı [0, 1) aralığında, navigasyon dışı süre negatif olmayan bir değer olmalı.")
    navigation = (N_TUR_YUKLU + N_TUR_BOS) * route_info["t_total_lap_s"]
    reserve = MAX_UCUS_SURESI_SN * UCUS_SURE_EMNIYET_ORANI
    usable = MAX_UCUS_SURESI_SN - reserve
    total = navigation + NAVIGASYON_DISI_SURE_SN
    return dict(navigasyon_suresi_sn=navigation,
                navigasyon_disi_sure_sn=NAVIGASYON_DISI_SURE_SN,
                toplam_sure_sn=total, sure_emniyet_payi_sn=reserve,
                kullanilabilir_ucus_suresi_sn=usable,
                navigasyon_sure_butcesi_sn=usable - NAVIGASYON_DISI_SURE_SN,
                sinira_kalan_sure_sn=MAX_UCUS_SURESI_SN - total,
                pay_sonrasi_kalan_sure_sn=usable - total,
                sure_uygun=total <= usable)


def _csc_candidates(p0, h0, p1, h1, radius):
    """Dört CSC bağlantısını döndür; kısıtlar değerlendirilmeden seçenek eleme."""
    options = []
    for s0 in (-1, 1):
        for s1 in (-1, 1):
            c0 = p0 + s0 * radius * _left_normal(h0)
            c1 = p1 + s1 * radius * _left_normal(h1)
            delta = c1 - c0
            distance = float(np.linalg.norm(delta))
            offset = (s1 - s0) * radius
            if distance < 1e-9 or distance < abs(offset):
                continue
            heading = float(np.arctan2(delta[1], delta[0])
                            - np.arcsin(np.clip(offset / distance, -1.0, 1.0)))
            sweep0 = _turn_sweep(h0, heading, s0)
            sweep1 = _turn_sweep(heading, h1, s1)
            straight = float(np.sqrt(max(0.0, distance**2-offset**2)))
            options.append(dict(length=radius*(sweep0+sweep1)+straight,
                                c0=c0, c1=c1, heading=heading, sweep0=sweep0, sweep1=sweep1,
                                s0=s0, s1=s1, straight=straight))
    return sorted(options, key=lambda item: item["length"])


def _csc_segments(p0, h0, p1, h1, radius, data):
    tangent0 = data["c0"] - data["s0"]*radius*_left_normal(data["heading"])
    tangent1 = data["c1"] - data["s1"]*radius*_left_normal(data["heading"])
    return [
        dict(kind="turn", start=p0, end=tangent0, center=data["c0"], radius=radius,
             direction=data["s0"], sweep=data["sweep0"], angle0=h0-data["s0"]*np.pi/2,
             heading_start=h0, heading_end=data["heading"], length=radius*data["sweep0"]),
        dict(kind="straight", start=tangent0, end=tangent1, heading_start=data["heading"],
             heading_end=data["heading"], length=data["straight"]),
        dict(kind="turn", start=tangent1, end=p1, center=data["c1"], radius=radius,
             direction=data["s1"], sweep=data["sweep1"], angle0=data["heading"]-data["s1"]*np.pi/2,
             heading_start=data["heading"], heading_end=h1, length=radius*data["sweep1"]),
    ]


def _route_leg_valid(segments, waypoints, boundary):
    for segment in segments:
        if any(_segment_point_distance(segment, point) < WP_CLEARANCE_M-1e-6 for point in waypoints):
            return False
        if not _points_in_polygon(_sample_segment(segment, ROTA_SINIR_ORNEK_ADIM_M), boundary).all():
            return False
    return True


def _best_feasible_csc(p0, h0, p1, h1, radius, waypoints, boundary):
    for data in _csc_candidates(p0, h0, p1, h1, radius):
        segments = _csc_segments(p0, h0, p1, h1, radius, data)
        if _route_leg_valid(segments, waypoints, boundary):
            return {**data, "segments": segments}
    return None


def _shortest_heading_cycle(costs):
    """Katmanlı yön grafiğinde kapalı tur; sonsuz maliyetli geçişler yasak."""
    count, width, _ = costs.shape
    best_cost, best_sequence = np.inf, None
    for start in range(width):
        distance = np.full(width, np.inf)
        distance[start] = 0.0
        parents = []
        for edge in costs[:-1]:
            choices = distance[:, None] + edge
            parent = choices.argmin(axis=0)
            distance = choices[parent, np.arange(width)]
            parents.append(parent)
        final = distance + costs[-1, :, start]
        end = int(final.argmin())
        if final[end] < best_cost:
            best_cost = float(final[end])
            sequence = [end]
            for parent in reversed(parents):
                sequence.append(int(parent[sequence[-1]]))
            best_sequence = list(reversed(sequence))
    return best_sequence


def compute_waypoint_route_geometry(V_cruise, bank_angle_deg):
    """Kısıtlı CSC rotası: geçerli yön grafiği + yalnız geçerli yerel iyileştirmeler.

    Küresel sürekli optimum/CCC iddiası yoktur. Alan sınırı 0.5 m örneklemeyle,
    waypoint açıklıkları analitik mesafeyle her aday bağlantıda denetlenir.
    """
    if not np.isfinite([V_cruise, bank_angle_deg]).all() or V_cruise <= 0 or not 0 < bank_angle_deg < 90:
        raise ValueError("Hız pozitif, yatış açısı 0 ile 90 derece arasında olmalı.")
    if not np.isfinite(WP_CLEARANCE_M) or WP_CLEARANCE_M < 0:
        raise ValueError("Waypoint açıklığı sonlu ve negatif olmayan olmalı.")
    reference = map_reference_geometry()
    wp, boundary = reference["wp_coords"], reference["boundary_coords"]
    count = len(wp)
    if count != N_WAYPOINT:
        raise ValueError("N_WAYPOINT ile haritadaki waypoint sayısı uyuşmuyor.")
    phi = np.radians(bank_angle_deg)
    radius = V_cruise**2/(G_YERCEKIMI*np.tan(phi))
    edges = np.roll(wp, -1, axis=0)-wp
    lengths = np.linalg.norm(edges, axis=1)
    if (lengths <= 1e-9).any():
        raise ValueError("Ardışık waypoint noktaları aynı olamaz.")
    unit = edges/lengths[:, None]
    incoming = np.roll(unit, 1, axis=0)
    directions = incoming+unit
    # Tam geri dönüşte açıortay tanımsız; başlangıç yönü çıkış kenarından alınır.
    degenerate = np.linalg.norm(directions, axis=1) < 1e-10
    directions[degenerate] = unit[degenerate]
    initial = np.arctan2(directions[:, 1], directions[:, 0])
    sides = np.where(incoming[:, 0]*unit[:, 1]-incoming[:, 1]*unit[:, 0] >= 0, -1.0, 1.0)

    def gate(i, heading):
        return wp[i]+sides[i]*WP_CLEARANCE_M*_left_normal(heading)

    @lru_cache(maxsize=50000)
    def leg(i, h0, h1):
        j = (i+1) % count
        p0, p1 = gate(i, h0), gate(j, h1)
        if not _points_in_polygon(np.array([p0, p1]), boundary).all():
            return None
        return _best_feasible_csc(p0, h0, p1, h1, radius, wp, boundary)

    headings = None
    # Önce kaba grid; çözüm yoksa daha sık yönlerle yeniden ara.
    for step_deg in (30.0, 15.0, 10.0):
        offsets = np.radians(np.arange(0.0, 360.0, step_deg))
        choices = initial[:, None]+offsets[None, :]
        width = len(offsets)
        costs = np.full((count, width, width), np.inf)
        for i in range(count):
            for a in range(width):
                for b in range(width):
                    data = leg(i, float(choices[i, a]), float(choices[(i+1) % count, b]))
                    if data is not None:
                        costs[i, a, b] = data["length"]
        sequence = _shortest_heading_cycle(costs)
        if sequence is not None:
            headings = choices[np.arange(count), sequence].copy()
            break
    if headings is None:
        raise ValueError("Taranan CSC yönlerinde sınır/açıklık kısıtlarını sağlayan kapalı rota bulunamadı. "
                         "Bu sonuç tüm olası rotaların imkânsız olduğunu göstermez.")

    def local_cost(i, heading):
        before = leg((i-1) % count, float(headings[i-1]), float(heading))
        after = leg(i, float(heading), float(headings[(i+1) % count]))
        if before is None or after is None:
            return np.inf
        return before["length"]+after["length"]

    for step in (15.0, 5.0, 1.0, 0.25):
        for _ in range(5):
            changed = False
            for i in range(count):
                options = headings[i]+np.radians(step)*np.arange(-2, 3)
                best = min(options, key=lambda heading: local_cost(i, heading))
                changed |= abs(best-headings[i]) > 1e-12
                headings[i] = best
            if not changed:
                break

    segments, leg_lengths = [], []
    for i in range(count):
        data = leg(i, float(headings[i]), float(headings[(i+1) % count]))
        if data is None:
            raise RuntimeError("İç hata: seçilen rota bağlantısı geçersiz.")
        leg_lengths.append(data["length"])
        for segment in data["segments"]:
            segments.append({**segment, "from_wp": reference["wp_names"][i],
                             "to_wp": reference["wp_names"][(i+1) % count]})
    fly_pts = np.vstack([_sample_segment(seg, ROTA_SINIR_ORNEK_ADIM_M) for seg in segments])
    clearances = np.array([min(_segment_point_distance(seg, point) for seg in segments) for point in wp])
    if not _route_leg_valid(segments, wp, boundary):
        raise RuntimeError("İç hata: son rota doğrulaması başarısız.")
    straight = sum(seg["length"] for seg in segments if seg["kind"] == "straight")
    turn = sum(seg["length"] for seg in segments if seg["kind"] == "turn")
    return {**reference, "fly_pts": fly_pts, "segments": segments,
            "gate_coords": np.array([gate(i, h) for i, h in enumerate(headings)]),
            "gate_headings_rad": headings, "leg_lengths_m": np.array(leg_lengths),
            "wp_min_clearance_m": clearances, "boundary_sample_check_ok": True,
            "bank_angle_deg": bank_angle_deg, "R_turn_m": radius, "n_load_turn": 1/np.cos(phi),
            "L_straight_lap_m": straight, "L_turn_lap_m": turn, "L_total_lap_m": straight+turn,
            "t_straight_lap_s": straight/V_cruise, "t_turn_lap_s": turn/V_cruise,
            "t_total_lap_s": (straight+turn)/V_cruise,
            "rota_yontemi": "kisitli_CSC_yon_grafigi_ve_yerel_iyilestirme"}


def compute_electrical_power_and_current(T_req_N, thrust_info):
    """İtki tablosundan I(A); sabit 24 V ile P(W)=V*I."""
    if not np.isfinite(T_req_N) or T_req_N < 0:
        raise ValueError("Gerekli itki sonlu ve negatif olmayan bir değer olmalı.")
    if T_req_N > thrust_info["max_itki_n"]:
        return (np.nan,) * 4
    throttle_pct = float(np.interp(T_req_N, thrust_info["itki_n"], thrust_info["gaz_yuzde"]))
    current_a = float(np.interp(T_req_N, thrust_info["itki_n"], thrust_info["current_a"]))
    if not np.isfinite(current_a) or current_a < 0:
        raise ValueError("Tablodan bulunan akım sonlu ve negatif olmayan bir değer olmalı.")
    return throttle_pct, current_a, V_BAT_SABIT, V_BAT_SABIT * current_a


def load_tail_lookup(explicit=None):
    """Hazır aday-Re LLT tablosu; kuyruk poları yeniden çözülmez."""
    path = Path(explicit) if explicit is not None else Path(find_csv_file("llt_flapsiz_aday_re.csv"))
    table = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    required = {"aday_no", "profil", "b_w_m", "AR_w", "S_w_m2", "c_w_m",
                "c_HT_m", "S_HT_m2", "Re_HT", "CL_alpha_HT_per_deg", "e_HT",
                "c_VT_m", "S_VT_m2", "Re_VT", "CL_alpha_VT_per_deg", "e_VT"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Kuyruk LLT tablosunda eksik sütunlar: {', '.join(sorted(missing))}")
    if table.empty:
        raise ValueError("Kuyruk LLT tablosu boş olamaz.")
    table["profil"] = table["profil"].fillna("").astype(str).str.strip().str.lower()
    for column in required - {"profil"}:
        # Hatalı bir satır diğer adayların okunmasını engellemez.
        table[column] = pd.to_numeric(table[column], errors="coerce")
    return table


def tail_for_candidate(row, table):
    """Kanat ve kuyruk geometrisiyle eşleştir; HT katsayılarını CSV'den al."""
    profile = re.sub(r"[^a-z0-9]", "", str(row["kuyruk_airfoil"]).lower())
    if profile != "naca0010":
        raise ValueError("Kuyruk sıfır-taşıma açısı kabulü NACA0010 içindir; başka profil için geçerli değildir.")
    mask = table["profil"].eq(str(row["kanat_airfoil"]).strip().lower()).to_numpy(copy=True)
    fields = {"b_w_m": "b_w_m", "AR_w": "AR_w", "S_w_m2": "S_w_m2",
              "c_w_m": "c_root_w_m", "c_HT_m": "c_root_HT_m", "S_HT_m2": "S_HT_m2",
              "c_VT_m": "c_root_VT_m", "S_VT_m2": "S_VT_m2"}
    for reference, candidate in fields.items():
        mask &= np.isclose(table[reference].to_numpy(dtype=float), float(row[candidate]),
                           rtol=GEOMETRI_RTOL, atol=GEOMETRI_ATOL)
    matched = table.loc[mask]
    if matched.empty:
        raise ValueError("llt_flapsiz_aday_re.csv içinde aynı kanat/kuyruk geometrisi bulunamadı.")
    context = "Kuyruk LLT eşleşmesi"
    values = {key: unique_value(matched, key, context) for key in
              ("CL_alpha_HT_per_deg", "e_HT", "Re_HT", "CL_alpha_VT_per_deg", "e_VT", "Re_VT")}
    if min(values.values()) <= 0:
        raise ValueError("Kuyruk taşıma eğimleri, e ve Reynolds değerleri pozitif olmalı.")
    reference_ar = unique_value(matched, "S_HT_m2", context) / unique_value(matched, "c_HT_m", context)**2
    if not np.isclose(reference_ar, float(row["AR_HT"]), rtol=GEOMETRI_RTOL, atol=GEOMETRI_ATOL):
        raise ValueError("Dikdörtgen kuyruk için CSV alan/veter oranı aday AR_HT ile uyuşmuyor.")
    target_ht = RHO_HAVA * V_SEYIR * float(row["c_root_HT_m"]) / MU_HAVA
    target_vt = RHO_HAVA * V_SEYIR * float(row["c_root_VT_m"]) / MU_HAVA
    re_ok = bool(np.isclose(values["Re_HT"], target_ht, rtol=1e-3, atol=1.)
                 and np.isclose(values["Re_VT"], target_vt, rtol=1e-3, atol=1.))
    # Dosyadaki Cl_max_2d / alpha_stall_2d_deg kanada aittir; kuyruğa uygulanmaz.
    # -4..6 derece mevcut doğrusal model aralığıdır, CSV'den doğrulanmış stall sınırı değildir.
    return dict(slope=values["CL_alpha_HT_per_deg"], e=values["e_HT"],
                Re=values["Re_HT"], Re_target=target_ht,
                Re_VT=values["Re_VT"], Re_VT_target=target_vt,
                slope_VT=values["CL_alpha_VT_per_deg"], e_VT=values["e_VT"],
                AR=float(row["AR_HT"]), S=float(row["S_HT_m2"]),
                alpha_min=-4.0, alpha_max=6.0, reynolds_match=re_ok,
                stall_verified=False, source="llt_flapsiz_aday_re.csv")


def validate_candidate(row):
    for span, area, aspect, chord in (
        ("b_w_m", "S_w_m2", "AR_w", "c_root_w_m"),
        ("b_HT_m", "S_HT_m2", "AR_HT", "c_root_HT_m"),
        ("h_VT_m", "S_VT_m2", "AR_VT", "c_root_VT_m"),
    ):
        b, s, ar, c = [float(row[key]) for key in (span, area, aspect, chord)]
        if not np.isfinite([b, s, ar, c]).all() or min(b, s, ar, c) <= 0:
            raise ValueError(f"Geometri: {span}, {area}, {aspect}, {chord} pozitif ve sonlu olmalı.")
        if not np.isclose(ar, b*b/s, rtol=GEOMETRI_RTOL, atol=GEOMETRI_ATOL):
            raise ValueError(f"Geometri: {aspect} != {span}²/{area}.")
    # Kuyruk LLT modeli dikdörtgen; bu kabulü veriyle doğrula.
    if not np.isclose(float(row["S_HT_m2"]), float(row["b_HT_m"])*float(row["c_root_HT_m"]),
                      rtol=GEOMETRI_RTOL, atol=GEOMETRI_ATOL):
        raise ValueError("Kuyruk LLT hesabı dikdörtgen yatay kuyruk gerektiriyor: S_HT != b_HT*c_HT.")
    loaded, empty, water = [float(row[k]) for k in ("total_kutle_kg", "total_susuz_kutle_kg", "govde_su_g")]
    water /= 1000.0
    if not np.isfinite([loaded, empty, water]).all() or min(loaded, empty) <= 0 or water < 0:
        raise ValueError("Kütleler sonlu; yüklü/boş kütle pozitif, su kütlesi negatif olmayan olmalı.")
    if not np.isclose(loaded, empty + water, rtol=1e-4, atol=1e-5):
        raise ValueError("Kütle tutarsız: yüklü kütle != susuz kütle + su kütlesi.")


def validate_wing_reference(row, reference):
    for candidate_field, reference_field in (("S_w_m2", "S_w_m2"), ("c_root_w_m", "c_w_m")):
        if not np.isclose(float(row[candidate_field]), reference[reference_field],
                          rtol=GEOMETRI_RTOL, atol=GEOMETRI_ATOL):
            raise ValueError(f"Kanat LLT ve aday geometrisi uyuşmuyor: {candidate_field}.")


def evaluate_cruise_phase(mass_kg, S_w, AR_w, CD_0, e_oswald, CL_alpha_deg, alpha_L0_deg, CL_max_3d, route_info, thrust_info,
                          alpha_min_deg=-4.0, alpha_max_deg=6.0, tail_data=None):
    """Kanat+kuyruk taşıma dengesi; i_HT sabit, downwash=0, moment trimi çözülmez."""
    positive = [mass_kg, S_w, AR_w, e_oswald, CL_alpha_deg, CL_max_3d]
    if not np.isfinite(positive).all() or min(positive) <= 0 or not np.isfinite(CD_0) or CD_0 < 0:
        raise ValueError("Kütle/alan/aerodinamik girdiler geçersiz.")
    if tail_data is None:
        raise ValueError("Yatay kuyruk aerodinamik verisi gerekli.")
    q = 0.5 * RHO_HAVA * V_SEYIR ** 2
    weight = mass_kg * G_YERCEKIMI
    ratio = tail_data["S"] / S_w
    slope_tail = tail_data["slope"]

    def state(load):
        required = load * weight / (q * S_w)
        # CL_w=a_w*(alpha-alpha_L0); CL_HT=a_HT*(alpha+i_HT), NACA0010 alpha_L0=0.
        alpha = (required + CL_alpha_deg*alpha_L0_deg - ratio*slope_tail*KUYRUK_OTURMA_DEG) / (CL_alpha_deg + ratio*slope_tail)
        clw = CL_alpha_deg * (alpha - alpha_L0_deg)
        alpha_tail = alpha + KUYRUK_OTURMA_DEG
        clh = slope_tail * alpha_tail
        cdiw = clw ** 2 / (np.pi * AR_w * e_oswald)
        cdih = ratio * clh ** 2 / (np.pi * tail_data["AR"] * tail_data["e"])
        cd = CD_0 + cdiw + cdih
        thrust = q * S_w * cd
        throttle, current, voltage, power = compute_electrical_power_and_current(thrust, thrust_info)
        # Bu CSV'de kuyruk CL_max yok: stall kontrolü yalnız kanat için yapılır.
        # Kuyruk stall doğrulaması ayrıca False olarak raporlanır.
        stall = 0 <= clw <= STALL_EMNIYET_ORANI * CL_max_3d
        polar = (alpha_min_deg <= alpha <= alpha_max_deg
                 and tail_data["alpha_min"] <= alpha_tail <= tail_data["alpha_max"])
        return dict(CL=required, CL_w=clw, CL_HT=clh, CD=cd, CDi_w=cdiw, CDi_HT=cdih,
                    LD=required/cd, alpha=alpha, alpha_HT=alpha_tail, thrust=thrust,
                    throttle=throttle, current=current, voltage=voltage, power=power,
                    L_w=q*S_w*clw, L_HT=q*tail_data["S"]*clh,
                    lift_error=q*S_w*(clw+ratio*clh)-load*weight, stall=stall, polar=polar)

    straight, turn = state(1.0), state(route_info["n_load_turn"])
    stall_ok = bool(straight["stall"] and turn["stall"])
    thrust_ok = max(straight["thrust"], turn["thrust"]) <= thrust_info["max_itki_n"]
    polar_ok = bool(straight["polar"] and turn["polar"])
    energy_ok = bool(stall_ok and thrust_ok and polar_ok
                     and np.isfinite([straight["current"], turn["current"]]).all())
    t_s, t_t = route_info["t_straight_lap_s"], route_info["t_turn_lap_s"]
    mah = (straight["current"]*t_s + turn["current"]*t_t) * 1000.0/3600.0
    joule = straight["power"]*t_s + turn["power"]*t_t
    result = dict(mah_per_lap=mah if energy_ok else np.nan,
                  joule_per_lap=joule if energy_ok else np.nan,
                  wh_per_lap=joule/3600.0 if energy_ok else np.nan,
                  stall_ok=stall_ok, thrust_ok=bool(thrust_ok), polar_ok=polar_ok, energy_ok=energy_ok,
                  low_throttle_assumption=min(straight["thrust"], turn["thrust"]) < thrust_info["min_tablo_itki_n"])
    for name, phase in (("straight", straight), ("turn", turn)):
        for prefix, key, suffix in (("CL", "CL", ""), ("CD", "CD", ""), ("LD", "LD", ""),
                                    ("alpha", "alpha", "_deg"), ("T_req", "thrust", "_N"),
                                    ("throttle", "throttle", "_pct"), ("current", "current", "_A"),
                                    ("vbat", "voltage", "_V"), ("power", "power", "_W"),
                                    ("CL_w", "CL_w", ""), ("CL_HT", "CL_HT", ""),
                                    ("CDi_w", "CDi_w", ""), ("CDi_HT", "CDi_HT", ""),
                                    ("L_w", "L_w", "_N"), ("L_HT", "L_HT", "_N"),
                                    ("L_denge_hatasi", "lift_error", "_N"), ("alpha_HT", "alpha_HT", "_deg")):
            result[prefix + "_" + name + suffix] = phase[key]
    return result


# ==============================================================================
# 5. TÜM ADAYLARIN TARANMASI VE BAĞIL ENERJİ PUANLANDIRMA SİMÜLASYONU
# ==============================================================================
def score_feasible_candidates(res_df):
    """Yalnız geçerli seyir adaylarını puanla; geçersiz puanları boş bırak."""
    res_df["puan_enerji_60"] = np.nan
    res_df["toplam_yarisma_puani"] = np.nan
    mask = (res_df["feasible"] & np.isfinite(res_df["ham_enerji_endeksi"])
            & (res_df["ham_enerji_endeksi"] > 0))
    if mask.any():
        best_index = res_df.loc[mask, "ham_enerji_endeksi"].max()
        res_df.loc[mask, "puan_enerji_60"] = (
            MAX_ENERJI_PUANI * res_df.loc[mask, "ham_enerji_endeksi"] / best_index)
        res_df.loc[mask, "toplam_yarisma_puani"] = (
            res_df.loc[mask, "puan_navigasyon"] + res_df.loc[mask, "puan_enerji_60"])
    return res_df


def run_full_cruise_mission_optimization(tail_csv_path=None):
    itki_df, vsp_df, llt_df, kutle_df = load_all_databases()
    llt_lookup, vsp_lookup, thrust_info = build_lookup_tables(itki_df, vsp_df, llt_df)
    tail_lookup = load_tail_lookup(tail_csv_path)
    for label, table in (("LLT", llt_lookup), ("OpenVSP", vsp_lookup)):
        for error in table.errors:
            print(f"UYARI: {label} veri grubu atlandı: {error}")
    route_info = compute_waypoint_route_geometry(V_SEYIR, BANK_ANGLE_DEG)
    mission_time = compute_mission_timing(route_info)

    results = []

    for idx, row in kutle_df.iterrows():
        try:
            validate_candidate(row)
            tail_data = tail_for_candidate(row, tail_lookup)
            profil = str(row["kanat_airfoil"]).strip()
            ar_w = float(row["AR_w"])
            b_w = float(row["b_w_m"])
            S_w = float(row["S_w_m2"])
            key = geometry_key(profil, row["AR_w"], row["b_w_m"])

            llt_data = match_geometry(llt_lookup, key, f"Kütle satırı {idx + 1}, LLT")
            vsp_data = match_geometry(vsp_lookup, key, f"Kütle satırı {idx + 1}, OpenVSP",
                                      wet_area=float(row["govde_S_toplam_m2"]), candidate=row)

            validate_wing_reference(row, llt_data)

            # Geometri birebir doğrulandı; başka gövde/kuyruktan ölçekleme yapılmaz.
            CD0_govde_aday = vsp_data["CD0_govde_base"]
            CD_0_total = vsp_data["CD0_w"] + vsp_data["CD0_HT"] + vsp_data["CD0_VT"] + CD0_govde_aday

            m_total_kg = float(row["total_kutle_kg"])
            m_empty_kg = float(row["total_susuz_kutle_kg"])
            m_water_kg = float(row["govde_su_g"]) / 1000.0

            # 1. YÜKLÜ SEYİR ANALİZİ (2 Tur)
            loaded_res = evaluate_cruise_phase(
                mass_kg=m_total_kg, S_w=S_w, AR_w=ar_w, CD_0=CD_0_total,
                e_oswald=llt_data["e_oswald"], CL_alpha_deg=llt_data["CL_alpha_3d_per_deg"],
                alpha_L0_deg=llt_data["alpha_L0_deg"], CL_max_3d=llt_data["CL_max_3d"],
                route_info=route_info, thrust_info=thrust_info,
                alpha_min_deg=llt_data["alpha_min_deg"], alpha_max_deg=llt_data["alpha_max_deg"], tail_data=tail_data,
            )

            # 2. BOŞ SEYİR ANALİZİ (1 Tur)
            empty_res = evaluate_cruise_phase(
                mass_kg=m_empty_kg, S_w=S_w, AR_w=ar_w, CD_0=CD_0_total,
                e_oswald=llt_data["e_oswald"], CL_alpha_deg=llt_data["CL_alpha_3d_per_deg"],
                alpha_L0_deg=llt_data["alpha_L0_deg"], CL_max_3d=llt_data["CL_max_3d"],
                route_info=route_info, thrust_info=thrust_info,
                alpha_min_deg=llt_data["alpha_min_deg"], alpha_max_deg=llt_data["alpha_max_deg"], tail_data=tail_data,
            )

            # 3. TOPLAM GÖREV SÜRESİ, KAPASİTE (mAh) VE ENERJİ (Wh / kJ) TÜKETİMİ
            toplam_tur = N_TUR_YUKLU + N_TUR_BOS
            toplam_mesafe_m = toplam_tur * route_info["L_total_lap_m"]
            # Navigasyon dışı 100 s bütçe ve 12 s emniyet payı ayrı tutulur.
            # Pay uçulmuş süreye eklenmez; izin verilen toplamdan düşülür.
            toplam_sure_sn = mission_time["toplam_sure_sn"]

            seyir_tuketim_mah = (N_TUR_YUKLU * loaded_res["mah_per_lap"]) + (N_TUR_BOS * empty_res["mah_per_lap"])
            butcelenen_toplam_tuketim_mah = seyir_tuketim_mah + SEYIR_DISI_KAPASITE_BUTCESI_MAH
            butceye_gore_iniste_kalan_batarya_yuzde = 100.0 * (1.0 - (butcelenen_toplam_tuketim_mah / BATARYA_TOPLAM_MAH))

            seyir_enerji_wh = (N_TUR_YUKLU * loaded_res["wh_per_lap"]) + (N_TUR_BOS * empty_res["wh_per_lap"])
            seyir_enerji_kj = ((N_TUR_YUKLU * loaded_res["joule_per_lap"]) + (N_TUR_BOS * empty_res["joule_per_lap"])) / 1000.0

            # 4. NAVİGASYON PUANI VE HAM ENERJİ ENDEKSİ
            # Su oranı = su kütlesi / toplam yüklü uçak kütlesi.
            su_orani = m_water_kg / m_total_kg
            # Yüklü tur: (1 + 2 * su oranı) * WP sayısı * 3; tur başına en fazla 60 puan.
            puan_yuklu_tur = min((1.0 + 2.0 * su_orani) * N_WAYPOINT * 3.0, 60.0)
            # Boş tur: WP sayısı * 2; tur başına en fazla 20 puan.
            puan_bos_tur = min(N_WAYPOINT * 2.0, 20.0)
            puan_yuklu_toplam = N_TUR_YUKLU * puan_yuklu_tur
            puan_bos_toplam = N_TUR_BOS * puan_bos_tur
            puan_navigasyon = puan_yuklu_toplam + puan_bos_toplam

            # Ham Enerji Endeksi = (Su Kütlesi * Waypoint Sayısı) / Harcanan Enerji (Wh)
            toplam_yuklu_wp = N_TUR_YUKLU * N_WAYPOINT
            ham_enerji_endeksi = ((m_water_kg * toplam_yuklu_wp) / seyir_enerji_wh
                                 if np.isfinite(seyir_enerji_wh) and seyir_enerji_wh > 0 else np.nan)

            # 5. FİZİBİLİTE KONTROLÜ (%20 İniş Rezervi Dahil!)
            feasible = (
                loaded_res["stall_ok"]
                and empty_res["stall_ok"]
                and loaded_res["thrust_ok"]
                and empty_res["thrust_ok"]
                and loaded_res["energy_ok"] and empty_res["energy_ok"]
                and np.isfinite(ham_enerji_endeksi) and ham_enerji_endeksi > 0
                and mission_time["sure_uygun"]
                and (seyir_tuketim_mah <= SEYIR_MAX_KULLANILABILIR_MAH)
                and (butceye_gore_iniste_kalan_batarya_yuzde >= INIS_REZERV_ORANI * 100.0)
            )
            reasons = []
            for label, phase in (("yuklu", loaded_res), ("bos", empty_res)):
                for check, reason in (("stall_ok", "stall"), ("thrust_ok", "itki"), ("polar_ok", "lineer_aci_araligi")):
                    if not phase[check]:
                        reasons.append(f"{label}_{reason}")
            if not mission_time["sure_uygun"]:
                reasons.append("sure")
            if np.isfinite(seyir_tuketim_mah) and seyir_tuketim_mah > SEYIR_MAX_KULLANILABILIR_MAH:
                reasons.append("batarya")
            if not np.isfinite(ham_enerji_endeksi) or ham_enerji_endeksi <= 0:
                reasons.append("gecersiz_enerji_endeksi")

            results.append({
                "kombinasyon_no": idx + 1,
                "kanat_aday_no": llt_data["kanat_aday_no"],
                "govde_aday_no": int(row["govde_aday_no"]),
                "kanat_airfoil": profil,
                "AR_w": ar_w,
                "b_w_m": b_w,
                "S_w_m2": S_w,
                "c_w_m": float(row["c_root_w_m"]),
                "govde_D_m": float(row["govde_D_m"]),
                "govde_L_m": float(row["govde_L_toplam_m"]),
                "su_kutlesi_kg": m_water_kg,
                "toplam_kutle_kg": m_total_kg,
                "bos_kutle_kg": m_empty_kg,
                "su_kutle_orani": m_water_kg / m_total_kg,
                "e_oswald": llt_data["e_oswald"],
                "CD_0": CD_0_total,
                "CL_max_3d": llt_data["CL_max_3d"],
                # Yüklü Seyir Çıktıları
                "CL_yuklu_duz": loaded_res["CL_straight"],
                "CD_yuklu_duz": loaded_res["CD_straight"],
                "LD_yuklu_duz": loaded_res["LD_straight"],
                "alpha_yuklu_deg": loaded_res["alpha_straight_deg"],
                "itki_yuklu_duz_N": loaded_res["T_req_straight_N"],
                "gaz_yuklu_duz_pct": loaded_res["throttle_straight_pct"],
                "akim_yuklu_duz_A": loaded_res["current_straight_A"],
                "guc_yuklu_duz_W": loaded_res["power_straight_W"],
                "CL_yuklu_donus": loaded_res["CL_turn"],
                "guc_yuklu_donus_W": loaded_res["power_turn_W"],
                # Boş Seyir Çıktıları
                "CL_bos_duz": empty_res["CL_straight"],
                "CD_bos_duz": empty_res["CD_straight"],
                "LD_bos_duz": empty_res["LD_straight"],
                "alpha_bos_deg": empty_res["alpha_straight_deg"],
                "itki_bos_duz_N": empty_res["T_req_straight_N"],
                "gaz_bos_duz_pct": empty_res["throttle_straight_pct"],
                "akim_bos_duz_A": empty_res["current_straight_A"],
                "guc_bos_duz_W": empty_res["power_straight_W"],
                # Batarya ve Enerji Çıktıları
                "toplam_mesafe_m": toplam_mesafe_m,
                "toplam_sure_sn": toplam_sure_sn,
                "navigasyon_suresi_sn": mission_time["navigasyon_suresi_sn"],
                "navigasyon_disi_sure_sn": mission_time["navigasyon_disi_sure_sn"],
                "sure_emniyet_payi_sn": mission_time["sure_emniyet_payi_sn"],
                "sinira_kalan_sure_sn": mission_time["sinira_kalan_sure_sn"],
                "pay_sonrasi_kalan_sure_sn": mission_time["pay_sonrasi_kalan_sure_sn"],
                "sure_uygun": mission_time["sure_uygun"],
                "seyir_tuketim_mah": seyir_tuketim_mah,
                "seyir_disi_kapasite_butcesi_mah": SEYIR_DISI_KAPASITE_BUTCESI_MAH,
                "hesaplanan_toplam_ucus_tuketim_mah": np.nan,
                "tam_gorev_tuketimi_hesaplandi": False,
                "puan_enerji_kapsami": "yalniz_seyir",
                "butcelenen_toplam_tuketim_mah": butcelenen_toplam_tuketim_mah,
                "butceye_gore_iniste_kalan_batarya_yuzde": butceye_gore_iniste_kalan_batarya_yuzde,
                "seyir_enerji_Wh": seyir_enerji_wh,
                "seyir_enerji_kJ": seyir_enerji_kj,
                # Puanlar
                "puan_yuklu_2tur": puan_yuklu_toplam,
                "puan_bos_1tur": puan_bos_toplam,
                "puan_navigasyon": puan_navigasyon,
                "ham_enerji_endeksi": ham_enerji_endeksi,
                "feasible": feasible,
                "uygunsuzluk_nedenleri": ";".join(reasons),
                "trim_dogrulandi": False,
                "kuyruk_oturma_deg": KUYRUK_OTURMA_DEG,
                "downwash_deg": 0.0,
                "Re_HT": tail_data["Re"],
                "e_HT": tail_data["e"],
                "CL_alpha_HT_per_deg": tail_data["slope"],
                "Re_HT_seyir": tail_data["Re_target"],
                "Re_VT": tail_data["Re_VT"],
                "Re_VT_seyir": tail_data["Re_VT_target"],
                "CL_alpha_VT_per_deg": tail_data["slope_VT"],
                "e_VT": tail_data["e_VT"],
                "kuyruk_veri_kaynagi": tail_data["source"],
                "kuyruk_Re_uyumlu": tail_data["reynolds_match"],
                "kuyruk_stall_dogrulandi": tail_data["stall_verified"],
                "kuyruk_lineer_araligi_varsayim": True,
                "kuyruk_model_notu": "CSV kuyruk stall sınırı içermiyor; doğrusal aralık -4..6 derece kabulüdür."
                    + ("" if tail_data["reynolds_match"] else " CSV Reynolds değerleri seyirden farklı; katsayılar doğrudan kullanıldı."),
                "gerilim_V": V_BAT_SABIT,
                "hata_aciklamasi": "",
                "dusuk_gaz_varsayimi": loaded_res["low_throttle_assumption"] or empty_res["low_throttle_assumption"],
                "seyir_hizi_m_s": V_SEYIR,
                "rho_kg_m3": RHO_HAVA,
                "mu_pa_s": MU_HAVA,
                "pervane": thrust_info["pervane"],
            })

            for label, phase in (("yuklu", loaded_res), ("bos", empty_res)):
                for flight, suffix in (("straight", "duz"), ("turn", "donus")):
                    for field in ("CL_w", "CL_HT", "CDi_w", "CDi_HT"):
                        results[-1][f"{field}_{label}_{suffix}"] = phase[f"{field}_{flight}"]
                    for field in ("L_w", "L_HT", "L_denge_hatasi"):
                        results[-1][f"{field}_{label}_{suffix}_N"] = phase[f"{field}_{flight}_N"]
                    results[-1][f"alpha_HT_{label}_{suffix}_deg"] = phase[f"alpha_HT_{flight}_deg"]
        except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
            results.append({
                "kombinasyon_no": idx + 1,
                "kanat_airfoil": row.get("kanat_airfoil", ""),
                "govde_aday_no": row.get("govde_aday_no", np.nan),
                "feasible": False, "ham_enerji_endeksi": np.nan,
                "puan_navigasyon": np.nan, "toplam_sure_sn": mission_time["toplam_sure_sn"],
                "uygunsuzluk_nedenleri": "aday_veri_hatasi",
                "hata_aciklamasi": f"{type(exc).__name__}: {exc}",
                "trim_dogrulandi": False, "tam_gorev_tuketimi_hesaplandi": False,
            })

    if not results:
        raise ValueError("Değerlendirilecek aday yok: kütle tablosunu ve veri eşleşmelerini kontrol edin.")
    res_df = pd.DataFrame(results)
    print("Kuyruk: llt_flapsiz_aday_re.csv katsayıları doğrudan kullanıldı. "
          "Dosyada kuyruk CL_max bulunmadığından kuyruk stall doğrulanmadı; uygunluk ön model kapsamındadır.")
    if "kuyruk_Re_uyumlu" in res_df:
        mismatch = res_df["kuyruk_Re_uyumlu"].eq(False).sum()
        if mismatch:
            print(f"Kuyruk Reynolds: {mismatch} adayda CSV koşulu seyir koşulundan farklı; "
                  "Re_HT/Re_HT_seyir ve Re_VT/Re_VT_seyir sütunlarına bakın.")

    # --- BAĞIL ENERJİ PUANI HESABI (1. Sıradaki Aday = 60 Puan) ---
    res_df = score_feasible_candidates(res_df)

    res_df.to_csv("seyir_optimizasyon_sonuclari.csv", index=False)
    return res_df, route_info, thrust_info


# ==============================================================================
# 6. SONUÇ RAPORLAMA VE GÖRSELLEŞTİRME
# ==============================================================================
def plot_route_geometry(ax, route_info):
    wp = route_info["wp_coords"]
    closed = np.vstack([wp, wp[0]])
    boundary = route_info["boundary_coords"]
    boundary = np.vstack([boundary, boundary[0]])
    ax.plot(boundary[:, 0], boundary[:, 1], color="firebrick", linestyle="--", alpha=.65,
            label="Görseldeki yaklaşık kırmızı sınır")
    ax.plot(closed[:, 0], closed[:, 1], color="grey", linestyle=":", alpha=.55,
            label=f"WP1–10 düz bağlantısı: {L_TABAN_ROTA_M:.0f} m")
    path = route_info["fly_pts"]
    ax.plot(path[:, 0], path[:, 1], color="teal", linewidth=2,
            label=f"Teğet dönüşlü tur: {route_info['L_total_lap_m']:.0f} m")
    ax.scatter(wp[:, 0], wp[:, 1], c="dodgerblue", s=45, zorder=5)
    for name, point in zip(route_info["wp_names"], wp):
        ax.annotate(name, point, xytext=(7, 7), textcoords="offset points", fontsize=9)
    for i in range(0, len(route_info["segments"]), 3):
        chunk = np.vstack([_sample_segment(seg) for seg in route_info["segments"][i:i+3]])
        k = len(chunk) // 2
        ax.annotate("", xy=chunk[min(k + 12, len(chunk) - 1)], xytext=chunk[max(k - 12, 0)],
                    arrowprops=dict(arrowstyle="->", color="teal", lw=1.8))
    ax.scatter([0], [0], marker="s", color="black", s=40, label="TOAL (referans)")
    drops = route_info["drop_coords"]
    ax.scatter(drops[:, 0], drops[:, 1], color="purple", s=25, label="P1–P3 (referans)")
    timing = compute_mission_timing(route_info)
    ax.set_title(f"WP1 → WP2 → … → WP10 → WP1 | {route_info['bank_angle_deg']:.0f}° yatış\n"
                 f"R={route_info['R_turn_m']:.1f} m | Toplam uçuş ≈ {timing['toplam_sure_sn']:.1f} sn")
    ax.set_xlabel("TOAL'a göre yerel X (m)")
    ax.set_ylabel("TOAL'a göre yerel Y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=.2)
    ax.legend(fontsize=7, loc="best")


def preview_route_only():
    """Aerodinamik CSV'ler olmadan: python seyir_optimizasyonu.py --rota"""
    import matplotlib.pyplot as plt
    route = compute_waypoint_route_geometry(V_SEYIR, BANK_ANGLE_DEG)
    timing = compute_mission_timing(route)
    print("Örnek harita; düz WP1–10 çevresi 2800 m kabul edilerek ölçeklendi.")
    print(f"Tur: {route['L_total_lap_m']:.1f} m | Navigasyon: {timing['navigasyon_suresi_sn']:.1f} sn")
    print(f"Diğer aşamalar: {NAVIGASYON_DISI_SURE_SN:.1f} sn | Toplam: {timing['toplam_sure_sn']:.1f} sn")
    print(f"Emniyet payı: {timing['sure_emniyet_payi_sn']:.1f} sn | Ek boşluk: {timing['pay_sonrasi_kalan_sure_sn']:.1f} sn")
    print("Bu süre kontrolü, aday uçağın stall/itki/batarya uygunluğunu doğrulamaz.")
    fig, ax = plt.subplots(figsize=(10, 9))
    plot_route_geometry(ax, route)
    plt.tight_layout()
    plt.show()


def plot_mission_and_route_summary(res_df, route_info):
    import matplotlib.pyplot as plt
    print("KAPSAM: Ön seyir değerlendirmesi; trim/moment dengesi doğrulanmadı. "
          "Seyir dışı süre ve kapasite ayrı bütçedir; tam görev tüketimi hesaplanmadı. Puanlar seyir enerjisine dayalıdır.")
    print("MODEL: Rüzgârsız, sabit hızlı/yatışlı dönüş; yatış geçişleri modellenmiyor. "
          "CSV akim_a amper; P=24*I. Kuyruk taşıma eğimi ve e katsayısı llt_flapsiz_aday_re.csv dosyasından alınır.")
    if res_df.empty:
        print("UYARI: Sonuç tablosu boş; optimum aday seçilemedi.")
        return
    feasible_df = res_df[res_df["feasible"]].copy()
    if feasible_df.empty:
        print("UYARI: Kısıtları sağlayan aday bulunamadı. Optimum aday seçilmedi; "
              "uygun olmayan adaylar yalnızca sonuç tablosunda tutuldu.")
        usable_time = MAX_UCUS_SURESI_SN * (1.0 - UCUS_SURE_EMNIYET_ORANI)
        if "toplam_sure_sn" in res_df and (res_df["toplam_sure_sn"] > usable_time).all():
            print(f"Süre kısıtı: en kısa toplam uçuş {res_df['toplam_sure_sn'].min():.1f} sn; "
                  f"pay ayrıldıktan sonraki sınır {usable_time:.1f} sn. Tüm adaylar süre bütçesini aşıyor.")
        return

    best = feasible_df.sort_values("toplam_yarisma_puani", ascending=False).iloc[0]

    print("\n" + "=" * 80)
    print("=== 1. WAYPOINT DIŞINDAN DÖNÜŞ ROTA VE BATARYA KISIT ÖZETİ ===")
    print("=" * 80)
    print(f"Seyir Hızı / Dönüş Yatış Açısı   : {V_SEYIR:.1f} m/s | {route_info['bank_angle_deg']:.1f}° (n = {route_info['n_load_turn']:.3f} g)")
    print("Rota modeli                    : Haritadan yaklaşık WP1 -> WP2 -> ... -> WP10 -> WP1; teğet yay-düz-yay bağlantıları.")
    mission_time = compute_mission_timing(route_info)
    print(f"Navigasyon / Diğer Aşamalar     : {mission_time['navigasyon_suresi_sn']:.1f} / {mission_time['navigasyon_disi_sure_sn']:.1f} sn")
    print(f"Toplam Uçuş / Sınır             : {mission_time['toplam_sure_sn']:.1f} / {MAX_UCUS_SURESI_SN:.1f} sn")
    print(f"Ayrılan Süre Payı / Ek Boşluk   : {mission_time['sure_emniyet_payi_sn']:.1f} / {mission_time['pay_sonrasi_kalan_sure_sn']:.1f} sn")
    print(f"1 Tur / 3 Tur Toplam Mesafe      : {route_info['L_total_lap_m']:.1f} m / {3 * route_info['L_total_lap_m']:.1f} m ({3 * route_info['t_total_lap_s']:.1f} sn)")
    print(f"Batarya Kapasitesi & %20 Rezerv  : {BATARYA_TOPLAM_MAH:.0f} mAh Toplam -> Min. {BATARYA_TOPLAM_MAH*INIS_REZERV_ORANI:.0f} mAh (%20) İnişte Kalmalı")
    print(f"Kullanılabilir Net / Seyir Sınırı: {KULLANILABILIR_TOPLAM_MAH:.0f} mAh Net ({SEYIR_DISI_KAPASITE_BUTCESI_MAH:.0f} mAh Kalkış/İniş Payı -> Max {SEYIR_MAX_KULLANILABILIR_MAH:.0f} mAh Seyir)")

    print("\n" + "=" * 80)
    print("=== 2. ÖN SEYİR MODELİNDE EN YÜKSEK PUANLI ADAY ===")
    print("=" * 80)
    print(f"Kombinasyon / Kanat / Gövde No   : #{int(best['kombinasyon_no'])} (Kanat #{int(best['kanat_aday_no'])}, Gövde #{int(best['govde_aday_no'])})")
    print(f"Kanat Profili & Geometrisi       : {best['kanat_airfoil']} | b = {best['b_w_m']:.2f} m | AR = {best['AR_w']:.1f} | S = {best['S_w_m2']:.4f} m²")
    print(f"Gövde Çapı & Uzunluğu            : D = {best['govde_D_m']:.2f} m | L = {best['govde_L_m']:.3f} m")
    print(f"Kütle Dağılımı                   : Tam Dolu = {best['toplam_kutle_kg']:.3f} kg | Boş = {best['bos_kutle_kg']:.3f} kg | Su = {best['su_kutlesi_kg']:.2f} kg")
    print("-" * 80)
    print(f"[YÜKLÜ SEYİR - 2 TUR]            : CL = {best['CL_yuklu_duz']:.3f} | CD = {best['CD_yuklu_duz']:.5f} | L/D = {best['LD_yuklu_duz']:.2f}")
    print(f"                                 : İtki = {best['itki_yuklu_duz_N']:.2f} N | Akım = {best['akim_yuklu_duz_A']:.2f} A | Elektriksel Güç = {best['guc_yuklu_duz_W']:.1f} W")
    print(f"[BOŞ SEYİR - 1 TUR]              : CL = {best['CL_bos_duz']:.3f} | CD = {best['CD_bos_duz']:.5f} | L/D = {best['LD_bos_duz']:.2f}")
    print(f"                                 : İtki = {best['itki_bos_duz_N']:.2f} N | Akım = {best['akim_bos_duz_A']:.2f} A | Elektriksel Güç = {best['guc_bos_duz_W']:.1f} W")
    print("-" * 80)
    print(f"3 Tur Seyir Enerji Tüketimi      : {best['seyir_enerji_Wh']:.2f} Wh ({best['seyir_enerji_kJ']:.1f} kJ)")
    print(f"Seyir Tüketimi / Toplam Bütçe    : {best['seyir_tuketim_mah']:.1f} mAh Seyir | {best['butcelenen_toplam_tuketim_mah']:.1f} mAh Bütçelenen Toplam")
    batarya_uygun = best['butceye_gore_iniste_kalan_batarya_yuzde'] >= INIS_REZERV_ORANI * 100.0
    batarya_durumu = "UYGUN" if batarya_uygun else "UYGUN DEĞİL"
    print(f"Bütçeye Göre İnişte Kalan       : %{best['butceye_gore_iniste_kalan_batarya_yuzde']:.1f} (Yarışma Alt Sınırı: >= %20.0) -> [{batarya_durumu}]")
    print("-" * 80)
    print(f"1) Yüklü Navigasyon Puanı (2 Tur): {best['puan_yuklu_2tur']:.2f} Puan")
    print(f"2) Boş Navigasyon Puanı (1 Tur)  : {best['puan_bos_1tur']:.2f} Puan")
    print(f"3) Bağıl Enerji Puanı (Max 60)   : {best['puan_enerji_60']:.2f} Puan (Endeks: {best['ham_enerji_endeksi']:.2f} kg*WP/Wh)")
    print(f"SEYİR TABANLI MODEL PUANI       : {best['toplam_yarisma_puani']:.2f} Puan")
    print("=" * 80)

    # --- GÖRSELLEŞTİRME ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    plot_route_geometry(ax1, route_info)

    # Panel 2: Harcanan Enerji (Wh) vs Toplam Yarışma Puanı (Navigasyon + 60p Enerji)
    sc = ax2.scatter(
        feasible_df["seyir_enerji_Wh"],
        feasible_df["toplam_yarisma_puani"],
        c=feasible_df["su_kutlesi_kg"],
        cmap="plasma",
        alpha=0.8,
        s=28,
    )
    plt.colorbar(sc, ax=ax2, label="Taşınan Su Kütlesi (kg)")

    ax2.plot(
        best["seyir_enerji_Wh"],
        best["toplam_yarisma_puani"],
        marker="*",
        color="lime",
        markersize=18,
        markeredgecolor="black",
        label=f"Optimum Aday #{int(best['kombinasyon_no'])} ({best['kanat_airfoil']}, Su={best['su_kutlesi_kg']:.1f}kg, Puan={best['toplam_yarisma_puani']:.1f})",
    )

    ax2.set_title(f"2. {len(feasible_df)} Uygun Aday: Seyir Enerjisi (Wh) ve Model Puanı", fontsize=11, fontweight="bold")
    ax2.set_xlabel("3 Tur Harcanan Elektriksel Enerji (Wh)")
    ax2.set_ylabel("Toplam Görev Puanı (Navigasyon + 60p Bağıl Enerji)")
    ax2.legend(loc="lower left", fontsize=9.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rota", action="store_true", help="CSV veritabanları olmadan örnek rotayı göster")
    parser.add_argument("--kuyruk-llt", type=Path, help="Kuyruk katsayıları: llt_flapsiz_aday_re.csv")
    args = parser.parse_args()
    if args.rota:
        preview_route_only()
    else:
        res_df, route_info, thrust_info = run_full_cruise_mission_optimization(args.kuyruk_llt)
        plot_mission_and_route_summary(res_df, route_info)