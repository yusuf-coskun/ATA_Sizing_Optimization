import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# 1. SABİTLER, BATARYA KISITLARI VE GÖREV PARAMETRELERİ
# ==============================================================================
V_SEYIR = 20.0                  # Seyir hızı (m/s)
RHO_HAVA = 1.225                # Hava yoğunluğu (kg/m^3)
G_YERCEKIMI = 9.81              # Yerçekimi ivmesi (m/s^2)

N_WAYPOINT = 10                 # Mavi waypoint sayısı (WP1 - WP10)
L_TABAN_ROTA_M = 2800.0         # Waypoint düz birleştirme çevresi (m)
N_TUR_YUKLU = 2                 # Tam dolu (su yüklü) uçulacak tur sayısı
N_TUR_BOS = 1                   # Yüksüz (boş) uçulacak tur sayısı

BANK_ANGLE_DEG = 30.0           # Waypoint dışından dönüş yatış açısı (derece)
WP_CLEARANCE_M = 5.0            # Waypoint dış güvenlik mesafesi (m)

# --- Batarya ve Elektriksel Güç Modeli (6S 12000 mAh LiPo) ---
BATARYA_HUCRE_SAYISI = 6        # 6S LiPo (Nominal: 6 * 3.7V = 22.2V)
V_BAT_ACIK_DEVRE = 22.8         # Seyir esnasında ortalama açık devre gerilimi (V)
R_BAT_IC_DIRENC_OHM = 0.018     # Paket toplam iç direnci (Ohm) -> V_bat = V_oc - I * R_ic

BATARYA_TOPLAM_MAH = 12000.0    # Toplam batarya kapasitesi (mAh)
INIS_REZERV_ORANI = 0.20        # İnişte bataryada kalması zorunlu %20 rezerv (2400 mAh)
KULLANILABILIR_TOPLAM_MAH = BATARYA_TOPLAM_MAH * (1.0 - INIS_REZERV_ORANI)  # 9600 mAh

# Kalkış, tırmanış, su bırakma yaklaşması ve iniş için ayrılan kapasite rezervi (mAh)
KALKIS_TIRMANIS_INIS_REZERV_MAH = 2000.0
# Sadece 3 turluk seyir (cruise) için kullanılabilecek maksimum kapasite sınırı (7600 mAh)
SEYIR_MAX_KULLANILABILIR_MAH = KULLANILABILIR_TOPLAM_MAH - KALKIS_TIRMANIS_INIS_REZERV_MAH

MAX_ENERJI_PUANI = 60.0         # Enerji sıralamasında 1. olan adayın alacağı maksimum puan
MAX_UCUS_SURESI_SN = 600.0      # Kalkıştan inişe TÜM uçuşun süre sınırı (sn)
NAVIGASYON_DISI_SURE_SN = 100.0 # Kullanıcının geçici bütçesi: kalkış, tırmanış,
                               # TOAL/rota geçişleri, yük bırakma yaklaşması ve iniş dahil.
UCUS_SURE_EMNIYET_ORANI = 0.02 # 600 saniyenin %2'si = 12 sn; hedef toplam <= 588 sn.
STALL_EMNIYET_ORANI = 0.85      # Dönüşte CL_max emniyet sınırı


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
    vsp_path = find_csv_file("openvsp_llt_polar_detay.csv")
    llt_path = find_csv_file("llt_coklu_re_tum_sonuclar.csv")
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


def add_lookup_record(table, key, value):
    # Aynı kanadı kullanan farklı ıslak alanlı adaylar geçerlidir.
    # Önceki kaydı ezmeden bütün referans kayıtlarını sakla.
    table.setdefault(key, []).append(value)


def match_geometry(table, key, context, wet_area=None):
    # Yalnızca çok küçük dosya yazım hassasiyeti farklarına izin ver.
    matches = [k for k in table if k[0] == key[0]
               and np.allclose(k[1:], key[1:], rtol=1e-9, atol=1e-6)]
    if len(matches) != 1:
        raise ValueError(f"{context}: {key} için {len(matches)} eşleşme bulundu; tam bir eşleşme gerekli.")
    records = table[matches[0]]
    if wet_area is not None:
        if not np.isfinite(wet_area) or wet_area <= 0:
            raise ValueError(f"{context}: gövde ıslak alanı sonlu ve pozitif olmalı.")
        areas = np.asarray([record["S_wet_govde_base"] for record in records])
        if not np.isfinite(areas).all() or (areas <= 0).any():
            raise ValueError(f"{context}: referans gövde ıslak alanları pozitif olmalı.")
        # Varsa aynı alanlı referansı seç. Yoksa en yakın alanlı referansı
        # mevcut CD0_govde * (aday alanı / referans alanı) hesabında kullan.
        reference_area = min(areas, key=lambda area: (abs(area - wet_area), area))
        records = [record for record in records
                   if record["S_wet_govde_base"] == reference_area]
    reference = records[0]
    # Aynı kanadın ortak LLT verisi tekrar edebilir; kimlik farkı hata değildir.
    # Aynı referansın gerçekten çelişkili aerodinamik verisini gizleme.
    fields = [field for field in reference if field != "kanat_aday_no"]
    for record in records[1:]:
        if not np.allclose([reference[field] for field in fields],
                           [record[field] for field in fields], rtol=1e-9, atol=1e-12):
            raise ValueError(f"{context}: aynı referans için çelişkili aerodinamik veri var: {key}.")
    return reference


def build_lookup_tables(itki_df, vsp_df, llt_df):
    if llt_df.empty or vsp_df.empty:
        raise ValueError("LLT ve OpenVSP tabloları boş olamaz.")
    if llt_df["aday_no"].isna().any() or vsp_df["aday_no"].isna().any():
        raise ValueError("LLT/OpenVSP aday_no sütununda eksik kimlik var.")
    if "Re_aday_hiz_m_s" not in llt_df.columns:
        warnings.warn("LLT Re_aday_hiz_m_s sütunu yok: mevcut 18 m/s referans varsayımı kullanılıyor.",
                      RuntimeWarning)

    # --- A. LLT Çoklu Reynolds Enterpolasyonu (V = 20 m/s için) ---
    llt_lookup = {}
    for aday_no, grp in llt_df.groupby("aday_no"):
        context = f"LLT aday {aday_no}"
        key = geometry_key(unique_value(grp, "profil", context, numeric=False),
                           unique_value(grp, "AR_w", context),
                           unique_value(grp, "b_w_m", context))
        re_reference = unique_value(grp, "Re_aday", context)
        reference_speed = (unique_value(grp, "Re_aday_hiz_m_s", context)
                           if "Re_aday_hiz_m_s" in grp.columns else 18.0)
        if re_reference <= 0 or reference_speed <= 0:
            raise ValueError(f"{context}: Reynolds ve referans hız pozitif olmalı.")
        re_20 = re_reference * (V_SEYIR / reference_speed)
        polar_columns = ["CL_alpha_3d_per_deg", "alpha_L0_deg", "e_oswald",
                         "CL_max_3d_tahmini", "CDp_seyir"]
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
        values = [float(np.interp(re_20, re_polars, polars[:, j + 1])) for j in range(5)]
        cl_alpha, alpha_l0, e_oswald, cl_max_3d, cdp_llt = values
        add_lookup_record(llt_lookup, key, {
            "kanat_aday_no": int(aday_no),
            "Re_20": re_20,
            "CL_alpha_3d_per_deg": cl_alpha,
            "alpha_L0_deg": alpha_l0,
            "e_oswald": e_oswald,
            "CL_max_3d": cl_max_3d,
            "CDp_seyir_llt": cdp_llt,
        })

    # --- B. OpenVSP CD0 Tablosu: tüm kimlikleri tara, çelişkili satırları reddet ---
    vsp_lookup = {}
    if vsp_df["S_wet_govde_m2"].isna().any():
        raise ValueError("OpenVSP gövde ıslak alanı sütununda eksik değer var.")
    for (aday_no, _wet_area), sub in vsp_df.groupby(["aday_no", "S_wet_govde_m2"], sort=False):
        context = f"OpenVSP aday {aday_no}"
        key = geometry_key(unique_value(sub, "kanat_airfoil", context, numeric=False),
                           unique_value(sub, "AR_w", context),
                           unique_value(sub, "b_w_m", context))
        add_lookup_record(vsp_lookup, key, {
            "CD0_w": unique_value(sub, "CD0_w", context),
            "CD0_HT": unique_value(sub, "CD0_HT", context),
            "CD0_VT": unique_value(sub, "CD0_VT", context),
            "CD0_govde_base": unique_value(sub, "CD0_govde", context),
            "CD_0_base": unique_value(sub, "CD_0", context),
            "S_wet_govde_base": unique_value(sub, "S_wet_govde_m2", context),
        })

    # --- C. İtki Tablosu (V = 20 m/s Hız Dilimi + Kısık Gaz Orijin Uzatması) ---
    itki_20 = itki_df[np.isclose(itki_df["hiz_m_s"], V_SEYIR)].sort_values("itki_n")
    if itki_20.empty:
        raise ValueError(f"itki_tablosu.csv içinde hiz_m_s == {V_SEYIR} verisi bulunamadı!")

    # %50 gazın (12.02 N) altındaki kısık gaz seyir bölgesi için (0 N, %0 Gaz, 0 A) orijin bağlantısı
    itki_vals = np.insert(itki_20["itki_n"].values, 0, 0.0)
    gaz_vals = np.insert(itki_20["gaz_yuzde"].values, 0, 0.0)
    akim_vals = np.insert(itki_20["akim_a"].values, 0, 0.0)

    thrust_table_info = {
        "pervane": str(itki_20["pervane"].iloc[0]),
        "itki_n": itki_vals,
        "gaz_yuzde": gaz_vals,
        "akim_a": akim_vals,
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
    """Dört yay-düz-yay (CSC) seçeneğinden en kısası; tüm Dubins ailesi değil.

    Çemberlerin ortak teğeti analitik olarak bulunur. Nokta VE yön süreklidir.
    Üç yaylı CCC seçenekleri ve yatışa giriş/çıkış dinamiği modellenmez.
    """
    best = None
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
            straight = float(np.sqrt(max(0.0, distance ** 2 - offset ** 2)))
            length = radius * (sweep0 + sweep1) + straight
            if best is None or length < best["length"]:
                best = dict(length=length, c0=c0, c1=c1, heading=heading,
                            sweep0=sweep0, sweep1=sweep1, s0=s0, s1=s1,
                            straight=straight)
    if best is None:
        raise ValueError("Bu başlangıç/bitiş yönleri için yay-düz-yay bağlantısı bulunamadı.")
    return best


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


def compute_waypoint_route_geometry(V_cruise, bank_angle_deg):
    if not np.isfinite([V_cruise, bank_angle_deg]).all() or V_cruise <= 0 or not 0 < bank_angle_deg < 90:
        raise ValueError("Hız pozitif, yatış açısı 0 ile 90 derece arasında olmalı.")
    if WP_CLEARANCE_M < 0:
        raise ValueError("Waypoint geçiş açıklığı negatif olamaz.")
    reference = map_reference_geometry()
    wp_coords = reference["wp_coords"]
    if N_WAYPOINT != len(wp_coords):
        raise ValueError("N_WAYPOINT ile haritadaki waypoint sayısı uyuşmuyor.")
    phi_rad = np.radians(bank_angle_deg)
    radius = V_cruise ** 2 / (G_YERCEKIMI * np.tan(phi_rad))
    edges = np.roll(wp_coords, -1, axis=0) - wp_coords
    unit = edges / np.linalg.norm(edges, axis=1)[:, None]
    incoming = np.roll(unit, 1, axis=0)
    directions = incoming + unit
    headings = np.arctan2(directions[:, 1], directions[:, 0])
    # Sayısal sıranın her köşesinde yerel dönüşün dış tarafında geçiş hedefi.
    # Kendini kesen bir parkur için tek bir küresel "dış taraf" yoktur.
    cross = incoming[:, 0] * unit[:, 1] - incoming[:, 1] * unit[:, 0]
    sides = np.where(cross >= 0.0, -1.0, 1.0)

    def gate(i, heading):
        return wp_coords[i] + sides[i] * WP_CLEARANCE_M * _left_normal(heading)

    def leg(i, h0, h1):
        j = (i + 1) % len(wp_coords)
        return _shortest_csc(gate(i, h0), h0, gate(j, h1), h1, radius)

    def local_cost(i, heading):
        return (leg((i - 1) % len(wp_coords), headings[i - 1], heading)["length"]
                + leg(i, heading, headings[(i + 1) % len(wp_coords)])["length"])

    # Geçiş yönlerini yerel olarak iyileştir; küresel optimum iddiası yoktur.
    for step_deg in (15.0, 5.0, 1.0, 0.25):
        for _ in range(5):
            changed = False
            for i in range(len(wp_coords)):
                options = headings[i] + np.radians(step_deg) * np.arange(-2, 3)
                best = min(options, key=lambda angle: local_cost(i, angle))
                changed = changed or abs(best - headings[i]) > 1e-12
                headings[i] = best
            if not changed:
                break

    segments, leg_lengths = [], []
    gates = np.array([gate(i, heading) for i, heading in enumerate(headings)])
    for i in range(len(wp_coords)):
        j = (i + 1) % len(wp_coords)
        data = leg(i, headings[i], headings[j])
        tangent0 = data["c0"] - data["s0"] * radius * _left_normal(data["heading"])
        tangent1 = data["c1"] - data["s1"] * radius * _left_normal(data["heading"])
        arc0 = dict(kind="turn", start=gates[i], end=tangent0, center=data["c0"],
                    radius=radius, direction=data["s0"], sweep=data["sweep0"],
                    angle0=headings[i] - data["s0"] * np.pi / 2,
                    heading_start=headings[i], heading_end=data["heading"],
                    length=radius * data["sweep0"])
        straight = dict(kind="straight", start=tangent0, end=tangent1,
                        heading_start=data["heading"], heading_end=data["heading"],
                        length=data["straight"])
        arc1 = dict(kind="turn", start=tangent1, end=gates[j], center=data["c1"],
                    radius=radius, direction=data["s1"], sweep=data["sweep1"],
                    angle0=data["heading"] - data["s1"] * np.pi / 2,
                    heading_start=data["heading"], heading_end=headings[j],
                    length=radius * data["sweep1"])
        for segment in (arc0, straight, arc1):
            segment["from_wp"] = reference["wp_names"][i]
            segment["to_wp"] = reference["wp_names"][j]
            segments.append(segment)
        leg_lengths.append(data["length"])

    fly_pts = np.vstack([_sample_segment(segment) for segment in segments])
    clearances = np.array([min(_segment_point_distance(seg, wp) for seg in segments) for wp in wp_coords])
    if (clearances < WP_CLEARANCE_M - 1e-6).any():
        raise ValueError("Oluşturulan rota bir waypoint açıklığını ihlal ediyor; rota yeniden planlanmalı.")
    boundary_ok = bool(_points_in_polygon(fly_pts, reference["boundary_coords"]).all())
    if not boundary_ok:
        raise ValueError("Rota örnek haritanın yaklaşık kırmızı sınırını aşıyor; geometri yeniden planlanmalı.")
    straight_length = sum(seg["length"] for seg in segments if seg["kind"] == "straight")
    turn_length = sum(seg["length"] for seg in segments if seg["kind"] == "turn")
    total = straight_length + turn_length
    return {
        **reference,
        "fly_pts": fly_pts, "segments": segments, "gate_coords": gates,
        "gate_headings_rad": headings, "leg_lengths_m": np.array(leg_lengths),
        "wp_min_clearance_m": clearances, "boundary_sample_check_ok": boundary_ok,
        "bank_angle_deg": bank_angle_deg, "R_turn_m": radius,
        "n_load_turn": 1.0 / np.cos(phi_rad),
        "L_straight_lap_m": straight_length, "L_turn_lap_m": turn_length,
        "L_total_lap_m": total,
        "t_straight_lap_s": straight_length / V_cruise,
        "t_turn_lap_s": turn_length / V_cruise,
        "t_total_lap_s": total / V_cruise,
    }


def compute_electrical_power_and_current(T_req_N, thrust_info):
    """
    Gerekli itki (T_req_N) için:
    1. itki_tablosu.csv'den gaz yüzdesi (%) ve akım (Amper) enterpolasyonu yapar.
    2. Batarya iç direnci düşümüyle anlık batarya gerilimini hesaplar: V_bat = V_oc - I * R_ic
    3. Elektriksel gücü hesaplar: P_el (Watt) = V_bat * I
    """
    throttle_pct = float(np.interp(T_req_N, thrust_info["itki_n"], thrust_info["gaz_yuzde"]))
    current_a = float(np.interp(T_req_N, thrust_info["itki_n"], thrust_info["akim_a"]))

    # Yük altında batarya uç gerilimi (Volt) ve Elektriksel Güç (Watt)
    v_bat_loaded = V_BAT_ACIK_DEVRE - (current_a * R_BAT_IC_DIRENC_OHM)
    power_elec_w = v_bat_loaded * current_a

    return throttle_pct, current_a, v_bat_loaded, power_elec_w


def evaluate_cruise_phase(mass_kg, S_w, AR_w, CD_0, e_oswald, CL_alpha_deg, alpha_L0_deg, CL_max_3d, route_info, thrust_info):
    q_dyn = 0.5 * RHO_HAVA * (V_SEYIR ** 2)
    W_newton = mass_kg * G_YERCEKIMI

    # 1. Düz Uçuş Fazı (n = 1.0)
    CL_straight = W_newton / (q_dyn * S_w)
    CDi_straight = (CL_straight ** 2) / (np.pi * AR_w * e_oswald)
    CD_straight = CD_0 + CDi_straight
    T_req_straight = q_dyn * S_w * CD_straight
    alpha_straight_deg = alpha_L0_deg + (CL_straight / CL_alpha_deg)

    thr_s, cur_s, vbat_s, pow_s = compute_electrical_power_and_current(T_req_straight, thrust_info)

    # 2. Waypoint Dışı Dönüş Fazı (n = 1 / cos(phi))
    n_load = route_info["n_load_turn"]
    CL_turn = CL_straight * n_load
    CDi_turn = (CL_turn ** 2) / (np.pi * AR_w * e_oswald)
    CD_turn = CD_0 + CDi_turn
    T_req_turn = q_dyn * S_w * CD_turn
    alpha_turn_deg = alpha_L0_deg + (CL_turn / CL_alpha_deg)

    thr_t, cur_t, vbat_t, pow_t = compute_electrical_power_and_current(T_req_turn, thrust_info)

    # Fizibilite kontrolleri
    stall_ok = CL_turn <= (STALL_EMNIYET_ORANI * CL_max_3d)
    thrust_ok = (T_req_straight <= thrust_info["max_itki_n"]) and (T_req_turn <= thrust_info["max_itki_n"])

    # 1 Turda Tüketilen Kapasite (mAh) = I(A) * t(s) * (1000 / 3600)
    t_s = route_info["t_straight_lap_s"]
    t_t = route_info["t_turn_lap_s"]
    mah_per_lap = (cur_s * t_s + cur_t * t_t) * (1000.0 / 3600.0)

    # 1 Turda Harcanan Elektriksel Enerji (Joule ve Wh) = P_el(W) * t(s)
    joule_per_lap = (pow_s * t_s) + (pow_t * t_t)
    wh_per_lap = joule_per_lap / 3600.0

    return {
        "CL_straight": CL_straight,
        "CD_straight": CD_straight,
        "LD_straight": CL_straight / CD_straight,
        "alpha_straight_deg": alpha_straight_deg,
        "T_req_straight_N": T_req_straight,
        "throttle_straight_pct": thr_s,
        "current_straight_A": cur_s,
        "vbat_straight_V": vbat_s,
        "power_straight_W": pow_s,
        "CL_turn": CL_turn,
        "CD_turn": CD_turn,
        "LD_turn": CL_turn / CD_turn,
        "alpha_turn_deg": alpha_turn_deg,
        "T_req_turn_N": T_req_turn,
        "throttle_turn_pct": thr_t,
        "current_turn_A": cur_t,
        "power_turn_W": pow_t,
        "mah_per_lap": mah_per_lap,
        "wh_per_lap": wh_per_lap,
        "joule_per_lap": joule_per_lap,
        "stall_ok": stall_ok,
        "thrust_ok": thrust_ok,
    }


# ==============================================================================
# 5. TÜM ADAYLARIN TARANMASI VE BAĞIL ENERJİ PUANLANDIRMA SİMÜLASYONU
# ==============================================================================
def run_full_cruise_mission_optimization():
    itki_df, vsp_df, llt_df, kutle_df = load_all_databases()
    llt_lookup, vsp_lookup, thrust_info = build_lookup_tables(itki_df, vsp_df, llt_df)
    route_info = compute_waypoint_route_geometry(V_SEYIR, BANK_ANGLE_DEG)
    mission_time = compute_mission_timing(route_info)

    results = []

    for idx, row in kutle_df.iterrows():
        profil = str(row["kanat_airfoil"]).strip()
        ar_w = round(float(row["AR_w"]), 3)
        b_w = round(float(row["b_w_m"]), 3)
        S_w = float(row["S_w_m2"])
        key = geometry_key(profil, row["AR_w"], row["b_w_m"])

        llt_data = match_geometry(llt_lookup, key, f"Kütle satırı {idx + 1}, LLT")
        vsp_data = match_geometry(vsp_lookup, key, f"Kütle satırı {idx + 1}, OpenVSP",
                                  wet_area=float(row["govde_S_toplam_m2"]))

        # Gövde ıslak alanına göre CD0_govde ölçeklemesi
        S_wet_govde_aday = float(row["govde_S_toplam_m2"])
        govde_ratio = S_wet_govde_aday / max(vsp_data["S_wet_govde_base"], 1e-4)
        CD0_govde_aday = vsp_data["CD0_govde_base"] * govde_ratio
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
        )

        # 2. BOŞ SEYİR ANALİZİ (1 Tur)
        empty_res = evaluate_cruise_phase(
            mass_kg=m_empty_kg, S_w=S_w, AR_w=ar_w, CD_0=CD_0_total,
            e_oswald=llt_data["e_oswald"], CL_alpha_deg=llt_data["CL_alpha_3d_per_deg"],
            alpha_L0_deg=llt_data["alpha_L0_deg"], CL_max_3d=llt_data["CL_max_3d"],
            route_info=route_info, thrust_info=thrust_info,
        )

        # 3. TOPLAM GÖREV SÜRESİ, KAPASİTE (mAh) VE ENERJİ (Wh / kJ) TÜKETİMİ
        toplam_tur = N_TUR_YUKLU + N_TUR_BOS
        toplam_mesafe_m = toplam_tur * route_info["L_total_lap_m"]
        # Navigasyon dışı 100 s bütçe ve 12 s emniyet payı ayrı tutulur.
        # Pay uçulmuş süreye eklenmez; izin verilen toplamdan düşülür.
        toplam_sure_sn = mission_time["toplam_sure_sn"]

        seyir_tuketim_mah = (N_TUR_YUKLU * loaded_res["mah_per_lap"]) + (N_TUR_BOS * empty_res["mah_per_lap"])
        toplam_ucus_tuketim_mah = seyir_tuketim_mah + KALKIS_TIRMANIS_INIS_REZERV_MAH
        iniste_kalan_batarya_yuzde = 100.0 * (1.0 - (toplam_ucus_tuketim_mah / BATARYA_TOPLAM_MAH))

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
        ham_enerji_endeksi = (m_water_kg * toplam_yuklu_wp) / max(seyir_enerji_wh, 1e-6)

        # 5. FİZİBİLİTE KONTROLÜ (%20 İniş Rezervi Dahil!)
        feasible = (
            loaded_res["stall_ok"]
            and empty_res["stall_ok"]
            and loaded_res["thrust_ok"]
            and empty_res["thrust_ok"]
            and mission_time["sure_uygun"]
            and (seyir_tuketim_mah <= SEYIR_MAX_KULLANILABILIR_MAH)
            and (iniste_kalan_batarya_yuzde >= INIS_REZERV_ORANI * 100.0)
        )

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
            "toplam_ucus_tuketim_mah": toplam_ucus_tuketim_mah,
            "iniste_kalan_batarya_yuzde": iniste_kalan_batarya_yuzde,
            "seyir_enerji_Wh": seyir_enerji_wh,
            "seyir_enerji_kJ": seyir_enerji_kj,
            # Puanlar
            "puan_yuklu_2tur": puan_yuklu_toplam,
            "puan_bos_1tur": puan_bos_toplam,
            "puan_navigasyon": puan_navigasyon,
            "ham_enerji_endeksi": ham_enerji_endeksi,
            "feasible": feasible,
        })

    if not results:
        raise ValueError("Değerlendirilecek aday yok: kütle tablosunu ve veri eşleşmelerini kontrol edin.")
    res_df = pd.DataFrame(results)

    # --- BAĞIL ENERJİ PUANI HESABI (1. Sıradaki Aday = 60 Puan) ---
    feasible_mask = res_df["feasible"]
    if feasible_mask.any():
        max_enerji_endeksi = float(res_df.loc[feasible_mask, "ham_enerji_endeksi"].max())
    else:
        max_enerji_endeksi = float(res_df["ham_enerji_endeksi"].max())

    res_df["puan_enerji_60"] = MAX_ENERJI_PUANI * (res_df["ham_enerji_endeksi"] / max_enerji_endeksi)
    res_df["toplam_yarisma_puani"] = res_df["puan_navigasyon"] + res_df["puan_enerji_60"]

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
    print(f"Kullanılabilir Net / Seyir Sınırı: {KULLANILABILIR_TOPLAM_MAH:.0f} mAh Net ({KALKIS_TIRMANIS_INIS_REZERV_MAH:.0f} mAh Kalkış/İniş Payı -> Max {SEYIR_MAX_KULLANILABILIR_MAH:.0f} mAh Seyir)")

    print("\n" + "=" * 80)
    print("=== 2. EN YÜKSEK PUANLI OPTİMUM UÇAK KONFİGÜRASYONU ===")
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
    print(f"Kapasite Tüketimi (Seyir / Top.) : {best['seyir_tuketim_mah']:.1f} mAh Seyir | {best['toplam_ucus_tuketim_mah']:.1f} mAh Toplam Uçuş")
    batarya_uygun = best['iniste_kalan_batarya_yuzde'] >= INIS_REZERV_ORANI * 100.0
    batarya_durumu = "UYGUN" if batarya_uygun else "UYGUN DEĞİL"
    print(f"İnişte Kalan Batarya Doluluğu    : %{best['iniste_kalan_batarya_yuzde']:.1f} (Yarışma Alt Sınırı: >= %20.0) -> [{batarya_durumu}]")
    print("-" * 80)
    print(f"1) Yüklü Navigasyon Puanı (2 Tur): {best['puan_yuklu_2tur']:.2f} Puan")
    print(f"2) Boş Navigasyon Puanı (1 Tur)  : {best['puan_bos_1tur']:.2f} Puan")
    print(f"3) Bağıl Enerji Puanı (Max 60)   : {best['puan_enerji_60']:.2f} Puan (Endeks: {best['ham_enerji_endeksi']:.2f} kg*WP/Wh)")
    print(f"TOPLAM YARIŞMA GÖREV PUANI       : {best['toplam_yarisma_puani']:.2f} Puan")
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

    ax2.set_title("2. 2871 Aday: Harcanan Seyir Enerjisi (Wh) vs. Toplam Yarışma Puanı (Nav + Enerji)", fontsize=11, fontweight="bold")
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
    args = parser.parse_args()
    if args.rota:
        preview_route_only()
    else:
        res_df, route_info, thrust_info = run_full_cruise_mission_optimization()
        plot_mission_and_route_summary(res_df, route_info)