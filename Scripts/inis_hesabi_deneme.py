"""Frensiz ve tum surec boyunca flapli yer kosusuna sahip ucak adaylari icin Gudmundsson Bolum 23 ve IMechE 2027 inis modulu.

Fiziksel Senaryo:
- Havada yaklasma, teker koyma ve tum yer kosusu boyunca FLAPLAR ACIKTIR (llt_flapli_coklu_re.csv
  ve openvsp_llt_flapli_aday_ozet.csv kullanilir).
- Susuz kutlede V_S0, V_REF = 1.3 * V_S0, V_TD = 1.1 * V_S0 ve V_eff = V_TD / sqrt(2) hesaplanir.
- Tekerlek freni ve ters flap yoktur; ucak flapli kanat aerodinamigi ve frensiz yuvarlanma
  surtunmesiyle (mu_pist = 0.08, Gudmundsson Tablo 23-3 Wet grass brakes off) yavaslayip durur.
- Puanlama (IMechE 2027): M1 (M3.2.3: 60p) + M2 (M3.3.5: 60p) + M3 (M3.4.4: 20p) = Toplam 140 puan.
  * Kural M3.2.3 & M3.3.5 (M1/M2 Otonom Inis):
    - 20 m kutu icinde teker koyup tam durma (S_G <= 20 m): 60 puan
    - 20 m kutu icinde teker koyup kutu disinda durma (S_G > 20 m): 40 puan
    - Kutu disinda otonom inis: 20 puan
    - Manuel inis veya hasarli inis: 10 puan
  * Kural M3.4.4 (M3 Manuel Ucus Degerlendirmesi Inisi):
    - Maksimum 20 puan (ilk yarim gun hazirsa 20 tam puan, son gun 10 yarim puan; M3'te 20m kutu kesintisi yoktur).
"""
from __future__ import annotations

import argparse
import bisect
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJE_KLASORU = ROOT.parent if ROOT.name.lower() == "scripts" else ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJE_KLASORU) not in sys.path:
    sys.path.insert(0, str(PROJE_KLASORU))

from aday_pervane import csv_yaz, kimlik_alanlari, performans_adaylarini_hazirla
from llt_csv_okuyucu import LLTError, LLTTable
from tirmanis_hesabi import calc_cd0, find_drag, wing_key

MAX_AIRSPEED_40KTS_MS = 40.0 * 0.514444  # Kural T2.6.1: 40 kts = 20.5778 m/s


class ModelError(ValueError):
    def __init__(self, status: str, detail: str):
        super().__init__(detail)
        self.status = status


def aday_degeri_al(row: dict, *keys: str, varsayilan: float | None = None) -> float:
    for key in keys:
        if row.get(key) not in (None, ""):
            return number(row, key)
    if varsayilan is not None:
        return float(varsayilan)
    raise ModelError("veri_hatasi", f"Eksik alan: {keys}")


def number(row: dict, key: str, positive: bool = False) -> float:
    try:
        val = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelError("veri_hatasi", f"Eksik/gecersiz sayi: {key}") from exc
    if not math.isfinite(val) or (positive and val <= 0):
        raise ModelError("veri_hatasi", f"Gecersiz {key}: {val}")
    return val


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(4096)
        stream.seek(0)
        delimiter = ";" if sample.splitlines()[0].count(";") > sample.splitlines()[0].count(",") else ","
        rows = list(csv.DictReader(stream, delimiter=delimiter))
    if not rows:
        raise ValueError(f"Bos CSV: {path}")
    return rows


def bracket(values, target: float, label: str) -> tuple[float, float, float]:
    xs = sorted(set(values))
    if not xs or target < xs[0] - 1e-8 or target > xs[-1] + 1e-8:
        raise ModelError("veri_araligi_disinda", f"{label}: hedef={target:g}, mevcut=[{xs[0]:g}..{xs[-1]:g}]")
    target = min(max(target, xs[0]), xs[-1])
    i = bisect.bisect_left(xs, target)
    if i < len(xs) and abs(xs[i] - target) < 1e-8:
        return xs[i], xs[i], 0.0
    lo, hi = xs[i - 1], xs[i]
    return lo, hi, (target - lo) / (hi - lo)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def interp_curve(curve: dict[float, float], alpha: float) -> float:
    lo, hi, t = bracket(curve, alpha, "Hucum acisi")
    return lerp(curve[lo], curve[hi], t)


def validate_geometry_consistency(aday: dict) -> None:
    for span_key, area_key, ar_key in (
        ("b_w_m", "S_w_m2", "AR_w"),
        ("b_HT_m", "S_HT_m2", "AR_HT"),
        ("h_VT_m", "S_VT_m2", "AR_VT"),
    ):
        b = number(aday, span_key, True)
        s = number(aday, area_key, True)
        ar = number(aday, ar_key, True)
        if not math.isclose(b * b / s, ar, rel_tol=1e-4, abs_tol=6e-6):
            raise ModelError("geometri_hatasi", f"{ar_key} boyutlarla uyusmuyor: {b}^2/{s} != {ar}")


def solve_landing_for_mass(
    mass_kg: float,
    aday: dict,
    args,
    llt_cache: dict,
) -> dict:
    weight_n = mass_kg * args.g
    sw = aday_degeri_al(aday, "S_w_m2")
    sh = aday_degeri_al(aday, "S_HT_m2", "S_h_m2")
    bw = aday_degeri_al(aday, "b_w_m")
    cw = aday_degeri_al(aday, "c_root_w_m", "c_w_m")
    arw = aday_degeri_al(aday, "AR_w")
    arh = aday_degeri_al(aday, "AR_HT", "AR_h", varsayilan=4.0)

    flap_table = llt_cache["_csv_flap"]
    flap_local_cache: dict = {}

    def state_at_re(re_val: float):
        if re_val not in flap_local_cache:
            velocity = re_val * args.mu / (args.rho * cw)
            flap_local_cache[re_val] = flap_table.values_for(aday, velocity, args.rho, args.mu, args.llt_re_kontrol)
        return flap_local_cache[re_val]

    # 1) Havada Yaklasma ve Teker Koyma: Flapli Inis Stall Hizi (V_S0)
    v_s0 = 10.0
    wing_stall = None
    for _ in range(80):
        re_stall = (args.rho * max(v_s0, 1.0) * cw) / args.mu
        wing_stall = state_at_re(re_stall)[0]
        v_new = math.sqrt((2.0 * weight_n) / (args.rho * sw * wing_stall["cl_max"]))
        if abs(v_new - v_s0) < 1e-4:
            v_s0 = v_new
            break
        v_s0 = v_new
    else:
        raise ModelError("stall_yakinsamadi", "Reynolds/stall hizi iterasyonu yakinsamadi.")

    stall_meta = state_at_re(args.rho * v_s0 * cw / args.mu)[2]

    # Gudmundsson Tablo 23-4 (GA / Kucuk Ucak Hizlari)
    v_ref = 1.3 * v_s0
    v_td = args.v_td_katsayi * v_s0  # Varsayilan 1.1 * V_S0
    v_eff = v_td / math.sqrt(2.0)    # Gudmundsson Denklem (23-16) etkin ortalama hiz

    # 2) Yer Kosusu Aerodinamigi: Tum Surec Boyunca Flaplar Acik (llt_flapli_coklu_re.csv + openvsp_llt_flapli)
    drag_row = find_drag(aday, llt_cache["_drag_flap"].get(wing_key(aday), []))
    cd0_info = calc_cd0(aday, drag_row, v_eff, args.rho, args.mu)
    wing_eff, tail_eff, llt_meta, _ = state_at_re(cd0_info["Re_w"])
    llt_meta = dict(llt_meta)
    llt_meta.update(
        llt_Re_w_stall=stall_meta["llt_Re_w_ucus"],
        llt_stall_alt_sinir_varsayimi=stall_meta["llt_kanat_alt_sinir_varsayimi"],
    )

    cl_w_ldg = interp_curve(wing_eff["curve"], args.alpha_ldg)
    cl_ht_ldg = tail_eff["slope"] * (args.alpha_ldg + args.kuyruk_oturma)
    cl_ldg = cl_w_ldg + (sh / sw) * cl_ht_ldg

    if args.h_kanat > 0:
        phi_d = ((16.0 * args.h_kanat / bw) ** 2) / (1.0 + (16.0 * args.h_kanat / bw) ** 2)
    else:
        phi_d = 1.0

    cdi_w = phi_d * (cl_w_ldg ** 2) / (math.pi * arw * wing_eff["e"])
    cdi_ht = phi_d * (sh / sw) * (cl_ht_ldg ** 2) / (math.pi * arh * tail_eff["e"])
    cdi_ldg = cdi_w + cdi_ht
    cd_ldg = cd0_info["CD0_toplam"] + args.cd_ldg + args.cd_flap_ek + cdi_ldg

    # 3) Referans Yaklasma ve Flare (Denklem 23-9..23-13, SI birimlerinde)
    gamma_app_rad = math.radians(args.gamma_app_deg)
    n_flare = 1.296
    r_flare = ((1.2 * v_s0) ** 2) / (args.g * (n_flare - 1.0))
    h_flare = r_flare * (1.0 - math.cos(gamma_app_rad))
    s_flare = r_flare * math.sin(gamma_app_rad)
    s_approach = max(0.0, (args.h_engel_m - h_flare) / math.tan(gamma_app_rad)) if args.h_engel_m > 0 else 0.0

    # 4) Serbest Yuvarlanma (Flaplar hep acik oldugu icin kapatma gecikmesi yoktur; varsayilan t_tepki = 0.0 s)
    t_fr = args.t_tepki
    s_fr = t_fr * v_td

    # 5) Yontem 1: Gudmundsson Ortalama Ivme ile Flapli/Frensiz Yavaslama (Denklem 23-16)
    q_eff = 0.5 * args.rho * (v_eff ** 2)
    lift_eff = q_eff * sw * cl_ldg
    drag_eff = q_eff * sw * cd_ldg
    slope_rad = math.radians(args.pist_egim_deg)
    thrust_idle = args.itki_rolanti_n

    # Fren olmadan, flaplar acikken: T_idle - D_ldg - mu_pist * (W - L) + W * sin(gamma)
    net_decel_force_eff = thrust_idle - drag_eff - args.mu_pist * max(0.0, weight_n - lift_eff) + weight_n * math.sin(slope_rad)
    if net_decel_force_eff >= -1e-4:
        raise ModelError(
            "yavaslama_yetersiz_duramaz",
            f"V_eff={v_eff:.2f} m/s hizinda net yavaslama kuvveti >= 0 ({net_decel_force_eff:.3f} N)",
        )

    a_decel_avg = (args.g / weight_n) * net_decel_force_eff
    s_br_avg = -(v_td ** 2) / (2.0 * a_decel_avg)
    t_br_avg = -v_td / a_decel_avg

    # 6) Yontem 2: Zaman Adimli Sayisal Integrasyon (Denklem 23-1 / 23-2, dt = 0.01 s)
    dt = 0.01
    v, s_br_num, t_br_num = v_td, 0.0, 0.0
    durabildi_num = False
    for _ in range(int(120.0 / dt)):
        q = 0.5 * args.rho * v * v
        l_n = q * sw * cl_ldg
        d_n = q * sw * cd_ldg
        f_net = thrust_idle - d_n - args.mu_pist * max(0.0, weight_n - l_n) + weight_n * math.sin(slope_rad)
        if f_net >= -1e-4:
            break
        accel = (args.g / weight_n) * f_net
        v_next = v + accel * dt
        if v_next <= 0.0:
            dt_frac = -v / accel
            s_br_num += v * dt_frac + 0.5 * accel * (dt_frac ** 2)
            t_br_num += dt_frac
            durabildi_num = True
            break
        s_br_num += v * dt + 0.5 * accel * (dt ** 2)
        t_br_num += dt
        v = v_next

    if not durabildi_num:
        raise ModelError("yavaslama_yetersiz_duramaz", "Zaman adimli integrasyonda ucak duramadi.")

    s_g_avg = s_fr + s_br_avg
    s_g_num = s_fr + s_br_num
    t_g_avg = t_fr + t_br_avg
    t_g_num = t_fr + t_br_num

    s_g_kriter = s_g_num if args.yontem == "sayisal" else s_g_avg
    t_g_kriter = t_g_num if args.yontem == "sayisal" else t_g_avg

    # Kural T2.8.1, M3.2.3 & M3.3.5 (Otonom M1 & M2):
    # - 20 m kutu icinde teker koyup tam durma: 60 puan
    # - 20 m kutu icinde teker koyup kutu disinda durma: 40 puan
    # - Kutu disinda otonom inis: 20 puan
    # Kural M3.4.4 (M3 Manuel Degerlendirme Inisi):
    # - M3'te 20 m kutu kesintisi yoktur; ilk yarim gun ucusa hazir olan takima tam 20 puan verilir.
    tam_kutu_icinde = s_g_kriter <= args.hedef_mesafe
    teker_kutu_icinde = s_fr <= args.hedef_mesafe
    if tam_kutu_icinde:
        kural_puani_60 = 60.0
    elif teker_kutu_icinde:
        kural_puani_60 = 40.0
    else:
        kural_puani_60 = 20.0
    kural_puani_m3_20 = 20.0

    return {
        "llt_meta": llt_meta,
        "CL_max_flapli": wing_stall["cl_max"],
        "alpha_stall_flapli_deg": wing_stall["stall"],
        "V_S0_m_s": v_s0,
        "V_REF_m_s": v_ref,
        "V_TD_m_s": v_td,
        "V_eff_m_s": v_eff,
        "Re_w_eff": cd0_info["Re_w"],
        "CL_w_LDG": cl_w_ldg,
        "CL_HT_LDG": cl_ht_ldg,
        "CL_LDG": cl_ldg,
        "CD0_toplam": cd0_info["CD0_toplam"],
        "CDi_LDG": cdi_ldg,
        "CD_LDG": cd_ldg,
        "L_eff_N": lift_eff,
        "D_eff_N": drag_eff,
        "a_decel_avg_m_s2": a_decel_avg,
        "S_A_yaklasma_m": s_approach,
        "h_F_flare_m": h_flare,
        "S_F_flare_m": s_flare,
        "S_FR_tepki_m": s_fr,
        "S_BR_ort_ivme_m": s_br_avg,
        "S_BR_sayisal_m": s_br_num,
        "S_G_inis_ort_ivme_m": s_g_avg,
        "sure_inis_ort_ivme_s": t_g_avg,
        "S_G_inis_sayisal_m": s_g_num,
        "sure_inis_sayisal_s": t_g_num,
        "S_G_inis_kriter_m": s_g_kriter,
        "sure_inis_kriter_s": t_g_kriter,
        "S_LDG_toplam_m": s_approach + s_flare + s_g_kriter,
        "tam_kutu_icinde": tam_kutu_icinde,
        "kural_puani_60": kural_puani_60,
        "kural_puani_m3_20": kural_puani_m3_20,
    }


def calculate(mass_rows: list[dict], args) -> list[dict]:
    mass_rows = performans_adaylarini_hazirla(mass_rows, args.pervane)
    flap_table = LLTTable.read(resolve_input(args.llt_flap, ["llt_flapli_coklu_re.csv"]), flapped=True)
    drag_flap = read_csv(
        resolve_input(
            args.drag_flap,
            ["openvsp_llt_flapli_aday_ozet.csv", "openvsp_llt_flapli_polar_detay.csv"],
        )
    )

    drag_flap_idx: dict = {}
    for row in drag_flap:
        drag_flap_idx.setdefault(wing_key(row), []).append(row)

    llt_cache = {
        "_csv_flap": flap_table,
        "_drag_flap": drag_flap_idx,
    }
    # Inis itkiden bagimsiz oldugu icin ayni temel_aday_id (farkli pervaneler) icin sonucu onbellekle.
    base_landing_cache: dict[str, dict | Exception] = {}

    output = []
    for index, mass in enumerate(mass_rows, 1):
        kanat_profil = Path(str(mass.get("kanat_airfoil") or mass.get("profil") or "")).stem.strip()
        flap_profil = f"{kanat_profil}_flap" if kanat_profil else ""
        out = {
            **kimlik_alanlari(mass),
            "aday_no": index,
            "kanat_airfoil": kanat_profil,
            "flap_airfoil": flap_profil,
            "durum": "",
            "aciklama": "",
            "inis_20m_basarili": False,
            "puan_M1_inis_60": 0.0,
            "puan_M2_inis_60": 0.0,
            "puan_M3_manuel_inis_20": 0.0,
            "resmi_inis_puani_140": 0.0,
            "yontem": args.yontem,
            "rho_kg_m3": args.rho,
            "mu_pist_frensiz": args.mu_pist,
            "alpha_LDG_deg": args.alpha_ldg,
            "hedef_mesafe_m": args.hedef_mesafe,
        }
        try:
            if not flap_profil:
                raise ModelError("flap_bulunamadi", "Adayda kanat_airfoil bos.")

            validate_geometry_consistency(mass)
            m_yuklu = number(mass, "total_kutle_kg", True) + args.ek_cihaz_kutlesi_kg
            m_susuz_raw = (
                number(mass, "total_susuz_kutle_kg", True)
                if mass.get("total_susuz_kutle_kg") not in (None, "")
                else number(mass, "total_kutle_kg", True)
            )
            m_susuz = m_susuz_raw + args.ek_cihaz_kutlesi_kg
            if m_susuz > m_yuklu + 1e-8:
                raise ModelError("kutle_hatasi", f"Susuz kutle ({m_susuz:.3f} kg) dolu kutleden ({m_yuklu:.3f} kg) buyuk.")

            sw = aday_degeri_al(mass, "S_w_m2")
            bw = aday_degeri_al(mass, "b_w_m")
            l_govde = aday_degeri_al(mass, "govde_uzunlugu_m", varsayilan=1.2)
            b_flap = aday_degeri_al(mass, "b_flap_m", varsayilan=min(1.0, bw))

            out.update(
                kutle_inis_susuz_kg=m_susuz,
                S_w_m2=sw,
                b_w_m=bw,
                b_flap_m=b_flap,
                flap_aciklik_orani=b_flap / bw,
                AR_w=aday_degeri_al(mass, "AR_w"),
                L_govde_m=l_govde,
                govde_aday_no=mass.get("govde_aday_no", ""),
            )

            if m_yuklu > 10.0:
                raise ModelError("mtom_10kg_asimi", f"Kural T2.2.1 ihlali: MTOM={m_yuklu:.3f} kg > 10.0 kg")
            if max(bw, l_govde) > 4.0:
                raise ModelError("boyut_4m_asimi", f"Kural T2.3.1 ihlali: Maks boyut={max(bw, l_govde):.3f} m > 4.0 m")

            base_id = out["temel_aday_id"]
            if base_id not in base_landing_cache:
                try:
                    base_landing_cache[base_id] = solve_landing_for_mass(m_susuz, mass, args, llt_cache)
                except (ModelError, LLTError, KeyError, FileNotFoundError) as exc:
                    base_landing_cache[base_id] = exc
            cached = base_landing_cache[base_id]
            if isinstance(cached, Exception):
                raise cached
            res = cached
            out.update(res["llt_meta"])

            if res["V_REF_m_s"] > MAX_AIRSPEED_40KTS_MS:
                raise ModelError(
                    "hiz_40kts_asimi",
                    f'Kural T2.6.1 ihlali: V_REF={res["V_REF_m_s"]:.2f} m/s > 40 kts ({MAX_AIRSPEED_40KTS_MS:.2f} m/s)',
                )

            puan_60 = res["kural_puani_60"]
            puan_m3_20 = res["kural_puani_m3_20"]
            basarili_20m = res["tam_kutu_icinde"]
            out.update(
                CL_max_flapli=res["CL_max_flapli"],
                alpha_stall_flapli_deg=res["alpha_stall_flapli_deg"],
                V_S0_m_s=res["V_S0_m_s"],
                V_REF_m_s=res["V_REF_m_s"],
                V_TD_m_s=res["V_TD_m_s"],
                V_eff_m_s=res["V_eff_m_s"],
                Re_w_eff=res["Re_w_eff"],
                CL_LDG=res["CL_LDG"],
                CD0_toplam=res["CD0_toplam"],
                CDi_LDG=res["CDi_LDG"],
                CD_LDG=res["CD_LDG"],
                L_eff_N=res["L_eff_N"],
                D_eff_N=res["D_eff_N"],
                a_decel_avg_m_s2=res["a_decel_avg_m_s2"],
                S_A_yaklasma_m=res["S_A_yaklasma_m"],
                h_F_flare_m=res["h_F_flare_m"],
                S_F_flare_m=res["S_F_flare_m"],
                S_FR_tepki_m=res["S_FR_tepki_m"],
                S_BR_ort_ivme_m=res["S_BR_ort_ivme_m"],
                S_BR_sayisal_m=res["S_BR_sayisal_m"],
                S_G_inis_ort_ivme_m=res["S_G_inis_ort_ivme_m"],
                sure_inis_ort_ivme_s=res["sure_inis_ort_ivme_s"],
                S_G_inis_sayisal_m=res["S_G_inis_sayisal_m"],
                sure_inis_sayisal_s=res["sure_inis_sayisal_s"],
                S_G_inis_kriter_m=res["S_G_inis_kriter_m"],
                sure_inis_kriter_s=res["sure_inis_kriter_s"],
                S_LDG_toplam_m=res["S_LDG_toplam_m"],
                inis_20m_basarili=basarili_20m,
                puan_M1_inis_60=puan_60,
                puan_M2_inis_60=puan_60,
                puan_M3_manuel_inis_20=puan_m3_20,
                resmi_inis_puani_140=2.0 * puan_60 + puan_m3_20,
                durum="uygun" if basarili_20m else "mesafe_asimi_20m",
                aciklama=(
                    "" if basarili_20m
                    else f'Flapli/frensiz S_G_inis={res["S_G_inis_kriter_m"]:.2f} m > 20 m (M1/M2: {puan_60:.0f}/60, M3: {puan_m3_20:.0f}/20)'
                ),
            )
        except (ModelError, LLTError, KeyError, FileNotFoundError) as exc:
            status = getattr(exc, "status", "veri_hatasi")
            out["durum"], out["aciklama"] = status, str(exc)
        output.append(out)

    return output


def resolve_input(explicit, names):
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise ValueError(f"Girdi bulunamadi: {path}")
        return path
    directories = [ROOT / "veriler", ROOT, ROOT.parent, ROOT / "Scripts", ROOT.parent / "Scripts"]
    for directory in directories:
        for name in names:
            path = directory / name
            if path.exists():
                return path
    raise ValueError(f"Girdi bulunamadi: {names}. Dosya yolunu komut satirinda belirtin.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("kutle", "llt-flap", "drag-flap", "llt-flapsiz", "drag-flapsiz"):
        parser.add_argument("--" + flag, type=Path)
    parser.add_argument("--cikti", type=Path, default=ROOT / "inis_sonuclari.csv")
    for flag, default in [
        ("hedef-mesafe", 20.0),
        ("v-td-katsayi", 1.1),
        ("t-tepki", 0.0),         # Flaplar tum surec boyunca acik kaldigi icin tepki gecikmesi 0 s
        ("rho", 1.175),
        ("mu", 1.789e-5),
        ("g", 9.80665),
        ("mu-pist", 0.08),        # Tablo 23-3: Wet grass (Brakes off) - T2.7.1 & T2.8.3 (Hard turf: 0.05)
        ("alpha-ldg", 0.0),
        ("kuyruk-oturma", 0.0),
        ("h-kanat", 0.0),
        ("cd-ldg", 0.0),          # Varsa inis takimi ek parazit suruklemesi
        ("cd-flap-ek", 0.0),      # Varsa ek flap parazit suruklemesi
        ("itki-rolanti-n", 0.0),
        ("gamma-app-deg", 3.0),
        ("h-engel-m", 0.0),       # Kural T2.8.1: Engel asma sarti yok
        ("pist-egim-deg", 0.0),
        ("ek-cihaz-kutlesi-kg", 0.0),
    ]:
        parser.add_argument("--" + flag, type=float, default=default)
    parser.add_argument("--pervane", help="Yalniz pervane alani bos eski adaylar icin acik varsayilan.")
    parser.add_argument("--yontem", choices=["ort_ivme", "sayisal"], default="ort_ivme",
                        help="20m baraji ve puanlama icin baz alinacak yontem (her iki yontem de CSVye yazilir).")
    parser.add_argument("--llt-re-kontrol", action="store_true", help="Alt sinir katsayisi kabullerini reddet.")
    args = parser.parse_args(argv)

    try:
        masses = read_csv(resolve_input(args.kutle, ["kanat_kuyruk_govde_kutleli.csv"]))
        output = calculate(masses, args)
        csv_yaz(output, args.cikti)

        counts: dict[str, int] = {}
        for row in output:
            counts[row["durum"]] = counts.get(row["durum"], 0) + 1
        print(f"{len(output)} flapli/frensiz yer kosulu inis adayi yazildi: {args.cikti.resolve()}")
        print("Durumlar: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())