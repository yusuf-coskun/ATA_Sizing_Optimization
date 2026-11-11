"""Flaplı uçak adayları için Gudmundsson Bölüm 18 ve IMechE UAS Challenge 2027 Kuralları bazlı kalkış kodu.

Yeni '<kanat_airfoil>_flap' adlandırma standardına göre güncellenmiştir:
- Her adayın 'kanat_airfoil' adına (ör. 'clarkysm-il') doğrudan '_flap' eklenerek
  'clarkysm-il_flap' profili ve poları çağrılır.
- Adaylar 'itki_tablosu.csv' içindeki TÜM PERVANE TİPLERİYLE çaprazlanır.
- Pist sürtünme katsayısı mu = 0.03, ek flap parazit sürüklemesi = 0.0.
- UAS Challenge 2027 Kuralları (M3.2.1, M3.3.1, M3.3.6, T2.8.1) puanlaması uygulanır.
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

import drag_hesabi as dh
import lifting_line_cozucu as llt

MAX_AIRSPEED_40KTS_MS = 40.0 * 0.514444  # Kural T2.6.1: 40 kts = 20.5778 m/s


class ModelError(ValueError):
    def __init__(self, status: str, detail: str):
        super().__init__(detail)
        self.status = status


def number(row: dict, key: str, positive: bool = False) -> float:
    try:
        val = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelError('veri_hatasi', f'Eksik/gecersiz sayi: {key}') from exc
    if not math.isfinite(val) or (positive and val <= 0):
        raise ModelError('veri_hatasi', f'Gecersiz {key}: {val}')
    return val


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        sample = stream.read(4096)
        stream.seek(0)
        delimiter = ';' if sample.splitlines()[0].count(';') > sample.splitlines()[0].count(',') else ','
        rows = list(csv.DictReader(stream, delimiter=delimiter))
    if not rows:
        raise ValueError(f'Bos CSV: {path}')
    return rows


def bracket(values, target: float, label: str) -> tuple[float, float, float]:
    xs = sorted(set(values))
    if not xs or target < xs[0] - 1e-8 or target > xs[-1] + 1e-8:
        raise ModelError('veri_araligi_disinda', f'{label}: hedef={target:g}, mevcut=[{xs[0]:g}..{xs[-1]:g}]')
    target = min(max(target, xs[0]), xs[-1])
    i = bisect.bisect_left(xs, target)
    if i < len(xs) and abs(xs[i] - target) < 1e-8:
        return xs[i], xs[i], 0.0
    lo, hi = xs[i - 1], xs[i]
    return lo, hi, (target - lo) / (hi - lo)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def interp_curve(curve: dict[float, float], alpha: float) -> float:
    lo, hi, t = bracket(curve, alpha, 'Hucum acisi')
    return lerp(curve[lo], curve[hi], t)


def index_propellers(prop_rows: list[dict], throttle: float, voltage: float, mode: str) -> dict[str, dict[float, dict]]:
    prop_map: dict[str, dict[float, dict]] = {}
    for row in prop_rows:
        prop = row['pervane'].strip()
        gaz = number(row, 'gaz_yuzde')
        if not math.isclose(gaz, throttle, abs_tol=1e-4):
            continue
        v = number(row, 'hiz_m_s')
        itki = number(row, 'itki_n')
        if mode == 'akim_a':
            guc = number(row, 'akim_a') * voltage
            kaynak = 'akim_a_x_gerilim'
        else:
            if row.get('guc_w') not in (None, ''):
                guc = number(row, 'guc_w')
                kaynak = 'guc_w'
            else:
                guc = number(row, 'akim_a')
                kaynak = 'akim_a_W_kabul_gecici'
        prop_map.setdefault(prop, {})[v] = {'itki_N': itki, 'guc_W': guc, 'kaynak': kaynak}
    return prop_map


def thrust_at_speed(speed_map: dict[float, dict], speed: float) -> tuple[float, float, str]:
    lo, hi, f = bracket(speed_map, speed, 'Pervane hizi')
    a, b = speed_map[lo], speed_map[hi]
    return (
        lerp(a['itki_N'], b['itki_N'], f),
        lerp(a['guc_W'], b['guc_W'], f),
        a['kaynak'],
    )


def solve_takeoff_for_mass(
    mass_kg: float,
    aday: dict,
    kanat_profil: str,
    flap_profil: str,
    kuyruk_profil: str,
    oran_flap: float,
    speed_map: dict[float, dict],
    flapsiz_indeks: dict,
    flapli_indeks: dict,
    args,
    llt_cache: dict,
) -> dict:
    weight_n = mass_kg * args.g
    sw = llt.aday_degeri_al(aday, 'S_w_m2')
    sh = llt.aday_degeri_al(aday, 'S_HT_m2', 'S_h_m2')
    bw = llt.aday_degeri_al(aday, 'b_w_m')
    cw = llt.aday_degeri_al(aday, 'c_root_w_m', 'c_w_m')
    arw = llt.aday_degeri_al(aday, 'AR_w')
    arh = llt.aday_degeri_al(aday, 'AR_HT', 'AR_h', varsayilan=4.0)

    def get_flapped_wing(re_val: float) -> dict:
        key = ('w_flap', kanat_profil.lower(), flap_profil.lower(), round(arw, 6), round(oran_flap, 4), round(re_val, 1))
        if key not in llt_cache:
            llt_cache[key] = llt.flapli_kanat_coz_re(
                kanat_profil, flap_profil, arw, oran_flap, re_val, flapsiz_indeks, flapli_indeks
            )
        return llt_cache[key]

    def get_tail(re_val: float) -> dict:
        key = ('ht', kuyruk_profil.lower(), round(arh, 6), round(re_val, 1))
        if key not in llt_cache:
            llt_cache[key] = llt.kuyruk_coz_re(kuyruk_profil, arh, re_val, flapsiz_indeks)
        return llt_cache[key]

    # 1) Stall Hızı (V_S1) ve Re_stall Sabit Nokta İterasyonu
    v_stall = 10.0
    wing_stall = None
    for _ in range(8):
        re_stall = (args.rho * max(v_stall, 1.0) * cw) / args.mu
        wing_stall = get_flapped_wing(re_stall)
        v_new = math.sqrt((2.0 * weight_n) / (args.rho * sw * wing_stall['cl_max']))
        if abs(v_new - v_stall) < 1e-4:
            v_stall = v_new
            break
        v_stall = v_new

    v_lof = args.v_lof_katsayi * v_stall
    v_eff = v_lof / math.sqrt(2.0)  # Gudmundsson Denklem (18-22) etkin ortalama hız

    # 2) Etkin Hızda (V_eff) Gerçek Reynolds ile OpenVSP CD0 ve Flaplı LLT
    cd0_info = dh.aday_cd0_hesapla(
        aday, hiz_ms=v_eff, rho=args.rho, mu=args.mu,
        flapli_mi=True, flap_profil_adi=flap_profil, oran_flap=oran_flap,
    )
    wing_eff = get_flapped_wing(cd0_info['Re_w'])
    tail_eff = get_tail(cd0_info['Re_HT'])

    cl_w_to = interp_curve(wing_eff['curve'], args.alpha_to)
    cl_ht_to = tail_eff['slope'] * (args.alpha_to + args.kuyruk_oturma)
    cl_to = cl_w_to + (sh / sw) * cl_ht_to

    if args.h_kanat > 0:
        phi_d = ((16.0 * args.h_kanat / bw) ** 2) / (1.0 + (16.0 * args.h_kanat / bw) ** 2)
    else:
        phi_d = 1.0

    cdi_w = phi_d * (cl_w_to ** 2) / (math.pi * arw * wing_eff['e'])
    cdi_ht = phi_d * (sh / sw) * (cl_ht_to ** 2) / (math.pi * arh * tail_eff['e'])
    cdi_to = cdi_w + cdi_ht
    cd_to = cd0_info['CD0_toplam'] + args.cd_ldg + args.cd_flap_ek + cdi_to

    # 3) Yöntem 1: Gudmundsson Ortalama İvme Yöntemi (Denklem 18-22)
    q_eff = 0.5 * args.rho * (v_eff ** 2)
    lift_eff = q_eff * sw * cl_to
    drag_eff = q_eff * sw * cd_to
    thrust_eff, power_eff, guc_kaynagi = thrust_at_speed(speed_map, v_eff)
    thrust_statik, _, _ = thrust_at_speed(speed_map, 0.0)

    net_force_eff = thrust_eff - drag_eff - args.mu_pist * max(0.0, weight_n - lift_eff)
    if net_force_eff <= 1e-4:
        raise ModelError(
            'itki_yetersiz_kalkamaz',
            f'V_eff={v_eff:.2f} m/s hizinda net kuvvet <= 0 ({net_force_eff:.3f} N)',
        )

    a_avg = (args.g / weight_n) * net_force_eff
    s_g_avg = (v_lof ** 2) / (2.0 * a_avg)
    t_to_avg = v_lof / a_avg
    wh_to_avg = power_eff * t_to_avg / 3600.0
    mah_to_avg = 1000.0 * wh_to_avg / args.gerilim

    # 4) Yöntem 2: Gudmundsson Zaman Adımlı Sayısal İntegrasyon (Denklem 18-12..17)
    dt = 0.01
    v, s_num, t_num, energy_j = 0.0, 0.0, 0.0, 0.0
    kalkabildi_num = False
    for _ in range(int(60.0 / dt)):
        t_n, p_w, _ = thrust_at_speed(speed_map, v)
        q = 0.5 * args.rho * v * v
        l_n = q * sw * cl_to
        d_n = q * sw * cd_to
        f_net = t_n - d_n - args.mu_pist * max(0.0, weight_n - l_n)
        if f_net <= 1e-4:
            break
        accel = (args.g / weight_n) * f_net
        v_next = v + accel * dt
        if v_next >= v_lof:
            dt_frac = (v_lof - v) / accel
            s_num += v * dt_frac + 0.5 * accel * (dt_frac ** 2)
            t_num += dt_frac
            energy_j += p_w * dt_frac
            kalkabildi_num = True
            break
        s_num += v * dt + 0.5 * accel * (dt ** 2)
        t_num += dt
        energy_j += p_w * dt
        v = v_next

    if not kalkabildi_num:
        raise ModelError('itki_yetersiz_kalkamaz', 'Zaman adimli integrasyonda V_LOF hizina ulasilamadi.')

    wh_to_num = energy_j / 3600.0
    mah_to_num = 1000.0 * wh_to_num / args.gerilim

    s_kriter = s_num if args.yontem == 'sayisal' else s_g_avg
    wh_kriter = wh_to_num if args.yontem == 'sayisal' else wh_to_avg
    mah_kriter = mah_to_num if args.yontem == 'sayisal' else mah_to_avg

    kutu_icinde = s_kriter <= args.hedef_mesafe
    kural_puani_20 = 20.0 if kutu_icinde else 5.0

    return {
        'CL_max_flapli': wing_stall['cl_max'],
        'alpha_stall_deg': wing_stall['stall'],
        'V_stall_m_s': v_stall,
        'V_LOF_m_s': v_lof,
        'V_eff_m_s': v_eff,
        'Re_w_eff': cd0_info['Re_w'],
        'CL_w_TO': cl_w_to,
        'CL_HT_TO': cl_ht_to,
        'CL_TO': cl_to,
        'CD0_toplam': cd0_info['CD0_toplam'],
        'CDi_TO': cdi_to,
        'CD_TO': cd_to,
        'T_statik_N': thrust_statik,
        'T_W_statik': thrust_statik / weight_n,
        'L_eff_N': lift_eff,
        'D_eff_N': drag_eff,
        'T_eff_N': thrust_eff,
        'a_avg_m_s2': a_avg,
        'S_G_ort_ivme_m': s_g_avg,
        'sure_ort_ivme_s': t_to_avg,
        'tuketim_ort_ivme_mAh': mah_to_avg,
        'enerji_ort_ivme_Wh': wh_to_avg,
        'S_G_sayisal_m': s_num,
        'sure_sayisal_s': t_num,
        'tuketim_sayisal_mAh': mah_to_num,
        'enerji_sayisal_Wh': wh_to_num,
        'S_ROT_1s_m': v_lof * 1.0,
        'S_G_kriter_m': s_kriter,
        'tuketim_kriter_mAh': mah_kriter,
        'enerji_kriter_Wh': wh_kriter,
        'guc_kaynagi': guc_kaynagi,
        'kutu_20m_icinde': kutu_icinde,
        'kural_puani_20': kural_puani_20,
    }


def calculate(mass_rows: list[dict], prop_map: dict[str, dict[float, dict]], args) -> list[dict]:
    flapsiz_indeks = llt.polar_klasorunu_indeksle(llt.POLAR_KLASORU)
    flapli_indeks = llt.polar_klasorunu_indeksle(llt.FLAP_POLAR_KLASORU)
    all_props = sorted(prop_map.keys())
    llt_cache: dict = {}

    output = []
    aday_sayaci = 0

    for mass in mass_rows:
        kanat_profil = Path(str(mass.get('kanat_airfoil') or mass.get('profil') or '')).stem.strip()
        kuyruk_profil = Path(str(mass.get('kuyruk_airfoil') or mass.get('ht_airfoil') or 'naca0010')).stem.strip()
        flap_profil = f"{kanat_profil}_flap" if kanat_profil else ""

        prop_list = [(mass.get('pervane') or args.pervane).strip()] if args.tek_pervane else all_props

        for prop in prop_list:
            aday_sayaci += 1
            out = {
                'aday_no': aday_sayaci,
                'kanat_airfoil': kanat_profil,
                'flap_airfoil': flap_profil,
                'pervane': prop,
                'durum': '',
                'aciklama': '',
                'kalkis_20m_basarili': False,
                'puan_M1_bos_kalkis_20': 0.0,
                'puan_M2_yuklu_kalkis_20': 0.0,
                'puan_M3_manuel_kalkis_20': 0.0,
                'resmi_kalkis_puani_60': 0.0,
                'tasarim_kalkis_puani_100': 0.0,
                'gaz_yuzde': args.gaz_yuzde,
                'gerilim_V': args.gerilim,
                'rho_kg_m3': args.rho,
                'mu_pist': args.mu_pist,
                'alpha_TO_deg': args.alpha_to,
                'hedef_mesafe_m': args.hedef_mesafe,
            }
            try:
                if not flap_profil:
                    raise ModelError('flap_bulunamadi', 'Adayda kanat_airfoil bos.')
                if prop not in prop_map:
                    raise ModelError('pervane_bulunamadi', f'{prop} @ %{args.gaz_yuzde:g} gaz verisi yok.')

                m_yuklu = number(mass, 'total_kutle_kg', True) + args.ek_cihaz_kutlesi_kg
                m_susuz = float(mass.get('total_susuz_kutle_kg') or m_yuklu) + args.ek_cihaz_kutlesi_kg
                m_su_kg = max(0.0, m_yuklu - m_susuz)

                sw = llt.aday_degeri_al(mass, 'S_w_m2')
                sh = llt.aday_degeri_al(mass, 'S_HT_m2', 'S_h_m2')
                bw = llt.aday_degeri_al(mass, 'b_w_m')
                arw = llt.aday_degeri_al(mass, 'AR_w')
                arh = llt.aday_degeri_al(mass, 'AR_HT', 'AR_h', varsayilan=4.0)
                l_govde = llt.aday_degeri_al(mass, 'govde_uzunlugu_m', varsayilan=1.2)
                b_flap = llt.aday_degeri_al(mass, 'b_flap_m', varsayilan=min(1.0, bw))
                oran_flap = b_flap / bw

                out.update(
                    kutle_M2_yuklu_kg=m_yuklu,
                    kutle_M1_susuz_kg=m_susuz,
                    su_payload_kg=m_su_kg,
                    payload_kutle_orani=m_su_kg / m_yuklu if m_yuklu > 0 else 0.0,
                    S_w_m2=sw,
                    S_HT_m2=sh,
                    b_w_m=bw,
                    b_flap_m=b_flap,
                    flap_aciklik_orani=oran_flap,
                    AR_w=arw,
                    AR_HT=arh,
                    L_govde_m=l_govde,
                    govde_aday_no=mass.get('govde_aday_no', ''),
                )

                if m_yuklu > 10.0:
                    raise ModelError('mtom_10kg_asimi', f'Kural T2.2.1 ihlali: MTOM={m_yuklu:.3f} kg > 10.0 kg')
                if max(bw, l_govde) > 4.0:
                    raise ModelError('boyut_4m_asimi', f'Kural T2.3.1 ihlali: Maks boyut={max(bw, l_govde):.3f} m > 4.0 m')

                # Mission 2 (Su Yüklü)
                m2 = solve_takeoff_for_mass(
                    m_yuklu, mass, kanat_profil, flap_profil, kuyruk_profil,
                    oran_flap, prop_map[prop], flapsiz_indeks, flapli_indeks, args, llt_cache,
                )

                # Mission 1 & 3 (Susuz)
                m1 = solve_takeoff_for_mass(
                    m_susuz, mass, kanat_profil, flap_profil, kuyruk_profil,
                    oran_flap, prop_map[prop], flapsiz_indeks, flapli_indeks, args, llt_cache,
                )

                if m2['V_LOF_m_s'] > MAX_AIRSPEED_40KTS_MS:
                    raise ModelError(
                        'hiz_40kts_asimi',
                        f'Kural T2.6.1 ihlali: V_LOF={m2["V_LOF_m_s"]:.2f} m/s > 40 kts ({MAX_AIRSPEED_40KTS_MS:.2f} m/s)',
                    )

                puan_m1 = m1['kural_puani_20']
                puan_m2 = m2['kural_puani_20']
                puan_m3 = m1['kural_puani_20']
                resmi_toplam_60 = puan_m1 + puan_m2 + puan_m3
                basarili_20m = m2['kutu_20m_icinde']

                out.update(
                    CL_max_flapli=m2['CL_max_flapli'],
                    alpha_stall_flapli_deg=m2['alpha_stall_deg'],
                    V_stall_M2_m_s=m2['V_stall_m_s'],
                    V_LOF_M2_m_s=m2['V_LOF_m_s'],
                    V_eff_M2_m_s=m2['V_eff_m_s'],
                    Re_w_M2_eff=m2['Re_w_eff'],
                    CL_TO=m2['CL_TO'],
                    CD0_toplam=m2['CD0_toplam'],
                    CDi_TO=m2['CDi_TO'],
                    CD_TO=m2['CD_TO'],
                    T_statik_N=m2['T_statik_N'],
                    T_W_M2_statik=m2['T_W_statik'],
                    L_eff_M2_N=m2['L_eff_N'],
                    D_eff_M2_N=m2['D_eff_N'],
                    T_eff_M2_N=m2['T_eff_N'],
                    a_avg_M2_m_s2=m2['a_avg_m_s2'],
                    S_G_M2_ort_ivme_m=m2['S_G_ort_ivme_m'],
                    sure_M2_ort_ivme_s=m2['sure_ort_ivme_s'],
                    tuketim_M2_ort_ivme_mAh=m2['tuketim_ort_ivme_mAh'],
                    enerji_M2_ort_ivme_Wh=m2['enerji_ort_ivme_Wh'],
                    S_G_M2_sayisal_m=m2['S_G_sayisal_m'],
                    sure_M2_sayisal_s=m2['sure_sayisal_s'],
                    tuketim_M2_sayisal_mAh=m2['tuketim_sayisal_mAh'],
                    enerji_M2_sayisal_Wh=m2['enerji_sayisal_Wh'],
                    S_ROT_M2_1s_m=m2['S_ROT_1s_m'],
                    S_G_M2_kriter_m=m2['S_G_kriter_m'],
                    V_stall_M1_bos_m_s=m1['V_stall_m_s'],
                    V_LOF_M1_bos_m_s=m1['V_LOF_m_s'],
                    S_G_M1_bos_ort_ivme_m=m1['S_G_ort_ivme_m'],
                    S_G_M1_bos_sayisal_m=m1['S_G_sayisal_m'],
                    S_G_M1_bos_kriter_m=m1['S_G_kriter_m'],
                    tuketim_M1_bos_mAh=m1['tuketim_kriter_mAh'],
                    enerji_M1_bos_Wh=m1['enerji_kriter_Wh'],
                    kalkis_20m_basarili=basarili_20m,
                    puan_M1_bos_kalkis_20=puan_m1,
                    puan_M2_yuklu_kalkis_20=puan_m2,
                    puan_M3_manuel_kalkis_20=puan_m3,
                    resmi_kalkis_puani_60=resmi_toplam_60,
                    durum='uygun' if basarili_20m else 'mesafe_asimi_20m',
                    aciklama=(
                        '' if basarili_20m
                        else f'M2 Yuklu S_G={m2["S_G_kriter_m"]:.2f} m > 20 m (Kural M3.3.1: 20 yerine 5 puan)'
                    ),
                )
            except (ModelError, KeyError, FileNotFoundError) as exc:
                status = getattr(exc, 'status', 'veri_hatasi')
                out['durum'], out['aciklama'] = status, str(exc)
            output.append(out)

    uygunlar = [r for r in output if r.get('kalkis_20m_basarili')]
    if uygunlar:
        eff_list = [
            (r['su_payload_kg'] / r['enerji_M2_ort_ivme_Wh']) if r['enerji_M2_ort_ivme_Wh'] > 0 else 0.0
            for r in uygunlar
        ]
        max_eff = max(eff_list) or 1.0
        max_v_stall = max(r['V_stall_M2_m_s'] for r in uygunlar) or 1.0

        for r in uygunlar:
            s_skor = (args.hedef_mesafe - r['S_G_M2_kriter_m']) / args.hedef_mesafe
            aday_eff = (r['su_payload_kg'] / r['enerji_M2_ort_ivme_Wh']) if r['enerji_M2_ort_ivme_Wh'] > 0 else 0.0
            e_skor = aday_eff / max_eff
            v_skor = 1.0 - (r['V_stall_M2_m_s'] / max_v_stall)
            r['tasarim_kalkis_puani_100'] = round(100.0 * (0.50 * s_skor + 0.35 * e_skor + 0.15 * v_skor), 4)

    for r in output:
        if not r.get('kalkis_20m_basarili') and r.get('resmi_kalkis_puani_60', 0.0) > 0:
            r['tasarim_kalkis_puani_100'] = round(
                25.0 * (r['resmi_kalkis_puani_60'] / 60.0) * (args.hedef_mesafe / max(r.get('S_G_M2_kriter_m', 40.0), args.hedef_mesafe)),
                4,
            )

    return output


def resolve_input(explicit, names):
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise ValueError(f'Girdi bulunamadi: {path}')
        return path
    directories = [ROOT / 'veriler', ROOT, ROOT.parent, ROOT / 'Scripts', ROOT.parent / 'Scripts']
    for directory in directories:
        for name in names:
            path = directory / name
            if path.exists():
                return path
    raise ValueError(f'Girdi bulunamadi: {names}. Dosya yolunu komut satirinda belirtin.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ('kutle', 'itki'):
        parser.add_argument('--' + flag, type=Path)
    parser.add_argument('--cikti', type=Path, default=ROOT / 'kalkis_sonuclari.csv')
    for flag, default in [
        ('hedef-mesafe', 20.0),
        ('v-lof-katsayi', 1.1),
        ('gaz-yuzde', 100.0),
        ('gerilim', 22.2),
        ('rho', 1.175),
        ('mu', 1.789e-5),
        ('g', 9.80665),
        ('mu-pist', 0.03),
        ('alpha-to', 0.0),
        ('kuyruk-oturma', 0.0),
        ('h-kanat', 0.0),
        ('cd-ldg', 0.0),
        ('cd-flap-ek', 0.0),
        ('ek-cihaz-kutlesi-kg', 0.0),
    ]:
        parser.add_argument('--' + flag, type=float, default=default)
    parser.add_argument('--pervane', default='14x7', help='--tek-pervane secildiginde kullanilacak varsayilan pervane.')
    parser.add_argument('--tek-pervane', action='store_true',
                        help='Belirtilirse tum pervaneleri caprazlamak yerine sadece --pervane ile verilen pervaneyi cozer.')
    parser.add_argument('--yontem', choices=['ort_ivme', 'sayisal'], default='ort_ivme',
                        help='20m baraji ve puanlama icin baz alinacak yontem (her iki yontem de CSVye yazilir).')
    parser.add_argument('--tablo-degeri', choices=['akim_a', 'guc_w'], default='akim_a',
                        help='akim_a: gercek A * V (varsayilan). guc_w: guc_w/power_w veya gecici W.')
    args = parser.parse_args(argv)

    try:
        masses = read_csv(resolve_input(args.kutle, ['kanat_kuyruk_govde_kutleli.csv']))
        props = read_csv(resolve_input(args.itki, ['itki_tablosu.csv']))
        prop_map = index_propellers(props, args.gaz_yuzde, args.gerilim, args.tablo_degeri)

        output = calculate(masses, prop_map, args)
        fields = list(dict.fromkeys(key for row in output for key in row))
        args.cikti.parent.mkdir(parents=True, exist_ok=True)
        with args.cikti.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output)

        counts = {}
        for row in output:
            counts[row['durum']] = counts.get(row['durum'], 0) + 1
        print(f'{len(output)} flapli kalkis kombinasyonu (ucak x pervane) yazildi: {args.cikti.resolve()}')
        print('Durumlar: ' + ', '.join(f'{k}={v}' for k, v in counts.items()))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f'HATA: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())