"""Flapsiz adaylar icin sabit hizli tirmanis ve batarya tuketimi.

Python 3.10+, yalniz standart kutuphane. Ornek: python tirmanis_hesabi.py
Varsayilanlar: Vx=15, Vz=4 m/s, dh=50 m, U=24 V, pervane=14x7.
Trim/downwash yok; i_HT=0, NACA0010 icin alpha_L0_HT=0.
Itki tablosundaki akim_a amperdir; guc P=U*I ile hesaplanir (varsayilan sabit U=24 V).
"""
from __future__ import annotations

import argparse
import bisect
import csv
import math
import re
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent


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


def wing_key(row: dict) -> tuple:
    name = (row.get('kanat_airfoil') or row.get('profil') or '').strip().lower()
    return name, round(number(row, 'b_w_m', True), 6), round(number(row, 'AR_w', True), 6)


def bracket(values, target: float, label: str) -> tuple[float, float, float]:
    xs = sorted(set(values))
    if not xs or target < xs[0] - 1e-8 or target > xs[-1] + 1e-8:
        raise ModelError('veri_araligi_disinda', f'{label}: hedef={target:g}, mevcut={xs}')
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


def index_wing(rows: list[dict]) -> dict:
    result = {}
    for row in rows:
        key = wing_key(row)
        reynolds = number(row, 'Re_polar', True)
        alpha = number(row, 'alpha_deg')
        group = result.setdefault(key, {}).setdefault(reynolds, {'curve': {}, 'row': row})
        cl = number(row, 'CL')
        if alpha in group['curve'] and abs(group['curve'][alpha] - cl) > 1e-9:
            raise ValueError(f'Celisen LLT satirlari: {key}, Re={reynolds}, alpha={alpha}')
        group['curve'][alpha] = cl
    return result


def wing_at_re(re_map: dict, target: float) -> dict:
    lo, hi, f = bracket(re_map, target, 'Kanat Reynolds')
    a, b = re_map[lo], re_map[hi]
    left = max(min(a['curve']), min(b['curve']))
    right = min(max(a['curve']), max(b['curve']))
    angles = sorted({x for x in set(a['curve']) | set(b['curve']) if left <= x <= right})
    if len(angles) < 2:
        raise ModelError('veri_hatasi', 'Ortak LLT aci araligi yok.')
    curve = {x: lerp(interp_curve(a['curve'], x), interp_curve(b['curve'], x), f) for x in angles}
    def field(name):
        return lerp(number(a['row'], name), number(b['row'], name), f)
    return {'curve': curve, 'e': field('e_oswald'),
            'cl_max': field('CL_max_3d_tahmini'),
            'stall': field('alpha_stall_2d_deg'), 'Re_lo': lo, 'Re_hi': hi}


def linear_slope(points: dict[float, float]) -> float:
    # Mevcut lifting_line_cozucu.py ile ayni -4..6 derece fit araligi.
    pairs = [(math.radians(a), cl) for a, cl in sorted(points.items()) if -4 <= a <= 6]
    if len(pairs) < 3:
        raise ModelError('veri_hatasi', 'Kuyruk polarinda -4..6 derece araliginda en az 3 nokta gerekli.')
    xm = sum(x for x, y in pairs) / len(pairs)
    ym = sum(y for x, y in pairs) / len(pairs)
    slope = sum((x-xm)*(y-ym) for x, y in pairs) / sum((x-xm)**2 for x, y in pairs)
    if slope <= 0:
        raise ModelError('veri_hatasi', 'Kuyruk 2B tasima egimi pozitif olmali.')
    return slope


def solve_linear(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    # Kismi pivotlamali Gauss eliminasyonu; 25x25 LLT icin NumPy gerektirmez.
    n = len(rhs)
    a = [list(row) + [val] for row, val in zip(matrix, rhs)]
    for k in range(n):
        pivot = max(range(k, n), key=lambda j: abs(a[j][k]))
        a[k], a[pivot] = a[pivot], a[k]
        if abs(a[k][k]) < 1e-14:
            raise ModelError('veri_hatasi', 'LLT matrisi tekil.')
        for i in range(k+1, n):
            factor = a[i][k] / a[k][k]
            a[i][k] = 0.0
            for j in range(k+1, n+1):
                a[i][j] -= factor * a[k][j]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (a[i][n] - sum(a[i][j]*x[j] for j in range(i+1, n))) / a[i][i]
    return x


@lru_cache(maxsize=512)
def llt_rectangular(ar: float, a0_rad: float, terms: int = 25) -> tuple[float, float]:
    odd = [2*j+1 for j in range(terms)]
    theta = [(i+1)*math.pi/(2*terms) for i in range(terms)]
    matrix = [[math.sin(t*n)*(4*ar/a0_rad + n/math.sin(t)) for n in odd] for t in theta]
    coef = solve_linear(matrix, [1.0]*terms)
    slope_per_deg = math.pi*ar*coef[0]*math.pi/180
    e = 1/(1+sum(n*(an/coef[0])**2 for n, an in zip(odd[1:], coef[1:])))
    return slope_per_deg, e


def read_tail(path: Path) -> dict[float, dict[float, float]]:
    polars = {}
    if path.is_file() and path.suffix.lower() == '.csv':
        for row in read_csv(path):
            re_val, alpha = number(row, 'Re', True), number(row, 'alpha_deg')
            cl = number(row, 'CL_2d')
            curve = polars.setdefault(re_val, {})
            if alpha in curve and abs(curve[alpha]-cl) > 1e-9:
                raise ValueError(f'Celisen kuyruk verisi: Re={re_val}, alpha={alpha}')
            curve[alpha] = cl
    else:
        files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
        for file in files:
            text = file.read_text(encoding='utf-8-sig', errors='replace')
            for block in re.split(r'(?=Calculated polar for:)', text):
                header = re.search(r'Calculated polar for:\s*([^\r\n]+)', block)
                if not header or 'naca0010' not in re.sub(r'[^a-z0-9]', '', header[1].lower()):
                    continue
                match = re.search(r'\bRe\s*=\s*([\d.]+)\s*[eE]\s*([+-]?\d+)', block)
                if not match:
                    continue
                re_val = float(match[1])*10**int(match[2])
                curve = polars.setdefault(re_val, {})
                started = False
                for line in block.splitlines():
                    if '------' in line:
                        started = True
                        continue
                    parts = line.split()
                    if not started or len(parts) < 7:
                        continue
                    try:
                        alpha, cl = float(parts[0]), float(parts[1])
                    except ValueError:
                        continue
                    if not math.isfinite(alpha) or not math.isfinite(cl):
                        raise ValueError(f'Sonlu olmayan polar verisi: {file}')
                    if alpha in curve and abs(curve[alpha]-cl) > 1e-9:
                        raise ValueError(f'Celisen polar: {file}, Re={re_val}, alpha={alpha}')
                    curve[alpha] = cl
    if not polars:
        raise ValueError(f'NACA0010 polari bulunamadi: {path}')
    return polars


def tail_at_re(polars: dict, target: float, ar: float) -> dict:
    lo, hi, f = bracket(polars, target, 'Kuyruk Reynolds')
    def at(re_val):
        points = polars[re_val]
        slope, e = llt_rectangular(ar, linear_slope(points))
        stall = max(points, key=points.get)
        return {'slope': slope, 'e': e, 'stall': stall,
                'cl_max': min(slope*stall, points[stall]),
                'alpha_min': min(points), 'alpha_max': max(points)}
    a, b = at(lo), at(hi)
    result = {key: lerp(a[key], b[key], f) for key in a}
    result['alpha_min'] = max(a['alpha_min'], b['alpha_min'])
    result['alpha_max'] = min(a['alpha_max'], b['alpha_max'])
    result.update(Re_lo=lo, Re_hi=hi)
    return result


def find_drag(mass: dict, rows: list[dict]) -> dict:
    checks = [('S_w_m2','S_w_m2'), ('S_HT_m2','S_HT_m2'), ('S_VT_m2','S_VT_m2'),
              ('c_root_w_m','c_w_m'), ('c_root_HT_m','c_HT_m'), ('c_root_VT_m','c_VT_m'),
              ('b_HT_m','b_HT_m'), ('h_VT_m','h_VT_m'), ('govde_uzunlugu_m','L_govde_m'),
              ('govde_D_m','D_govde_m'), ('govde_S_toplam_m2','S_wet_govde_m2')]
    matches = []
    for row in rows:
        if all(math.isclose(number(mass,a), number(row,b), rel_tol=1e-4, abs_tol=6e-6) for a,b in checks):
            matches.append(row)
    if not matches:
        raise ModelError('geometri_eslesmedi', 'Ayni kanat/kuyruk/govde geometrisinde OpenVSP satiri yok.')
    # Polar detay satirlari aciya gore tekrar eder; yalniz geometri/parazit girdileri alinir.
    signature = lambda r: tuple(number(r,k) for k in ('t_c_w','t_c_HT','t_c_VT'))
    if len({signature(r) for r in matches}) > 1:
        raise ModelError('veri_hatasi', 'Ayni geometride celisen profil kalinliklari var.')
    return matches[0]


def calc_cd0(mass: dict, drag: dict, velocity: float, rho: float, mu: float) -> dict:
    sw = number(mass, 'S_w_m2', True)
    mach = velocity/340.294
    def cf(re_val):
        if re_val <= 1:
            raise ModelError('veri_hatasi', 'Cf hesabinda Re > 1 gerekli.')
        return 0.455/(math.log10(re_val)**2.58 * (1+0.144*mach**2)**0.65)
    result = {}
    for part, area, chord, thickness, q in [
        ('w','S_w_m2','c_root_w_m','t_c_w',1.0),
        ('HT','S_HT_m2','c_root_HT_m','t_c_HT',1.2),
        ('VT','S_VT_m2','c_root_VT_m','t_c_VT',1.2)]:
        s, c, tc = number(mass,area,True), number(mass,chord,True), number(drag,thickness,True)
        re_val = rho*velocity*c/mu
        wet = 2*s*(1+0.2*tc)
        ff = 1+3.3*tc-0.008*tc**2+27*tc**3 if part=='w' else 1+3.52*tc
        result['Re_'+part] = re_val
        result['CD0_'+part] = cf(re_val)*ff*q*wet/sw
    length = number(mass, 'govde_uzunlugu_m', True)
    diameter = number(mass, 'govde_D_m', True)
    wet = number(mass, 'govde_S_toplam_m2', True)
    slender = length/diameter
    ff = 1+2.2*slender**-1.5-0.9*slender**-3
    result['Re_govde'] = rho*velocity*length/mu
    result['CD0_govde'] = cf(result['Re_govde'])*ff*wet/sw
    result['CD0_toplam'] = sum(result['CD0_'+p] for p in ('w','HT','VT','govde'))
    return result


def operating_point(wing: dict, tail: dict, required: float, area_ratio: float, incidence: float) -> tuple:
    curve = wing['curve']
    low = max(min(curve), tail['alpha_min']-incidence)
    high = min(max(curve), wing['stall'], tail['alpha_max']-incidence, tail['stall']-incidence)
    if high <= low:
        raise ModelError('cl_araligi_disinda', 'Gecerli ortak aci araligi yok.')
    angles = sorted({low, high} | {a for a in curve if low <= a <= high})
    def components(alpha):
        cw = interp_curve(curve, alpha)
        ct = tail['slope']*(alpha+incidence)
        return cw, ct, cw+area_ratio*ct
    for left, right in zip(angles, angles[1:]):
        c0, c1 = components(left)[2], components(right)[2]
        if c1 <= c0 or not c0-1e-10 <= required <= c1+1e-10:
            continue
        alpha = left+(right-left)*(required-c0)/(c1-c0)
        cw, ct, total = components(alpha)
        if cw > wing['cl_max']+1e-9 or ct > tail['cl_max']+1e-9:
            continue
        return alpha, cw, ct, total
    raise ModelError('cl_araligi_disinda', f'Stall/CL sinirlari icinde gerekli CL={required:.6g} saglanamiyor.')


def prop_at_speed(rows: list[dict], prop: str, speed: float, voltage: float, mode: str) -> tuple:
    groups = {}
    for row in rows:
        if row['pervane'].strip() == prop:
            groups.setdefault(number(row,'gaz_yuzde'), []).append(row)
    if not groups:
        raise ModelError('pervane_bulunamadi', prop)
    points, skipped = [], []
    for throttle, group in sorted(groups.items()):
        by_speed = {}
        for row in group:
            v = number(row,'hiz_m_s')
            if v in by_speed and by_speed[v] != row:
                raise ModelError('veri_hatasi','Tekrarlanan pervane/gaz/hiz verisi.')
            by_speed[v] = row
        try:
            lo, hi, f = bracket(by_speed,speed,'Pervane hizi')
        except ModelError:
            skipped.append(throttle)
            continue
        a,b = by_speed[lo],by_speed[hi]
        def power(row):
            if mode=='akim_a':
                return number(row,'akim_a')*voltage, 'akim_a_x_gerilim'
            for name in ('guc_w','power_w'):
                if row.get(name) not in (None,''):
                    return number(row,name), name
            return number(row,'akim_a'), 'akim_a_W_kabul_gecici'
        pa,source = power(a)
        pb,source_b = power(b)
        if source != source_b or min(pa,pb) < 0:
            raise ModelError('veri_hatasi','Tutarsiz/negatif guc verisi.')
        points.append({'gaz':throttle,'itki':lerp(number(a,'itki_n'),number(b,'itki_n'),f),
                       'guc':lerp(pa,pb,f),'kaynak':source,
                       'yontem':'|'.join(sorted({a.get('itki_yontemi',''),b.get('itki_yontemi','')}))})
    return points, skipped


def required_power(points: list[dict], thrust: float) -> dict:
    if not points:
        raise ModelError('veri_araligi_disinda','Ucus hizini kapsayan pervane verisi yok.')
    if any(b['itki'] <= a['itki'] for a,b in zip(points,points[1:])):
        raise ModelError('veri_hatasi','Gaz arttikca itki artmiyor; ters interpolasyon belirsiz.')
    if thrust > points[-1]['itki']+1e-9:
        raise ModelError('itki_yetersiz',f'Gerekli {thrust:.4f} N > mevcut {points[-1]["itki"]:.4f} N')
    if thrust < points[0]['itki']-1e-9:
        raise ModelError('itki_verisi_altinda',f'Gerekli {thrust:.4f} N < tablonun alt siniri {points[0]["itki"]:.4f} N')
    lookup = {p['itki']:p for p in points}
    lo,hi,f = bracket(lookup,thrust,'Itki')
    a,b = lookup[lo],lookup[hi]
    return {'gaz_yuzde':lerp(a['gaz'],b['gaz'],f),'guc_W':lerp(a['guc'],b['guc'],f),
            'gaz_alt':a['gaz'],'gaz_ust':b['gaz'], 'itki_alt_N':lo,'itki_ust_N':hi,
            'guc_kaynagi':'|'.join(sorted({a['kaynak'],b['kaynak']})),
            'itki_veri_yontemi':'|'.join(sorted({a['yontem'],b['yontem']}))}


def calculate(mass_rows, wing_index, drag_rows, tail_polars, prop_rows, args):
    velocity = math.hypot(args.vx,args.vz)
    gamma = math.atan2(args.vz,args.vx)
    duration = args.irtifa/args.vz
    dynamic = 0.5*args.rho*velocity**2
    drag_index = {}
    for row in drag_rows:
        drag_index.setdefault(wing_key(row), []).append(row)
    prop_cache = {}
    output = []
    for i,mass in enumerate(mass_rows,1):
        prop = (mass.get('pervane') or args.pervane).strip()
        out = {'aday_no':i, 'kanat_airfoil':mass.get('kanat_airfoil',''), 'pervane':prop,
               'durum':'', 'aciklama':'', 'Vx_m_s':args.vx,'Vz_m_s':args.vz,'V_m_s':velocity,
               'tirmanis_acisi_deg':math.degrees(gamma),'irtifa_m':args.irtifa,'sure_s':duration,
               'gerilim_V':args.gerilim,'rho_kg_m3':args.rho,'q_Pa':dynamic,
               'kuyruk_oturma_deg':args.kuyruk_oturma,'downwash_deg':0.0,'trim_cozuldu':False}
        try:
            key = wing_key(mass)
            m = number(mass,'total_kutle_kg',True)
            weight = m*args.g
            sw = number(mass,'S_w_m2',True)
            sh = number(mass,'S_HT_m2',True)
            arw,arh = number(mass,'AR_w',True),number(mass,'AR_HT',True)
            out.update(kutle_kg=m,agirlik_N=weight,S_w_m2=sw,S_HT_m2=sh,
                       b_w_m=number(mass,'b_w_m'),AR_w=arw,AR_HT=arh,
                       AR_VT=number(mass,'AR_VT',True),L_govde_m=number(mass,'govde_uzunlugu_m'),
                       govde_aday_no=mass.get('govde_aday_no',''))
            for span,area,ar in [('b_w_m','S_w_m2','AR_w'),('b_HT_m','S_HT_m2','AR_HT'),('h_VT_m','S_VT_m2','AR_VT')]:
                if not math.isclose(number(mass,span)**2/number(mass,area),number(mass,ar),rel_tol=1e-5):
                    raise ModelError('geometri_eslesmedi',f'{ar} boyutlarla uyusmuyor.')
            drag = find_drag(mass,drag_index.get(key,[]))
            cd0 = calc_cd0(mass,drag,velocity,args.rho,args.mu)
            out.update(cd0)
            if key not in wing_index:
                raise ModelError('llt_bulunamadi',str(key))
            wing = wing_at_re(wing_index[key],out['Re_w'])
            tail = tail_at_re(tail_polars,out['Re_HT'],arh)
            required = weight*math.cos(gamma)/(dynamic*sw)
            out.update(CL_gerekli=required,e_w=wing['e'],e_HT=tail['e'],
                       CL_alpha_HT_per_deg=tail['slope'],CL_max_w=wing['cl_max'],CL_max_HT=tail['cl_max'],
                       Re_w_alt=wing['Re_lo'],Re_w_ust=wing['Re_hi'],
                       Re_HT_alt=tail['Re_lo'],Re_HT_ust=tail['Re_hi'])
            alpha,cw,ct,total = operating_point(wing,tail,required,sh/sw,args.kuyruk_oturma)
            cdiw = cw*cw/(math.pi*arw*wing['e'])
            cdih_surface = ct*ct/(math.pi*arh*tail['e'])
            cdih = (sh/sw)*cdih_surface
            cd = out['CD0_toplam']+cdiw+cdih
            drag_force = dynamic*sw*cd
            climb_force = weight*math.sin(gamma)
            thrust = drag_force+climb_force
            out.update(alpha_deg=alpha,alpha_HT_deg=alpha+args.kuyruk_oturma,
                       CL_w=cw,CL_HT=ct,CL_ucak=total,CDi_w=cdiw,
                       CDi_HT_kendi_alani=cdih_surface,CDi_HT_kanat_ref=cdih,
                       CDi_VT=0.0,CDi_toplam=cdiw+cdih,CD_toplam=cd,
                       L_w_N=dynamic*sw*cw,L_HT_N=dynamic*sh*ct,L_toplam_N=dynamic*sw*total,
                       D_N=drag_force,tirmanis_ek_itki_N=climb_force,T_gerekli_N=thrust,
                       L_denge_hatasi_N=dynamic*sw*total-weight*math.cos(gamma),
                       T_denge_hatasi_N=thrust-drag_force-climb_force)
            if prop not in prop_cache:
                prop_cache[prop] = prop_at_speed(prop_rows,prop,velocity,args.gerilim,args.tablo_degeri)
            points,skipped = prop_cache[prop]
            out['hiz_kapsami_disindaki_gazlar'] = '|'.join(f'{g:g}' for g in skipped)
            if points:
                out['mevcut_min_itki_N'],out['mevcut_max_itki_N'] = points[0]['itki'],points[-1]['itki']
            out.update(required_power(points,thrust))
            power = out['guc_W']
            out.update(enerji_J=power*duration,enerji_Wh=power*duration/3600,
                       esdeger_akim_A=power/args.gerilim,
                       tuketim_mAh=1000*power*duration/(3600*args.gerilim),durum='uygun')
            if 'gecici' in out['guc_kaynagi']:
                out['aciklama'] = 'akim_a sayilari W kabul edildi; guc ve tuketim gecicidir.'
        except ModelError as exc:
            out['durum'],out['aciklama'] = exc.status,str(exc)
        output.append(out)
    return output


def resolve_input(explicit, names):
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise ValueError(f'Girdi bulunamadi: {path}')
        return path
    directories = [ROOT/'veriler',ROOT,ROOT.parent,ROOT/'Scripts',ROOT.parent/'Scripts']
    for directory in directories:
        for name in names:
            path = directory/name
            if path.exists():
                return path
    raise ValueError(f'Girdi bulunamadi: {names}. Dosya yolunu komut satirinda belirtin.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ('kutle','llt','drag','kuyruk-polar','itki'):
        parser.add_argument('--'+flag,type=Path)
    parser.add_argument('--cikti',type=Path,default=ROOT/'tirmanis_sonuclari.csv')
    for flag,default in [('vx',15.0),('vz',4.0),('irtifa',50.0),('gerilim',24.0),
                         ('rho',1.175),('mu',1.789e-5),('g',9.80665),('kuyruk-oturma',0.0)]:
        parser.add_argument('--'+flag,type=float,default=default)
    parser.add_argument('--pervane',default='14x7',help='Adayda pervane sutunu yoksa kullanilir.')
    parser.add_argument('--tablo-degeri',choices=['guc_w','akim_a'],default='akim_a',
                        help='guc_w: guc_w/power_w veya gecici olarak akim_a W kabul edilir. akim_a: gercek A * V.')
    args = parser.parse_args(argv)
    for name in ('vx','vz','irtifa','gerilim','rho','mu','g'):
        val = getattr(args,name)
        if not math.isfinite(val) or val <= 0:
            parser.error(f'{name} sonlu ve pozitif olmali.')
    if not math.isfinite(args.kuyruk_oturma):
        parser.error('kuyruk-oturma sonlu olmali.')
    try:
        masses = read_csv(resolve_input(args.kutle,['kanat_kuyruk_govde_kutleli.csv']))
        wings = index_wing(read_csv(resolve_input(args.llt,['llt_flapsiz_tum_sonuclar.csv'])))
        drag = read_csv(resolve_input(args.drag,['openvsp_llt_flapsiz_aday_ozet.csv','openvsp_llt_flapsiz_polar_detay.csv']))
        tail = read_tail(resolve_input(args.kuyruk_polar,['naca0010_polarlari.csv','airfoils_polar']))
        props = read_csv(resolve_input(args.itki,['itki_tablosu.csv']))
        output = calculate(masses,wings,drag,tail,props,args)
        fields = list(dict.fromkeys(key for row in output for key in row))
        args.cikti.parent.mkdir(parents=True,exist_ok=True)
        with args.cikti.open('w',encoding='utf-8-sig',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            writer.writerows(output)
        counts = {}
        for row in output:
            counts[row['durum']] = counts.get(row['durum'],0)+1
        print(f'{len(output)} aday yazildi: {args.cikti.resolve()}')
        print('Durumlar: '+', '.join(f'{k}={v}' for k,v in counts.items()))
        return 0
    except (OSError,ValueError,KeyError) as exc:
        print(f'HATA: {exc}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())