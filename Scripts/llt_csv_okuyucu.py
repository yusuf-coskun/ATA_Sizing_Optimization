"""Geometri/Re/alpha tabanli LLT CSV okuyucusu. Alt sinir kabulleri raporlanir.
Flapli kanat Re<150000 ise 150000 satiri kullanilir; flapsiz alt sinir reddedilir.
strict_re=True alt sinir kabullerini reddeder; aralik ici interpolasyona izin verir.
HT/VT alt sinirlarinin altinda en dusuk mevcut katsayilar kullanilir ve raporlanir.
Polar üst sınırı dışı kuyruk noktaları yüklenir, ancak kullanımda LLTError verir.
"""
from __future__ import annotations
import csv
import math
import re
from pathlib import Path


class LLTError(ValueError):
    status = 'llt_csv_hatasi'


def num(row, key):
    try:
        value = float(row[key])
    except (ValueError, TypeError, KeyError) as exc:
        raise LLTError(f'LLT CSV: eksik/gecersiz {key}') from exc
    if not math.isfinite(value):
        raise LLTError(f'LLT CSV: sonlu olmayan {key}')
    return value


def profile(row):
    return Path(str(row.get('kanat_airfoil') or row.get('profil') or '')).stem.lower().strip()


GEOMETRY = {'b_w_m':'b_w_m', 'AR_w':'AR_w', 'S_w_m2':'S_w_m2',
            'c_w_m':'c_root_w_m', 'c_HT_m':'c_root_HT_m', 'S_HT_m2':'S_HT_m2',
            'c_VT_m':'c_root_VT_m', 'S_VT_m2':'S_VT_m2'}
CONSTANTS = ('Re_aday','e_oswald','CL_alpha_3d_per_deg','alpha_L0_deg',
             'alpha_stall_2d_deg','CL_max_3d_tahmini','Re_HT','CL_alpha_HT_per_deg',
             'e_HT','alpha_stall_HT_deg','CL_max_HT','Re_VT','CL_alpha_VT_per_deg','e_VT')


TAIL_COEFFICIENTS = {
    'HT': ('CL_alpha_HT_per_deg', 'e_HT', 'alpha_stall_HT_deg', 'CL_max_HT'),
    'VT': ('CL_alpha_VT_per_deg', 'e_VT'),
}


def tail_unavailable(row):
    """Yalnız üreticinin açıkça kapsam dışı işaretlediği boşluklara izin ver."""
    missing = set()
    reasons = []
    for part, fields in TAIL_COEFFICIENTS.items():
        if row.get('Re_'+part+'_durumu') != 'polar_araligi_disinda':
            continue
        if any(row.get(field) not in (None, '') for field in fields):
            raise LLTError(f'LLT CSV: {part} kapsam disi durumuyla katsayilar celisiyor.')
        missing.update(fields)
        reasons.append(f'{part} Re={num(row, "Re_"+part):g}: polar_araligi_disinda')
    return missing, '; '.join(reasons)


def same_value(left, right):
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-10)


class LLTTable:
    def __init__(self, rows, source='LLT CSV', flapped=False):
        self.source, self.flapped, self.errors = str(source), flapped, []
        groups = {}
        for row in rows:
            if Path(self.source).name.startswith('vlm_') and row.get('aero_modeli') != 'vlm_tek_egri_v1':
                raise LLTError('Eski VLM LLT tablosu: tek egri modeliyle yeniden uretin.')
            key = (profile(row),) + tuple(round(num(row,k),9) for k in GEOMETRY)
            if not key[0]: raise LLTError('LLT CSV: profil adi eksik.')
            if flapped:
                key += (str(row.get('flap_airfoil','')).strip().lower(),round(num(row,'b_flap_m'),9))
            missing, unavailable = tail_unavailable(row)
            v = {k:None if k in missing else num(row,k) for k in CONSTANTS}
            for k in CONSTANTS:
                if k not in ('alpha_L0_deg','alpha_stall_2d_deg','alpha_stall_HT_deg') and v[k] is not None and v[k]<=0:
                    raise LLTError(f'LLT CSV: {k} pozitif olmali.')
            used = num(row,'Re_VT_kullanilan') if row.get('Re_VT_kullanilan') not in (None,'') else v['Re_VT']
            if used<=0: raise LLTError('Gecersiz Re_VT_kullanilan.')
            v['Re_VT_kullanilan']=used
            ht_used = num(row,'Re_HT_kullanilan') if row.get('Re_HT_kullanilan') not in (None,'') else v['Re_HT']
            if ht_used<=0: raise LLTError('Gecersiz Re_HT_kullanilan.')
            v['Re_HT_kullanilan']=ht_used
            # Her satir ayni hizdaki kanat ve kuyruklari temsil etmeli.
            for part in ('HT','VT'):
                if not math.isclose(v['Re_'+part]/v['Re_aday'],num(row,'c_'+part+'_m')/num(row,'c_w_m'),rel_tol=1e-5):
                    raise LLTError('Kanat/kuyruk Reynolds ve kord oranlari uyusmuyor.')
            group=groups.setdefault(key,{'row':dict(row),'nodes':{}})
            node=group['nodes'].setdefault(v['Re_aday'],{'row':dict(row),'values':v,'curve':{},'unavailable':unavailable})
            if node['unavailable'] != unavailable:
                raise LLTError('Ayni geometri/Re icin celisen kuyruk kapsam durumu.')
            for k,value in v.items():
                if not same_value(value,node['values'][k]):
                    raise LLTError(f'Ayni geometri/Re icin celisen {k}')
            angle,cl=num(row,'alpha_deg'),num(row,'CL')
            if cl > v['CL_max_3d_tahmini'] + 2e-5:
                raise LLTError('LLT CSV: CL, ayni satirin CLmax degerini asiyor.')
            if row.get('aero_modeli') == 'vlm_tek_egri_v1':
                if any(row.get(k) not in (None, '') for k in ('CL_flapsiz_kisim','CL_flapli_kisim','CL_kanat')):
                    raise LLTError('LLT CSV: VLM modelinde eski flap bileseni CL alanlari bulunamaz.')
                for k, ana in (('CL_alpha_3d_kanat_per_deg','CL_alpha_3d_per_deg'),
                               ('e_oswald_kanat','e_oswald')):
                    if row.get(k) not in (None, '') and not math.isclose(num(row,k),v[ana],abs_tol=2e-6):
                        raise LLTError(f'LLT CSV: {k} ana kanat modeliyle uyusmuyor.')
                expected_stall = v['alpha_L0_deg'] + v['CL_max_3d_tahmini'] / v['CL_alpha_3d_per_deg']
                if not math.isclose(v['alpha_stall_2d_deg'], expected_stall, abs_tol=2e-3):
                    raise LLTError('LLT CSV: stall acisi, egim/alpha_L0/CLmax ile uyusmuyor.')
                if angle <= min(2.0, expected_stall) and angle >= -2.0:
                    expected_cl = min(v['CL_alpha_3d_per_deg'] * (angle - v['alpha_L0_deg']),
                                      v['CL_max_3d_tahmini'])
                    if not math.isclose(cl, expected_cl, abs_tol=2e-4):
                        raise LLTError('LLT CSV: CL egrisi dogrusal bolgede egim/alpha_L0 ile uyusmuyor.')
                if angle >= expected_stall and not math.isclose(cl, v['CL_max_3d_tahmini'], abs_tol=2e-4):
                    raise LLTError('LLT CSV: stall sonrasi CL, CLmax ile uyusmuyor.')
            if angle in node['curve'] and not math.isclose(cl,node['curve'][angle],rel_tol=1e-9,abs_tol=1e-10):
                raise LLTError('Ayni geometri/Re/alpha icin celisen CL.')
            node['curve'][angle]=cl
        if not groups: raise LLTError('LLT CSV bos.')
        self.groups=list(groups.values()); self.by_profile={}
        for group in self.groups:
            group['vt_nodes']={}
            group['ht_nodes']={}
            for node in group['nodes'].values():
                if len(node['curve'])<2: raise LLTError('En az iki alpha noktasi gerekli.')
                if node['row'].get('aero_modeli') == 'vlm_tek_egri_v1':
                    stall = node['values']['alpha_stall_2d_deg']
                    points = sorted((a, c) for a, c in node['curve'].items() if a <= stall)
                    if any(c2 < c1 - 2e-5 for (_, c1), (_, c2) in zip(points, points[1:])):
                        raise LLTError('LLT CSV: VLM CL egrisi stall oncesi azaliyor.')
                if node['unavailable']:
                    self.errors.append({
                        'profil':profile(node['row']),
                        'geometri':{k:node['row'][k] for k in GEOMETRY},
                        'Re_aday':node['values']['Re_aday'],
                        'aciklama':node['unavailable'],
                    })
                v=node['values']
                if all(v[k] is not None for k in TAIL_COEFFICIENTS['HT']):
                    hr=v['Re_HT_kullanilan']
                    hvalues=tuple(v[k] for k in TAIL_COEFFICIENTS['HT'])
                    previous=group['ht_nodes'].setdefault(hr,hvalues)
                    if any(not same_value(x,y) for x,y in zip(previous,hvalues)):
                        raise LLTError('Ayni kullanilan HT Reynolds icin celisen katsayi.')
                if v['CL_alpha_VT_per_deg'] is None or v['e_VT'] is None:
                    continue
                r=v['Re_VT_kullanilan']
                pair=(v['CL_alpha_VT_per_deg'],v['e_VT'])
                old=group['vt_nodes'].setdefault(r,pair)
                if any(not math.isclose(x,y,rel_tol=1e-9,abs_tol=1e-10) for x,y in zip(old,pair)):
                    raise LLTError('Ayni kullanilan VT Reynolds icin celisen katsayi.')
            self.by_profile.setdefault(profile(group['row']),[]).append(group)

    @classmethod
    def read(cls, path, flapped=False):
        with Path(path).open(encoding='utf-8-sig',newline='') as stream:
            first=stream.readline(); stream.seek(0)
            return cls(list(csv.DictReader(stream, delimiter=';' if first.count(';')>first.count(',') else ',')),path,flapped)

    def match(self, candidate):
        matches=[]
        for group in self.by_profile.get(profile(candidate),[]):
            row=group['row']
            if not all(math.isclose(num(row,a),num(candidate,b),rel_tol=1e-5,abs_tol=6e-6) for a,b in GEOMETRY.items()):
                continue
            if self.flapped:
                width=float(candidate.get('b_flap_m') or min(1.,num(candidate,'b_w_m')))
                if not math.isclose(num(row,'b_flap_m'),width,rel_tol=1e-5,abs_tol=6e-6):
                    continue
                expected=str(candidate.get('flap_airfoil') or profile(candidate)+'_flap').lower()
                if str(row['flap_airfoil']).strip().lower()!=expected:
                    continue
            matches.append(group)
        if len(matches)!=1:
            raise LLTError(f'{self.source}: {profile(candidate)} icin ayni kanat/kuyruk/flap geometrisinde tek kayit bekleniyor; bulunan={len(matches)}')
        group=matches[0]
        for area,chord,ar in [('S_HT_m2','c_HT_m','AR_HT'),('S_VT_m2','c_VT_m','AR_VT')]:
            if not math.isclose(num(group['row'],area)/num(group['row'],chord)**2,num(candidate,ar),rel_tol=1e-4,abs_tol=6e-6):
                raise LLTError(f'LLT CSV: dikdortgen kuyruk {ar} uyusmuyor.')
        tail_profile=re.sub('[^a-z0-9]','',str(candidate.get('kuyruk_airfoil','')).lower())
        if tail_profile!='naca0010':
            raise LLTError('CSV kuyruk profilini belirtmiyor; mevcut uretici NACA0010. Farkli kuyruk kabul edilmedi.')
        return group

    def values_for(self, candidate, velocity, rho, mu, strict_re=False):
        if any(not math.isfinite(x) or x<=0 for x in (velocity,rho,mu)):
            raise LLTError('Hiz, yogunluk ve viskozite pozitif/sonlu olmali.')
        group=self.match(candidate)
        target={p:rho*velocity*num(candidate,c)/mu for p,c in [('w','c_root_w_m'),('HT','c_root_HT_m'),('VT','c_root_VT_m')]}
        lo,hi,f,used=_bounds(group['nodes'],target['w'],allow_lower=(self.flapped and math.isclose(min(group['nodes']),150000.0,abs_tol=1e-6)))
        a,b=group['nodes'][lo],group['nodes'][hi]
        # Eksik düğümleri atlayıp daha uzak Re noktaları arasında interpolasyon yapma.
        for re_node in dict.fromkeys((lo, hi)):
            node = group['nodes'][re_node]
            if node['unavailable']:
                raise LLTError(
                    f'{self.source}: {profile(candidate)}, kanat Re={re_node:g}; '
                    f'kuyruk katsayilari kullanilamiyor: {node["unavailable"]}')
        v={k:a['values'][k]+f*(b['values'][k]-a['values'][k]) for k in a['values']}
        left=max(min(a['curve']),min(b['curve'])); right=min(max(a['curve']),max(b['curve']))
        angles=sorted({x for x in set(a['curve'])|set(b['curve']) if left<=x<=right})
        if len(angles)<2: raise LLTError('Re komsularinda ortak alpha araligi yok.')
        curve={x:_at(a['curve'],x)+f*(_at(b['curve'],x)-_at(a['curve'],x)) for x in angles}
        hl,hh,hf,hu=_bounds(group['ht_nodes'],target['HT'],allow_lower=True)
        ha,hb=group['ht_nodes'][hl],group['ht_nodes'][hh]
        for k,x,y in zip(TAIL_COEFFICIENTS['HT'],ha,hb):
            v[k]=x+hf*(y-x)
        v['Re_HT_kullanilan']=hu
        ht_lower=not math.isclose(hu,target['HT'],rel_tol=1e-9,abs_tol=1e-6)
        vl,vh,vf,vu=_bounds(group['vt_nodes'],target['VT'],allow_lower=True)
        va,vb=group['vt_nodes'][vl],group['vt_nodes'][vh]
        v['CL_alpha_VT_per_deg']=va[0]+vf*(vb[0]-va[0]);v['e_VT']=va[1]+vf*(vb[1]-va[1])
        v['Re_VT_kullanilan']=vu
        lower=not math.isclose(used,target['w'],rel_tol=1e-9,abs_tol=1e-6)
        vt_lower=not math.isclose(vu,target['VT'],rel_tol=1e-9,abs_tol=1e-6)
        if strict_re and (lower or ht_lower or vt_lower):
            raise LLTError('Alt sinir katsayisi gerekiyor; --llt-re-kontrol bu varsayimi reddeder.')
        refs={'w':used,'HT':hu,'VT':vu}
        meta={'llt_veri_kaynagi':self.source,'llt_katsayi_modu':'csv_alt_sinir_kabul' if (lower or ht_lower or vt_lower) else ('csv_re_interpolasyon' if lo!=hi else 'csv_tam_re'),
              'llt_Re_uyumlu':not (lower or ht_lower or vt_lower),'llt_Re_interpolasyon':lo!=hi,
              'llt_sabit_Re_varsayimi':lower or ht_lower or vt_lower,'llt_kanat_alt_sinir_varsayimi':lower,
              'llt_VT_alt_sinir_varsayimi':vt_lower,'llt_HT_alt_sinir_varsayimi':ht_lower,
              'llt_Re_w_alt':lo,'llt_Re_w_ust':hi,'llt_Re_agirlik':f,
              'llt_Re_HT_alt':hl,'llt_Re_HT_ust':hh,
              'llt_Re_VT_alt':vl,'llt_Re_VT_ust':vh,
              'llt_model_notu':'; '.join(s for s in [
                 f'Kanat Re={target["w"]:.1f}, alt sinir {used:.1f} satiri kullanildi.' if lower else '',
                 f'HT Re={target["HT"]:.1f}, alt sinir {hu:.1f} katsayilari kullanildi.' if ht_lower else '',
                 f'VT Re={target["VT"]:.1f}, alt sinir {vu:.1f} katsayilari kullanildi.' if vt_lower else ''] if s)}
        for p in refs:
            meta['llt_Re_'+p+'_kaynak']=refs[p];meta['llt_Re_'+p+'_kullanilan']=refs[p];meta['llt_Re_'+p+'_ucus']=target[p]
        wing=dict(curve=curve,e=v['e_oswald'],cl_max=v['CL_max_3d_tahmini'],stall=v['alpha_stall_2d_deg'],
                  Re_lo=lo,Re_hi=hi,slope=v['CL_alpha_3d_per_deg'],alpha_L0=v['alpha_L0_deg'])
        tail=dict(e=v['e_HT'],slope=v['CL_alpha_HT_per_deg'],cl_max=v['CL_max_HT'],stall=v['alpha_stall_HT_deg'],
                  alpha_min=-abs(v['alpha_stall_HT_deg']),alpha_max=v['alpha_stall_HT_deg'],
                  Re_lo=hl,Re_hi=hh)
        record={**a['row'],**v};record['Re_aday']=used
        record['HT_alt_sinir_varsayimi']=ht_lower
        record['Re_HT_durumu']=('alt_sinir_sabit_katsayi' if ht_lower else
                                'tam_eslesme' if hl==hh else f'ara deger {hl:g}-{hh:g}')
        return wing,tail,meta,record


def _bounds(values, target, allow_lower=False):
    xs=sorted(values)
    for x in xs:
        if math.isclose(x,target,rel_tol=1e-9,abs_tol=1e-7): return x,x,0.0,x
    if target<xs[0] and allow_lower: return xs[0],xs[0],0.0,xs[0]
    if target<xs[0] or target>xs[-1]:
        raise LLTError(f'Re/alpha hedefi {target:.6g}, veri araligi disinda [{xs[0]:.6g}, {xs[-1]:.6g}].')
    for lo,hi in zip(xs,xs[1:]):
        if lo<target<hi: return lo,hi,(target-lo)/(hi-lo),target
    raise LLTError('Re/alpha araligi bulunamadi.')


def _at(curve, alpha):
    lo,hi,f,_=_bounds(curve,alpha)
    return curve[lo]+f*(curve[hi]-curve[lo])


def cl_at_alpha(curve, alpha):
    """Tüm fazlar için aynı LLT CL eğrisinde doğrusal ara değerleme."""
    return _at(curve, alpha)
