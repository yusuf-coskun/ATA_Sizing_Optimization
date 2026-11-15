"""Aday-Re LLT CSV okuyucusu. Katsayilari yeniden cozmez/duzeltmez.

Tek Re kosulundaki CSV baska ucus hizinda kullanilirsa sabit katsayi
varsayimi acikca raporlanir. strict_re=True bu kullanimi reddeder.
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


class LLTTable:
    def __init__(self, rows, source='LLT CSV', flapped=False):
        self.source, self.flapped, self.errors = str(source), flapped, []
        groups = {}
        for row in rows:
            key = (profile(row),) + tuple(round(num(row,k),9) for k in GEOMETRY)
            if not key[0]:
                raise LLTError('LLT CSV: profil adi eksik.')
            if flapped:
                key += (str(row.get('flap_airfoil','')).strip().lower(), round(num(row,'b_flap_m'),9))
            constants = {k:num(row,k) for k in CONSTANTS}
            for k in ('Re_aday','e_oswald','CL_alpha_3d_per_deg','CL_max_3d_tahmini',
                      'Re_HT','CL_alpha_HT_per_deg','e_HT','CL_max_HT','Re_VT','CL_alpha_VT_per_deg','e_VT'):
                if constants[k] <= 0:
                    raise LLTError(f'LLT CSV: {k} pozitif olmali.')
            if key not in groups:
                groups[key] = {'row':dict(row), 'values':constants, 'curve':{}}
            group = groups[key]
            for field,value in constants.items():
                if not math.isclose(value, group['values'][field], rel_tol=1e-9, abs_tol=1e-10):
                    raise LLTError(f'LLT CSV: ayni geometride celisen {field}; aday-Re dosyasi tek kosul icermeli.')
            angle, cl = num(row,'alpha_deg'), num(row,'CL')
            if angle in group['curve'] and not math.isclose(cl,group['curve'][angle],rel_tol=1e-9,abs_tol=1e-10):
                raise LLTError(f'LLT CSV: ayni geometri/acida celisen CL: {key[0]}, {angle}')
            group['curve'][angle] = cl
        if not groups:
            raise LLTError('LLT CSV bos.')
        self.groups = list(groups.values())
        self.by_profile = {}
        for group in self.groups:
            if len(group['curve']) < 2:
                raise LLTError('LLT CSV: en az iki farkli alpha satiri gerekli.')
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
        group=self.match(candidate); row=group['row']; v=group['values']
        target={part:rho*velocity*num(candidate,chord)/mu for part,chord in
                [('w','c_root_w_m'),('HT','c_root_HT_m'),('VT','c_root_VT_m')]}
        refs={'w':v['Re_aday'],'HT':v['Re_HT'],'VT':v['Re_VT']}
        aligned=all(math.isclose(target[p],refs[p],rel_tol=1e-3,abs_tol=1.) for p in refs)
        if strict_re and not aligned:
            raise LLTError(f'LLT Re uyusmuyor: CSV kanat Re={refs["w"]:.1f}, ucus Re={target["w"]:.1f}. Ucus kosulunda CSV uretin.')
        wing=dict(curve=group['curve'], e=v['e_oswald'], cl_max=v['CL_max_3d_tahmini'],
                  stall=v['alpha_stall_2d_deg'], Re_lo=refs['w'],Re_hi=refs['w'],
                  slope=v['CL_alpha_3d_per_deg'],alpha_L0=v['alpha_L0_deg'])
        # Kuyruk aci siniri CSV stall degeridir; negatif sinir NACA0010 simetrisi varsayimidir.
        tail=dict(e=v['e_HT'],slope=v['CL_alpha_HT_per_deg'],cl_max=v['CL_max_HT'],
                  stall=v['alpha_stall_HT_deg'],alpha_min=-abs(v['alpha_stall_HT_deg']),alpha_max=v['alpha_stall_HT_deg'],
                  Re_lo=refs['HT'],Re_hi=refs['HT'])
        meta={'llt_veri_kaynagi':self.source,'llt_katsayi_modu':'csv_sabit_katsayi',
              'llt_Re_uyumlu':aligned,'llt_sabit_Re_varsayimi':not aligned,
              'llt_model_notu':'' if aligned else 'CSV katsayilari kaynak Re kosulundan sabit alindi; Reynolds enterpolasyonu yapilmadi.'}
        for p in refs:
            meta['llt_Re_'+p+'_kaynak']=refs[p]; meta['llt_Re_'+p+'_ucus']=target[p]
        return wing,tail,meta,row
