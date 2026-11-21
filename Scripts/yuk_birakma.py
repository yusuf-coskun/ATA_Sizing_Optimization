"""Sabit hizli dalis, esit debili su birakma ve susuz tirmanis hesabi.

Python 3.10+, standart kutuphane. python yuk_birakma_hesabi.py
Mevcut aday_pervane, tirmanis_hesabi ve llt_csv_okuyucu modullerini kullanir.
Eksik pervane verisinde toplam mAh bos kalir; bilinen kisim ayri raporlanir.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from aday_pervane import GERILIM_V, csv_yaz, kimlik_alanlari, performans_adaylarini_hazirla
from llt_csv_okuyucu import LLTTable, LLTError
from tirmanis_hesabi import (ModelError, number, read_csv, wing_key, find_drag,
                            calc_cd0, operating_point, prop_at_speed, required_power)

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Ayarlar:
    seyir_irtifasi: float = 50.0
    birakma_irtifasi: float = 5.0
    seyir_hizi: float = 20.0
    vx: float = 15.0
    vz: float = 4.0
    birakma_suresi: float = 4.0
    dt: float = 0.05
    gerilim: float = GERILIM_V
    rho: float = 1.175
    mu: float = 1.789e-5
    g: float = 9.80665
    kuyruk_oturma: float = 0.0
    yardimci_guc: float = 0.0
    dalis_motor_kapali: bool = True
    dalis_profili: str = 'sabit'
    alt_gaz_modeli: str = 'sifir-noktasi'
    llt_re_kontrol: bool = False
    tablo_degeri: str = 'auto'

    def validate(self):
        for key in ('seyir_irtifasi', 'seyir_hizi', 'vx', 'vz', 'birakma_suresi',
                    'dt', 'gerilim', 'rho', 'mu', 'g'):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} sonlu ve pozitif olmali.')
        for key in ('birakma_irtifasi', 'yardimci_guc'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f'{key} sonlu ve negatif olmayan bir sayi olmali.')
        if self.seyir_irtifasi <= self.birakma_irtifasi:
            raise ValueError('Seyir irtifasi birakma irtifasindan yuksek olmali.')
        if not math.isfinite(self.kuyruk_oturma):
            raise ValueError('Kuyruk oturma acisi sonlu olmali.')
        if self.alt_gaz_modeli not in ('veri', 'sifir-noktasi'):
            raise ValueError('Bilinmeyen alt gaz modeli.')
        if self.dalis_profili not in ('sabit','yumusak'):
            raise ValueError('Bilinmeyen dalis profili.')

    @property
    def egimli_sure(self):
        return (self.seyir_irtifasi-self.birakma_irtifasi)/self.vz


def kutleler(candidate):
    loaded = number(candidate, 'total_kutle_kg', True)
    dry = number(candidate, 'total_susuz_kutle_kg', True)
    water = loaded-dry
    if water < -1e-8:
        raise ModelError('kutle_hatasi', 'Susuz kutle dolu kutleden buyuk.')
    water = max(0.0, water)
    if candidate.get('govde_su_g') not in (None, ''):
        declared = number(candidate, 'govde_su_g')/1000.0
        if declared < 0 or not math.isclose(water, declared, rel_tol=1e-6, abs_tol=1e-6):
            raise ModelError('kutle_hatasi', 'Dolu-susuz kutle farki govde_su_g ile uyusmuyor.')
    return loaded, dry, water


def yumusak_dalis(t, duration, height):
    """Besinci derece irtifa profili; iki ucta Vz=az=0, ortalama Vz=-height/duration."""
    s = min(1.0,max(0.0,t/duration))
    drop = height*(10*s**3-15*s**4+6*s**5)
    vz = -height/duration*30*s*s*(1-s)**2
    az = -height/duration**2*60*s*(1-s)*(1-2*s)
    return drop, vz, az


class AdayModeli:
    def __init__(self, candidate, llt, drag_rows, props, settings, prop_cache):
        self.candidate, self.llt, self.settings = candidate, llt, settings
        self.props, self.prop_cache = props, prop_cache
        self.loaded, self.dry, self.water = kutleler(candidate)
        self.drag = find_drag(candidate, drag_rows)
        self.sw = number(candidate, 'S_w_m2', True)
        self.sh = number(candidate, 'S_HT_m2', True)
        self.arw = number(candidate, 'AR_w', True)
        self.arh = number(candidate, 'AR_HT', True)
        for span, area, ar in [('b_w_m','S_w_m2','AR_w'), ('b_HT_m','S_HT_m2','AR_HT'),
                               ('h_VT_m','S_VT_m2','AR_VT')]:
            if not math.isclose(number(candidate,span)**2/number(candidate,area,True),
                                number(candidate,ar,True), rel_tol=1e-5):
                raise ModelError('geometri_hatasi', f'{ar} boyutlarla uyusmuyor.')
        self.aero_cache = {}

    def point(self, mass, vx, vz, phase, az=0.0):
        a, c = self.settings, self.candidate
        velocity, gamma = math.hypot(vx,vz), math.atan2(vz,vx)
        q = .5*a.rho*velocity**2
        out = dict(kutle_kg=mass, kalan_su_kg=max(0.0,mass-self.dry),
                   Vx_m_s=vx, Vz_m_s=vz, V_m_s=velocity, gamma_deg=math.degrees(gamma),
                   az_m_s2=az, ax_m_s2=0.0,
                   gerilim_V=a.gerilim, q_Pa=q, durum='veri_eksik', aciklama='',
                   guc_W=None, gaz_yuzde=None, tahmini_guc=False,
                   dalis_frenleme_kosullu=False, ek_surukleme_gerekli_N=0.0)
        try:
            if velocity not in self.aero_cache:
                wing, tail, meta, _ = self.llt.values_for(c,velocity,a.rho,a.mu,a.llt_re_kontrol)
                cd0 = calc_cd0(c,self.drag,velocity,a.rho,a.mu)
                self.aero_cache[velocity] = wing, tail, meta, cd0
            wing, tail, meta, cd0 = self.aero_cache[velocity]
            out.update(meta)
            out.update(cd0)
            weight = mass*a.g
            # Yatay hiz sabit, az isaretli (yukari pozitif). Itki ucus yolu yonunde.
            normal_force = mass*(a.g+az)*math.cos(gamma)
            needed_cl = normal_force/(q*self.sw)
            alpha, cw, ct, cl = operating_point(wing,tail,needed_cl,self.sh/self.sw,a.kuyruk_oturma)
            cdi_w = cw*cw/(math.pi*self.arw*wing['e'])
            cdi_h = self.sh/self.sw*ct*ct/(math.pi*self.arh*tail['e'])
            cd = cd0['CD0_toplam']+cdi_w+cdi_h
            drag = q*self.sw*cd
            along_weight = weight*math.sin(gamma)
            acceleration_thrust = mass*az*math.sin(gamma)
            thrust = drag+along_weight+acceleration_thrust
            out.update(alpha_deg=alpha, alpha_HT_deg=alpha+a.kuyruk_oturma,
                       govde_yunuslama_deg=alpha+math.degrees(gamma),
                       CL_gerekli=needed_cl, CL_w=cw, CL_HT=ct, CL_ucak=cl,
                       e_w=wing['e'], e_HT=tail['e'], CDi_w=cdi_w, CDi_HT_kanat_ref=cdi_h,
                       CD_toplam=cd, D_N=drag, T_gerekli_N=thrust,
                       agirlik_yol_bileseni_N=along_weight, ivme_itki_bileseni_N=acceleration_thrust,
                       L_toplam_N=q*self.sw*cl, yuk_faktoru=q*self.sw*cl/weight,
                       L_denge_hatasi_N=q*self.sw*cl-normal_force,
                       ek_surukleme_gerekli_N=max(0.0,-thrust))
            # T<0: hava araci / pervane ek frenleme olmadan bu hiz profilini koruyamaz.
            # Bu secenek yalniz kosullu enerji senaryosudur; frenleme kaniti degildir.
            if phase == 'dalis' and a.dalis_motor_kapali and thrust <= 0:
                out.update(gaz_yuzde=0.0, guc_W=0.0, dalis_frenleme_kosullu=True,
                           durum='motor_kapali_dalis_kosullu', guc_kaynagi='acik_motor_kapali_varsayimi',
                           aciklama='Gerekli ek suruklemenin saglandigi varsayildi; pervane frenleme verisi yok.')
            else:
                cache_key = (c['pervane'], velocity, a.gerilim, a.tablo_degeri)
                if cache_key not in self.prop_cache:
                    self.prop_cache[cache_key] = prop_at_speed(self.props,c['pervane'],velocity,a.gerilim,a.tablo_degeri)
                points, skipped = self.prop_cache[cache_key]
                out['hiz_kapsami_disindaki_gazlar'] = '|'.join(f'{v:g}' for v in skipped)
                if points:
                    out.update(mevcut_min_itki_N=points[0]['itki'], mevcut_max_itki_N=points[-1]['itki'])
                try:
                    out.update(required_power(points,thrust))
                    out['durum'] = 'tablo_araliginda'
                except ModelError as exc:
                    if (exc.status == 'itki_verisi_altinda' and thrust >= 0 and
                        a.alt_gaz_modeli == 'sifir-noktasi' and points[0]['gaz'] > 0 and points[0]['itki'] > 0):
                        p = points[0]
                        fraction = thrust/p['itki']
                        out.update(gaz_yuzde=p['gaz']*fraction, guc_W=p['guc']*fraction,
                                   tahmini_guc=True, durum='alt_gaz_tahmini',
                                   guc_kaynagi='varsayimsal_0N_0W_ile_tablo_alt_noktasi',
                                   aciklama='Gaz/itki/guc sifir noktasi ile ilk tablo noktasi arasinda dogrusal varsayildi; olcum degil.')
                    elif thrust < 0 and exc.status == 'itki_verisi_altinda':
                        raise ModelError('frenleme_verisi_eksik',
                                         f'Motor kapaliyken { -thrust:.6g} N ek surukleme gerekiyor; tabloda bunu karsilayan veri yok.') from exc
                    else:
                        raise
            out['toplam_elektrik_gucu_W'] = out['guc_W']+a.yardimci_guc
            out['akim_A'] = out['toplam_elektrik_gucu_W']/a.gerilim
        except (ModelError, LLTError) as exc:
            out.update(durum=exc.status, aciklama=str(exc), toplam_elektrik_gucu_W=None, akim_A=None)
        return out


def integrate(samples, duration, voltage):
    """Orta-nokta integrasyonu. Eksik ornekler toplam enerjiye sifir yazilmaz."""
    known_j = 0.0
    known_duration = 0.0
    statuses, notes = set(), {}
    estimated = conditional = False
    for dt, point in samples:
        statuses.add(point['durum'])
        if point.get('aciklama'):
            notes.setdefault(point['durum'],point['aciklama'])
        power = point.get('toplam_elektrik_gucu_W')
        if power is not None:
            known_j += power*dt
            known_duration += dt
        estimated |= point.get('tahmini_guc',False)
        conditional |= point.get('dalis_frenleme_kosullu',False)
    complete = math.isclose(known_duration,duration,rel_tol=0,abs_tol=1e-8)
    return dict(sure_s=duration, hesaplanan_sure_s=known_duration,
                hesaplanamayan_sure_s=max(0.0,duration-known_duration),
                tuketim_tam=complete, bilinen_kisim_enerji_J=known_j,
                bilinen_kisim_mAh=known_j/voltage/3.6,
                enerji_J=known_j if complete else None,
                enerji_Wh=known_j/3600 if complete else None,
                tuketim_mAh=known_j/voltage/3.6 if complete else None,
                ortalama_guc_W=known_j/duration if complete else None,
                durum='|'.join(sorted(statuses)), aciklama=' | '.join(notes[k] for k in sorted(notes)),
                tahmini_guc=estimated, dalis_frenleme_kosullu=conditional)


def llt_summary(points):
    result = {'llt_Re_uyumlu':all(p.get('llt_Re_uyumlu',False) for p in points),
              'llt_Re_interpolasyon':any(p.get('llt_Re_interpolasyon',False) for p in points),
              'llt_sabit_Re_varsayimi':any(p.get('llt_sabit_Re_varsayimi',False) for p in points),
              'llt_VT_alt_sinir_varsayimi':any(p.get('llt_VT_alt_sinir_varsayimi',False) for p in points)}
    for part in ('w','HT','VT'):
        for kind in ('ucus','kullanilan','alt','ust'):
            key='llt_Re_'+part+'_'+kind
            values=[p[key] for p in points if p.get(key) is not None]
            if values:
                result[key+'_min']=min(values);result[key+'_max']=max(values)
    return result


def endpoints(out, start, end):
    for key in ('kutle_kg','kalan_su_kg','alpha_deg','alpha_HT_deg','T_gerekli_N','D_N',
                'gaz_yuzde','guc_W','akim_A'):
        out[key+'_baslangic'] = start.get(key)
        out[key+'_bitis'] = end.get(key)
    out['baslangic_durum'] = start['durum']
    out['bitis_durum'] = end['durum']
    out['uc_noktalar_hesaplandi'] = all(p.get('toplam_elektrik_gucu_W') is not None for p in (start,end))
    if not out['uc_noktalar_hesaplandi']:
        out.update(tuketim_tam=False,enerji_J=None,enerji_Wh=None,tuketim_mAh=None,ortalama_guc_W=None)
    return out


def calculate(mass_rows, llt, drag_rows, prop_rows, settings, default_prop=None, detail=False):
    settings.validate()
    candidates = performans_adaylarini_hazirla(mass_rows,default_prop)
    drag_index = {}
    for row in drag_rows:
        drag_index.setdefault(wing_key(row),[]).append(row)
    summaries, phases, seconds, details = [], [], [], []
    prop_cache = {}
    a = settings
    total_time = 2*a.egimli_sure+a.birakma_suresi
    for index, candidate in enumerate(candidates,1):
        ident = {**kimlik_alanlari(candidate), 'aday_no':index,
                 'kanat_airfoil':candidate.get('kanat_airfoil','')}
        summary = {**ident, 'durum':'veri_hatasi', 'aciklama':'',
                   'sure_s':total_time, 'yatay_mesafe_m':total_time*a.vx,
                   'seyir_irtifasi_m':a.seyir_irtifasi, 'birakma_irtifasi_m':a.birakma_irtifasi,
                   'ilk_seyir_hizi_m_s':a.seyir_hizi, 'tirmanis_hizi_m_s':math.hypot(a.vx,a.vz),
                   'dalis_sabit_hizi_m_s':math.hypot(a.vx,a.vz) if a.dalis_profili=='sabit' else None,
                   'dalis_profili':a.dalis_profili, 'dalis_gecisleri_dahil':a.dalis_profili=='yumusak',
                   'Vx_m_s':a.vx, 'Vz_buyukluk_m_s':a.vz, 'gerilim_V':a.gerilim,
                   'rho_kg_m3':a.rho, 'mu_Pa_s':a.mu, 'g_m_s2':a.g,
                   'dt_ust_sinir_s':a.dt, 'yardimci_guc_W':a.yardimci_guc,
                   'gecisler_dahil':False, 'trim_cozuldu':False, 'downwash_deg':0.0,
                   'kuyruk_oturma_deg':a.kuyruk_oturma, 'tuketim_mAh':None,
                   'puan_kg_katsayisi':30.0, 'puan_ust_siniri':120.0,
                   'model_notu':'Ruzgar, seyirden yavaslama, tirmanisa giris/cikis, trim, su cikis tepkisi ve yere dusme hesabi yok. Yumusak dalista az dahil.'}
        try:
            model = AdayModeli(candidate,llt,drag_index.get(wing_key(candidate),[]),prop_rows,a,prop_cache)
            water, dry, loaded = model.water, model.dry, model.loaded
            rate = water/a.birakma_suresi
            summary.update(dolu_kutle_kg=loaded, susuz_kutle_kg=dry, birakilan_su_kg=water,
                           su_debisi_kg_s=rate, ham_puan=water*30,
                           puan=min(water*30,120.0), puan_turu='planlanan_su_tam_birakilirsa',
                           hiz_azaltmada_kinetik_enerji_farki_J=.5*loaded*(a.seyir_hizi**2-(a.vx**2 if a.dalis_profili=='yumusak' else a.vx**2+a.vz**2)))
            cruise = model.point(loaded,a.seyir_hizi,0.0,'seyir_karsilastirma')
            for field in ('alpha_deg','D_N','gaz_yuzde','guc_W','durum'):
                summary['ilk_seyir_'+field] = cruise.get(field)
            all_samples, all_endpoints = [], []
            elapsed = 0.0
            for phase, duration, vz, mass, h0 in (
                ('dalis',a.egimli_sure,-a.vz,loaded,a.seyir_irtifasi),
                ('yuk_birakma',a.birakma_suresi,0.0,loaded,a.birakma_irtifasi),
                ('tirmanis',a.egimli_sure,a.vz,dry,a.birakma_irtifasi)):
                phase_samples = []
                smooth = phase=='dalis' and a.dalis_profili=='yumusak'
                if smooth:
                    start = end = model.point(loaded,a.vx,0.0,phase)
                    n = max(1,math.ceil(duration/a.dt))
                    dt = duration/n
                    for j in range(n):
                        t = (j+.5)*dt
                        drop, local_vz, az = yumusak_dalis(t,duration,a.seyir_irtifasi-a.birakma_irtifasi)
                        point = model.point(loaded,a.vx,local_vz,phase,az)
                        phase_samples.append((dt,point))
                        if detail:
                            details.append({**ident,'faz':phase,'t_s':elapsed+t,'faz_t_s':t,
                                            'dt_s':dt,'irtifa_m':h0-drop,'x_m':(elapsed+t)*a.vx,**point})
                elif phase == 'yuk_birakma':
                    start = model.point(loaded,a.vx,0.0,phase)
                    end = model.point(dry,a.vx,0.0,phase)
                    # Saniye sinirlarina tam oturan hucreler; son kesirli saniye de korunur.
                    for sec in range(math.ceil(duration)):
                        left, right = float(sec), min(float(sec+1),duration)
                        n = max(1, math.ceil((right-left)/a.dt))
                        dt = (right-left)/n
                        cell_samples = []
                        for j in range(n):
                            t = left+(j+.5)*dt
                            point = model.point(dry+water*(1-t/duration),a.vx,0.0,phase)
                            cell_samples.append((dt,point))
                            if detail:
                                details.append({**ident,'faz':phase,'t_s':elapsed+t,'faz_t_s':t,
                                                'dt_s':dt,'irtifa_m':h0,'x_m':(elapsed+t)*a.vx,**point})
                        interval = {**ident, 'faz':phase, 'saniye_no':sec+1,
                                    't_baslangic_s':elapsed+left, 't_bitis_s':elapsed+right,
                                    'faz_t_baslangic_s':left, 'faz_t_bitis_s':right,
                                    'birakilan_su_kg':rate*(right-left),'su_debisi_kg_s':rate}
                        interval.update(integrate(cell_samples,right-left,a.gerilim))
                        seconds.append(endpoints(interval,
                            model.point(dry+water*(1-left/duration),a.vx,0.0,phase),
                            model.point(dry+water*(1-right/duration),a.vx,0.0,phase)))
                        phase_samples.extend(cell_samples)
                else:
                    start = end = model.point(mass,a.vx,vz,phase)
                    phase_samples = [(duration,start)]
                    if detail:
                        details.append({**ident,'faz':phase,'t_s':elapsed+duration/2,
                                        'faz_t_s':duration/2,'dt_s':duration,
                                        'irtifa_m':h0+vz*duration/2,'x_m':(elapsed+duration/2)*a.vx,**start})
                phase_row = {**ident, 'faz':phase, 't_baslangic_s':elapsed,'t_bitis_s':elapsed+duration,
                             'irtifa_baslangic_m':h0,'irtifa_bitis_m':h0+vz*duration,
                             'yatay_mesafe_m':a.vx*duration, 'Vx_m_s':a.vx,'Vz_m_s':None if smooth else vz,
                             'Vz_ortalama_m_s':vz, 'Vz_baslangic_m_s':start['Vz_m_s'], 'Vz_bitis_m_s':end['Vz_m_s'],
                             'V_m_s':None if smooth else math.hypot(a.vx,vz),
                             'gamma_deg':None if smooth else math.degrees(math.atan2(vz,a.vx)),
                             'ek_surukleme_gerekli_N':max(p.get('ek_surukleme_gerekli_N',0.0) for _,p in phase_samples),
                             'e_w':start.get('e_w'),'e_HT':start.get('e_HT'),
                             'llt_Re_uyumlu':start.get('llt_Re_uyumlu'),
                             'llt_Re_w_kaynak':start.get('llt_Re_w_kaynak'),
                             'llt_Re_w_ucus':start.get('llt_Re_w_ucus'),
                             'llt_model_notu':start.get('llt_model_notu'),
                             'llt_veri_kaynagi':start.get('llt_veri_kaynagi')}
                phase_row.update(integrate(phase_samples,duration,a.gerilim))
                phase_row.update(llt_summary([p for _,p in phase_samples]+[start,end]))
                for field in ('Vz_m_s','az_m_s2','alpha_deg','T_gerekli_N','yuk_faktoru'):
                    values=[p[field] for _,p in phase_samples if field in p]+[p[field] for p in (start,end) if field in p]
                    if values:
                        phase_row[field+'_min']=min(values)
                        phase_row[field+'_max']=max(values)
                endpoints(phase_row,start,end)
                phases.append(phase_row)
                all_samples.extend(phase_samples)
                all_endpoints.extend((start,end))
                summary[phase+'_sure_s'] = duration
                summary[phase+'_mAh'] = phase_row['tuketim_mAh']
                summary[phase+'_durum'] = phase_row['durum']
                if phase == 'dalis':
                    summary['dalis_ek_surukleme_gerekli_N'] = phase_row['ek_surukleme_gerekli_N']
                    if smooth:
                        summary['dalis_tepe_dusus_hizi_m_s'] = 1.875*a.vz
                    if 'T_gerekli_N' in start and not smooth:
                        summary['dalis_motor_kapali_ek_suruklemesiz_ivme_m_s2'] = -start['T_gerekli_N']/loaded
                        summary['dalis_motor_kapali_toplam_surukleme_hedef_N'] = -start['agirlik_yol_bileseni_N']
                elapsed += duration
            summary.update(integrate(all_samples,total_time,a.gerilim))
            summary['uc_noktalar_hesaplandi'] = all(p.get('toplam_elektrik_gucu_W') is not None for p in all_endpoints)
            if not summary['uc_noktalar_hesaplandi']:
                summary.update(tuketim_tam=False,enerji_J=None,enerji_Wh=None,tuketim_mAh=None,ortalama_guc_W=None)
            summary.update(llt_summary([p for _,p in all_samples]+all_endpoints))
            summary['veriyle_tamamlandi'] = (summary['tuketim_tam'] and summary['uc_noktalar_hesaplandi']
                                           and not summary['tahmini_guc'] and not summary['dalis_frenleme_kosullu'])
            summary['durum'] = ('eksik_veri_veya_ucus_kisiti' if not summary['tuketim_tam'] or not summary['uc_noktalar_hesaplandi']
                                else 'kosullu_tahmin' if not summary['veriyle_tamamlandi'] else 'model_kapsaminda_hesaplandi')
        except (ModelError,LLTError) as exc:
            summary.update(durum=exc.status,aciklama=str(exc))
        summaries.append(summary)
    return summaries, phases, seconds, details


def resolve_input(explicit, names):
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise ValueError(f'Girdi bulunamadi: {path}')
        return path
    for directory in (ROOT/'veriler',ROOT,ROOT.parent,ROOT/'Scripts',ROOT.parent/'Scripts'):
        for name in names:
            path = directory/name
            if path.is_file():
                return path
    raise ValueError(f'Girdi bulunamadi: {names}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ('kutle','llt','drag','itki'):
        parser.add_argument('--'+flag,type=Path)
    parser.add_argument('--cikti',type=Path,default=ROOT/'yuk_birakma_sonuclari.csv')
    parser.add_argument('--faz-cikti',type=Path)
    parser.add_argument('--saniye-cikti',type=Path)
    parser.add_argument('--detay-cikti',type=Path,help='Istenirse ara integrasyon noktalarini da yazar.')
    parser.add_argument('--pervane',help='Yalniz pervanesiz eski adaylar icin varsayilan.')
    parser.add_argument('--konfigurasyon-id',help='Tek bir konfigurasyonu hesapla.')
    defaults = Ayarlar()
    for name in ('seyir-irtifasi','birakma-irtifasi','seyir-hizi','vx','vz','birakma-suresi',
                 'dt','gerilim','rho','mu','g','kuyruk-oturma','yardimci-guc'):
        parser.add_argument('--'+name,type=float,default=getattr(defaults,name.replace('-','_')))
    parser.add_argument('--dalis-motor-kapali', action=argparse.BooleanOptionalAction,
                        default=defaults.dalis_motor_kapali,
                        help='Ek surukleme saglandigi kosuluyla dalista T<=0 adimlarinda motor gucunu 0 W kabul et (varsayilan: acik).')
    parser.add_argument('--dalis-profili',choices=['sabit','yumusak'],default='sabit',
                        help='yumusak: Vz iki ucta 0; --vz ortalama dusus hizidir. Dikey ivme kuvvetlere dahil.')
    parser.add_argument('--alt-gaz-modeli', choices=['veri', 'sifir-noktasi'],
                        default=defaults.alt_gaz_modeli,
                        help='sifir-noktasi: 0 N / 0 W kabulune dayali tahmin (varsayilan). veri: tablo disina cikmaz.')
    parser.add_argument('--llt-re-kontrol',action='store_true')
    parser.add_argument('--tablo-degeri',choices=['auto','guc_w','akim_a'],default='auto')
    args = parser.parse_args(argv)
    try:
        settings = Ayarlar(**{name:getattr(args,name) for name in Ayarlar.__dataclass_fields__})
        settings.validate()
        paths = dict(kutle=resolve_input(args.kutle,['kanat_kuyruk_govde_kutleli.csv']),
                     llt=resolve_input(args.llt,['llt_flapsiz_coklu_re.csv']),
                     drag=resolve_input(args.drag,['openvsp_llt_flapsiz_aday_ozet.csv','openvsp_llt_flapsiz_polar_detay.csv']),
                     itki=resolve_input(args.itki,['itki_tablosu.csv']))
        masses = performans_adaylarini_hazirla(read_csv(paths['kutle']),args.pervane)
        if args.konfigurasyon_id:
            masses = [m for m in masses if m['konfigurasyon_id']==args.konfigurasyon_id]
            if not masses:
                raise ValueError('Konfigurasyon kimligi bulunamadi.')
        # Ciktilar girdilerin veya birbirlerinin uzerine yazilamaz.
        outputs = [args.cikti,
                   args.faz_cikti or args.cikti.with_name(args.cikti.stem+'_fazlar.csv'),
                   args.saniye_cikti or args.cikti.with_name(args.cikti.stem+'_saniyelik.csv')]
        if args.detay_cikti:
            outputs.append(args.detay_cikti)
        resolved = [p.resolve() for p in outputs]
        if len(set(resolved)) != len(resolved) or set(resolved)&{p.resolve() for p in paths.values()}:
            raise ValueError('Cikti yollari birbirinden ve girdi yollarindan farkli olmali.')
        results = calculate(masses,LLTTable.read(paths['llt']),read_csv(paths['drag']),
                            read_csv(paths['itki']),settings,detail=bool(args.detay_cikti))
        for rows, target in zip(results,outputs):
            if rows:
                csv_yaz(rows,target)
                print(f'{len(rows)} satir: {target.resolve()}')
            elif target.exists():
                # Bos sonucda eski basarili raporun yerinde kalmasini engelle.
                target.write_text('durum,aciklama\nveri_yok,Aday ozetindeki hatalara bakin.\n',encoding='utf-8-sig')
        print('Durumlar: '+str(dict(Counter(r['durum'] for r in results[0]))))
        print('Surelere seyirden yavaslama ve tirmanisa giris/cikis dahil degil; yumusak dalista dalis giris/cikisi dahil.')
        if any(not r.get('tuketim_tam',False) for r in results[0]):
            print('Eksik verili adaylarda toplam mAh bos; bilinen_kisim_mAh toplam tuketim degildir.')
        return 0
    except (OSError,ValueError,KeyError) as exc:
        print(f'HATA: {exc}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
