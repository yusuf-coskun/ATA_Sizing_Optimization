import math
import os
import csv
from datetime import datetime

def tirmanis_enerjisi_hesapla(m, v_yatay, v_dikey, h_hedef, CD0, K, S, rho, eta_prop, eta_motor, kaydet=True, dosya_yolu="Scripts/kanat_sonuclari/tirmanis_sonuclari.csv"):
    
    g = 9.81 
    
    v_toplam = math.sqrt(v_yatay**2 + v_dikey**2) 
    gamma = math.atan2(v_dikey, v_yatay)          
    
    W = m * g
    L = W * math.cos(gamma)
    
    q = 0.5 * rho * (v_toplam ** 2)
    CL = L / (q * S)
    
    CD = CD0 + K * (CL ** 2)
    D = q * S * CD
    
    T_req = D + W * math.sin(gamma)
    
    P_mekanik = T_req * v_toplam
    eta_toplam = eta_prop * eta_motor
    P_elektrik = P_mekanik / eta_toplam
    
    zaman = h_hedef / v_dikey
    enerji_j = P_elektrik * zaman
    enerji_wh = enerji_j / 3600 
    
    if kaydet:
        klasor = os.path.dirname(dosya_yolu)
        if klasor and not os.path.exists(klasor):
            os.makedirs(klasor)
            
        dosya_mevcut = os.path.isfile(dosya_yolu)
        
        with open(dosya_yolu, mode='a', newline='', encoding='utf-8') as f:
            yazici = csv.writer(f)
            if not dosya_mevcut:
                yazici.writerow(["Tarih", "Kutle_kg", "V_Yatay_m_s", "V_Dikey_m_s", "Hedef_Irtifa_m", "Sure_s", "Guc_W", "Enerji_Wh", "Enerji_Joule", "V_Toplam_m_s"])
            
            tarih = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            yazici.writerow([tarih, m, v_yatay, v_dikey, h_hedef, round(zaman, 2), round(P_elektrik, 2), round(enerji_wh, 2), round(enerji_j, 2), round(v_toplam, 2)])
    
    return enerji_wh, enerji_j, zaman, P_elektrik, v_toplam