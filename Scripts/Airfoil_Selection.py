import os
import time
import string
import requests
from bs4 import BeautifulSoup
import subprocess
import pandas as pd
import numpy as np

# --- KULLANICI PARAMETRELERİ ---
TARGET_ALPHA = 0.0        # İstenilen hücum açısı (derece)
MIN_CL = 0.2             # Target alpha'da minimum CL değeri
MAX_CD = 0.04              # Target alpha'da maksimum CD değeri
MIN_STALL_ANGLE = 12.0    # Minimum stall açısı (derece)
MIN_MAX_LD = 30          # Minimum istenen Max L/D oranı
REYNOLDS = 300000         # Analiz yapılacak Reynolds sayısı
ITERATIONS = 200          # Yakınsama şansını artırmak için 200'e çıkarıldı
XFOIL_TIMEOUT = 12        # XFOIL'in takılmasını önlemek için zaman aşımı (saniye)
# -------------------------------

BASE_URL = "http://airfoiltools.com"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def get_airfoil_links():
    airfoils = set()
    for char in string.ascii_lowercase:
        url = f"{BASE_URL}/search/airfoils?page=1&search={char}"
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code != 200:
                continue
            soup = BeautifulSoup(response.content, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/airfoil/details?airfoil=' in href:
                    airfoil_name = href.split('airfoil=')[-1].strip()
                    if airfoil_name:
                        airfoils.add(airfoil_name)
        except requests.RequestException:
            continue
        time.sleep(0.3)
    return sorted(list(airfoils))

def download_dat(airfoil_name, folder="airfoils"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    dat_url = f"{BASE_URL}/airfoil/seligdatfile?airfoil={airfoil_name}"
    filepath = os.path.join(folder, f"{airfoil_name}.dat")
    if not os.path.exists(filepath):
        response = requests.get(dat_url, headers=HEADERS, timeout=10)
        with open(filepath, 'w') as f:
            f.write(response.text)
        time.sleep(0.5)
    return filepath

def run_xfoil(dat_filepath, polar_filepath):
    if os.path.exists(polar_filepath):
        try:
            os.remove(polar_filepath)
        except OSError:
            pass
        
    commands = (
        f"LOAD {dat_filepath}\n"
        "PANE\n"
        "OPER\n"
        f"Iter {ITERATIONS}\n"
        f"Visc {REYNOLDS}\n"
        "PACC\n"
        f"{polar_filepath}\n\n"
        "ASEQ 0 20 1\n\n"
        "QUIT\n"
    )

    proc = None
    try:
        proc = subprocess.Popen(
            ["xfoil.exe"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True
        )
        proc.communicate(input=commands, timeout=XFOIL_TIMEOUT)
    except subprocess.TimeoutExpired:
        if proc:
            proc.kill()
            proc.communicate()
    except Exception:
        if proc:
            proc.kill()

def check_criteria(polar_filepath):
    if not os.path.exists(polar_filepath):
        return False, None, "Polar dosyası oluşmadı (XFOIL çöktü)"
    
    try:
        with open(polar_filepath, 'r') as f:
            lines = f.readlines()
            
        data = []
        # Satır sayısına bağlı kalmadan, sadece sayısal verileri filtrele
        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    alpha = float(parts[0])
                    cl = float(parts[1])
                    cd = float(parts[2])
                    data.append([alpha, cl, cd])
                except ValueError:
                    continue # Metin içeren başlık satırlarını atla
                    
        if not data:
            return False, None, "Veri okunamadı veya XFOIL yakınsamadı"

        df = pd.DataFrame(data, columns=['Alpha', 'CL', 'CD'])
        df = df[df['CD'] > 0]
        
        if df.empty:
            return False, None, "Geçerli CD değeri (CD > 0) bulunamadı"

        df['L_D'] = df['CL'] / df['CD']

        # Stall Açısı (CL'nin maksimum olduğu andaki Alpha)
        max_cl_idx = df['CL'].idxmax()
        stall_angle = df.loc[max_cl_idx, 'Alpha']
        
        # Max L/D
        max_ld_idx = df['L_D'].idxmax()
        max_ld = df.loc[max_ld_idx, 'L_D']
        alpha_at_max_ld = df.loc[max_ld_idx, 'Alpha']

        # 1. Stall Kontrolü
        if stall_angle < MIN_STALL_ANGLE:
            max_alpha_reached = df['Alpha'].max()
            return False, None, f"Stall açısı yetersiz ({stall_angle}° < {MIN_STALL_ANGLE}°). Analiz {max_alpha_reached}°'ye kadar yapılabildi."

        # 2. Max L/D Kontrolü
        if max_ld < MIN_MAX_LD:
            return False, None, f"Max L/D yetersiz ({max_ld:.1f} < {MIN_MAX_LD})"

        # 3. Target Alpha Aralık Kontrolü
        min_alpha = df['Alpha'].min()
        max_alpha = df['Alpha'].max()
        if TARGET_ALPHA < min_alpha or TARGET_ALPHA > max_alpha:
            return False, None, f"Hedef açı ({TARGET_ALPHA}°) hesaplanan aralıkta değil [{min_alpha}°, {max_alpha}°]"

        cl_at_target = np.interp(TARGET_ALPHA, df['Alpha'], df['CL'])
        cd_at_target = np.interp(TARGET_ALPHA, df['Alpha'], df['CD'])
        
        # 4. Target CL Kontrolü
        if cl_at_target < MIN_CL:
            return False, None, f"Hedef CL yetersiz ({cl_at_target:.3f} < {MIN_CL})"
            
        # 5. Target CD Kontrolü
        if cd_at_target > MAX_CD:
            return False, None, f"Hedef CD yüksek ({cd_at_target:.4f} > {MAX_CD})"
            
        return True, {
            'Stall_Angle': stall_angle, 
            'CL': cl_at_target, 
            'CD': cd_at_target,
            'Max_LD': max_ld,
            'Alpha_Max_LD': alpha_at_max_ld
        }, "Kriterler sağlandı"
        
    except Exception as e:
        return False, None, f"Okuma hatası: {str(e)}"

def main():
    print("Airfoil listesi çekiliyor...")
    airfoils = get_airfoil_links()
    print(f"Toplam {len(airfoils)} airfoil bulundu.")
    
    successful_airfoils = []
    
    for i, airfoil in enumerate(airfoils[:1638]): 
        dat_file = download_dat(airfoil)
        polar_file = f"polar_{airfoil}.txt"
        
        run_xfoil(dat_file, polar_file)
        
        passed, data, reason = check_criteria(polar_file)
        
        if passed:
            print(f"[{i+1}/{len(airfoils[:1638])}] [BAŞARILI] {airfoil} | Stall: {data['Stall_Angle']}° | CL: {data['CL']:.3f} | CD: {data['CD']:.4f} | L/D: {data['Max_LD']:.1f}")
            successful_airfoils.append(airfoil)
        else:
            print(f"[{i+1}/{len(airfoils[:1638])}] [REDDEDİLDİ] {airfoil} -> {reason}")
            # Kriterleri geçemeyen airfoil'in dat dosyasını sil
            if os.path.exists(dat_file):
                os.remove(dat_file)
            
        if os.path.exists(polar_file):
            os.remove(polar_file)
            
    print("\n--- SONUÇLAR ---")
    print("Kriterleri sağlayan airfoil sayısı:", len(successful_airfoils))
    for a in successful_airfoils:
        print(a)

if __name__ == "__main__":
    main()