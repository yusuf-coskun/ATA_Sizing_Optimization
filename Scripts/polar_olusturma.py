import os
import subprocess
import time

# ================= AYARLAR =================
DAT_FOLDER = "airfoils"
POLAR_FOLDER = "airfoils_polar"
XFOIL_EXE = "xfoil"

REYNOLDS = 300000
ALPHA_MIN = -5.0
ALPHA_MAX = 15.0
ALPHA_STEP = 0.5
ITERATIONS = 300
# ===========================================

def sort_polar_file(polar_path):
    """XFOIL'in ürettiği polar dosyasını sıralar."""
    if not os.path.exists(polar_path):
        return
        
    with open(polar_path, 'r') as f:
        lines = f.readlines()
        
    header = []
    data = []
    in_data_section = False
    
    for line in lines:
        if not in_data_section:
            header.append(line)
            if "------" in line and "--------" in line:
                in_data_section = True
        else:
            if line.strip():
                data.append(line)
                
    parsed_data = []
    for line in data:
        try:
            alpha = float(line.split()[0])
            parsed_data.append((alpha, line))
        except ValueError:
            pass
            
    # Açılara göre küçükten büyüğe sırala
    parsed_data.sort(key=lambda x: x[0])
    
    # Sadece eşsiz açıları yaz (kopyalara karşı son güvenlik)
    unique_data = []
    seen_alphas = set()
    for alpha, line in parsed_data:
        if alpha not in seen_alphas:
            unique_data.append(line)
            seen_alphas.add(alpha)
            
    with open(polar_path, 'w') as f:
        f.writelines(header)
        f.writelines(unique_data)

def run_xfoil_for_airfoil(airfoil_path: str, polar_path: str, airfoil_name: str):
    commands = [
        f"LOAD {airfoil_path}",
        "PANE",
        "OPER",
        "ALFA 0",              # Inviscid olarak 0'ı çöz (kayıt dışı, sınır tabakaya hazırlık)
        f"ITER {ITERATIONS}",
        f"VISC {REYNOLDS}",    # Viskoziteyi aç
        "PACC",                # Kaydı başlat
        f"{polar_path}",
        "" 
    ]
    
    # 1. Aşama: 0'dan 15'e pozitif tarama (0 burada sadece BİR KEZ çözülüp kaydedilir)
    a = 0.0
    while a <= ALPHA_MAX + 1e-5:
        commands.append(f"ALFA {a:.2f}")
        a += ALPHA_STEP
        
    # 2. Aşama: Sınır tabakayı sıfırla ve 0'ı TEKRAR ÇÖZMEDEN doğrudan -0.5'e geç
    commands.append("INIT")
    
    a = -ALPHA_STEP
    while a >= ALPHA_MIN - 1e-5:
        commands.append(f"ALFA {a:.2f}")
        a -= ALPHA_STEP

    commands.append("PACC") 
    commands.append("QUIT")
    
    xfoil_script = "\n".join(commands) + "\n"

    try:
        process = subprocess.run(
            [XFOIL_EXE],
            input=xfoil_script,
            text=True,
            capture_output=True,
            timeout=45
        )
        
        if os.path.exists(polar_path) and os.path.getsize(polar_path) > 0:
            sort_polar_file(polar_path)
            print(f"Başarılı: {airfoil_name} (Re={REYNOLDS})")
        else:
            print(f"Başarısız (Yakınsamadı): {airfoil_name}")
             
    except subprocess.TimeoutExpired:
        print(f"Zaman Aşımı: {airfoil_name} atlanıyor.")
    except FileNotFoundError:
        print("Hata: XFOIL bulunamadı! XFOIL_EXE yolunu kontrol edin.")
        exit(1)

def main():
    if not os.path.exists(DAT_FOLDER):
        print(f"Hata: {DAT_FOLDER} klasörü bulunamadı.")
        return
        
    os.makedirs(POLAR_FOLDER, exist_ok=True)
    
    dat_files = [f for f in os.listdir(DAT_FOLDER) if f.endswith(".dat")]
    total_files = len(dat_files)
    
    if total_files == 0:
        print("Klasörde .dat dosyası bulunamadı.")
        return
        
    print(f"Toplam {total_files} adet airfoil bulundu. XFOIL işlemi başlıyor...\n" + "="*50)
    
    start_time = time.time()
    
    for i, filename in enumerate(dat_files, 1):
        airfoil_name = os.path.splitext(filename)[0]
        dat_path = os.path.join(DAT_FOLDER, filename)
        polar_path = os.path.join(POLAR_FOLDER, f"{airfoil_name}_re{REYNOLDS}.pol")
        
        if os.path.exists(polar_path):
            if os.path.getsize(polar_path) > 0:
                print(f"Atlanıyor (Zaten var): {airfoil_name} [{i}/{total_files}]")
                continue
            else:
                os.remove(polar_path)
            
        print(f"[{i}/{total_files}] İşleniyor: {airfoil_name} ...", end="\r")
        run_xfoil_for_airfoil(dat_path, polar_path, airfoil_name)
        
    elapsed = time.time() - start_time
    print("\n" + "="*50)
    print(f"İşlem tamamlandı. Süre: {elapsed:.2f} saniye")

if __name__ == "__main__":
    main()