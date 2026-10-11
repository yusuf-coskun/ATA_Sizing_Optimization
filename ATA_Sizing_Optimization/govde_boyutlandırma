import numpy as np

def generate_fuselage_geometry(D_range, V_water_range, L_tail_range, V_avionics):
   
     # Parametre kombinasyonları için 3 boyutlu grid oluşturma
    D_grid, V_su_grid, L_tail_grid = np.meshgrid(D_range, V_water_range, L_tail_range)
    
    D_array = D_grid.flatten()
    V_su = V_su_grid.flatten()
    L_tail_array = L_tail_grid.flatten()
    
    # Yarıçap
    R_array = D_array / 2.0
    
    # 1. Aviyonik Bölümü (Silindir)
    # Hacim sabit, uzunluk çapa göre değişiyor
    S_aviyonik = np.pi * D_array * L_aviyonik
    
    # 2. Su Haznesi Bölümü (Silindir)
    # Uzunluk çapa ve hedeflenen su hacmine göre değişiyor
    L_su = V_su / (np.pi * (R_array ** 2))
    S_su = np.pi * D_array * L_su
    
    # 3. Kuyruk Bölümü (Kesik Koni)
    # Çap lineer olarak 8 cm'ye (0.08 m) düşüyor
    D_min = 0.08
    R_min = D_min / 2.0
    
    # Kesik koni hacmi: V = (1/3) * pi * h * (r1^2 + r1*r2 + r2^2)
    V_kuyruk_array = (1.0 / 3.0) * np.pi * L_tail_array * (R_array*2 + R_array * R_min + R_min*2)
    
    # Kesik koni yanal yüzey alanı: A = pi * (r1 + r2) * s
    # s (eğik yanal uzunluk) = sqrt((r1 - r2)^2 + h^2)
    slant_height = np.sqrt((R_array - R_min)*2 + L_tail_array*2)
    S_kuyruk_array = np.pi * (R_array + R_min) * slant_height
    
    # Toplam Değerler
    L_total = L_aviyonik + L_su + L_tail_array
    L_total *= 1.1
    Volume_total_array = V_aviyonik + V_su + V_kuyruk_array
    S_total = S_aviyonik + S_su + S_kuyruk_array
    
    # Kuyruk momenti kolu (Kavramsal tasarım için orijinal koddaki yaklaşımla bırakıldı)
    # İhtiyaca göre CG noktasına veya kanat yerleşimine (muhtemelen su tankı üzeri) revize edilebilir.
    tail_arm_array = L_total * 0.55
    
    return (D_array, V_su, L_tail_array, L_total, 
            L_aviyonik, L_su, Volume_total_array, 
            S_total, tail_arm_array)

# Girdi Parametreleri
# Çap seçenekleri: 20, 25 ve 30 cm (metre cinsinden)
D_secenekleri = np.array([0.14, 0.17, 0.20])  

# Su hacmi seçenekleri (m^3) (Örnek: 5 litreden 20 litreye kadar)
V_su_secenekleri = np.linspace(0.003, 0.005, 10)  

# Kuyruk bölgesi uzunluğu seçenekleri (m)
L_kuyruk_secenekleri = np.linspace(0.5, 1.5, 5)  

# Sabit aviyonik hacmi (m^3) (Örnek: 2 litre)
V_aviyonik = 0.0046
L_aviyonik = 0.3

# Fonksiyonu çalıştır
(D_arr, V_su_arr, L_kuyruk_arr, L_total, 
 L_aviyonik_arr, L_su_arr, V_toplam_arr, 
 S_toplam, Tail_Arm_arr) = generate_fuselage_geometry(D_secenekleri, 
                                                           V_su_secenekleri, 
                                                           L_kuyruk_secenekleri, 
                                                           V_aviyonik)