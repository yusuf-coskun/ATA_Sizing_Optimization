import os
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# NumPy sürüm uyumluluğu (NumPy >= 2.0 için trapezoid, alt sürümler için trapz)
trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

# ==============================================================================
# 1. POLAR DOSYASI OKUYUCU (XFLR5 / XFOIL / AIRFOILTOOLS UYUMLU)
# ==============================================================================
def load_polar_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"'{file_path}' dosyası bulunamadı. Lütfen dosya yolunu kontrol et.")

    alpha_list, cl_list, cd_list = [], [], []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    a_val = float(parts[0])
                    cl_val = float(parts[1])
                    cd_val = float(parts[2])
                    alpha_list.append(a_val)
                    cl_list.append(cl_val)
                    cd_list.append(cd_val)
                except ValueError:
                    continue

    if len(alpha_list) < 5:
        raise ValueError("Polar dosyasından yeterli veri okunamadı. Sütun formatını kontrol et.")

    return np.array(alpha_list), np.array(cl_list), np.array(cd_list)

def extract_polar_derivatives(alpha_deg, Cl):
    # Stabil lineer aralıktan regresyon
    mask = (alpha_deg >= -1.0) & (alpha_deg <= 4.5)
    if np.sum(mask) < 3:
        mask = (alpha_deg >= -1.0) & (alpha_deg <= 6.0)

    p = np.polyfit(np.radians(alpha_deg[mask]), Cl[mask], 1)
    a0 = p[0]                      
    alpha_0L_rad = -p[1] / p[0]   
    alpha_0L_deg = np.degrees(alpha_0L_rad)
    cl_max = np.max(Cl)
    alpha_stall = alpha_deg[np.argmax(Cl)]

    return a0, alpha_0L_rad, alpha_0L_deg, cl_max, alpha_stall

# ==============================================================================
# 2. VİSKOZ PROFİL POLARLI PRANDTL LIFTING-LINE ÇÖZÜCÜSÜ
# ==============================================================================
def solve_lifting_line_with_polar(b, S, y_stations, c_stations, alpha_geom_deg, 
                                  a0, alpha_0L_rad, polar_cl, polar_cd, num_terms=30):
    alpha_eff_rad = np.radians(alpha_geom_deg) - alpha_0L_rad
    
    i_indices = np.arange(1, num_terms + 1)
    theta = (2 * i_indices - 1) * np.pi / (4 * num_terms)
    y_colloc = (b / 2) * np.cos(theta)
    c_colloc = np.interp(y_colloc, y_stations, c_stations)
    
    mu = (c_colloc * a0) / (4 * b)
    odd_n = 2 * i_indices - 1
    
    K = np.zeros((num_terms, num_terms))
    R = mu * np.sin(theta) * alpha_eff_rad
    
    for i in range(num_terms):
        for j in range(num_terms):
            n_val = odd_n[j]
            th = theta[i]
            K[i, j] = np.sin(n_val * th) * (np.sin(th) + n_val * mu[i])
            
    A = np.linalg.solve(K, R)
    A1 = A[0]
    
    AR = b**2 / S
    CL_3D = A1 * np.pi * AR
    delta = np.sum(odd_n[1:] * (A[1:] / A1)**2)
    e = 1.0 / (1.0 + delta)
    CDi = (CL_3D**2) / (np.pi * AR * e)
    
    # 1. Fourier sirkülasyon toplamından gerçek boylamasına taşıma yükü: c(y) * Cl(y)
    theta_y = np.arccos(np.clip(2 * y_stations / b, 0.0, 1.0))
    gamma_sum = np.zeros_like(y_stations)
    for j, n_val in enumerate(odd_n):
        gamma_sum += A[j] * np.sin(n_val * theta_y)
        
    lift_distribution_actual = 4 * b * gamma_sum  # c(y) * Cl(y) [m]
    
    # Yerel Cl(y) kesit katsayısı
    c_safe = np.where(c_stations < 1e-4, 1e-4, c_stations)
    cl_dist = lift_distribution_actual / c_safe
    
    # 2. İdeal Eliptik Taşıma Dağılımı: (c * Cl)_ideal
    c0_ideal_loading = (4 * S * CL_3D) / (np.pi * b)
    lift_distribution_ideal = c0_ideal_loading * np.sqrt(np.clip(1.0 - (2 * y_stations / b)**2, 0.0, None))
    
    # Profil sürtünmesi (CDp) integrali
    cd_local = np.interp(cl_dist, polar_cl, polar_cd)
    CDp = (2 / S) * trapz(cd_local * c_stations, y_stations)
    CD_total = CDi + CDp
    
    return CL_3D, CD_total, CDi, CDp, e, delta, cl_dist, lift_distribution_actual, lift_distribution_ideal

# ==============================================================================
# 3. KANAT GEOMETRİSİ MOTORU
# ==============================================================================
def design_multi_section_wing(S, c_root, rect_fraction, n, num_sections=120):
    x_int = np.linspace(0, 1, 1000)
    integral_val = trapz((1 - x_int**n)**(1/n), x_int)
    k = rect_fraction + (1 - rect_fraction) * integral_val
    semi_span = (S / 2) / (c_root * k)
    b = 2 * semi_span
    y_rect = rect_fraction * semi_span
    
    y = np.linspace(0, semi_span, num_sections)
    c = np.zeros_like(y)
    for i, yi in enumerate(y):
        if yi <= y_rect:
            c[i] = c_root
        else:
            x_val = (yi - y_rect) / (semi_span - y_rect)
            c[i] = c_root * (1 - x_val**n)**(1/n)
            
    c_root_ideal = (4 * S) / (np.pi * b)
    c_ideal = c_root_ideal * np.sqrt(1 - (y / semi_span)**2)
    return y, c, c_ideal, b, semi_span

# ==============================================================================
# 4. GİRDİLER VE POLAR YÜKLEME
# ==============================================================================
POLAR_DOSYASI = r"C:\Users\d2808\Desktop\ATA\ATA 27\airfoils_polar\wb140-il_re300000.pol"

# Kanat isterleri
S_input = 0.729         # m^2
c_root_input = 0.34    # m
L_box = 1.50            # m (Kutu uzunluk sınırı)
w_mfg = 0.035           # Üretim cezası ağırlığı
alpha_cruise = 1.5      # Seyir hücum açısı (derece)

# Polar Dosyasını Yükle ve Analiz Et
alpha_arr, cl_arr, cd_arr = load_polar_file(POLAR_DOSYASI)
a0_real, a0L_rad, a0L_deg, cl_max_real, a_stall = extract_polar_derivatives(alpha_arr, cl_arr)

# ==============================================================================
# 5. KISITLI PLANFORM OPTİMİZASYONU
# ==============================================================================
def objective_balanced(vars):
    rect_frac, n_val = vars
    y, c, c_ideal, _, _ = design_multi_section_wing(S_input, c_root_input, rect_frac, n_val)
    rmse = np.sqrt(np.mean((c - c_ideal)**2))
    mfg_penalty = w_mfg * (1.0 - rect_frac)**2
    return rmse + mfg_penalty

def constraint_center_piece(vars):
    rect_frac, n_val = vars
    _, _, _, b, _ = design_multi_section_wing(S_input, c_root_input, rect_frac, n_val)
    return L_box - (rect_frac * b)

def constraint_tip_piece(vars):
    rect_frac, n_val = vars
    _, _, _, b, _ = design_multi_section_wing(S_input, c_root_input, rect_frac, n_val)
    return L_box - ((1.0 - rect_frac) * (b / 2.0))

constraints = [
    {'type': 'ineq', 'fun': constraint_center_piece},
    {'type': 'ineq', 'fun': constraint_tip_piece}
]

opt_res = minimize(
    objective_balanced, 
    [0.35, 2.0], 
    method='SLSQP', 
    bounds=[(0.05, 0.85), (1.3, 3.5)], 
    constraints=constraints
)
opt_rect, opt_n = opt_res.x

y_opt, c_opt, c_ideal_opt, b_opt, semi_span = design_multi_section_wing(
    S_input, c_root_input, opt_rect, opt_n
)

# Nihai Aerodinamik Hesaplama
CL_3D, CD_total, CDi, CDp, e_factor, delta_val, cl_dist, L_actual, L_ideal = solve_lifting_line_with_polar(
    b=b_opt, S=S_input, y_stations=y_opt, c_stations=c_opt, 
    alpha_geom_deg=alpha_cruise, a0=a0_real, alpha_0L_rad=a0L_rad, 
    polar_cl=cl_arr, polar_cd=cd_arr
)

L_orta = opt_rect * b_opt
L_uc = (1.0 - opt_rect) * semi_span

# ==============================================================================
# 6. DETAYLI ÇIKTILAR
# ==============================================================================
print("=== YÜKLENEN POLAR VE AIRFOIL ÖZELLİKLERİ ===")
print(f"Kaynak Dosya                   : {POLAR_DOSYASI}")
print(f"Çıkarılan Taşıma Eğimi (a0)    : {a0_real:.3f} 1/rad (Teorik 2π = {2*np.pi:.3f} yerine)")
print(f"Sıfır Taşıma Açısı (alpha_0L)  : {a0L_deg:.2f}°")
print(f"Airfoil Cl_max / Stall Açısı   : {cl_max_real:.2f} @ {a_stall:.1f}°")

print("\n=== OPTİMİZASYON VE 3B KANAT AERODİNAMİĞİ ===")
print(f"Optimum Kanat Açıklığı (b)     : {b_opt:.3f} m (AR = {(b_opt**2 / S_input):.2f})")
print(f"Optimum Dikdörtgen Oranı       : %{opt_rect * 100:.1f} (Elips Üssü n: {opt_n:.3f})")
print(f"Oswald Faktörü (e)             : {e_factor:.4f} (Fourier delta: {delta_val:.5f})")
print(f"Seyir Açısı ({alpha_cruise}°) için Toplam CL : {CL_3D:.4f}")
print(f"Toplam Sürükleme (CD)          : {CD_total:.5f} (CDi = {CDi:.5f}, CDp = {CDp:.5f})")
print(f"Aerodinamik Verim (L/D)        : {(CL_3D / CD_total):.2f}")
print(f"Maksimum Yerel Cl Değeri       : {np.max(cl_dist):.3f} (Stall Marjı: %{((cl_max_real - np.max(cl_dist))/cl_max_real)*100:.1f})")

print("\n=== KUTU BOYUTU ANALİZİ ===")
print(f"Kutu Sınırı                    : {L_box:.3f} m")
print(f"Orta Parça Boyu                : {L_orta:.3f} m  -> {'[UYGUN]' if L_orta <= L_box else '[SIĞMIYOR]'}")
print(f"Uç Parça Boyu                  : {L_uc:.3f} m  -> {'[UYGUN]' if L_uc <= L_box else '[SIĞMIYOR]'}")

# ==============================================================================
# 7. GÖRSELLEŞTİRME (3 PANELLİ DETAYLI ANALİZ)
# ==============================================================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

# --- PANEL 1: KANAT PLANFORMU (Standart c/4 Çizgisi Düz) ---
# Hücum kenarı c/4, Firar kenarı -3c/4
ax1.plot(y_opt, c_opt / 4, 'b-', label=f'Tasarım Geometrisi (rect={opt_rect:.2f}, n={opt_n:.2f})', linewidth=2.5)
ax1.plot(y_opt, -3 * c_opt / 4, 'b-', linewidth=2.5)
ax1.plot(y_opt, c_ideal_opt / 4, 'r--', label='İdeal Eliptik Referans', linewidth=1.5)
ax1.plot(y_opt, -3 * c_ideal_opt / 4, 'r--', linewidth=1.5)

ax1.axvline(x=opt_rect * semi_span, color='darkorange', linestyle='--', linewidth=2, 
            label=f'Parça Ayrım Hattı (y={opt_rect * semi_span:.2f}m)')
ax1.axhline(y=0, color='black', linewidth=0.5, linestyle=':')

ax1.text((opt_rect * semi_span) / 2, 0, f'ORTA PARÇA YARISI\n(Toplam Boy: {L_orta:.2f}m)', 
         ha='center', va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
ax1.text((opt_rect * semi_span) + L_uc / 2, 0, f'UÇ ELİPTİK PARÇA\n(Boy: {L_uc:.2f}m)', 
         ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

ax1.set_title(f'1. Standart Kanat Planform Geometrisi (c/4 Düz Referans)\n'
              f'Alan={S_input} m² | Açıklık={b_opt:.2f} m | L/D = {(CL_3D/CD_total):.1f} | e = {e_factor:.4f}', 
              fontsize=12, fontweight='bold')
ax1.set_ylabel('Veter / X Konumu (m)')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# --- PANEL 2: İDEAL ELİPTİK VS GERÇEK TAŞIMA DAĞILIMI (c * Cl) ---
ax2.plot(y_opt, L_ideal, 'r--', label='İdeal Eliptik Taşıma Dağılımı ($e=1.0$ Referansı)', linewidth=2.2)
ax2.plot(y_opt, L_actual, 'g-', label=f'Gerçek Taşıma Dağılımı ($c \cdot C_l$, $e={e_factor:.4f}$)', linewidth=2.5)

ax2.fill_between(y_opt, L_ideal, L_actual, color='red', alpha=0.15, 
                 label='Eliptik Dağılımdan Sapma Bölgesi (İndüklenmiş Kayıp)')

ax2.axvline(x=opt_rect * semi_span, color='darkorange', linestyle='--', linewidth=2, label='Parça Ayrım Hattı')
ax2.set_title('2. Boylamasına Taşıma Yükü Dağılımı: İdeal Eliptik vs Gerçek LLT ($c \cdot C_l$)', 
              fontsize=12, fontweight='bold')
ax2.set_xlabel('Yarı Kanat Açıklığı - y (m)')
ax2.set_ylabel('Taşıma İndeksi $c(y) \cdot C_l(y)$ (m)')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, semi_span)
ax2.set_ylim(0, max(np.max(L_ideal), np.max(L_actual)) * 1.15)

# --- PANEL 3: LOKAL Cl DAĞILIMI VE STALL EMNİYETİ ---
ax3.plot(y_opt, cl_dist, 'b-', label='Lokal Kesit Katsayısı $C_l(y)$ (Gerçek Polar Çözümü)', linewidth=2.5)
ax3.axhline(y=CL_3D, color='green', linestyle=':', label=f'Ortalama 3B $C_L$ ({CL_3D:.3f})')
ax3.axhline(y=cl_max_real, color='red', linestyle='--', label=f'Profil $C_{{l,\\max}}$ Limiti ({cl_max_real:.2f})')

ax3.axvline(x=opt_rect * semi_span, color='darkorange', linestyle='--', linewidth=2, label='Parça Ayrım Hattı')
ax3.set_title('3. Açıklık Boyunca Kesit $C_l(y)$ Dağılımı ve Airfoil Stall Sınırı', 
              fontsize=12, fontweight='bold')
ax3.set_xlabel('Yarı Kanat Açıklığı - y (m)')
ax3.set_ylabel('Lokal $C_l$')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, semi_span)
ax3.set_ylim(0, cl_max_real * 1.25)

plt.tight_layout(pad=3.0)
plt.show()