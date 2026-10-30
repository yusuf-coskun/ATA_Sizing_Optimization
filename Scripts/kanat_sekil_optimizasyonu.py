"""Kanat optimizasyonu: VLM + polar tabanli yaklasik profil duzeltmesi.

Kurulum: python -m pip install numpy scipy matplotlib aerosandbox
Calistirma: python kanat_optimizasyonu_duzeltilmis.py
Secenekler: --polar DOSYA --no-show --sections 26 --final-sections 50
            --strict-polar --maxiter 35 --output SONUC_KLASORU

Varsayimlar: simetrik kanat, sifir yan kayma, sabit yogunluk; flap etkisi
kullanici tarafindan verilen esdeger aci ve kesit Cl_max artisidir.
CL_max kritik kesit yaklasimidir; ayrilmis akis cozulmez. Sonuclar yalnizca
kanada aittir. Gercek flap polarlari ve dusuk Re verileriyle dogrulanmalidir.
API: https://aerosandbox.readthedocs.io/en/master/autoapi/aerosandbox/
     aerodynamics/aero_3D/vortex_lattice_method/index.html
"""
from pathlib import Path
from dataclasses import dataclass, replace
import argparse
import json
from datetime import datetime
import re
import warnings
import time
import threading
from contextlib import contextmanager
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
try:
    import aerosandbox as asb
except ImportError as exc:
    raise ImportError("AeroSandbox gerekli: python -m pip install aerosandbox") from exc

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
DEFAULT_POLAR = r"C:\Users\d2808\Desktop\ATA\ATA 27\airfoils_polar\sg6043-il_re300000.pol"


_START_TIME = time.perf_counter()
_PROGRESS = False
_VLM_CALLS = 0
_DEADLINE = None
_LOG_PATH = None
_OUTPUT_PATH = None
_BEST_EXPORT = None


class ComputationStopped(RuntimeError):
    """Hesap butcesi veya basarisiz optimizasyon nedeniyle kontrollu durus."""


def check_budget():
    if _DEADLINE is not None and time.perf_counter() >= _DEADLINE:
        raise ComputationStopped("Toplam sure sinirina ulasildi. Optimum sonuc elde edilmis sayilmiyor.")


def progress(message):
    if _PROGRESS:
        line = f"[{time.perf_counter()-_START_TIME:7.1f} s] {message}"
        print(line, flush=True)
        if _LOG_PATH is not None:
            with _LOG_PATH.open("a", encoding="utf-8") as log:
                log.write(line + "\n")


@contextmanager
def calculation_status(label):
    """Uzun tek cozumlerde de terminale yasam sinyali yaz."""
    started = time.perf_counter()
    stopped = threading.Event()
    def heartbeat():
        while not stopped.wait(15):
            progress(f"{label}: hesap suruyor ({time.perf_counter()-started:.0f} s).")
    worker = threading.Thread(target=heartbeat, daemon=True) if _PROGRESS else None
    if worker:
        worker.start()
    try:
        yield
    finally:
        stopped.set()
        if worker:
            worker.join(timeout=0.2)
        if time.perf_counter()-started >= 5:
            progress(f"{label}: sona erdi ({time.perf_counter()-started:.1f} s).")


@dataclass(frozen=True)
class Config:
    S: float = 0.8306
    c_root: float = 0.41
    box: float = 1.50
    mass: float = 9.5
    cruise_speed: float = 20.0
    takeoff_speed: float = 13.0
    speed_margin: float = 1.15
    rho: float = 1.225
    mu: float = 1.81e-5
    flap_delta_cl: float = 0.80
    flap_alpha: float = 6.0
    flap_usage: float = 0.95
    chordwise: int = 4

    @property
    def stall_speed(self):
        return self.takeoff_speed / self.speed_margin

    def cl_required(self, speed):
        return self.mass * 9.81 / (0.5 * self.rho * speed**2 * self.S)

    def reynolds(self, speed, chord):
        return self.rho * speed * np.asarray(chord) / self.mu


def load_polar(path):
    """XFOIL basligindan Re oku; uzantisiz dosyalari da destekle."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    match = re.search(r"\bRe\s*=\s*(" + number + r")(?:\s*[eEdD]\s*([+-]?\d+))?", text)
    if match:
        reynolds = float(match[1]) * 10 ** int(match[2] or 0)
    else:
        match = re.search(r"re[_-]?(\d+(?:\.\d+)?(?:e[+-]?\d+)?)", Path(path).name, re.I)
        if not match:
            raise ValueError(f"Re bulunamadi: {path}")
        reynolds = float(match[1])
        warnings.warn(f"{Path(path).name}: Re baslikta yok, dosya adi kullanildi.")
    if not np.isfinite(reynolds) or reynolds <= 0:
        raise ValueError(f"Gecersiz Reynolds: {path}")
    profile = re.search(r"Calculated polar for:\s*([^\r\n]+)", text)
    rows = []
    started = False
    for line in text.splitlines():
        if re.match(r"\s*alpha\s+CL\s+CD\b", line, re.I):
            started = True
            continue
        if not started:
            continue
        try:
            values = [float(s.replace("D", "E")) for s in line.split()[:3]]
        except ValueError:
            continue
        if len(values) == 3 and np.all(np.isfinite(values)) and values[2] > 0:
            rows.append(values)
    if len(rows) < 5:
        raise ValueError(f"XFOIL alpha/CL/CD tablosunda en az 5 gecerli satir gerekli: {path}")
    data = np.asarray(rows)
    alpha, indices = np.unique(data[:, 0], return_index=True)
    data = data[indices]
    linear = (alpha >= -1) & (alpha <= 4.5)
    if np.count_nonzero(linear) < 3:
        raise ValueError(f"[-1, 4.5] derece araliginda en az 3 nokta gerekli: {path}")
    slope, intercept = np.polyfit(np.radians(alpha[linear]), data[linear, 1], 1)
    if slope <= 0:
        raise ValueError(f"Pozitif tasima egimi bulunamadi: {path}")
    # Ilk belirgin yerel tepeyi kullan; post-stall toparlanmasini Cl_max sayma.
    cl = data[:, 1]
    peak = int(np.argmax(cl))
    for i in range(1, len(cl) - 2):
        if alpha[i] >= 4 and cl[i] > cl[i-1] and np.all(cl[i+1:i+3] < cl[i]):
            peak = i
            break
    return dict(path=str(path), re=reynolds, alpha=alpha, cl=cl, cd=data[:, 2],
                a0=slope, a0L=np.degrees(-intercept/slope), clmax=float(cl[peak]),
                peak_at_end=(peak == len(cl)-1),
                profile=profile[1].strip() if profile else None)


class PolarDatabase:
    """Her polar kendi alpha ekseninde sorgulanir; Re arasinda lineer enterpolasyon.

    Veri disinda varsayilan sinir polari kullanilir ve tanilama raporlanir.
    Sentetik Re polari uretilmez; --strict-polar veri disi nihai sonucu reddeder.
    """
    def __init__(self, primary):
        primary = Path(primary)
        reference = load_polar(primary)
        prefix = re.split(r"re[_-]?\d", primary.name, flags=re.I)[0]
        polars = {reference["re"]: reference}
        for path in sorted(primary.parent.iterdir()):
            if not path.is_file() or path == primary or not path.name.lower().startswith(prefix.lower()):
                continue
            try:
                polar = load_polar(path)
            except ValueError as exc:
                warnings.warn(str(exc))
                continue
            if reference["profile"] and polar["profile"] != reference["profile"]:
                warnings.warn(f"Farkli profil atlandi: {path.name}")
                continue
            if polar["re"] in polars:
                warnings.warn(f"Ayni Re icin ikinci dosya atlandi: {path.name}")
                continue
            polars[polar["re"]] = polar
        self.polars = [polars[r] for r in sorted(polars)]
        self.re = np.array(sorted(polars))
        for p in self.polars:
            if p["peak_at_end"]:
                warnings.warn(f"{Path(p['path']).name}: polar tepeye ulasmadan bitmis olabilir; Cl_max alt sinir tahmini.")

    def properties(self, reynolds):
        return {key: np.interp(reynolds, self.re, [p[key] for p in self.polars])
                for key in ("a0", "a0L", "clmax")}

    def outside_re(self, reynolds):
        return (reynolds < self.re[0]) | (reynolds > self.re[-1])

    def evaluate(self, reynolds, alpha):
        reynolds, alpha = np.broadcast_arrays(reynolds, alpha)
        cl, cd = np.empty_like(alpha, dtype=float), np.empty_like(alpha, dtype=float)
        outside = np.zeros(alpha.shape, dtype=bool)
        for idx in np.ndindex(alpha.shape):
            r, a = float(reynolds[idx]), float(alpha[idx])
            hi = int(np.searchsorted(self.re, r, side="right"))
            if hi == 0:
                lo = hi = 0
            elif hi == len(self.re):
                lo = hi = len(self.re)-1
            else:
                lo = hi-1
            weight = 0.0 if hi == lo else (r-self.re[lo])/(self.re[hi]-self.re[lo])
            result = np.zeros(2)
            for j, w in ((lo, 1-weight), (hi, weight)):
                if w == 0:
                    continue
                p = self.polars[j]
                outside[idx] |= a < p["alpha"][0] or a > p["alpha"][-1]
                result += w * np.array([np.interp(a, p["alpha"], p[key]) for key in ("cl", "cd")])
            cl[idx], cd[idx] = result
        return cl, cd, outside


def design_multi_section_wing(S, c_root, rect_fraction, n=2.0,
                              tip_washout_deg=3.5, num_sections=26):
    if not (S > 0 and c_root > 0 and 0 < rect_fraction < 1 and n > 0 and num_sections >= 8):
        raise ValueError("Gecersiz kanat geometri parametreleri.")
    # Ayrim noktasini agda tam temsil et. Gercek sonlu uc veteri alana dahil.
    eta = np.unique(np.r_[np.linspace(0, 1, num_sections), rect_fraction])
    x = np.clip((eta-rect_fraction)/(1-rect_fraction), 0, 1)
    chord = np.maximum(c_root*np.sqrt(np.maximum(1-x**n, 0)), 0.008)
    semi_span = S/(2*trapz(chord, eta))
    y = eta*semi_span
    twist = np.where(eta <= rect_fraction,
                     -0.8*eta/rect_fraction,
                     -0.8-(tip_washout_deg-0.8)*x)
    b = 2*semi_span
    ideal = 4*S/(np.pi*b)*np.sqrt(np.maximum(1-eta**2, 0))
    return y, chord, ideal, b, semi_span, twist



def planar_wake_drag(y_edges, loading, CL, S):
    """VLM serit yuklerinden esdeger DUZLEMSEL iz/Fourier surukleme tahmini.

    Yeni bir LLT geometri cozumu yapilmaz. loading=2*Gamma/V=c*Cl,
    Gamma=2*b*V*sum(A_n*sin(n*theta)), y=(b/2)*cos(theta).
    CL=pi*AR*A1; CDi=pi*AR*sum(n*A_n**2).
    Bu, bukulmus gercek 3B izin tam Trefftz integrali DEGILDIR.
    Sonlu Fourier aciliminin uyum ve mod hassasiyeti ayrica denetlenir.
    Kaynak: https://pressbooks.lib.vt.edu/configurationaerodynamics/
             chapter/fundamentals-of-aerodynamic-drag/
    """
    y_edges, loading = np.asarray(y_edges, dtype=float), np.asarray(loading, dtype=float)
    if (y_edges.ndim != 1 or loading.shape != (len(y_edges)-1,) or len(loading) < 6
            or not np.all(np.isfinite(y_edges)) or not np.all(np.isfinite(loading))
            or not np.isfinite(CL) or not np.isfinite(S) or S <= 0
            or abs(y_edges[0]) > 1e-12 or np.any(np.diff(y_edges) <= 0)):
        raise ValueError("Duzlemsel iz icin kokten uca en az 6 gecerli serit gerekli.")
    b, dy = 2*y_edges[-1], np.diff(y_edges)
    AR = b*b/S
    integrated_CL = 2*np.sum(loading*dy)/S
    if not np.isclose(integrated_CL, CL, atol=1e-7, rtol=1e-5):
        raise ValueError("VLM yuk integrali ile CL tutarsiz; Fourier hesabina gecilemez.")
    theta = np.arccos(np.clip(2*y_edges/b, 0, 1))
    weight = np.sqrt(dy/y_edges[-1])
    target = loading/(4*b)
    A1 = CL/(np.pi*AR)

    def fit(count):
        odd = np.arange(1, 2*count, 2)
        # Serit ortalamalari: sin(n*theta) bazinin y boyunca ANALITIK integrali.
        # Boylece ilk harmonik toplam CL'yi tam korur, yuksek tek harmoniklerin
        # toplam tasimasi sifirdir. Uc bolgesinde noktasal enterpolasyon yapilmaz.
        primitives = np.empty((len(theta), count))
        primitives[:, 0] = theta/2-np.sin(2*theta)/4
        for j, n in enumerate(odd[1:], start=1):
            primitives[:, j] = (np.sin((n-1)*theta)/(2*(n-1))
                                 -np.sin((n+1)*theta)/(2*(n+1)))
        basis = (b/2)*(primitives[:-1]-primitives[1:])/dy[:, None]
        matrix = basis[:, 1:]*weight[:, None]
        rhs = (target-A1*basis[:, 0])*weight
        rest, _, rank, singular = np.linalg.lstsq(matrix, rhs, rcond=1e-10)
        if rank < count-1 or singular[-1] <= 0:
            raise ValueError("Fourier baz matrisi tekil; ag/mod sayisi kontrol edilmeli.")
        A = np.r_[A1, rest]
        fit_error = np.linalg.norm((basis@A-target)*weight)/max(np.linalg.norm(target*weight), 1e-12)
        energy = float(np.sum(odd*A*A))
        return dict(CDi=float(np.pi*AR*energy),
                    e=float(A1*A1/energy) if energy > 1e-24 else np.nan,
                    fit_relative=float(fit_error), condition=float(singular[0]/singular[-1]),
                    modes=count, coefficients=A.tolist())

    low_count = min(12, max(3, len(loading)//3))
    high_count = min(len(loading)-1, low_count+3)
    low, high = fit(low_count), fit(high_count)
    high["mode_change_relative"] = abs(high["CDi"]-low["CDi"])/max(abs(high["CDi"]), 1e-12)
    high["resolved"] = bool(high["fit_relative"] <= 0.02
                            and high["mode_change_relative"] <= 0.02
                            and high["condition"] < 1e8)
    return high


def run_vlm(cfg, y, chord, twist, alpha, speed, chordwise=None):
    """Yuk: VLM; CDi ve e_span: ayni yukten duzlemsel Fourier iz modeli."""
    chordwise = cfg.chordwise if chordwise is None else chordwise
    check_budget()
    b = 2*y[-1]
    wing = asb.Wing(name="Main_Wing", symmetric=True, xsecs=[
        asb.WingXSec(xyz_le=[-0.25*float(c), float(yi), 0], chord=float(c),
                    twist=float(t), airfoil=asb.Airfoil("naca0012"))
        for yi, c, t in zip(y, chord, twist)])
    airplane = asb.Airplane(name="UAV_Wing", wings=[wing], s_ref=cfg.S,
                           b_ref=b, c_ref=cfg.S/b)
    vlm = asb.VortexLatticeMethod(airplane=airplane,
        op_point=asb.OperatingPoint(velocity=float(speed), alpha=float(alpha)),
        spanwise_resolution=1, chordwise_resolution=chordwise)
    global _VLM_CALLS
    _VLM_CALLS += 1
    if _VLM_CALLS == 1:
        progress("Ilk VLM cozumu basliyor.")
    with calculation_status(f"VLM #{_VLM_CALLS}, alpha={alpha:.3f} derece"):
        solution = vlm.run()
    check_budget()
    CL, CDi = float(solution["CL"]), float(solution["CD"])
    if not np.isfinite(CL) or not np.isfinite(CDi) or CDi < -1e-7:
        raise RuntimeError("VLM sonlu olmayan CL/CD veya negatif induklenmis surukleme verdi.")
    CDi = max(CDi, 0.0)
    centers = np.asarray(vlm.vortex_centers)[:, 1]
    gamma = np.asarray(vlm.vortex_strengths).reshape(-1)
    midpoint = (y[:-1]+y[1:])/2
    c_mid = (chord[:-1]+chord[1:])/2
    loading = np.zeros_like(midpoint)
    for j in range(len(midpoint)):
        mask = (centers > y[j]) & (centers < y[j+1])
        if not np.any(mask):
            raise RuntimeError("VLM panel seridi bulunamadi; AeroSandbox API/ag kontrol edilmeli.")
        loading[j] = 2*np.sum(gamma[mask])/speed
    integral = 2*np.sum(loading*np.diff(y))/cfg.S
    # Gamma tabanli yerel yukleri toplam ruzgar-ekseni CL ile tutarli kil.
    if abs(CL) > 1e-7:
        if abs(integral) < 1e-9 or CL*integral <= 0:
            raise RuntimeError("VLM yerel yuk ve toplam CL tutarsiz.")
        loading *= CL/integral
    e_nearfield = CL**2/(np.pi*(b*b/cfg.S)*CDi) if CDi > 1e-10 and abs(CL)>1e-7 else np.nan
    wake = planar_wake_drag(y, loading, CL, cfg.S)
    return dict(CL=CL, CDi=wake["CDi"], e=wake["e"], y=midpoint, c=c_mid,
                CD_nearfield=CDi, e_nearfield=e_nearfield, wake=wake,
                loading=loading, cl=loading/c_mid)


def trim_vlm(cfg, y, c, twist, target, speed, alpha_start=3.0):
    zero = run_vlm(cfg, y, c, twist, alpha_start, speed)
    if abs(target-zero["CL"]) < 2e-4:
        return alpha_start, zero
    one = run_vlm(cfg, y, c, twist, alpha_start+1, speed)
    slope = one["CL"]-zero["CL"]
    if slope <= 1e-4:
        raise RuntimeError("Trim icin pozitif CL-alpha egimi bulunamadi.")
    alpha, result = alpha_start, zero
    for _ in range(10):
        error = target-result["CL"]
        if abs(error) < 2e-4:
            return alpha, result
        alpha += np.clip(error/slope, -4, 4)
        if not -15 <= alpha <= 25:
            raise RuntimeError("Trim acisi modelin izin verilen araligini asti.")
        result = run_vlm(cfg, y, c, twist, alpha, speed)
    raise RuntimeError("CL trim yakinşamadi.")


def solve_cruise(cfg, db, geometry, max_iter=30):
    y, c, _, b, _, geom_twist = geometry
    ym, cm = (y[:-1]+y[1:])/2, (c[:-1]+c[1:])/2
    reynolds = cfg.reynolds(cfg.cruise_speed, cm)
    properties = db.properties(reynolds)
    a0L_nodes = db.properties(cfg.reynolds(cfg.cruise_speed, c))["a0L"]
    correction = np.zeros_like(y)
    alpha = 3.0
    residual = np.inf
    target = cfg.cl_required(cfg.cruise_speed)
    best_residual, stale = np.inf, 0
    stop_reason = "polar_iteration_limit"
    for iteration in range(max_iter):
        progress(f"Seyir polar iterasyonu {iteration+1}/{max_iter}; trim hesaplaniyor.")
        twist = geom_twist-a0L_nodes+correction
        alpha, result = trim_vlm(cfg, y, c, twist, target, cfg.cruise_speed, alpha)
        # Ince profil esdeger VLM acisindan yalnizca yapay polar duzeltmesini cikar.
        # Geometrik washout fiziksel acinin parcasidir, burada tekrar cikarilmaz.
        alpha_eff = (np.degrees(result["cl"]/(2*np.pi))
                     + np.interp(ym, y, a0L_nodes)-np.interp(ym, y, correction))
        cl_polar, cd, outside_alpha = db.evaluate(reynolds, alpha_eff)
        difference = cl_polar-result["cl"]
        residual = float(np.max(np.abs(difference)))
        progress(f"  CL={result['CL']:.5f}, polar hatasi={residual:.5f} (hedef <0.004)")
        if residual < 0.004:
            stop_reason = "converged"
            break
        if residual < best_residual - 1e-5:
            best_residual, stale = residual, 0
        else:
            stale += 1
        if stale >= 6:
            stop_reason = "polar_stagnation"
            progress("Polar hatasi 6 adimdir iyilesmiyor; bu aday yakinşamadi, ic dongu durduruldu.")
            break
        if iteration < max_iter-1:
            delta = np.degrees(difference/(2*np.pi))
            correction += 0.4*np.interp(y, ym, delta)
    CDp = float(2*np.sum(cd*cm*np.diff(y))/cfg.S)
    result.update(alpha=alpha, CDp=CDp, CD_total=result["CDi"]+CDp,
                  re=reynolds, clmax=properties["clmax"], alpha_eff=alpha_eff,
                  outside_re=db.outside_re(reynolds), outside_alpha=outside_alpha,
                  converged=residual < 0.004, residual=residual, iterations=iteration+1, stop_reason=stop_reason)
    return result


def critical_section(alpha1, alpha2, cl1, cl2, limits):
    slopes = (cl2-cl1)/(alpha2-alpha1)
    if np.any(slopes <= 1e-6):
        raise RuntimeError("Stall hesabinda pozitif olmayan yerel tasima egimi.")
    angles = alpha1+(limits-cl1)/slopes
    idx = int(np.argmin(angles))
    alpha = float(angles[idx])
    distribution = cl1+slopes*(alpha-alpha1)
    return alpha, distribution, idx


def solve_takeoff(cfg, db, geometry):
    y, c, _, b, semi, twist = geometry
    ym, cm = (y[:-1]+y[1:])/2, (c[:-1]+c[1:])/2
    # Istenen stall hizindaki Reynolds ile kapasiteyi hesapla.
    speed = cfg.stall_speed
    reynolds = cfg.reynolds(speed, cm)
    props = db.properties(reynolds)
    a0L = db.properties(cfg.reynolds(speed, c))["a0L"]
    rect_end = y[np.argmin(np.abs(twist+0.8))]
    flap_end = cfg.flap_usage*rect_end
    # Yalnizca fiziksel flap sinirini yumusat; 3B yayilim VLM tarafindan hesaplanir.
    transition = max(0.01*semi, 1e-6)
    weights = np.clip((flap_end+transition/2-y)/transition, 0, 1)
    mid_weights = np.interp(ym, y, weights)
    answer = {"y":ym, "c":cm, "re":reynolds, "outside_re":db.outside_re(reynolds)}
    for name, flap in (("clean", False), ("flapped", True)):
        progress(f"Stall analizi: {name}")
        effective_twist = twist-a0L+(weights*cfg.flap_alpha if flap else 0)
        low = run_vlm(cfg, y, c, effective_twist, 2, speed)
        high = run_vlm(cfg, y, c, effective_twist, 10, speed)
        limits = props["clmax"]+(mid_weights*cfg.flap_delta_cl if flap else 0)
        alpha, distribution, idx = critical_section(2, 10, low["cl"], high["cl"], limits)
        CLmax = float(2*np.sum(distribution*cm*np.diff(y))/cfg.S)
        answer[name] = dict(alpha=alpha, cl=distribution, limits=limits,
                           index=idx, y_crit=float(ym[idx]), CLmax=CLmax,
                           region="FLAP" if ym[idx] <= flap_end else "FLAP DISI",
                           extrapolated_angle=not 2 <= alpha <= 10)
    return answer


def evaluate_design(x, cfg, db, sections):
    geometry = design_multi_section_wing(cfg.S, cfg.c_root, float(x[0]),
                 tip_washout_deg=float(x[1]), num_sections=sections)
    cruise = solve_cruise(cfg, db, geometry)
    takeoff = solve_takeoff(cfg, db, geometry)
    b = geometry[3]
    constraints = np.array([
        cfg.box-x[0]*b,
        cfg.box-(1-x[0])*b/2,
        takeoff["flapped"]["CLmax"]-cfg.cl_required(cfg.stall_speed),
        np.min(0.85*cruise["clmax"]-cruise["cl"]),
        0.004-cruise["residual"],
        0.02-cruise["wake"]["fit_relative"],
        0.02-cruise["wake"]["mode_change_relative"],
    ])
    return geometry, cruise, takeoff, constraints



def wing_lengths(x, geometry):
    """Aciklik yonundeki parca boylari; iki uc birbirine eklenmez."""
    b, semi = float(geometry[3]), float(geometry[4])
    rect_half = float(x[0])*semi
    return dict(span_m=b, semi_span_m=semi,
                rectangular_half_m=rect_half, rectangular_full_m=2*rect_half,
                elliptical_tip_each_m=semi-rect_half)


def save_best_outputs(context):
    """Tamamlanan en iyi adayi bellekteki verilerden kaydet; yeni VLM CALISTIRMA."""
    if context is None or context.get("evaluation") is None:
        progress("Tamamlanmis aday yok; en iyi aday grafigi/raporu uretilmedi.")
        return
    cfg, x = context["cfg"], context["x"]
    geometry, cruise, takeoff, margins = context["evaluation"]
    destination = context["destination"]
    destination.mkdir(parents=True, exist_ok=True)
    lengths = wing_lengths(x, geometry)
    feasible = bool(context["candidate"]["model_feasible"])
    label = "MODEL KISITLARI SAGLANIYOR" if feasible else "KISITLARIN TAMAMI SAGLANMIYOR"
    outside = bool(context["candidate"]["has_outside_polar"])
    names = ["Orta parca kutu payi (m)", "Tek uc parca kutu payi (m)",
             "Kalkis CLmax payi", "Seyir yerel Cl emniyet payi",
             "Polar yakinsama payi", "Fourier uyum payi", "Fourier mod hassasiyet payi"]
    constraints = [dict(name=names[i] if i<len(names) else f"Kisit {i+1}",
                        margin=float(v), passed=bool(v >= -1e-4)) for i,v in enumerate(margins)]
    q = 0.5*cfg.rho*cfg.cruise_speed**2
    def number(v):
        return float(v) if np.isfinite(v) else None
    report = dict(
        title="En iyi tamamlanan aday", created_at=datetime.now().astimezone().isoformat(),
        status=label, completed_candidates=context["completed"],
        selected_candidate_number=context["selected_number"],
        selection_rule="Once model kisitlarini saglayanlar arasinda en dusuk CD; yoksa en az toplam kisit ihlali, sonra CD.",
        grid_note=context.get("grid_note","Aday tarama agi. Bu rapor tek basina son ag dogrulamasi veya global optimum kaniti degildir."),
        mesh_check_passed=context.get("mesh_check_passed"),
        config=cfg.__dict__, polar_files=context["polar_files"],
        rect_fraction=float(x[0]), tip_washout_deg=float(x[1]), geometry=lengths,
        area_m2=cfg.S, root_chord_m=cfg.c_root, aspect_ratio=lengths["span_m"]**2/cfg.S,
        cruise=dict(alpha_deg=float(cruise["alpha"]), CL=float(cruise["CL"]),
                    CL_target=cfg.cl_required(cfg.cruise_speed), CD_total=float(cruise["CD_total"]),
                    CDi=float(cruise["CDi"]), CDp=float(cruise["CDp"]),
                    e_span=number(cruise["e"]), e_nearfield_diagnostic=number(cruise["e_nearfield"]),
                    lift_drag=number(cruise["CL"]/cruise["CD_total"]),
                    lift_N=q*cfg.S*float(cruise["CL"]), drag_N=q*cfg.S*float(cruise["CD_total"])),
        takeoff=dict(CLmax_clean=float(takeoff["clean"]["CLmax"]),
                     CLmax_flapped=float(takeoff["flapped"]["CLmax"]),
                     CLmax_required=cfg.cl_required(cfg.stall_speed),
                     alpha_stall_deg=float(takeoff["flapped"]["alpha"]),
                     critical_y_m=float(takeoff["flapped"]["y_crit"])),
        validation=dict(model_feasible=feasible, has_outside_polar=outside,
                        cruise_converged=bool(cruise["converged"]),
                        polar_residual=float(cruise["residual"]),
                        wake=cruise["wake"], constraints=constraints),
        actual_span_strips=len(geometry[0])-1, chordwise_panels=cfg.chordwise)
    # Veri raporlari once yazilir; grafik ortami sorunlu olsa da sayilar korunur.
    (destination/"en_iyi_aday.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["EN İYİ TAMAMLANAN ADAY", "="*72,
             f"Durum                 : {label}",
             f"Tamamlanan aday       : {context['completed']}",
             f"Seçilen aday numarası : {context['selected_number']}",
             f"Kayıt zamanı          : {report['created_at']}",
             "", "GEOMETRİ — AÇIKLIK YÖNÜNDEKİ UZUNLUKLAR", "-"*72]
    def row(name, value, unit="", digits=4):
        text = "tanımsız" if value is None else f"{value:.{digits}f}"
        lines.append(f"{name:<42}: {text} {unit}".rstrip())
    row("Toplam kanat açıklığı", lengths["span_m"], "m")
    row("Yarı kanat açıklığı", lengths["semi_span_m"], "m")
    row("Orta dikdörtgen parça — TAM (sol+sağ)", lengths["rectangular_full_m"], "m")
    row("Dikdörtgen bölge — bir yarı kanatta", lengths["rectangular_half_m"], "m")
    row("Eliptik uç parça — HER BİRİ", lengths["elliptical_tip_each_m"], "m")
    row("Alan", cfg.S, "m²")
    row("Kök veter", cfg.c_root, "m")
    row("Açıklık oranı AR", report["aspect_ratio"])
    row("Dikdörtgen oranı", 100*float(x[0]), "%",2)
    row("Uç washout büyüklüğü", float(x[1]), "derece",3)
    lines += ["", "SEYİR", "-"*72]
    for name,key,unit in [("Hız",None,"m/s"),("Trim hücum açısı","alpha_deg","derece"),
        ("CL","CL",""),("Hedef CL","CL_target",""),("CD toplam","CD_total",""),
        ("CDi — Fourier iz modeli","CDi",""),("CDp — profil","CDp",""),
        ("e_span","e_span",""),("L/D","lift_drag",""),("Kaldırma","lift_N","N"),("Sürükleme","drag_N","N")]:
        row(name, cfg.cruise_speed if key is None else report["cruise"][key],unit,5)
    lines += ["", "KALKIŞ / STALL", "-"*72]
    row("İstenen stall hızı",cfg.stall_speed,"m/s")
    for name,key,unit in [("Temiz CLmax","CLmax_clean",""),("Flaplı CLmax","CLmax_flapped",""),
        ("Gereken CLmax","CLmax_required",""),("Flaplı stall açısı","alpha_stall_deg","derece"),
        ("Kritik kesit konumu y","critical_y_m","m")]:
        row(name,report["takeoff"][key],unit)
    lines += ["", "KISITLAR VE SAYISAL KONTROLLER", "-"*72]
    for item in constraints:
        lines.append(f"{item['name']:<42}: {item['margin']:+.6f}  [{'SAĞLANDI' if item['passed'] else 'SAĞLANMADI'}]")
    lines += [f"Polar yakınsadı        : {'Evet' if cruise['converged'] else 'Hayır'}",
              f"Fourier kontrolü      : {'Geçti' if cruise['wake']['resolved'] else 'Geçmedi'}",
              f"Polar aralığı dışı    : {'VAR' if outside else 'Yok'}",
              f"Kullanılan ağ         : {report['actual_span_strips']} yarı kanat şeridi / {cfg.chordwise} veter paneli",
              "", "SEÇİM VE YORUM", "-"*72, report["selection_rule"], report["grid_note"],
              "e_span düzlemsel iz modelidir; viskoz Oswald sayısı değildir.",
              "Uygun aday yoksa bu çıktı en az ihlalli adayı gösterir; uygun tasarım anlamına gelmez.",
              "", "POLAR DOSYALARI", *context["polar_files"]]
    (destination/"en_iyi_aday_raporu.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    np.savetxt(destination/"en_iyi_aday_geometri.csv",
               np.column_stack((geometry[0],geometry[1],geometry[5])), delimiter=",",
               header="y_m,chord_m,geometric_twist_deg",comments="")
    caption = (f"EN İYİ TAMAMLANAN ADAY — {context['completed']} değerlendirme\n"
               f"{label} | " + ("İnce ağ kontrolü: " + ("GEÇTİ" if context["mesh_check_passed"] else "GEÇMEDİ")
                                 if "mesh_check_passed" in context else "Aday tarama ağı; son ağ kontrolü değildir"))
    plot_results(cfg,x,geometry,cruise,takeoff,destination,False,
                 filename="en_iyi_aday.png",caption=caption)
    progress(f"En iyi aday grafiği: {destination / 'en_iyi_aday.png'}")
    progress(f"En iyi aday raporu: {destination / 'en_iyi_aday_raporu.txt'}")


def plot_results(cfg, x, geometry, cruise, takeoff, destination, show,
                 filename="kanat_analizi.png", caption=None):
    y, c, ideal, b, semi, twist = geometry
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))
    if caption:
        fig.suptitle(caption, fontsize=12, fontweight="bold", y=0.99)
    ax = axes[0]
    for chord, style, label in ((c, "b-", "Aday kanat"), (ideal, "r--", "Eliptik referans")):
        ax.plot(y, chord/4, style, label=label)
        ax.plot(y, -3*chord/4, style)
    flap = y <= cfg.flap_usage*x[0]*semi
    ax.fill_between(y[flap], -0.75*c[flap], -0.50*c[flap], color="limegreen", alpha=0.4,
                    label="Flap konumu (veter orani sematik)")
    lengths = wing_lengths(x,geometry)
    rect_half = lengths["rectangular_half_m"]
    ax.axvline(rect_half, color="darkorange", ls="--", label="Parça ayrımı")
    max_c = float(max(np.max(c),np.max(ideal)))
    dimension_y = -0.75*max_c-0.20*max_c
    for left,right,label,color in (
        (0,rect_half,f"Dikdörtgen (yarım)\n{rect_half:.3f} m","#9a5800"),
        (rect_half,semi,f"Eliptik uç (tek parça)\n{lengths['elliptical_tip_each_m']:.3f} m","#075985")):
        ax.annotate("",xy=(right,dimension_y),xytext=(left,dimension_y),
                    arrowprops=dict(arrowstyle="<->",color=color,lw=1.5))
        ax.text((left+right)/2,dimension_y-0.08*max_c,label,
                ha="center",va="top",fontsize=9,color=color,
                bbox=dict(facecolor="white",edgecolor="none",alpha=0.9,pad=2))
    ax.text(0,0.52*max_c,
            f"Orta dikdörtgen parça TAM (sol + sağ): {lengths['rectangular_full_m']:.3f} m\n"
            f"Yarı açıklık: {semi:.3f} m  |  Toplam açıklık: {b:.3f} m",
            fontsize=9,ha="left",va="center",
            bbox=dict(boxstyle="round,pad=0.45",facecolor="#f1f5f9",edgecolor="#cbd5e1"))
    ax.set_title(f"Kanat | b={b:.3f} m | S={cfg.S:.3f} m² | washout={x[1]:.2f}°")
    ax.set_ylabel("Veter yonu (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.025*semi,1.025*semi)
    ax.set_ylim(dimension_y-0.38*max_c,0.82*max_c)
    ax.legend(loc="upper right",fontsize=8)
    ax = axes[1]
    ideal_loading = 4*cfg.S*cruise["CL"]/(np.pi*b)*np.sqrt(1-(cruise["y"]/semi)**2)
    ax.plot(cruise["y"], cruise["loading"], "g-", label="Seyir c·Cl")
    ax.plot(cruise["y"], ideal_loading, "r--", label="Eliptik yuk")
    ax.set_ylabel("c·Cl (m)")
    ax.set_title(f"Seyir: CL={cruise['CL']:.3f}, CD={cruise['CD_total']:.5f}, e_span={cruise['e']:.3f}")
    ax2 = ax.twinx()
    ax2.plot(cruise["y"], cruise["re"]/1000, "m-.", label="Gercek yerel Re / 1000")
    ax2.set_ylabel("Re / 1000", color="m")
    lines, labels = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines+l2, labels+lab2)
    ax = axes[2]
    flap = takeoff["flapped"]
    ax.plot(cruise["y"], cruise["cl"], "b-", label="Seyir Cl")
    ax.plot(cruise["y"], cruise["clmax"], "b--", label="Seyir Re: temiz Cl limiti")
    ax.plot(takeoff["y"], flap["cl"], "m-", label="Flapli ilk stall dagilimi")
    ax.plot(takeoff["y"], flap["limits"], "r--", label="Stall Re: flapli Cl limiti")
    ax.plot(flap["y_crit"], flap["limits"][flap["index"]], "rx", ms=12, mew=3, label="Kritik kesit")
    ax.set_title(f"Kritik kesit yaklasimi | CLmax={flap['CLmax']:.3f} | alpha={flap['alpha']:.2f}°")
    ax.set_ylabel("Yerel Cl")
    ax.legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_xlabel("Yari aciklik y (m)")
    fig.tight_layout(rect=(0,0,1,0.945) if caption else (0,0,1,1), h_pad=3.0)
    fig.savefig(destination / filename, dpi=160)
    if show:
        plt.show()
    plt.close(fig)



def design_rank(evaluation):
    """Mevcut secim olcutu: uygunluk, toplam negatif marj, surukleme."""
    _, cruise, _, margins = evaluation
    feasible = bool(np.all(margins >= -1e-4) and cruise["converged"] and cruise["wake"]["resolved"])
    return (0 if feasible else 1,
            0.0 if feasible else float(np.sum(np.maximum(-margins, 0))),
            float(cruise["CD_total"]))


def diverse_designs(samples, count, min_distance=0.18):
    """Normalize edilmis iki degiskende birbirinden uzak, iyi adaylari sec."""
    ranked = sorted(((x, value) for x,value in samples.items() if value is not None),
                    key=lambda item: design_rank(item[1]))
    selected = []
    scale = np.array([0.48-0.25, 5.0-1.0])
    for x,value in ranked:
        if all(np.linalg.norm((np.asarray(x)-np.asarray(previous[0]))/scale) >= min_distance
               for previous in selected):
            selected.append((x,value))
            if len(selected) >= count:
                break
    return selected


def grid_multistart_search(evaluate, budget, maxiter, local_starts=8, finalists=3):
    """Tum izgara once tamamlanir. Yerel erken durus diger baslangiclari iptal etmez.

    Butce yeni tasarim/ag degerlendirmelerini sayar. Her finaliste iki ek ag
    cozumu ayrilir; bir yerel arama diger baslangiclarin butcesini tuketemez.
    """
    final_count = min(finalists, max(1, budget//8)) if budget >= 3 else 0
    reserve = 2*final_count
    search_cap = budget-reserve
    side = int(np.sqrt(max(4, int(0.40*search_cap)))) if search_cap >= 4 else 0
    if side:
        grid = [(float(r),float(w)) for r in np.linspace(0.25,0.48,side)
                for w in np.linspace(1.0,5.0,side)]
        # Uzun bir tarama kesilse bile ilk noktalarda merkez ve koseler gorulsun.
        first = [(side//2)*side+side//2, 0, side-1, (side-1)*side, side*side-1]
        order = list(dict.fromkeys(first+list(range(len(grid)))))
        grid = [grid[i] for i in order]
    else:
        grid = [(0.365,3.0),(0.25,1.0),(0.48,5.0)][:search_cap]
    samples = {}
    info = dict(method="regular_grid_then_diverse_multistart", total_budget=budget,
                grid_shape=[side,side] if side else None, grid_planned=len(grid),
                grid_completed=0, fine_grid_reserve=reserve, local_runs=[],
                search_completed=False, global_optimum_proven=False)
    def sample(x):
        key = tuple(float(v) for v in x)
        if key not in samples:
            samples[key] = evaluate(np.asarray(x))
        return samples[key]
    progress(f"ASAMA 1/3: {len(grid)} nokta ile tum araligin izgara taramasi. Son aglar icin {reserve} hesap ayrildi.")
    for i,x in enumerate(grid):
        progress(f"IZGARA {i+1}/{len(grid)}")
        sample(x)
        info["grid_completed"] += 1
    if not any(v is not None for v in samples.values()):
        raise ComputationStopped("Izgarada hicbir aday hesaplanamadi; ilerleme.log icindeki cozum hatalarini kontrol edin.")
    remaining = search_cap-len(samples)
    seeds = diverse_designs(samples, min(local_starts, max(0,remaining//3))) if remaining >= 3 else []
    class LocalBudgetReached(Exception):
        pass
    progress(f"ASAMA 2/3: {len(seeds)} farkli bolgeden yerel arama; en fazla {remaining} yeni hesap.")
    for j,(seed,_) in enumerate(seeds):
        quota = (search_cap-len(samples))//(len(seeds)-j)
        start_count, iterations = len(samples), 0
        def local_sample(x):
            key = tuple(float(v) for v in x)
            if key not in samples and len(samples)-start_count >= quota:
                raise LocalBudgetReached()
            return sample(x)
        def objective(x):
            ev = local_sample(x)
            if ev is None:
                return 1e6
            cr = ev[1]
            return 100*cr["CD_total"]+100*max(cr["residual"]-0.004,0)**2
        def constraints(x):
            ev = local_sample(x)
            return np.full(7,-1e3) if ev is None else ev[3]
        def callback(x):
            nonlocal iterations
            iterations += 1
            progress(f"YEREL {j+1}/{len(seeds)}: adim={iterations}, yeni aday={len(samples)-start_count}/{quota}")
            check_budget()
        progress(f"YEREL BASLANGIC {j+1}/{len(seeds)}: rect={seed[0]:.5f}, washout={seed[1]:.4f}, butce={quota}")
        run = dict(start=list(seed), budget=quota, success=False)
        try:
            res = minimize(objective, np.asarray(seed), method="SLSQP",
                           bounds=[(0.25,0.48),(1.0,5.0)],
                           constraints=[dict(type="ineq",fun=constraints)], callback=callback,
                           options=dict(eps=1e-3,ftol=1e-5,maxiter=maxiter,disp=False))
            local_sample(res.x)
            run.update(success=bool(res.success),message=str(res.message))
        except LocalBudgetReached:
            run["message"] = "Bu baslangicin aday butcesi doldu; diger baslangiclara geciliyor."
        except (ValueError, np.linalg.LinAlgError) as exc:
            run["message"] = f"Yerel cozum hatasi: {exc}; diger baslangiclara geciliyor."
        run.update(new_evaluations=len(samples)-start_count,iterations=iterations)
        info["local_runs"].append(run)
        progress(f"YEREL {j+1} TAMAMLANDI: {run['message']}")
    chosen = diverse_designs(samples,final_count,min_distance=0.08) if final_count else []
    info.update(search_completed=True, search_evaluations=len(samples),
                failed_evaluations=sum(v is None for v in samples.values()),
                finalists=[list(x) for x,_ in chosen])
    return chosen,info


def compare_grids(coarse, span_fine, fine, args):
    def difference(first,second):
        return dict(CDi_relative=float(abs(first[1]["CDi"]-second[1]["CDi"])/max(abs(second[1]["CDi"]),1e-12)),
                    e_span_absolute=float(abs(first[1]["e"]-second[1]["e"])),
                    CLmax_absolute=float(abs(first[2]["flapped"]["CLmax"]-second[2]["flapped"]["CLmax"])))
    change = dict(span=difference(coarse,span_fine),chord=difference(span_fine,fine))
    refined = args.final_sections > args.sections and args.final_chordwise > args.chordwise
    ok = bool(refined and all(d["CDi_relative"]<=0.02 and d["e_span_absolute"]<=0.02
                             and d["CLmax_absolute"]<=0.03 for d in change.values())
              and all(ev[1]["wake"]["resolved"] for ev in (coarse,span_fine,fine)))
    return change,ok


def ask_candidate_limit(default=10000):
    while True:
        try:
            answer = input(f"En fazla kaç aday denensin? [Enter = {default}]: ").strip()
        except EOFError:
            print(f"Girdi okunamadı; varsayılan {default} aday kullanılacak.", flush=True)
            return default
        if not answer:
            return default
        try:
            count = int(answer)
        except ValueError:
            count = 0
        if count > 0:
            return count
        print("Lütfen 1 veya daha büyük bir tam sayı girin.", flush=True)


def run_main():
    global _PROGRESS, _START_TIME, _DEADLINE, _LOG_PATH, _OUTPUT_PATH, _VLM_CALLS, _BEST_EXPORT
    _PROGRESS = True
    _START_TIME = time.perf_counter()
    _VLM_CALLS = 0
    _BEST_EXPORT = None
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--polar", default=DEFAULT_POLAR)
    parser.add_argument("--sections", type=int, default=26)
    parser.add_argument("--final-sections", type=int, default=50)
    parser.add_argument("--chordwise", type=int, default=4)
    parser.add_argument("--final-chordwise", type=int, default=8)
    parser.add_argument("--check-design", type=float, nargs=2, metavar=("RECT", "WASHOUT"),
                        help="Optimizasyon yapmadan tek tasarimi uc agda denetle")
    parser.add_argument("--local-starts", type=int, default=8)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=None,
                        help="Ana adim siniri; verilmezse aday butcesi kadar")
    parser.add_argument("--max-evals", type=int, default=None,
                        help="Aday ust siniri; verilmezse program baslangicta sorar (Enter: 10000)")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Istege bagli sure siniri; varsayilan sure siniri yok")
    parser.add_argument("--diagnose", action="store_true", help="Yalnizca bir baslangic adayi hesapla")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--strict-polar", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "kanat_sonuclari")
    args = parser.parse_args()
    if args.max_evals is None:
        # Tek tasarim modlarinda aday taramasi yapilmaz; gereksiz soru sorma.
        args.max_evals = 1 if args.diagnose else (3 if args.check_design else ask_candidate_limit())
    if args.maxiter is None:
        args.maxiter = args.max_evals
    # Kullanici cevap verirken gecen sureyi hesap butcesinden dusme.
    _START_TIME = time.perf_counter()
    if (args.max_evals < 1
            or (args.max_seconds is not None and (args.max_seconds <= 0 or not np.isfinite(args.max_seconds)))
            or args.maxiter < 1 or args.local_starts < 1 or args.finalists < 1
            or min(args.sections, args.final_sections) < 8
            or min(args.chordwise, args.final_chordwise) < 1):
        parser.error("Butceler pozitif, kesit sayilari en az 8 olmali.")
    if args.check_design and (args.diagnose or not 0.25 <= args.check_design[0] <= 0.48
                              or not 1 <= args.check_design[1] <= 5):
        parser.error("--check-design icin rect=[0.25,0.48], washout=[1,5]; --diagnose ile birlikte kullanilamaz.")
    if args.check_design and args.max_evals < 3:
        parser.error("--check-design uc ag icin en az --max-evals 3 gerektirir.")
    _OUTPUT_PATH = args.output
    args.output.mkdir(parents=True, exist_ok=True)
    _LOG_PATH = args.output / "ilerleme.log"
    _LOG_PATH.write_text("", encoding="utf-8")
    _DEADLINE = None if args.max_seconds is None else _START_TIME + args.max_seconds
    (args.output / "calisma_durumu.json").write_text(json.dumps(
        dict(status="running", max_evals=args.max_evals, max_seconds=args.max_seconds)), encoding="utf-8")
    (args.output / "ara_sonuc.json").write_text(json.dumps(
        dict(status="running_no_completed_candidate", best_candidate=None)), encoding="utf-8")
    time_limit_text = "sure siniri yok" if args.max_seconds is None else f"{args.max_seconds:g} saniye"
    progress(f"Sinirlar: {args.max_evals} aday, {time_limit_text}, {args.maxiter} ana adim.")
    if args.max_seconds is not None:
        progress("Sure denetimi VLM cagrilari arasindadir; devam eden tek cozumun bitmesi beklenebilir.")
    cfg, db = replace(Config(), chordwise=args.chordwise), PolarDatabase(args.polar)
    _BEST_EXPORT = dict(cfg=cfg,destination=args.output,evaluation=None,completed=0,
                        polar_files=[item["path"] for item in db.polars])
    print("Polar Reynolds degerleri:", db.re)
    print(f"Gereken seyir CL: {cfg.cl_required(cfg.cruise_speed):.4f}")
    print(f"Stall hizi: {cfg.stall_speed:.3f} m/s; gereken CLmax: {cfg.cl_required(cfg.stall_speed):.4f}")
    print("Flap modeli: esdeger aci + kullanici tanimli Cl_max artisi.")
    if cfg.reynolds(cfg.stall_speed, cfg.c_root) < db.re[0]:
        print("UYARI: Stall kosulunda kok dahil tum kanat mevcut Re verisinin altinda.")
        if args.strict_polar:
            raise ValueError("--strict-polar: once dusuk Reynolds polarlari eklenmeli.")
    progress("Optimizasyon basliyor; her aday birden fazla VLM cozumu gerektirir. Ctrl+C ile durdurabilirsiniz.")
    cache = {}
    candidate_records = {}
    best_rank, best_candidate = None, None
    def evaluate(x, sections=None, chordwise=None, track_best=True):
        nonlocal best_rank, best_candidate
        check_budget()
        sections = args.sections if sections is None else sections
        chordwise = args.chordwise if chordwise is None else chordwise
        key = tuple(float(v) for v in x)+(int(sections),int(chordwise))
        if key not in cache:
            if len(cache) >= args.max_evals:
                raise ComputationStopped(f"{args.max_evals} aday sinirina ulasildi; optimizasyon tamamlanmadi.")
            progress(f"Aday {len(cache)+1}/{args.max_evals}: rect={x[0]:.5f}, washout={x[1]:.4f} derece, kesit={sections}, veter={chordwise}")
            try:
                cache[key] = evaluate_design(x,replace(cfg,chordwise=chordwise),db,sections)
            except ComputationStopped:
                raise
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                cache[key] = None
                _BEST_EXPORT["completed"] = sum(ev is not None for ev in cache.values())
                progress(f"ADAY HESAPLANAMADI: {exc}; diger adaylarla devam ediliyor.")
                return None
            _, cruise, takeoff, margins = cache[key]
            violation = float(np.sum(np.maximum(-margins, 0)))
            feasible = bool(np.all(margins >= -1e-4) and cruise["converged"] and cruise["wake"]["resolved"])
            candidate = dict(x=[float(v) for v in x], CD=float(cruise["CD_total"]),
                CDi=float(cruise["CDi"]), e_span=float(cruise["e"]),
                e_nearfield_diagnostic=float(cruise["e_nearfield"]) if np.isfinite(cruise["e_nearfield"]) else None,
                wake_diagnostics=cruise["wake"],
                CLmax=float(takeoff["flapped"]["CLmax"]), constraint_margins=margins.tolist(),
                model_feasible=feasible, polar_residual=float(cruise["residual"]),
                polar_stop_reason=cruise["stop_reason"],
                has_outside_polar=bool(np.any(cruise["outside_re"]) or np.any(cruise["outside_alpha"])
                                       or np.any(takeoff["outside_re"])))
            # Once uygun adaylar; uygun aday yoksa en az toplam kisit ihlali.
            rank = (0 if feasible else 1, 0.0 if feasible else violation, candidate["CD"])
            candidate_records[key] = candidate
            if track_best and (best_rank is None or rank < best_rank):
                best_rank, best_candidate = rank, candidate
                _BEST_EXPORT.update(x=np.array(x,copy=True),evaluation=cache[key],
                                    candidate=candidate,selected_number=len(cache))
            _BEST_EXPORT["completed"] = sum(ev is not None for ev in cache.values())
            checkpoint = dict(status="INTERMEDIATE_NOT_AN_OPTIMUM", completed_candidates=len(cache),
                elapsed_seconds=time.perf_counter()-_START_TIME, last_candidate=candidate,
                best_candidate=best_candidate, note="Son ag dogrulamasi henuz yapilmadi.")
            (args.output / "ara_sonuc.json").write_text(
                json.dumps(checkpoint, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            progress(f"Aday bitti: CD={cruise['CD_total']:.6f}, e_span={cruise['e']:.5f}, Fourier kontrol={cruise['wake']['resolved']}, kisitlar={margins}")
            check_budget()
        return cache[key]
    if args.diagnose:
        ev = evaluate(np.array([0.35,3.0]))
        if ev is None:
            raise ComputationStopped("Tek aday hesabi basarisiz; ilerleme.log dosyasini kontrol edin.")
        progress("Tek aday testi tamamlandi; optimizasyon yapilmadi. Ayrintilar: ara_sonuc.json")
        (args.output/"calisma_durumu.json").write_text(
            json.dumps(dict(status="diagnostic_only",optimizer_run=False)),encoding="utf-8")
        return 0
    if args.check_design:
        x = np.asarray(args.check_design)
        coarse = evaluate(x)
        if coarse is None:
            raise ComputationStopped("Secilen tasarim hesaplanamadi.")
        finalists = [(tuple(x),coarse)]
        search_info = dict(method="single_design_check",local_runs=[],search_completed=True)
    else:
        finalists,search_info = grid_multistart_search(
            evaluate,args.max_evals,args.maxiter,args.local_starts,args.finalists)
    (args.output/"arama_ozeti.json").write_text(
        json.dumps(search_info,ensure_ascii=False,indent=2),encoding="utf-8")
    if not finalists:
        raise ComputationStopped("Tarama tamamlandi; ince ag kontrolune yetecek butce yok. En iyi tarama adayi kaydediliyor.")
    progress(f"ASAMA 3/3: {len(finalists)} finalistin aciklik ve veter agi kontrolu.")
    verified = []
    validation_rows = []
    for j,(x,coarse) in enumerate(finalists):
        progress(f"FINALIST {j+1}/{len(finalists)}: rect={x[0]:.5f}, washout={x[1]:.4f}")
        span_fine = evaluate(x,sections=args.final_sections,track_best=False)
        fine = evaluate(x,sections=args.final_sections,chordwise=args.final_chordwise,track_best=False)
        if span_fine is None or fine is None:
            validation_rows.append(dict(x=list(x),accepted=False,error="Ince ag hesabi basarisiz"))
            continue
        mesh_change,mesh_ok = compare_grids(coarse,span_fine,fine,args)
        cr,to = fine[1],fine[2]
        has_outside = bool(np.any(cr["outside_re"]) or np.any(cr["outside_alpha"]) or np.any(to["outside_re"]))
        model_feasible = design_rank(fine)[0] == 0
        accepted = bool(model_feasible and mesh_ok and not(args.strict_polar and has_outside))
        validation_rows.append(dict(x=list(x),CD=float(cr["CD_total"]),e_span=float(cr["e"]),
                                    model_feasible=model_feasible,mesh_check_passed=mesh_ok,
                                    mesh_change=mesh_change,accepted=accepted))
        verified.append(dict(x=np.asarray(x),fine=fine,mesh_change=mesh_change,mesh_ok=mesh_ok,
                             has_outside=has_outside,model_feasible=model_feasible,accepted=accepted))
    search_info["final_validation"] = validation_rows
    search_info["total_evaluations"] = len(cache)
    (args.output/"arama_ozeti.json").write_text(
        json.dumps(search_info,ensure_ascii=False,indent=2),encoding="utf-8")
    if not verified:
        raise ComputationStopped("Finalistlerin ince ag hesaplari basarisiz; en iyi tamamlanan tarama adayi kaydediliyor.")
    selected = min(verified,key=lambda item:(0 if item["accepted"] else 1,*design_rank(item["fine"])))
    x = selected["x"]
    geometry,cruise,takeoff,margins = selected["fine"]
    mesh_change,mesh_ok = selected["mesh_change"],selected["mesh_ok"]
    has_outside,model_feasible,accepted = selected["has_outside"],selected["model_feasible"],selected["accepted"]
    selected_key = tuple(float(v) for v in x)+(args.final_sections,args.final_chordwise)
    _BEST_EXPORT.update(x=x,evaluation=selected["fine"],candidate=candidate_records[selected_key],
                        cfg=replace(cfg,chordwise=args.final_chordwise),
                        selected_number=list(cache).index(selected_key)+1,
                        grid_note="Finalistler ince agda karsilastirildi. Global optimum garantisi yoktur.",
                        mesh_check_passed=mesh_ok)
    result = argparse.Namespace(success=True,message="Izgara/yerel aramalar ve finalist kontrolleri tamamlandi; global optimum kaniti degildir.")
    status = ("MODEL ICIN UYGUN ADAY" if accepted else "DOGRULANMAMIS / KISITLARI SAGLAMAYAN ADAY")
    if args.check_design:
        status = "TEK TASARIM KONTROLU (OPTIMIZASYON YOK): " + status
    if has_outside:
        status += " — POLAR VERI DISI YAKLASIM ICERIYOR"
    summary = dict(status=status, optimizer_run=not bool(args.check_design),
                   optimizer_success=None if args.check_design else any(r["success"] for r in search_info["local_runs"]), optimizer_message=str(result.message),
                   search=search_info,
                   drag_method="planar_wake_fourier_from_vlm_strip_loading",
                   config=cfg.__dict__, grid_sizes=dict(span=[args.sections,args.final_sections], chord=[args.chordwise,args.final_chordwise]),
                   rect_fraction=float(x[0]), tip_washout_deg=float(x[1]), span_m=float(geometry[3]),
                   alpha_cruise_deg=float(cruise["alpha"]), CL=float(cruise["CL"]),
                   CD_total=float(cruise["CD_total"]), CDi=float(cruise["CDi"]), CDp=float(cruise["CDp"]),
                   e_span=float(cruise["e"]) if np.isfinite(cruise["e"]) else None,
                   CD_nearfield_diagnostic=float(cruise["CD_nearfield"]),
                   e_nearfield_diagnostic=float(cruise["e_nearfield"]) if np.isfinite(cruise["e_nearfield"]) else None,
                   wake_diagnostics=cruise["wake"],
                   lift_drag=float(cruise["CL"]/cruise["CD_total"]),
                   CLmax_clean=float(takeoff["clean"]["CLmax"]), CLmax_flapped=float(takeoff["flapped"]["CLmax"]),
                   CLmax_required=cfg.cl_required(cfg.stall_speed),
                   alpha_stall_deg=takeoff["flapped"]["alpha"], critical_y_m=takeoff["flapped"]["y_crit"],
                   stall_linear_angle_extrapolation=bool(takeoff["flapped"]["extrapolated_angle"]),
                   cruise_converged=bool(cruise["converged"]), cruise_residual=cruise["residual"],
                   constraint_margins=margins.tolist(), mesh_change=mesh_change, mesh_check_passed=bool(mesh_ok),
                   cruise_re_outside_count=int(np.sum(cruise["outside_re"])),
                   cruise_alpha_outside_count=int(np.sum(cruise["outside_alpha"])),
                   stall_re_outside_count=int(np.sum(takeoff["outside_re"])),
                   polar_files=[p["path"] for p in db.polars],
                   note="Kanat-only; duzlemsel Fourier iz yaklasimi, ampirik flap/kritik kesit. e_span viskoz Oswald degildir; deneysel dogrulama yoktur.")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"sonuc.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for key, value in summary.items():
        print(f"{key}: {value}")
    np.savetxt(args.output/"geometri.csv", np.column_stack((geometry[0], geometry[1], geometry[5])),
               delimiter=",", header="y_m,chord_m,geometric_twist_deg", comments="")
    # En iyi aday dosyalari GUI acilmadan once de yazilmis olsun.
    save_best_outputs(_BEST_EXPORT)
    _BEST_EXPORT["saved"] = True
    plot_results(cfg, x, geometry, cruise, takeoff, args.output, not args.no_show)
    (args.output / "calisma_durumu.json").write_text(json.dumps(
        dict(status="completed", accepted=accepted, elapsed_seconds=time.perf_counter()-_START_TIME)), encoding="utf-8")
    print("Ciktilar:", args.output.resolve())
    # Basarisiz cozum otomasyonlarda da basarili sayilmasin; tanilama dosyalari korunur.
    return 0 if accepted else 2


def main():
    try:
        return run_main()
    except (ComputationStopped, KeyboardInterrupt) as exc:
        reason = str(exc) if isinstance(exc, ComputationStopped) else "Kullanici Ctrl+C ile durdurdu."
        progress("DURDURULDU: " + reason)
        if _OUTPUT_PATH is not None:
            (_OUTPUT_PATH / "calisma_durumu.json").write_text(json.dumps(
                dict(status="stopped_without_optimum", reason=reason,
                     elapsed_seconds=time.perf_counter()-_START_TIME), ensure_ascii=False, indent=2), encoding="utf-8")
            progress(f"Son tamamlanan aday: {_OUTPUT_PATH / 'ara_sonuc.json'}")
            progress(f"Hesap gunlugu: {_OUTPUT_PATH / 'ilerleme.log'}")
            progress("Nihai optimum uretilmedi; en iyi tamamlanan aday ayrica grafik ve rapor olarak kaydedilecek.")
        return 2
    finally:
        if _BEST_EXPORT is not None and not _BEST_EXPORT.get("saved", False):
            try:
                save_best_outputs(_BEST_EXPORT)
            except Exception as export_error:
                progress(f"En iyi aday disari aktarimi tamamlanamadi: {export_error}")


if __name__ == "__main__":
    raise SystemExit(main())
