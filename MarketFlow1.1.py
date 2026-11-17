#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║       PolyMarket — Quant Markets Analysis Terminal Interface v0.2    ║
║       MARKETS PREDICTIVE MODEL FOR FINANCIAL PRODUCTS                ║
║       © 2026 Vintage AI, Casablanca                                  ║
╚══════════════════════════════════════════════════════════════════════╝

Run:  python3 SMC0.2.py
Requires: Python 3.6+  (no external dependencies)
"""

import os
import sys
import math
import json
import time
import hashlib
import getpass
import shutil
from datetime import datetime, timezone

# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def term_width():
    return shutil.get_terminal_size((80, 24)).columns

def clr():
    os.system('cls' if os.name == 'nt' else 'clear')

class C:
    GREEN  = '\033[32m'
    BGREEN = '\033[92m'
    DIM    = '\033[2m'
    AMBER  = '\033[33m'
    BAMBER = '\033[93m'
    RED    = '\033[91m'
    RESET  = '\033[0m'
    BOLD   = '\033[1m'

def g(t):   return f"{C.BGREEN}{t}{C.RESET}"
def dim(t): return f"{C.DIM}{C.GREEN}{t}{C.RESET}"
def am(t):  return f"{C.BAMBER}{t}{C.RESET}"
def rd(t):  return f"{C.RED}{t}{C.RESET}"

def println(text='', colour=None, delay=0.0):
    if colour:
        print(f"{colour}{text}{C.RESET}")
    else:
        print(text)
    if delay:
        time.sleep(delay)

def divider(char='─', width=None):
    w = width or min(term_width(), 72)
    println(dim(char * w))

def header_line(char='═', width=None):
    w = width or min(term_width(), 72)
    println(am(char * w))

def centre(text, width=None):
    w = width or min(term_width(), 72)
    println(am(text.center(w)))

def box_line(label, value, unit='', width=58):
    label_str = f"  {label}"
    value_str = f"{value} {unit}".strip()
    dots = width - len(label_str) - len(value_str)
    dots = max(1, dots)
    println(f"{C.DIM}{C.GREEN}  {label}{C.RESET}{C.GREEN}{'.' * dots}{C.RESET}{C.BAMBER}{value_str}{C.RESET}")

def highlight_line(label, value, unit='', width=58):
    label_str = f"  {label}"
    value_str = f">>> {value} {unit}".strip()
    dots = width - len(label_str) - len(value_str)
    dots = max(1, dots)
    println(f"{C.BAMBER}  {label}{'.' * dots}{value_str}{C.RESET}")

def section(title):
    println()
    println(f"  {C.BAMBER}◆ {title}{C.RESET}")
    divider('─')

def beep():
    print('\a', end='', flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# MARKETS CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------
# QUANT EQUATIONS
# -----------------------------------------------------------------------------

class PHY:
    G = 6.674e-11
    MU_SUN    = 1.32712440018e20
    MU_EARTH  = 3.986004418e14
    MU_MARS   = 4.2828e13
    AU        = 1.496e11
    R_EARTH   = 6_371_000
    R_MARS    = 3_389_500
    EARTH_SMA = 1.496e11
    MARS_SMA  = 2.279e11
    ISP_CHEM  = 450
    G0        = 9.80665
    LEO_ALT   = 400_000
    LMO_ALT   = 300_000
    C_LIGHT   = 299_792_458
    SYNODIC   = 779.9

# ─────────────────────────────────────────────────────────────────────────────
# ORBITAL MECHANICS
# ─────────────────────────────────────────────────────────────────────────────

def hohmann(r1, r2, mu):
    a   = (r1 + r2) / 2
    v1c = math.sqrt(mu / r1)
    v2c = math.sqrt(mu / r2)
    v1t = math.sqrt(mu * (2/r1 - 1/a))
    v2t = math.sqrt(mu * (2/r2 - 1/a))
    dv1 = abs(v1t - v1c)
    dv2 = abs(v2c - v2t)
    T   = math.pi * math.sqrt(a**3 / mu)
    return dict(a=a, v1c=v1c, v2c=v2c, v1t=v1t, v2t=v2t,
                dv1=dv1, dv2=dv2, dv_total=dv1+dv2, T=T)

def tsiolkovsky(dv_ms, isp, m_dry):
    ve         = isp * PHY.G0
    mass_ratio = math.exp(dv_ms / ve)
    m_prop     = m_dry * (mass_ratio - 1)
    return dict(ve=ve, ratio=mass_ratio, m_prop=m_prop, m_total=m_dry+m_prop)

def compute_mission(probe_mass_kg):
    r_LEO = PHY.R_EARTH + PHY.LEO_ALT
    r_LMO = PHY.R_MARS  + PHY.LMO_ALT

    hel = hohmann(PHY.EARTH_SMA, PHY.MARS_SMA, PHY.MU_SUN)

    v_inf_E  = hel['dv1']
    v_LEO    = math.sqrt(PHY.MU_EARTH / r_LEO)
    v_hyp_E  = math.sqrt(v_inf_E**2 + 2*PHY.MU_EARTH/r_LEO)
    dv_dep   = v_hyp_E - v_LEO

    v_inf_M  = hel['dv2']
    v_LMO    = math.sqrt(PHY.MU_MARS / r_LMO)
    v_hyp_M  = math.sqrt(v_inf_M**2 + 2*PHY.MU_MARS/r_LMO)
    dv_cap   = v_hyp_M - v_LMO

    dv_total = dv_dep + dv_cap

    prop     = tsiolkovsky(dv_total, PHY.ISP_CHEM, probe_mass_kg)

    T_days   = hel['T'] / 86400
    n_mars   = 360 / 686.97
    phase    = 180 - (n_mars * T_days)

    T_earth  = 2*math.pi * math.sqrt(PHY.EARTH_SMA**3 / PHY.MU_SUN) / 86400
    T_mars   = 2*math.pi * math.sqrt(PHY.MARS_SMA**3  / PHY.MU_SUN) / 86400
    T_xfer   = 2*math.pi * math.sqrt(hel['a']**3      / PHY.MU_SUN) / 86400

    comm_s   = (1.5 * PHY.AU) / PHY.C_LIGHT
    solar    = (PHY.EARTH_SMA / PHY.MARS_SMA)**2 * 100
    esc_E    = math.sqrt(2 * PHY.MU_EARTH / r_LEO)
    esc_M    = math.sqrt(2 * PHY.MU_MARS  / r_LMO)

    return {
        'a_AU':        hel['a'] / PHY.AU,
        'v_earth_kms': hel['v1c'] / 1000,
        'v_mars_kms':  hel['v2c'] / 1000,
        'v_peri_kms':  hel['v1t'] / 1000,
        'v_aphe_kms':  hel['v2t'] / 1000,
        'r_LEO_km':    r_LEO / 1000,
        'v_LEO_kms':   v_LEO / 1000,
        'v_hyp_E_kms': v_hyp_E / 1000,
        'v_inf_E_kms': v_inf_E / 1000,
        'esc_E_kms':   esc_E / 1000,
        'dv_dep_kms':  dv_dep / 1000,
        'c3_km2s2':    (v_inf_E/1000)**2,
        'r_LMO_km':    r_LMO / 1000,
        'v_LMO_kms':   v_LMO / 1000,
        'v_hyp_M_kms': v_hyp_M / 1000,
        'v_inf_M_kms': v_inf_M / 1000,
        'esc_M_kms':   esc_M / 1000,
        'dv_cap_kms':  dv_cap / 1000,
        'dv_total_kms':dv_total / 1000,
        've_ms':       prop['ve'],
        'mass_ratio':  prop['ratio'],
        'm_prop_kg':   prop['m_prop'],
        'm_total_kg':  prop['m_total'],
        'T_xfer_days': T_days,
        'T_earth_days':T_earth,
        'T_mars_days': T_mars,
        'T_orbit_days':T_xfer,
        'phase_deg':   phase,
        'synodic_days':PHY.SYNODIC,
        'comm_min':    comm_s / 60,
        'comm_s':      comm_s,
        'solar_pct':   solar,
    }

# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

SAVE_FILE = os.path.join(os.path.expanduser('~'), '.mars_missions.json')

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def load_db():
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_db(db):
    with open(SAVE_FILE, 'w') as f:
        json.dump(db, f, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def print_boot():
    clr()
    W = min(term_width(), 72)
    header_line('█', W)
    centre('PolyMarket — Quant Markets Analysis Terminal Interface v0.2', W)
    centre('MARKETS PREDICTIVE MODEL FOR FINANCIAL PRODUCTS', W)
    centre('© 2026 Vintage AI, Casablanca ', W)
    header_line('█', W)
    println()
    checks = [
        ('MEMORY CHECK',      '65536 BYTES', 'OK'),
        ('MATH CO-PROCESSOR', '8087',        'OK'),
        ('ORBITAL DATABASE',  'LOADED',      'OK'),
        ('EPHEMERIS FILE',    '1977-2000',   'OK'),
        ('TRAJECTORY ENGINE', 'ONLINE',      'OK'),
    ]
    for label, val, status in checks:
        println(dim(f"  {label:<22} {val:<14} {status}"), delay=0.07)
    println()
    println(dim('  NASA/JPL INTERPLANETARY MISSION DESIGN SYSTEM'))
    println(dim('  COPYRIGHT (C) 1977  JET PROPULSION LABORATORY'))
    println(dim('  PASADENA, CALIFORNIA   ALL RIGHTS RESERVED'))
    println()
    divider()

def print_main_menu():
    println()
    println(am('  COMMANDS:'))
    println(g('  [N]  NEW MISSION'))
    println(g('  [L]  LOAD MISSION'))
    println(g('  [H]  HELP / CONSTANTS'))
    println(g('  [Q]  QUIT'))
    println()

def print_mission_report(name, probe_mass, R, computed_at):
    W = min(term_width(), 72)
    println()
    header_line('═', W)
    centre(f'MISSION: {name}', W)
    centre('EARTH  →  MARS   HOHMANN TRANSFER ORBIT', W)
    centre(f'COMPUTED: {computed_at}', W)
    header_line('═', W)

    B  = box_line
    HL = highlight_line

    section('PROBE SPECIFICATIONS')
    B('DRY MASS (PAYLOAD)',       f'{probe_mass:.2f}',       'KG')
    B('PROPULSION SYSTEM',        'CHEMICAL (HYDRAZINE/LOX)')
    B('SPECIFIC IMPULSE (ISP)',   f'{PHY.ISP_CHEM}',         'S')
    B('EXHAUST VELOCITY (VE)',    f"{R['ve_ms']:.1f}",        'M/S')

    section('HELIOCENTRIC TRANSFER ORBIT')
    B('EARTH ORBIT RADIUS',       f"{PHY.EARTH_SMA/PHY.AU:.4f}", 'AU  (1.000)')
    B('MARS  ORBIT RADIUS',       f"{PHY.MARS_SMA/PHY.AU:.4f}",  'AU  (1.524)')
    B('TRANSFER SEMI-MAJOR AXIS', f"{R['a_AU']:.4f}",             'AU')
    B('PERIHELION VELOCITY',      f"{R['v_peri_kms']:.4f}",       'KM/S')
    B('APHELION  VELOCITY',       f"{R['v_aphe_kms']:.4f}",       'KM/S')
    B('EARTH CIRCULAR VELOCITY',  f"{R['v_earth_kms']:.4f}",      'KM/S')
    B('MARS  CIRCULAR VELOCITY',  f"{R['v_mars_kms']:.4f}",       'KM/S')

    section('EARTH DEPARTURE BURN')
    B('PARKING ORBIT ALTITUDE',   f"{PHY.LEO_ALT/1000:.0f}",      'KM')
    B('LEO ORBITAL VELOCITY',     f"{R['v_LEO_kms']:.4f}",        'KM/S')
    B('HYPERBOLIC VELOCITY',      f"{R['v_hyp_E_kms']:.4f}",      'KM/S')
    B('HYPERBOLIC EXCESS  V-INF', f"{R['v_inf_E_kms']:.4f}",      'KM/S')
    B('EARTH ESCAPE VELOCITY',    f"{R['esc_E_kms']:.4f}",        'KM/S')
    B('LAUNCH ENERGY (C3)',       f"{R['c3_km2s2']:.4f}",         'KM2/S2')
    HL('DV1  DEPARTURE BURN',     f"{R['dv_dep_kms']:.4f}",       'KM/S')

    section('MARS ARRIVAL BURN')
    B('CAPTURE ORBIT ALTITUDE',   f"{PHY.LMO_ALT/1000:.0f}",      'KM')
    B('LMO ORBITAL VELOCITY',     f"{R['v_LMO_kms']:.4f}",        'KM/S')
    B('HYPERBOLIC ARRIVAL VEL',   f"{R['v_hyp_M_kms']:.4f}",      'KM/S')
    B('HYPERBOLIC EXCESS  V-INF', f"{R['v_inf_M_kms']:.4f}",      'KM/S')
    B('MARS  ESCAPE VELOCITY',    f"{R['esc_M_kms']:.4f}",        'KM/S')
    HL('DV2  CAPTURE BURN',       f"{R['dv_cap_kms']:.4f}",       'KM/S')

    section('MISSION DELTA-V BUDGET')
    B('DV1  EARTH DEPARTURE',     f"{R['dv_dep_kms']:.4f}",       'KM/S')
    B('DV2  MARS CAPTURE',        f"{R['dv_cap_kms']:.4f}",       'KM/S')
    println(f"\n  {C.RED}{C.BOLD}DV   TOTAL MISSION  >>>  {R['dv_total_kms']:.4f} KM/S{C.RESET}\n")

    section('PROPELLANT REQUIREMENTS  (TSIOLKOVSKY EQ.)')
    B('PAYLOAD / DRY MASS',       f"{probe_mass:.2f}",             'KG')
    B('MASS RATIO',               f"{R['mass_ratio']:.6f}")
    HL('PROPELLANT MASS REQUIRED',f"{R['m_prop_kg']:.2f}",         'KG')
    HL('TOTAL LAUNCH MASS',       f"{R['m_total_kg']:.2f}",        'KG')

    section('MISSION TIMELINE')
    B('TRANSFER ORBIT PERIOD',    f"{R['T_orbit_days']:.2f}",      'DAYS')
    HL('TRANSIT TIME EARTH->MARS',f"{R['T_xfer_days']:.1f}",       'DAYS')
    B('EARTH ORBITAL PERIOD',     f"{R['T_earth_days']:.2f}",      'DAYS')
    B('MARS  ORBITAL PERIOD',     f"{R['T_mars_days']:.2f}",       'DAYS')
    B('SYNODIC PERIOD (WINDOW)',  f"{R['synodic_days']:.1f}",      'DAYS')
    B('REQUIRED PHASE ANGLE',     f"{R['phase_deg']:.2f}",         'DEG')

    section('COMMUNICATIONS & POWER')
    B('COMM DELAY  (AVG 1.5 AU)', f"{R['comm_min']:.2f}",          'MIN')
    B('COMM DELAY  (AVG 1.5 AU)', f"{R['comm_s']:.1f}",            'SEC')
    B('SOLAR POWER AT MARS',      f"{R['solar_pct']:.2f}",         '% OF EARTH')

    println()
    header_line('═', W)
    println(g('  COMPUTATION COMPLETE — ALL SYSTEMS NOMINAL'))
    header_line('═', W)

def print_constants():
    section('PHYSICAL CONSTANTS')
    box_line('G  (GRAVITATIONAL CONST)', f'{PHY.G:.4e}',        'M3/KG·S2')
    box_line('MU-SUN (GRAV PARAM)',      f'{PHY.MU_SUN:.4e}',   'M3/S2')
    box_line('MU-EARTH (GRAV PARAM)',    f'{PHY.MU_EARTH:.4e}', 'M3/S2')
    box_line('MU-MARS  (GRAV PARAM)',    f'{PHY.MU_MARS:.4e}',  'M3/S2')
    box_line('AU  (ASTRONOMICAL UNIT)',  f'{PHY.AU:.4e}',        'M')
    box_line('R-EARTH  (EARTH RADIUS)',  f'{PHY.R_EARTH:.4e}',  'M')
    box_line('R-MARS   (MARS  RADIUS)',  f'{PHY.R_MARS:.4e}',   'M')
    box_line('c   (SPEED OF LIGHT)',     f'{PHY.C_LIGHT:.4e}',  'M/S')
    box_line('g0  (STANDARD GRAVITY)',   f'{PHY.G0:.5f}',        'M/S2')
    box_line('ISP (CHEMICAL ENGINE)',    f'{PHY.ISP_CHEM}',      'S')
    println()

def print_tsiolkovsky(R=None, probe_mass=None):
    section('TSIOLKOVSKY ROCKET EQUATION')
    println(am('  DV = Isp x g0 x ln(m0 / m1)'))
    println()
    println(g('  WHERE:'))
    println(g('    DV   = REQUIRED VELOCITY CHANGE  (M/S)'))
    println(g('    Isp  = SPECIFIC IMPULSE           (SECONDS)'))
    println(g('    g0   = STANDARD GRAVITY           (9.80665 M/S2)'))
    println(g('    m0   = INITIAL MASS (PAYLOAD + PROPELLANT)'))
    println(g('    m1   = FINAL MASS   (PAYLOAD / DRY MASS)'))
    println(g('    VE   = Isp x g0    = EFFECTIVE EXHAUST VELOCITY'))
    if R and probe_mass:
        println()
        println(am('  CURRENT MISSION VALUES:'))
        box_line('DV TOTAL',    f"{R['dv_total_kms']*1000:.1f}", 'M/S')
        box_line('VE',          f"{R['ve_ms']:.1f}",             'M/S')
        box_line('MASS RATIO',  f"{R['mass_ratio']:.6f}")
        box_line('m1 (DRY)',    f"{probe_mass:.2f}",             'KG')
        box_line('m0 (TOTAL)',  f"{R['m_total_kg']:.2f}",        'KG')
        box_line('PROPELLANT',  f"{R['m_prop_kg']:.2f}",         'KG')
    println()

def print_orbital_notes():
    section('HOHMANN TRANSFER — ORBITAL MECHANICS')
    notes = [
        'THE HOHMANN TRANSFER IS THE MINIMUM-ENERGY ORBITAL',
        'MANEUVER BETWEEN TWO COPLANAR CIRCULAR ORBITS.',
        '',
        'STEP 1 — EARTH DEPARTURE:',
        '  PROBE FIRES ENGINE IN LEO TO ENTER ELLIPTICAL',
        '  HELIOCENTRIC ORBIT. PERIHELION = EARTH ORBIT,',
        '  APHELION = MARS ORBIT.',
        '',
        'STEP 2 — COAST PHASE (~259 DAYS):',
        '  NO THRUST. PROBE COASTS ALONG THE ELLIPSE.',
        '  EARTH MOVES FASTER, MARS SLOWER, GEOMETRY ALIGNS.',
        '',
        'STEP 3 — MARS ARRIVAL:',
        '  RETROGRADE BURN TO SLOW DOWN AND CAPTURE INTO LMO.',
        '  WITHOUT BURN: PROBE FLIES PAST ON HYPERBOLIC ORBIT.',
        '',
        'PATCHED CONIC APPROXIMATION:',
        '  NEAR EARTH   -> EARTH 2-BODY PROBLEM',
        '  HELIOCENTRIC -> SUN   2-BODY PROBLEM',
        '  NEAR MARS    -> MARS  2-BODY PROBLEM',
    ]
    for ln in notes:
        println(g(f'  {ln}') if ln else '')
    println()

def print_help():
    section('HELP — COMMANDS')
    println(am('  MAIN MENU:'))
    println(g('  [N]  CREATE NEW MISSION'))
    println(g('  [L]  LOAD EXISTING MISSION FROM DISK'))
    println(g('  [H]  SHOW HELP AND PHYSICAL CONSTANTS'))
    println(g('  [Q]  QUIT'))
    println()
    println(am('  MISSION MODE:'))
    println(g('  [1]  REDISPLAY FULL MISSION REPORT'))
    println(g('  [2]  PHYSICAL CONSTANTS'))
    println(g('  [3]  RECALCULATE WITH NEW PROBE MASS'))
    println(g('  [4]  TSIOLKOVSKY ROCKET EQUATION'))
    println(g('  [5]  ORBITAL MECHANICS NOTES'))
    println(g('  [M]  RETURN TO MAIN MENU'))
    println(g('  [Q]  QUIT'))
    println()

# ─────────────────────────────────────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────────────────────────────────────

def prompt(label='SYS'):
    try:
        val = input(f"{C.BAMBER}  {label}>{C.BGREEN} ")
        print(C.RESET, end='')
        return val.strip()
    except (KeyboardInterrupt, EOFError):
        println()
        return '__QUIT__'

def secure_prompt(label='PW'):
    try:
        print(f"{C.BAMBER}  {label}>{C.RESET} ", end='', flush=True)
        return getpass.getpass('')
    except (KeyboardInterrupt, EOFError):
        println()
        return '__QUIT__'

def ask_mass():
    while True:
        println()
        println(g('  ENTER PROBE DRY MASS IN KILOGRAMS'))
        println(dim('  (PAYLOAD MASS, EXCLUDING PROPELLANT)'))
        println(dim('  EXAMPLES: VOYAGER=722  MARINER=998  CURIOSITY=900'))
        val = prompt('MASS')
        if val == '__QUIT__':
            return None
        try:
            m = float(val)
            if 0 < m <= 1_000_000:
                return m
            println(rd('  ERROR: ENTER A POSITIVE VALUE IN KG.'))
        except ValueError:
            println(rd('  ERROR: INVALID NUMBER.'))

# ─────────────────────────────────────────────────────────────────────────────
# MISSION FLOWS
# ─────────────────────────────────────────────────────────────────────────────

def new_mission(db):
    println()
    header_line('─')
    println(am('  NEW MISSION SETUP'))
    header_line('─')

    # Mission name
    while True:
        println()
        println(g('  ENTER MISSION NAME  (E.G. VOYAGER-1, ARES-PROBE):'))
        name = prompt('NAME')
        if name == '__QUIT__':
            return
        name = name.upper()
        if len(name) < 2:
            println(rd('  ERROR: NAME TOO SHORT.')); continue
        if name in db:
            println(rd(f'  ERROR: MISSION "{name}" ALREADY EXISTS.')); continue
        break

    # Password
    println()
    println(g('  PASSWORD PROTECTION?  [Y/N]:'))
    pw_hash = None
    while True:
        choice = prompt('Y/N').upper()
        if choice == '__QUIT__':
            return
        if choice == 'Y':
            while True:
                pw1 = secure_prompt('PW')
                if pw1 == '__QUIT__':
                    return
                if len(pw1) < 4:
                    println(rd('  ERROR: MIN 4 CHARACTERS.')); continue
                pw2 = secure_prompt('CONFIRM')
                if pw2 == '__QUIT__':
                    return
                if pw1 != pw2:
                    println(rd('  ERROR: PASSWORDS DO NOT MATCH.')); continue
                pw_hash = hash_pw(pw1)
                println(g('  PASSWORD SET  OK'))
                break
            break
        elif choice == 'N':
            break
        else:
            println(rd('  ENTER Y OR N.'))

    # Mass
    probe_mass = ask_mass()
    if probe_mass is None:
        return

    # Compute
    println()
    for msg in [
        '  COMPUTING HOHMANN TRANSFER TRAJECTORY...',
        '  RUNNING PATCHED CONIC APPROXIMATION...',
        '  SOLVING KEPLER EQUATIONS...',
    ]:
        println(dim(msg))
        time.sleep(0.3)

    R  = compute_mission(probe_mass)
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

    db[name] = {
        'probe_mass': probe_mass,
        'password':   pw_hash,
        'computed':   ts,
        'results':    R,
    }
    save_db(db)

    print_mission_report(name, probe_mass, R, ts)
    mission_menu(name, db)


def load_mission(db):
    if not db:
        println()
        println(rd('  NO MISSIONS STORED ON DISK.'))
        return

    println()
    println(am('  STORED MISSIONS:'))
    for i, (n, m) in enumerate(db.items(), 1):
        lock = '  [PROTECTED]' if m.get('password') else ''
        println(g(f'  [{i}]  {n}  —  {m["probe_mass"]:.0f} KG  —  {m["computed"]}{lock}'))
    println()
    println(g('  ENTER MISSION NAME TO LOAD:'))
    name = prompt('LOAD').upper()
    if name == '__QUIT__' or name not in db:
        println(rd('  MISSION NOT FOUND.'))
        return

    entry = db[name]
    if entry.get('password'):
        pw = secure_prompt('PW')
        if pw == '__QUIT__':
            return
        if hash_pw(pw) != entry['password']:
            println(rd('  ACCESS DENIED — INCORRECT PASSWORD.'))
            beep()
            return
        println(g('  ACCESS GRANTED  OK'))

    print_mission_report(name, entry['probe_mass'], entry['results'], entry['computed'])
    mission_menu(name, db)


def mission_menu(name, db):
    entry = db[name]
    while True:
        println()
        println(am(f'  MISSION: {name}'))
        println(g('  [1] REPORT  [2] CONSTANTS  [3] NEW MASS  [4] TSIOLKOVSKY  [5] NOTES  [M] MENU  [Q] QUIT'))
        cmd = prompt('CMD').upper()
        if cmd in ('__QUIT__', 'Q'):
            shutdown()
        elif cmd == 'M':
            return
        elif cmd == '1':
            print_mission_report(name, entry['probe_mass'], entry['results'], entry['computed'])
        elif cmd == '2':
            print_constants()
        elif cmd == '3':
            m = ask_mass()
            if m is None:
                continue
            R  = compute_mission(m)
            ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            entry.update(probe_mass=m, results=R, computed=ts)
            save_db(db)
            print_mission_report(name, m, R, ts)
        elif cmd == '4':
            print_tsiolkovsky(entry['results'], entry['probe_mass'])
        elif cmd == '5':
            print_orbital_notes()
        else:
            println(rd('  UNKNOWN COMMAND.'))

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def shutdown():
    println()
    println(am('  SYSTEM SHUTDOWN INITIATED...'), delay=0.1)
    println(dim('  SAVING SESSION DATA...'), delay=0.2)
    println(am('  GOODBYE.  END OF TRANSMISSION.'))
    println()
    sys.exit(0)

def main():
    db = load_db()
    print_boot()
    print_main_menu()

    while True:
        cmd = prompt('SYS').upper()
        if cmd in ('__QUIT__', 'Q'):
            shutdown()
        elif cmd == 'N':
            new_mission(db)
            print_main_menu()
        elif cmd == 'L':
            load_mission(db)
            print_main_menu()
        elif cmd == 'H':
            print_help()
            print_constants()
        else:
            println(rd('  UNKNOWN COMMAND. [N] NEW  [L] LOAD  [H] HELP  [Q] QUIT'))

if __name__ == '__main__':
    main()
