"""
Induction Motor Digital Twin — dq0 Model
==========================================
Reference frame: Synchronous (rotating at ωe)

Key references:
  - Krause et al., "Analysis of Electric Machinery and Drive Systems" (2002)
    Chapters 3–4: dq0 transformation and induction motor equations
  - Vas, P., "Electrical Machines and Drives: A Space-Vector Theory Approach" (1992)
  - Bose, B.K., "Modern Power Electronics and AC Drives" (2002), Ch.2

Physics summary:
  The 3-phase induction motor is transformed into two orthogonal axes (d and q)
  rotating synchronously with the supply frequency. This eliminates the time-varying
  mutual inductances and turns the problem into a set of constant-coefficient ODEs.

  State vector: x = [λds, λqs, λdr, λqr, ωr]
    λds, λqs = stator flux linkages (d and q axes)
    λdr, λqr = rotor flux linkages  (d and q axes)
    ωr       = rotor electrical angular velocity [rad/s]

  The elegance: from fluxes you algebraically recover all currents,
  then compute torque, then integrate the mechanical equation.
  Five ODEs. That's the whole motor.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ═══════════════════════════════════════════════════════════════════════════
#  MOTOR PARAMETERS — 3HP / 2.2kW, 4-pole, 60Hz, 220V induction motor
#  Source: Krause et al. Appendix C (standard benchmark machine)
# ═══════════════════════════════════════════════════════════════════════════
class MotorParams:
    # Electrical
    Rs   = 0.435     # Stator resistance [Ω]
    Rr   = 0.816     # Rotor resistance referred to stator [Ω]
    Lls  = 0.002     # Stator leakage inductance [H]
    Llr  = 0.002     # Rotor leakage inductance referred to stator [H]
    Lm   = 0.0679    # Magnetizing inductance [H]
    Ls   = Lls + Lm  # Total stator inductance [H]
    Lr   = Llr + Lm  # Total rotor inductance [H]
    
    # Mechanical
    P    = 4         # Number of poles
    J    = 0.089     # Moment of inertia [kg·m²]
    B    = 0.001     # Viscous friction coefficient [N·m·s/rad]
    
    # Supply
    Vll  = 220.0     # Line-to-line voltage RMS [V]
    fe   = 60.0      # Supply frequency [Hz]
    we   = 2*np.pi*fe  # Electrical angular frequency [rad/s]
    Vpk  = Vll * np.sqrt(2/3)  # Peak phase voltage [V]
    
    # Derived
    D    = Ls*Lr - Lm**2  # Inductance determinant (always > 0)

p = MotorParams()

# ═══════════════════════════════════════════════════════════════════════════
#  ALGEBRAIC RECOVERY OF CURRENTS FROM FLUX LINKAGES
#  From the flux linkage definitions:
#    λds = Ls*ids + Lm*idr  →  solve for ids, idr
#    λdr = Lr*idr + Lm*ids
#  Inverting the 2×2 system:
# ═══════════════════════════════════════════════════════════════════════════
def fluxes_to_currents(lam_ds, lam_qs, lam_dr, lam_qr):
    ids = ( p.Lr * lam_ds - p.Lm * lam_dr) / p.D
    iqs = ( p.Lr * lam_qs - p.Lm * lam_qr) / p.D
    idr = ( p.Ls * lam_dr - p.Lm * lam_ds) / p.D
    iqr = ( p.Ls * lam_qr - p.Lm * lam_qs) / p.D
    return ids, iqs, idr, iqr

# ═══════════════════════════════════════════════════════════════════════════
#  SUPPLY VOLTAGE IN dq FRAME (synchronous frame → DC quantities at steady state)
#  In synchronous reference frame, balanced 3-phase voltages become:
#    vds =  Vpk * cos(δ)  (here δ=0 → vds = Vpk, vqs = 0)
#  This is the Park transform applied to Va=Vpk·cos(ωe·t), with θ=ωe·t
# ═══════════════════════════════════════════════════════════════════════════
def supply_voltage(t):
    # Synchronous frame: balanced supply maps to constant DC-like values
    # vds = Vpk (d-axis aligned with voltage vector)
    # vqs = 0
    vds = p.Vpk
    vqs = 0.0
    return vds, vqs

# ═══════════════════════════════════════════════════════════════════════════
#  ELECTROMAGNETIC TORQUE
#  Te = (3/2) * (P/2) * (λds*iqs - λqs*ids)
#  Derivation: power balance in dq frame → Te = (3/2)(P/2)(λ × i)
#  Ref: Krause eq. 6.5-5
# ═══════════════════════════════════════════════════════════════════════════
def torque(lam_ds, lam_qs, ids, iqs):
    return (3/2) * (p.P/2) * (lam_ds * iqs - lam_qs * ids)

# ═══════════════════════════════════════════════════════════════════════════
#  THE ODE SYSTEM — this is the heart of the digital twin
#
#  5 coupled first-order ODEs:
#
#  Stator flux (voltage equations, synchronous frame):
#    dλds/dt = vds - Rs*ids + ωe*λqs         [Krause eq. 6.5-1]
#    dλqs/dt = vqs - Rs*iqs - ωe*λds         [Krause eq. 6.5-2]
#
#  Rotor flux (short-circuited rotor, vdr=vqr=0):
#    dλdr/dt = -Rr*idr + (ωe - ωr)*λqr      [Krause eq. 6.5-3]
#    dλqr/dt = -Rr*iqr - (ωe - ωr)*λdr      [Krause eq. 6.5-4]
#
#  Mechanical (Newton's 2nd law for rotation):
#    dωr/dt  = (P/2)/J * (Te - TL - B*ωr)   [Krause eq. 6.5-6]
#
#  The (ωe - ωr) term is the slip angular frequency — it's what drives
#  rotor currents. At synchronous speed ωr=ωe, slip=0, rotor currents→0.
# ═══════════════════════════════════════════════════════════════════════════
def motor_odes(t, x, T_load):
    lam_ds, lam_qs, lam_dr, lam_qr, wr = x

    # Recover currents from fluxes
    ids, iqs, idr, iqr = fluxes_to_currents(lam_ds, lam_qs, lam_dr, lam_qr)

    # Supply voltages
    vds, vqs = supply_voltage(t)

    # Slip angular frequency
    slip_w = p.we - wr

    # Electromagnetic torque
    Te = torque(lam_ds, lam_qs, ids, iqs)

    # The 5 ODEs
    dlam_ds = vds - p.Rs * ids + p.we  * lam_qs
    dlam_qs = vqs - p.Rs * iqs - p.we  * lam_ds
    dlam_dr =     - p.Rr * idr + slip_w * lam_qr
    dlam_qr =     - p.Rr * iqr - slip_w * lam_dr
    dwr     = (p.P/2) / p.J * (Te - T_load - p.B * wr)

    return [dlam_ds, dlam_qs, dlam_dr, dlam_qr, dwr]

# ═══════════════════════════════════════════════════════════════════════════
#  SIMULATION SETUP
# ═══════════════════════════════════════════════════════════════════════════

# Initial conditions: motor at rest, no flux
x0 = [0.0, 0.0, 0.0, 0.0, 0.0]

# Time span: 2 seconds, enough to reach steady state
t_start, t_end = 0.0, 2.0
t_eval = np.linspace(t_start, t_end, 20000)

# Load profile: starts unloaded, full load applied at t=0.8s
def T_load(t):
    T_rated = 11.9  # Nm — rated torque for 2.2kW at 1750 RPM
    if t < 0.8:
        return 0.0           # Free acceleration
    else:
        return T_rated * 0.75  # 75% rated load applied

print("Solving motor ODE system...")
print(f"  Motor: {p.P}-pole, {p.fe}Hz, {p.Vll}V line-to-line")
print(f"  Parameters: Rs={p.Rs}Ω, Rr={p.Rr}Ω, Lm={p.Lm*1000:.1f}mH")
print(f"  Simulation: {t_start}s → {t_end}s, {len(t_eval)} points")

sol = solve_ivp(
    fun=lambda t, x: motor_odes(t, x, T_load(t)),
    t_span=(t_start, t_end),
    y0=x0,
    method='RK45',       # Runge-Kutta 4/5 — standard for this class of ODE
    t_eval=t_eval,
    rtol=1e-6,           # Relative tolerance
    atol=1e-8,           # Absolute tolerance
    max_step=1e-4        # Limit step size — motor ODEs have fast electrical dynamics
)

print(f"  Solver: {sol.message}")
print(f"  Steps taken: {sol.t.shape[0]}")

# ═══════════════════════════════════════════════════════════════════════════
#  POST-PROCESSING — recover all physical quantities from solution
# ═══════════════════════════════════════════════════════════════════════════
t = sol.t
lam_ds, lam_qs, lam_dr, lam_qr, wr = sol.y

# Currents from fluxes
ids, iqs, idr, iqr = fluxes_to_currents(lam_ds, lam_qs, lam_dr, lam_qr)

# Electromagnetic torque
Te_arr = torque(lam_ds, lam_qs, ids, iqs)

# Shaft speed in RPM
RPM = wr * (60 / (2*np.pi)) * (2/p.P)  # electrical rad/s → mechanical RPM

# Slip
slip_pct = (p.we - wr) / p.we * 100

# Reconstruct approximate phase A current (inverse Park)
# ia ≈ ids*cos(ωe*t) - iqs*sin(ωe*t)  [stationary frame]
ia = ids * np.cos(p.we * t) - iqs * np.sin(p.we * t)

# RMS stator current magnitude in dq frame
I_rms = np.sqrt(ids**2 + iqs**2) / np.sqrt(2)

# Steady-state values (last 10% of simulation)
ss_idx = int(0.9 * len(t))
print(f"\nSteady-state results:")
print(f"  Speed:    {RPM[ss_idx:].mean():.1f} RPM  (sync: {60*p.fe/(p.P/2):.0f} RPM)")
print(f"  Slip:     {slip_pct[ss_idx:].mean():.2f}%")
print(f"  Torque:   {Te_arr[ss_idx:].mean():.2f} Nm")
print(f"  I_rms:    {I_rms[ss_idx:].mean():.2f} A")

# ═══════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════
plt.style.use('dark_background')
fig = plt.figure(figsize=(14, 10), facecolor='#070b10')
fig.suptitle('Induction Motor — dq0 Model Simulation\n'
             '4-pole, 60Hz, 220V | Load applied at t=0.8s',
             color='#f0a500', fontsize=13, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

GRID_C  = '#1c2840'
LINE_C  = {'speed':'#f0a500', 'torque':'#00e676', 'slip':'#00d4e8',
           'ids':'#f0a500', 'iqs':'#00d4e8', 'ia':'#00d4e8',
           'lam_ds':'#f0a500', 'lam_qs':'#00d4e8'}

def styled_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor('#0c1018')
    ax.tick_params(colors='#3d5568', labelsize=8)
    ax.xaxis.label.set_color('#3d5568')
    ax.yaxis.label.set_color('#3d5568')
    ax.set_title(title, color='#b8ccd8', fontsize=9, pad=6)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, color=GRID_C, linewidth=0.5, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.axvline(0.8, color='#ff4040', linewidth=0.8, linestyle='--', alpha=0.6, label='Load applied')

# ─── 1. Shaft Speed ───────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
styled_ax(ax1, 'Shaft Speed', 'Time [s]', 'RPM')
ax1.plot(t, RPM, color=LINE_C['speed'], linewidth=1.2, label='Rotor speed')
ax1.axhline(1800, color='#ffffff', linewidth=0.6, linestyle=':', alpha=0.3, label='Sync speed')
ax1.legend(fontsize=7, loc='lower right')
ax1.set_ylim(-100, 2000)

# ─── 2. Electromagnetic Torque ────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
styled_ax(ax2, 'Electromagnetic Torque', 'Time [s]', 'Torque [N·m]')
ax2.plot(t, Te_arr, color=LINE_C['torque'], linewidth=1.0)
ax2.set_ylim(-5, max(Te_arr.max()*1.1, 30))

# ─── 3. dq Stator Currents ────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
styled_ax(ax3, 'Stator Currents — dq Frame', 'Time [s]', 'Current [A]')
ax3.plot(t, ids, color=LINE_C['ids'],  linewidth=0.9, label='ids (d-axis)')
ax3.plot(t, iqs, color=LINE_C['iqs'],  linewidth=0.9, label='iqs (q-axis)', alpha=0.85)
ax3.legend(fontsize=7)

# ─── 4. Phase A Current (reconstructed) ──────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
styled_ax(ax4, 'Phase A Current (inverse Park)', 'Time [s]', 'Current [A]')
# Show last 100ms of startup and last 100ms of loaded steady state
mask_ss = (t > 1.85)
ax4.plot(t[mask_ss], ia[mask_ss], color=LINE_C['ia'], linewidth=1.0, label='ia steady-state')
ax4_twin = ax4.twinx()
mask_start = (t < 0.15)
ax4_twin.plot(t[mask_start], ia[mask_start], color='#ff8c00', linewidth=0.8,
              alpha=0.7, label='ia startup')
ax4_twin.tick_params(colors='#3d5568', labelsize=8)
ax4_twin.set_ylabel('Startup current [A]', color='#3d5568', fontsize=8)
ax4.legend(fontsize=7, loc='upper left')
ax4_twin.legend(fontsize=7, loc='upper right')

# ─── 5. Stator Flux Linkages ──────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
styled_ax(ax5, 'Stator Flux Linkages — dq Frame', 'Time [s]', 'Flux [Wb]')
ax5.plot(t, lam_ds, color=LINE_C['lam_ds'], linewidth=0.9, label='λds')
ax5.plot(t, lam_qs, color=LINE_C['lam_qs'], linewidth=0.9, label='λqs', alpha=0.85)
ax5.legend(fontsize=7)

# ─── 6. Slip ──────────────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
styled_ax(ax6, 'Rotor Slip', 'Time [s]', 'Slip [%]')
ax6.plot(t, slip_pct, color=LINE_C['slip'], linewidth=1.0)
ax6.set_ylim(-5, 105)
# Annotate steady-state slip
ss_slip = slip_pct[ss_idx:].mean()
ax6.annotate(f'Steady-state: {ss_slip:.2f}%',
             xy=(1.8, ss_slip), xytext=(1.2, ss_slip+8),
             color='#b8ccd8', fontsize=8,
             arrowprops=dict(arrowstyle='->', color='#3d5568', lw=0.8))

plt.savefig('/mnt/user-data/outputs/motor_dq_simulation.png',
            dpi=150, bbox_inches='tight', facecolor='#070b10')
print("\nPlot saved.")
