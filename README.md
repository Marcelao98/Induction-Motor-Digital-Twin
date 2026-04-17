# Induction Motor Digital Twin (dq0 Model)

A high-fidelity electromagnetic simulator for 3-phase induction motors, developed as the core of a multi-physics Digital Twin for fault detection and predictive maintenance research.

---

## Project Overview

This project implements the dynamic modeling of a squirrel-cage induction motor using the dq0 transformation (Park’s Transform) in a synchronous reference frame.

Unlike steady-state models, this digital twin core captures transient behaviors essential for condition monitoring, including:

- Startup inrush currents and torque pulsations  
- Dynamic load response and transient stability  
- Flux linkage dynamics in stator and rotor circuits  

---

## Technical Foundation

The simulation is based on constant-coefficient ODEs derived from standard power systems literature:

- Krause, P. C. et al. — *Analysis of Electric Machinery and Drive Systems*  
- Bose, B. K. — *Modern Power Electronics and AC Drives*  

The state vector is:

x = [λds, λqs, λdr, λqr, ωr]

The system is solved using RK45 (Runge-Kutta 4/5), ensuring accuracy during fast electromagnetic transients.

---

## Key Features

### dq0 Reference Frame
Transforms the 3-phase system into a rotating reference frame, removing time-varying inductances and reducing the system to constant-coefficient differential equations.

### Algebraic Recovery
Real-time computation of:
- Stator and rotor currents  
- Electromagnetic torque  

from flux linkages.

### Flexible Load Profiling
Supports scenarios such as:
- Free acceleration
- Sudden torque steps (e.g., rated load application)

---

## Roadmap (Future Development)

This is the foundation for a full Multi-Physics Digital Twin:

- Thermal coupling (temperature-dependent Rs and Rr)
- Vibration signatures (imbalance and eccentricity modeling)
- Fault injection:
  - Stator turn-to-turn short circuits
  - Broken rotor bars
- AI-based diagnostic model training

---

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib

---

## How to Use

```bash
git clone <repo-url>
cd <repo-folder>
python motor_dq.py
