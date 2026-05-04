"""
Tennessee Eastman Process (TEP) Simulation
Based on: Downs & Vogel (1993), Comput. Chem. Eng., 17(3), 245-255
Adapted for cybersecurity research benchmarking.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

# ─── TEP Process Constants ────────────────────────────────────────────────────
# 41 measured variables (XMEAS), 12 manipulated variables (XMV)
N_MEASURED   = 41
N_MANIPULATED = 12
N_DISTURBANCE = 20

# Safe operating ranges for key process variables (from Downs & Vogel 1993)
# [min, max] for each of the 41 measured variables
SAFE_RANGES = {
    0:  (0.0,   100.0),   # A feed (kscmh)
    1:  (0.0,   100.0),   # D feed (kg/h)
    2:  (0.0,   100.0),   # E feed (kg/h)
    3:  (0.0,   100.0),   # Total feed (kscmh)
    4:  (0.0,   100.0),   # Recycle flow (kscmh)
    5:  (0.0,   100.0),   # Reactor feed rate (kscmh)
    6:  (2500., 3000.),   # Reactor pressure (kPa)
    7:  (70.,   90.),     # Reactor level (%)
    8:  (100.,  190.),    # Reactor temperature (deg C)
    9:  (0.0,   100.0),   # Purge rate (kscmh)
    10: (60.,   100.),    # Product separator temp (deg C)
    11: (20.,   80.),     # Product separator level (%)
    12: (2500., 3000.),   # Product separator pressure (kPa)
    13: (0.0,   100.0),   # Product separator underflow (m3/h)
    14: (20.,   80.),     # Stripper level (%)
    15: (2500., 3000.),   # Stripper pressure (kPa)
    16: (0.0,   100.0),   # Stripper underflow (m3/h)
    17: (50.,   130.),    # Stripper temperature (deg C)
    18: (0.0,   100.0),   # Stripper steam flow (kg/h)
    19: (0.0,   100.0),   # Compressor work (kW)
    20: (100.,  200.),    # Reactor cooling water outlet temp (deg C)
    21: (50.,   120.),    # Separator cooling water outlet temp (deg C)
}

# Nominal operating point (steady state) for 41 measured variables
NOMINAL_STATE = np.array([
    0.250,  # XMEAS(1)  - A feed flow
    3664.,  # XMEAS(2)  - D feed flow
    4509.,  # XMEAS(3)  - E feed flow
    9.35,   # XMEAS(4)  - A and C feed flow
    26.9,   # XMEAS(5)  - recycle flow
    42.3,   # XMEAS(6)  - reactor feed rate
    2705.,  # XMEAS(7)  - reactor pressure
    75.0,   # XMEAS(8)  - reactor level
    120.4,  # XMEAS(9)  - reactor temperature
    0.337,  # XMEAS(10) - purge rate
    80.1,   # XMEAS(11) - product sep temp
    50.0,   # XMEAS(12) - product sep level
    2633.,  # XMEAS(13) - product sep pressure
    25.2,   # XMEAS(14) - product sep underflow
    50.0,   # XMEAS(15) - stripper level
    3102.,  # XMEAS(16) - stripper pressure
    22.9,   # XMEAS(17) - stripper underflow
    65.7,   # XMEAS(18) - stripper temperature
    230.,   # XMEAS(19) - stripper steam flow
    341.,   # XMEAS(20) - compressor work
    94.6,   # XMEAS(21) - reactor cooling outlet temp
    77.3,   # XMEAS(22) - separator cooling outlet temp
    # Component compositions (22-41)
    32.2, 8.89, 26.5, 6.88, 18.8, 1.66, 14.5, 53.4,
    43.8, 0.238, 48.6, 2.66, 0.00, 0.893, 0.00, 0.665,
    0.533, 42.3, 0.00,
], dtype=float)

# Nominal manipulated variable setpoints
NOMINAL_MV = np.array([
    63.1,   # XMV(1)  - D feed valve
    53.3,   # XMV(2)  - E feed valve
    24.6,   # XMV(3)  - A feed valve
    61.3,   # XMV(4)  - total feed valve
    22.2,   # XMV(5)  - compressor recycle valve
    40.1,   # XMV(6)  - purge valve
    38.1,   # XMV(7)  - separator underflow valve
    46.6,   # XMV(8)  - stripper underflow valve
    47.6,   # XMV(9)  - stripper steam valve
    41.1,   # XMV(10) - reactor cooling water valve
    18.1,   # XMV(11) - condenser cooling water valve
    50.0,   # XMV(12) - agitator speed
], dtype=float)


class TEPSimulator:
    """
    Simplified but faithful TEP simulator for cybersecurity experiments.
    Uses linearized dynamics around the nominal operating point with
    realistic nonlinear corrections and disturbance propagation.
    """

    def __init__(self, seed: int = 42, noise_level: float = 0.02):
        self.rng = np.random.default_rng(seed)
        self.noise_level = noise_level
        self.reset()

    def reset(self):
        """Reset to nominal operating point with small perturbation."""
        self.state = NOMINAL_STATE.copy()
        self.mv = NOMINAL_MV.copy()
        self.t = 0.0
        self.disturbance = np.zeros(N_DISTURBANCE)
        self.fault_active = False
        self.history = []
        return self.state.copy()

    def step(self, mv_setpoints: Optional[np.ndarray] = None, dt: float = 0.1):
        """
        Advance simulation by dt hours.
        mv_setpoints: optional array of manipulated variable commands (0-100 scale)
        Returns: (state, reward, done, info)
        """
        if mv_setpoints is not None:
            # Valve rate-of-change limits (realistic actuator constraints)
            delta = np.clip(mv_setpoints - self.mv, -5.0, 5.0)
            self.mv = np.clip(self.mv + delta, 0.0, 100.0)

        # ── Core process dynamics (linearized TEP around nominal) ──────────
        s = self.state
        mv = self.mv

        # Reactor temperature dynamics (most safety-critical variable)
        # Driven by cooling water valve (mv[9]) and feed composition
        T_reactor = s[8]
        dT_reactor = (
            0.08 * (120.4 - T_reactor)           # mean reversion
            - 0.12 * (mv[9] - 41.1)              # cooling effect
            + 0.05 * (mv[3] - 61.3)              # feed rate effect
            + self.rng.normal(0, self.noise_level)
        )

        # Reactor pressure dynamics
        P_reactor = s[6]
        dP_reactor = (
            0.06 * (2705. - P_reactor)
            + 0.08 * (mv[4] - 22.2)              # recycle valve
            - 0.10 * (mv[5] - 40.1)              # purge valve
            + self.rng.normal(0, self.noise_level * 10)
        )

        # Reactor level dynamics
        L_reactor = s[7]
        dL_reactor = (
            0.04 * (75.0 - L_reactor)
            + 0.06 * (mv[3] - 61.3)
            - 0.05 * (mv[7] - 46.6)
            + self.rng.normal(0, self.noise_level * 0.5)
        )

        # Product separator temperature
        T_sep = s[10]
        dT_sep = (
            0.05 * (80.1 - T_sep)
            + 0.03 * (T_reactor - 120.4) * 0.1
            - 0.08 * (mv[10] - 18.1)
            + self.rng.normal(0, self.noise_level * 0.5)
        )

        # Stripper temperature
        T_strip = s[17]
        dT_strip = (
            0.04 * (65.7 - T_strip)
            + 0.05 * (mv[8] - 47.6)
            + self.rng.normal(0, self.noise_level * 0.3)
        )

        # Apply disturbances (IDV signals)
        disturbance_effect = np.sum(self.disturbance[:5]) * 0.1

        # Update critical states
        new_state = self.state.copy()
        new_state[8]  = T_reactor  + dT_reactor  * dt
        new_state[6]  = P_reactor  + dP_reactor  * dt
        new_state[7]  = L_reactor  + dL_reactor  * dt
        new_state[10] = T_sep      + dT_sep      * dt
        new_state[17] = T_strip    + dT_strip    * dt

        # Flow variables follow valve positions with lag
        for i, (mv_idx, nom_state, nom_mv, gain) in enumerate([
            (3, 42.3, 61.3, 0.3),   # reactor feed
            (4, 26.9, 22.2, 0.25),  # recycle flow
            (5, 0.337, 40.1, 0.01), # purge rate
        ]):
            new_state[5+i] += 0.1 * (nom_state + gain*(mv[mv_idx]-nom_mv) - new_state[5+i]) * dt
            new_state[5+i] += self.rng.normal(0, self.noise_level * 0.1)

        # Add global process noise to remaining variables
        noise = self.rng.normal(0, self.noise_level * 0.1, N_MEASURED)
        noise[6] = noise[8] = noise[17] = 0  # already handled above
        new_state += noise
        new_state = np.clip(new_state, 0, None)  # no negative flows/levels

        self.state = new_state
        self.t += dt
        self.history.append(self.state.copy())

        # Check safety boundaries
        done = self._check_safety_violation()
        info = {
            'time': self.t,
            'reactor_temp': self.state[8],
            'reactor_pressure': self.state[6],
            'reactor_level': self.state[7],
            'safety_violation': done,
        }
        return self.state.copy(), done, info

    def _check_safety_violation(self) -> bool:
        """Return True if any process variable is outside safe operating range."""
        checks = [
            self.state[8]  > 190.0,   # reactor temp too high
            self.state[8]  < 100.0,   # reactor temp too low
            self.state[6]  > 3000.0,  # reactor pressure too high
            self.state[6]  < 2500.0,  # reactor pressure too low
            self.state[7]  > 90.0,    # reactor level too high
            self.state[7]  < 70.0,    # reactor level too low
            self.state[17] > 130.0,   # stripper temp too high
            self.state[17] < 50.0,    # stripper temp too low
        ]
        return any(checks)

    def inject_fdi(self, variable_idx: int, bias: float):
        """
        Inject False Data Injection: corrupt sensor reading by bias.
        The physical process continues normally; only the reading is spoofed.
        Returns (true_value, spoofed_value).
        """
        true_val = self.state[variable_idx]
        spoofed_val = true_val + bias
        return true_val, spoofed_val

    def inject_setpoint_manipulation(self, mv_idx: int, malicious_value: float):
        """
        Inject malicious setpoint command.
        The actual MV is changed to the malicious value.
        """
        original = self.mv[mv_idx]
        self.mv[mv_idx] = np.clip(malicious_value, 0.0, 100.0)
        return original, self.mv[mv_idx]

    def activate_disturbance(self, idv: int, magnitude: float = 1.0):
        """Activate one of the 20 IDV disturbances."""
        if 0 <= idv < N_DISTURBANCE:
            self.disturbance[idv] = magnitude

    def get_state_vector(self) -> np.ndarray:
        return self.state.copy()

    def get_history_array(self) -> np.ndarray:
        return np.array(self.history)
