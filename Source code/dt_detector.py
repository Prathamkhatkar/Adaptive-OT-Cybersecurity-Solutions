"""
Digital Twin Anomaly Detection + DRL Red-Blue Agent
Implements Pillar II of the AOCSF framework.
"""
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass, field


# ── Digital Twin Anomaly Detector ─────────────────────────────────────────────
class DigitalTwinDetector:
    """
    Physics-aware anomaly detection using the TEP process model.
    Evaluates every control command against the physics model
    over a look-ahead window L to detect FDI and setpoint manipulation.
    """

    def __init__(self, tep_simulator, lookahead_steps: int = 10,
                 safety_margin: float = 0.05, seed: int = 42):
        self.twin = tep_simulator
        self.L = lookahead_steps        # look-ahead window (steps × dt = L*0.1 hours)
        self.safety_margin = safety_margin
        self.rng = np.random.default_rng(seed)
        self.detection_log = []

        # Safety limits for critical variables (index: [low, high])
        self.safety_limits = {
            8:  (100.0, 190.0),    # reactor temperature
            6:  (2500., 3000.),    # reactor pressure
            7:  (70.,   90.),      # reactor level
            17: (50.,   130.),     # stripper temperature
        }

    def predict_trajectory(self, current_state: np.ndarray,
                           command: np.ndarray) -> np.ndarray:
        """
        Predict process state trajectory over look-ahead window
        given a proposed control command, using the DT physics model.
        """
        # Save twin state
        saved_state = self.twin.state.copy()
        saved_mv    = self.twin.mv.copy()
        saved_t     = self.twin.t

        # Set twin to current state and apply command
        self.twin.state = current_state.copy()
        trajectory = []

        for _ in range(self.L):
            state, _, _ = self.twin.step(mv_setpoints=command)
            trajectory.append(state.copy())

        # Restore twin state
        self.twin.state = saved_state
        self.twin.mv    = saved_mv
        self.twin.t     = saved_t

        return np.array(trajectory)

    def is_safe_trajectory(self, trajectory: np.ndarray) -> Tuple[bool, str]:
        """
        Check if predicted trajectory stays within safety bounds.
        Returns (is_safe, reason).
        """
        for var_idx, (low, high) in self.safety_limits.items():
            margin_low  = low  + (high - low) * self.safety_margin
            margin_high = high - (high - low) * self.safety_margin
            traj_values = trajectory[:, var_idx]
            if np.any(traj_values < margin_low):
                return False, f"VAR_{var_idx}_below_safe_range"
            if np.any(traj_values > margin_high):
                return False, f"VAR_{var_idx}_above_safe_range"
        return True, "OK"

    def evaluate_command(self, current_state: np.ndarray,
                         command: np.ndarray,
                         is_malicious: bool = False) -> dict:
        """
        Evaluate a control command using physics-aware DT detection.
        Returns detection result dict.
        """
        trajectory = self.predict_trajectory(current_state, command)
        is_safe, reason = self.is_safe_trajectory(trajectory)
        detected = not is_safe

        result = {
            'detected': detected,
            'is_malicious': is_malicious,
            'safe_predicted': is_safe,
            'violation_reason': reason,
            'true_positive': detected and is_malicious,
            'false_positive': detected and not is_malicious,
            'true_negative': not detected and not is_malicious,
            'false_negative': not detected and is_malicious,
            'max_temp_deviation': float(np.max(np.abs(
                trajectory[:, 8] - 120.4))),
            'max_pressure_deviation': float(np.max(np.abs(
                trajectory[:, 6] - 2705.))),
        }
        self.detection_log.append(result)
        return result

    def generate_benign_command(self, rng) -> np.ndarray:
        """Generate a realistic benign setpoint command."""
        cmd = self.twin.mv.copy()
        # Small normal variation around current setpoints
        cmd += rng.normal(0, 1.5, size=cmd.shape)
        return np.clip(cmd, 0.0, 100.0)

    def generate_fdi_command(self, rng, severity: str = 'medium') -> np.ndarray:
        """
        Generate an FDI attack command.
        The command is syntactically valid but drives the process unsafe.
        severity: 'low', 'medium', 'high'
        """
        cmd = self.twin.mv.copy()
        magnitudes = {'low': 8, 'medium': 18, 'high': 30}
        mag = magnitudes[severity]

        # Target reactor cooling (most impactful): drive cooling water DOWN
        cmd[9]  = max(0, cmd[9]  - rng.uniform(mag * 0.8, mag * 1.2))
        # Also slightly increase feed to compound the effect
        cmd[3]  = min(100, cmd[3] + rng.uniform(mag * 0.3, mag * 0.6))
        return np.clip(cmd, 0.0, 100.0)


# ── DRL Red-Blue Agent (simplified co-evolutionary framework) ──────────────────
@dataclass
class AttackChain:
    """Represents a discovered multi-step attack chain."""
    steps: List[str]
    total_reward: float
    safety_violated: bool
    detection_evaded: bool
    discovery_episode: int


class DRLRedAgent:
    """
    Simplified DRL Red Agent that learns to find attack chains.
    Uses a tabular Q-learning approximation for the TEP state space
    (full DRL would require GPU training; this captures the key dynamics).
    """

    ACTIONS = [
        'increase_reactor_feed',
        'decrease_cooling',
        'manipulate_pressure',
        'inject_setpoint_bias',
        'replay_safe_command',
        'idle',
    ]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.q_table: dict = {}
        self.discovered_chains: List[AttackChain] = []
        self.epsilon = 1.0      # exploration rate
        self.alpha   = 0.1      # learning rate
        self.gamma   = 0.95     # discount factor
        self.episode = 0

    def _state_key(self, state: np.ndarray) -> str:
        """Discretize continuous state to Q-table key."""
        # Discretize key variables into bins
        T_bin  = int(np.clip((state[8]  - 100) / 10, 0, 9))
        P_bin  = int(np.clip((state[6]  - 2500) / 50, 0, 9))
        L_bin  = int(np.clip((state[7]  - 70) / 2, 0, 9))
        return f"{T_bin},{P_bin},{L_bin}"

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, len(self.ACTIONS))
        key = self._state_key(state)
        if key not in self.q_table:
            return self.rng.integers(0, len(self.ACTIONS))
        return int(np.argmax(self.q_table[key]))

    def update(self, state: np.ndarray, action: int,
               reward: float, next_state: np.ndarray):
        """Q-learning update."""
        key      = self._state_key(state)
        next_key = self._state_key(next_state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(len(self.ACTIONS))
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(len(self.ACTIONS))
        td_target = reward + self.gamma * np.max(self.q_table[next_key])
        self.q_table[key][action] += self.alpha * (
            td_target - self.q_table[key][action])
        self.epsilon = max(0.05, self.epsilon * 0.9995)

    def run_episode(self, tep_sim, dt_detector,
                    max_steps: int = 50) -> AttackChain:
        """Run one attack discovery episode."""
        state = tep_sim.reset()
        chain_steps = []
        total_reward = 0.0
        safety_violated = False
        detection_evaded = True
        self.episode += 1

        for step_i in range(max_steps):
            action_idx = self.select_action(state)
            action_name = self.ACTIONS[action_idx]
            chain_steps.append(action_name)

            # Execute action in DT sandbox
            cmd = tep_sim.mv.copy()
            if action_name == 'increase_reactor_feed':
                cmd[3] = min(100, cmd[3] + self.rng.uniform(5, 15))
            elif action_name == 'decrease_cooling':
                cmd[9] = max(0, cmd[9] - self.rng.uniform(5, 20))
            elif action_name == 'manipulate_pressure':
                cmd[4] = min(100, cmd[4] + self.rng.uniform(8, 18))
            elif action_name == 'inject_setpoint_bias':
                cmd[9] = max(0, cmd[9] - self.rng.uniform(10, 25))
                cmd[3] = min(100, cmd[3] + self.rng.uniform(5, 12))
            elif action_name == 'replay_safe_command':
                cmd = tep_sim.mv.copy()  # benign-looking
            # idle: no change

            next_state, done, info = tep_sim.step(cmd)

            # Compute reward: maximize physical disruption, minimize detection
            temp_deviation  = abs(next_state[8]  - 120.4) / 70.0
            press_deviation = abs(next_state[6]  - 2705.) / 500.0
            dt_result = dt_detector.evaluate_command(
                state, cmd, is_malicious=(action_name != 'idle'))
            detection_penalty = -2.0 if dt_result['detected'] else 0.0
            if dt_result['detected']:
                detection_evaded = False

            reward = temp_deviation + press_deviation + detection_penalty
            total_reward += reward

            self.update(state, action_idx, reward, next_state)

            if info['safety_violation']:
                safety_violated = True
                total_reward += 5.0  # bonus for achieving unsafe state
                break

            state = next_state

        chain = AttackChain(
            steps=chain_steps,
            total_reward=total_reward,
            safety_violated=safety_violated,
            detection_evaded=detection_evaded,
            discovery_episode=self.episode,
        )
        self.discovered_chains.append(chain)
        return chain


class BlueAgent:
    """
    Blue Agent that learns from Red Agent discoveries to improve detection.
    Updates detection thresholds based on observed attack patterns.
    """

    def __init__(self, dt_detector: DigitalTwinDetector, seed: int = 42):
        self.detector = dt_detector
        self.rng = np.random.default_rng(seed)
        self.detection_improvements: List[float] = []

    def update_from_chain(self, chain: AttackChain):
        """Update detection policy based on a discovered attack chain."""
        if chain.detection_evaded and chain.safety_violated:
            # Tighten safety margin to catch this class of attack
            old_margin = self.detector.safety_margin
            self.detector.safety_margin = min(0.12, old_margin + 0.005)
            self.detection_improvements.append(0.005)
        else:
            self.detection_improvements.append(0.0)
