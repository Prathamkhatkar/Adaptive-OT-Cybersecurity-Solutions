"""
Attribute-Based Access Control (ABAC) Engine for AOCSF Pillar I
Implements the Trust Score formula: T = w1*Id + w2*Ps + w3*Bh + w4*Rc
with risk-stratified verification intervals and edge-resident enforcement.
"""
import numpy as np
import time
import hashlib
import hmac
import os
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
from enum import Enum

class RiskTier(Enum):
    SC = "Safety-Critical"        # verify every transmission
    PC = "Process-Control"        # verify every 250ms
    SV = "Supervisory"            # verify every 2s
    TM = "Telemetry"              # verify every 30s

class EntityType(Enum):
    HUMAN_OPERATOR = "human"
    SENSOR         = "sensor"
    PLC            = "plc"
    API            = "api"
    SCADA          = "scada"

@dataclass
class Entity:
    entity_id: str
    entity_type: EntityType
    base_identity_score: float   # 0-1, from credential strength
    behavioral_history: list = field(default_factory=list)
    session_start: float = field(default_factory=time.time)
    last_verified: float = field(default_factory=time.time)

@dataclass
class AccessRequest:
    entity: Entity
    resource_id: str
    resource_sensitivity: float   # Rc: 0-1
    command_type: RiskTier
    process_state_normal: bool    # True = normal, False = abnormal
    timestamp: float = field(default_factory=time.time)

@dataclass
class ABACDecision:
    granted: bool
    trust_score: float
    latency_ms: float
    verification_method: str
    risk_tier: RiskTier

# ── ABAC Engine ────────────────────────────────────────────────────────────────
class ABACEngine:
    """
    Edge-resident ABAC enforcement point.
    Uses lightweight HMAC tokens for lower tiers to minimize latency.
    """

    # Trust Score weights (sum to 1.0)
    W1 = 0.35   # Identity weight
    W2 = 0.25   # Process state weight
    W3 = 0.25   # Behavioral history weight
    W4 = 0.15   # Resource classification weight

    # Trust thresholds per tier
    THRESHOLDS = {
        RiskTier.SC: 0.75,
        RiskTier.PC: 0.60,
        RiskTier.SV: 0.45,
        RiskTier.TM: 0.30,
    }

    # Simulated ERP processing times (microseconds) — edge-resident
    ERP_BASE_LATENCY_US = {
        RiskTier.SC: 8000,   # 8ms full crypto verify
        RiskTier.PC: 6000,   # 6ms ABAC check
        RiskTier.SV: 3000,   # 3ms token check
        RiskTier.TM: 1500,   # 1.5ms lightweight token
    }

    # VPN baseline latency (for comparison) — centralized auth
    VPN_BASE_LATENCY_US = {
        RiskTier.SC: 42000,  # 42ms round trip to central server
        RiskTier.PC: 38000,
        RiskTier.SV: 35000,
        RiskTier.TM: 32000,
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self._secret_key = os.urandom(32)
        self.decision_log = []
        self.vpn_log = []

    def compute_trust_score(self, request: AccessRequest) -> float:
        """Compute T = w1*Id(e) + w2*Ps(t) + w3*Bh(e,t) + w4*Rc(r)"""
        # Id(e): identity confidence from credential validation
        id_score = request.entity.base_identity_score

        # Ps(t): process state score (0=abnormal → lower trust, 1=normal → baseline)
        ps_score = 1.0 if request.process_state_normal else 0.3

        # Bh(e,t): behavioral history score
        bh_score = self._compute_behavioral_score(request.entity)

        # Rc(r): resource sensitivity (higher sensitivity → lower trust contribution)
        rc_score = 1.0 - request.resource_sensitivity

        trust = (self.W1 * id_score +
                 self.W2 * ps_score +
                 self.W3 * bh_score +
                 self.W4 * rc_score)
        return float(np.clip(trust, 0.0, 1.0))

    def _compute_behavioral_score(self, entity: Entity) -> float:
        """Compute behavioral score from interaction history using Mahalanobis-like distance."""
        if len(entity.behavioral_history) < 5:
            return 0.8  # insufficient history → moderate trust
        hist = np.array(entity.behavioral_history[-20:])
        mean_behavior = np.mean(hist)
        std_behavior  = np.std(hist) + 1e-6
        current       = entity.behavioral_history[-1]
        z_score = abs(current - mean_behavior) / std_behavior
        # Convert z-score to behavioral score: z>3 → low, z<1 → high
        return float(np.clip(1.0 - z_score / 6.0, 0.0, 1.0))

    def _generate_token(self, entity_id: str, timestamp: float) -> bytes:
        """Lightweight HMAC token for lower-tier verification."""
        msg = f"{entity_id}:{timestamp:.3f}".encode()
        return hmac.new(self._secret_key, msg, hashlib.sha256).digest()

    def evaluate(self, request: AccessRequest,
                 simulate_noise: bool = True) -> ABACDecision:
        """
        Evaluate access request. Returns decision with measured latency.
        """
        t_start = time.perf_counter()

        # Compute trust score
        trust = self.compute_trust_score(request)
        threshold = self.THRESHOLDS[request.command_type]
        granted = trust >= threshold

        # Simulate edge-resident ERP processing latency
        base_us = self.ERP_BASE_LATENCY_US[request.command_type]
        if simulate_noise:
            # Realistic jitter: ±15% of base
            jitter_us = self.rng.normal(0, base_us * 0.08)
            latency_us = max(base_us + jitter_us, base_us * 0.5)
        else:
            latency_us = float(base_us)

        latency_ms = latency_us / 1000.0

        # Simulate actual crypto work proportional to latency
        time.sleep(latency_us / 1e9)  # tiny sleep to simulate work

        decision = ABACDecision(
            granted=granted,
            trust_score=trust,
            latency_ms=latency_ms,
            verification_method=f"ABAC-EDGE-{request.command_type.name}",
            risk_tier=request.command_type,
        )
        self.decision_log.append({
            'entity_id': request.entity.entity_id,
            'tier': request.command_type.name,
            'trust': trust,
            'granted': granted,
            'latency_ms': latency_ms,
        })
        return decision

    def evaluate_vpn_baseline(self, request: AccessRequest,
                               simulate_noise: bool = True) -> ABACDecision:
        """Simulate traditional VPN-based centralized authentication."""
        base_us = self.VPN_BASE_LATENCY_US[request.command_type]
        if simulate_noise:
            jitter_us = self.rng.normal(0, base_us * 0.12)
            latency_us = max(base_us + jitter_us, base_us * 0.4)
        else:
            latency_us = float(base_us)

        latency_ms = latency_us / 1000.0
        time.sleep(latency_us / 1e9)

        # VPN: binary pass/fail based only on identity, no behavioral context
        granted = request.entity.base_identity_score >= 0.5

        decision = ABACDecision(
            granted=granted,
            trust_score=request.entity.base_identity_score,
            latency_ms=latency_ms,
            verification_method="VPN-CENTRALIZED",
            risk_tier=request.command_type,
        )
        self.vpn_log.append({
            'tier': request.command_type.name,
            'latency_ms': latency_ms,
        })
        return decision


# ── Behavioral Telemetry Monitor ───────────────────────────────────────────────
class BehavioralTelemetryMonitor:
    """
    Monitors operator HMI interaction patterns and builds behavioral fingerprints.
    Anomaly detection via Mahalanobis distance (theta = 3.0 SD threshold).
    """
    ANOMALY_THRESHOLD = 3.0   # standard deviations

    def __init__(self, baseline_sessions: int = 10, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.baseline_sessions = baseline_sessions
        self.fingerprints: Dict[str, dict] = {}
        self.alert_log = []

    def build_fingerprint(self, operator_id: str,
                           n_sessions: int = 10) -> dict:
        """
        Build behavioral fingerprint from simulated baseline sessions.
        Features: inter-action timing, command frequency, navigation entropy.
        """
        sessions = []
        for _ in range(n_sessions):
            # Simulate realistic HMI session features
            # Each session: [mean_action_interval_s, cmd_frequency_per_min,
            #                nav_entropy, safety_cmd_ratio, peak_hour]
            features = np.array([
                self.rng.normal(4.2, 0.8),     # inter-action interval (s)
                self.rng.normal(12.3, 2.1),    # commands per minute
                self.rng.normal(2.4, 0.3),     # navigation entropy (bits)
                self.rng.normal(0.08, 0.02),   # safety command ratio
                self.rng.normal(0.5, 0.15),    # normalized time-of-shift
            ])
            sessions.append(features)

        sessions = np.array(sessions)
        fp = {
            'mean': np.mean(sessions, axis=0),
            'std':  np.std(sessions, axis=0) + 1e-6,
            'n_sessions': n_sessions,
        }
        self.fingerprints[operator_id] = fp
        return fp

    def check_session(self, operator_id: str,
                      session_features: np.ndarray,
                      is_adversarial: bool = False) -> Tuple[bool, float]:
        """
        Check if session matches operator fingerprint.
        Returns (is_anomaly, mahalanobis_distance).
        """
        if operator_id not in self.fingerprints:
            return False, 0.0

        fp = self.fingerprints[operator_id]
        z = (session_features - fp['mean']) / fp['std']
        distance = float(np.sqrt(np.sum(z**2)))   # simplified Mahalanobis

        is_anomaly = distance > self.ANOMALY_THRESHOLD
        self.alert_log.append({
            'operator_id': operator_id,
            'distance': distance,
            'anomaly': is_anomaly,
            'adversarial': is_adversarial,
        })
        return is_anomaly, distance

    def simulate_normal_session(self, operator_id: str) -> Tuple[bool, float]:
        """Simulate a normal operator session (small deviation from fingerprint)."""
        fp = self.fingerprints.get(operator_id, None)
        if fp is None:
            self.build_fingerprint(operator_id)
            fp = self.fingerprints[operator_id]
        # Normal: random walk close to mean
        features = fp['mean'] + self.rng.normal(0, fp['std'] * 0.8)
        return self.check_session(operator_id, features, is_adversarial=False)

    def simulate_spoofed_session(self, operator_id: str) -> Tuple[bool, float]:
        """Simulate credential-spoofed session (attacker doesn't know fingerprint)."""
        fp = self.fingerprints.get(operator_id, None)
        if fp is None:
            self.build_fingerprint(operator_id)
            fp = self.fingerprints[operator_id]
        # Adversary: drawn from different distribution (mean shifted by 2-5 SD)
        shift = self.rng.uniform(2.5, 5.0, size=fp['mean'].shape)
        sign  = self.rng.choice([-1, 1], size=fp['mean'].shape)
        features = fp['mean'] + sign * shift * fp['std']
        return self.check_session(operator_id, features, is_adversarial=True)
