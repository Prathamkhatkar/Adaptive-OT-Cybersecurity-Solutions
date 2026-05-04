# AOCSF — Adaptive OT Cybersecurity Solutions Framework

**Experimental Code for the Research Paper:**
> *Adaptive OT Cybersecurity Solutions: An Integrated Zero-Trust, Digital-Twin, and Human-Centric Framework for Resilient Industrial Control Systems*

**Author:** Pratham Khatkar
**Institution:** Department of Computer Science and Engineering, Chitkara University, Rajpura, India
**Contact:** pratham674.be22@chitkara.edu.in

---

## What This Repository Contains

This repository holds the complete simulation code used to generate the experimental results in the paper. Every number reported in Tables I, II, and III, and every figure in the paper, was produced by running this code. The results are fully reproducible using a fixed random seed.

| File | What It Does |
|------|-------------|
| `tep_sim.py` | Tennessee Eastman Process simulator (chemical plant model) |
| `abac_engine.py` | Attribute-Based Access Control engine + behavioral telemetry |
| `dt_detector.py` | Digital Twin anomaly detector + DRL Red/Blue agents |
| `run_experiments.py` | Runs all four experiments and generates all figures |

---

## Background — What Is This Research About?

Modern industrial facilities (power plants, water treatment, oil refineries) increasingly connect their control systems to the internet as part of Industry 4.0. This creates a serious security problem because the equipment was designed decades ago with no network threats in mind.

This project proposes and tests a three-part security framework called AOCSF:

1. **Zero-Trust Enforcement** — Every device and operator is continuously verified using a trust score, not just checked once at login
2. **Digital Twin Simulations** — A virtual copy of the plant runs in the background and flags any command that would cause a physically dangerous outcome, even if the command looks legitimate
3. **Human-Centric Training** — Operators are monitored for unusual behavior (catching credential theft) and trained through realistic simulated attack drills

---

## How to Run the Code

### Requirements

- Python 3.10 or higher
- pip (comes with Python)

### Installation

**Step 1 — Clone or download this repository**

If you have Git installed:
```bash
git clone https://github.com/YOUR_USERNAME/AOCSF-Experiments.git
cd AOCSF-Experiments
```

Or just download the ZIP from GitHub and unzip it.

**Step 2 — Install the required libraries**

Open a terminal (Command Prompt on Windows, Terminal on Mac/Linux) in the project folder and run:

```bash
pip install numpy scipy pandas matplotlib scikit-learn
```

This installs everything needed. It will take about 1–2 minutes.

**Step 3 — Run the experiments**

```bash
python run_experiments.py
```

That is all. The script runs all four experiments automatically and saves everything.

**Step 4 — Find your results**

After the script finishes (about 1–3 minutes), two new folders appear:

```
results/
    attack_mitigation_final.json    ← Table I data
    latency_benchmark.json          ← Table II data
    human_factor.json               ← Table III data
    drl_coevolution.json            ← DRL learning curve data

figures/
    fig1_architecture.png           ← Figure 1 (framework diagram)
    fig2_attack_mitigation.png      ← Figure 2 (attack detection results)
    fig3_latency_humanfactor.png    ← Figure 3 (latency + human factor)
    fig4_drl_coevolution.png        ← Figure 4 (DRL learning curves)
```

---

## What Each Experiment Does

### Experiment 1 — Attack Vector Mitigation
Tests how well the framework detects four types of attacks compared to a traditional VPN+IDS baseline:
- **False Data Injection (FDI)** — Injecting fake sensor readings into the process
- **Man-in-the-Middle (MitM)** — Intercepting control commands between operator and equipment
- **Replay Attacks** — Re-sending previously captured valid commands at a wrong time
- **Credential Spoofing** — An attacker pretending to be a legitimate operator

**How it works:** Runs 50 independent attack trials per attack type. For each trial, both the baseline system and the AOCSF are asked to detect the attack. Results are stored in `attack_mitigation_final.json`.

### Experiment 2 — Latency Benchmarking
Measures how long authentication takes in the AOCSF vs. the traditional VPN approach.

**How it works:** Generates 1,000 simulated authentication requests across four risk tiers (Safety-Critical, Process-Control, Supervisory, Telemetry) and times both systems. The AOCSF uses edge-resident processing, which is much faster than sending requests to a central server.

### Experiment 3 — Human Factor Study
Simulates 15 operators before and after a 3-week training programme using digital twin scenarios.

**How it works:** Builds a behavioral fingerprint for each operator, then measures how quickly they identify incidents and how many errors they make under stress. Pre-training and post-training results are compared using paired t-tests.

### Experiment 4 — DRL Red-Blue Co-evolution
Runs a deep reinforcement learning agent (the "Red Agent") inside the digital twin sandbox for 200 episodes. The Red Agent tries to find attack sequences that cause physical damage. A "Blue Agent" learns from each discovered attack and improves detection.

**How it works:** Uses tabular Q-learning (α=0.1, γ=0.95, ε-greedy decay to 0.05). The safety margin of the detection system increases as the Blue Agent adapts.

---

## Key Results

All results are reproducible with `seed=42`.

| Metric | Baseline | AOCSF | Improvement |
|--------|----------|-------|-------------|
| FDI Detection Rate | 14.0% | 96.0% | +82 pp |
| MitM Detection Rate | 34.0% | 82.0% | +48 pp |
| Auth Handshake Latency | 36.5 ms | 4.6 ms | 87.3% faster |
| Operator Incident ID Time | 314 s | 150 s | 52.1% faster |
| Stress-Induced Error Rate | 15.9% | 4.4% | 72.5% lower |

The most important finding is that **no single component achieves these results on its own**. When tested in isolation, Zero-Trust achieved 0% FDI detection, the Digital Twin achieved 0% credential spoofing detection, and Human-Centric training achieved 0% FDI detection. The three-pillar integrated design is what makes comprehensive coverage possible.

---

## Project Structure

```
AOCSF-Experiments/
│
├── tep_sim.py              # Tennessee Eastman Process simulator
├── abac_engine.py          # ABAC engine + behavioral telemetry
├── dt_detector.py          # Digital twin detector + DRL agents
├── run_experiments.py      # Main experiment runner
│
├── results/                # Generated automatically after running
│   ├── attack_mitigation_final.json
│   ├── latency_benchmark.json
│   ├── human_factor.json
│   └── drl_coevolution.json
│
├── figures/                # Generated automatically after running
│   ├── fig1_architecture.png
│   ├── fig2_attack_mitigation.png
│   ├── fig3_latency_humanfactor.png
│   └── fig4_drl_coevolution.png
│
└── README.md               # This file
```

---

## The Tennessee Eastman Process

The TEP is a standard benchmark in ICS (Industrial Control System) security research. It was originally published by Downs and Vogel (1993) in *Computers & Chemical Engineering* as a test case for process control. It models a chemical manufacturing process with:

- 41 measured process variables (temperatures, pressures, flow rates, compositions)
- 12 manipulated variables (valve positions, setpoints)
- 20 possible disturbance inputs

It is used in this research because its inherent instability means that FDI and setpoint manipulation attacks can escalate from subtle to dangerous quickly, which makes it a realistic and demanding test environment.

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.24 | Array operations, random number generation |
| scipy | ≥1.10 | Statistical tests (t-tests, confidence intervals) |
| pandas | ≥2.0 | Data handling |
| matplotlib | ≥3.7 | Figure generation |
| scikit-learn | ≥1.3 | Utility functions |

All dependencies install with one command:
```bash
pip install numpy scipy pandas matplotlib scikit-learn
```

---

## Reproducibility

All experiments use `SEED = 42` for the random number generator. Running `python run_experiments.py` on any machine with the same Python version will produce the same numerical results.

If you want to run with a different seed, open `run_experiments.py` and change the line near the top:
```python
MASTER_SEED = 42   # change this to any integer
```

---

## How to Cite

If you use this code in your own research, please cite the paper:

```
P. Khatkar, "Adaptive OT Cybersecurity Solutions: An Integrated Zero-Trust,
Digital-Twin, and Human-Centric Framework for Resilient Industrial Control
Systems," Chitkara University, Rajpura, India, 2025.
```

---

## License

This code is released for academic and research purposes. If you build on it, please cite the original paper and link back to this repository.

---

## Common Problems

**"ModuleNotFoundError: No module named 'numpy'"**
Run `pip install numpy scipy pandas matplotlib scikit-learn` again and make sure you are using the same Python installation.

**"python is not recognized as a command"**
On Windows, try `python3` instead of `python`. Or reinstall Python and make sure you checked "Add Python to PATH" during installation.

**The script runs but no figures folder appears**
Check that you are running the script from inside the project folder, not from a different directory.

**The numbers I get are slightly different from the paper**
This should not happen if SEED=42 is unchanged. If you changed the seed, different numbers are expected and correct.

---

*This research was conducted during an internship at Rockwell Automation, Noida, and developed further at Chitkara University, Rajpura, India.*
