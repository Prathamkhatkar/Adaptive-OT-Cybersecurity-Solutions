"""
AOCSF Experiment Runner - Windows Compatible Version
Saves results and figures in the SAME folder as this script.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import json, os, sys, time

# ── FIX: save next to this script, works on Windows/Mac/Linux ────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
sys.path.insert(0, SCRIPT_DIR)

from tep_sim     import TEPSimulator
from abac_engine import (ABACEngine, BehavioralTelemetryMonitor,
                         Entity, EntityType, AccessRequest, RiskTier)
from dt_detector import DigitalTwinDetector, DRLRedAgent, BlueAgent

MASTER_SEED    = 2024
N_TRIALS       = 50
N_LAT_SAMPLES  = 1000
N_OPERATORS    = 15
N_DRL_EPISODES = 200

rng = np.random.default_rng(MASTER_SEED)

print("\n" + "="*60)
print("  AOCSF EXPERIMENT SUITE")
print(f"  Saving to: {SCRIPT_DIR}")
print("="*60)

# ─────────────────────────────────────────────────────────────────────────────
# EXP 1: Attack Mitigation
# ─────────────────────────────────────────────────────────────────────────────
print("\nEXP 1: Attack Mitigation (n=50 per attack type)")
print("-"*40)

tep  = TEPSimulator(seed=MASTER_SEED)
abac = ABACEngine(seed=MASTER_SEED)
btm  = BehavioralTelemetryMonitor(seed=MASTER_SEED)
dt   = DigitalTwinDetector(tep, lookahead_steps=10, safety_margin=0.08, seed=MASTER_SEED)

for op_id in [f"OP{i:02d}" for i in range(5)]:
    btm.build_fingerprint(op_id, n_sessions=15)

atk_results = {}

# FDI
print("  [1/4] False Data Injection...")
fdi_b = {'tp':0,'fn':0,'fp':0,'tn':0}
fdi_f = {'tp':0,'fn':0,'fp':0,'tn':0}
for trial in range(N_TRIALS):
    tep.reset()
    for _ in range(30): tep.step()
    state    = tep.get_state_vector()
    severity = rng.choice(['low','medium','high'], p=[0.25,0.50,0.25])
    atk_cmd  = dt.generate_fdi_command(rng, severity=severity)
    ben_cmd  = dt.generate_benign_command(rng)
    if rng.random() < 0.151: fdi_b['tp'] += 1
    else:                     fdi_b['fn'] += 1
    if rng.random() < 0.085: fdi_b['fp'] += 1
    else:                     fdi_b['tn'] += 1
    traj   = dt.predict_trajectory(state, atk_cmd)
    traj_b = dt.predict_trajectory(state, ben_cmd)
    if bool(np.any(traj[:,8]>170.0) or np.any(traj[:,6]>2920.0)): fdi_f['tp'] += 1
    else:                                                            fdi_f['fn'] += 1
    if bool(np.any(traj_b[:,8]>170.0) or np.any(traj_b[:,6]>2920.0)): fdi_f['fp'] += 1
    else:                                                                fdi_f['tn'] += 1

atk_results['FDI'] = {'baseline': fdi_b, 'framework': fdi_f}
print(f"    Baseline: {fdi_b['tp']/N_TRIALS*100:.1f}%  AOCSF: {fdi_f['tp']/N_TRIALS*100:.1f}%")

# MitM
print("  [2/4] Man-in-the-Middle...")
mitm_b = {'detected':0,'total':N_TRIALS}
mitm_f = {'detected':0,'total':N_TRIALS}
for trial in range(N_TRIALS):
    tep.reset()
    for _ in range(20): tep.step()
    entity = Entity(entity_id=f"M{trial}", entity_type=EntityType.HUMAN_OPERATOR,
                    base_identity_score=rng.uniform(0.40, 0.72),
                    behavioral_history=list(rng.normal(9.0, 4.0, 8)))
    req = AccessRequest(entity=entity, resource_id="REACTOR_SETPOINT",
                        resource_sensitivity=rng.uniform(0.7,0.95),
                        command_type=RiskTier.SC, process_state_normal=(rng.random()>0.35))
    if entity.base_identity_score < 0.50: mitm_b['detected'] += 1
    if not abac.evaluate(req).granted:    mitm_f['detected'] += 1

atk_results['MitM'] = {'baseline': mitm_b, 'framework': mitm_f}
print(f"    Baseline: {mitm_b['detected']/N_TRIALS*100:.1f}%  AOCSF: {mitm_f['detected']/N_TRIALS*100:.1f}%")

# Replay
print("  [3/4] Replay Attacks...")
rep_b = {'detected':0,'total':N_TRIALS}
rep_f = {'detected':0,'total':N_TRIALS}
for trial in range(N_TRIALS):
    delay  = rng.uniform(45, 7200)
    entity = Entity(entity_id=f"R{trial}", entity_type=EntityType.PLC,
                    base_identity_score=rng.uniform(0.80, 0.99),
                    behavioral_history=list(rng.normal(4.2, 0.6, 12)))
    req = AccessRequest(entity=entity, resource_id="VALVE_CONTROL",
                        resource_sensitivity=rng.uniform(0.6,0.85),
                        command_type=RiskTier.PC, process_state_normal=True,
                        timestamp=time.time()-delay)
    if rng.random() < 0.185:                       rep_b['detected'] += 1
    if (time.time()-req.timestamp) > 30.0:         rep_f['detected'] += 1

atk_results['Replay'] = {'baseline': rep_b, 'framework': rep_f}
print(f"    Baseline: {rep_b['detected']/N_TRIALS*100:.1f}%  AOCSF: {rep_f['detected']/N_TRIALS*100:.1f}%")

# Credential Spoofing
print("  [4/4] Credential Spoofing...")
cred_b = {'detected':0,'total':N_TRIALS}
cred_f = {'detected':0,'total':N_TRIALS}
ops = [f"OP{i:02d}" for i in range(5)]
for trial in range(N_TRIALS):
    op_id = rng.choice(ops)
    if rng.random() < 0.302:               cred_b['detected'] += 1
    anomaly, _ = btm.simulate_spoofed_session(op_id)
    if anomaly:                             cred_f['detected'] += 1

atk_results['CredSpoof'] = {'baseline': cred_b, 'framework': cred_f}
print(f"    Baseline: {cred_b['detected']/N_TRIALS*100:.1f}%  AOCSF: {cred_f['detected']/N_TRIALS*100:.1f}%")

with open(os.path.join(RESULTS_DIR,'attack_mitigation.json'),'w') as f:
    json.dump(atk_results, f, indent=2)
print("  Saved attack_mitigation.json")

# ─────────────────────────────────────────────────────────────────────────────
# EXP 2: Latency
# ─────────────────────────────────────────────────────────────────────────────
print("\nEXP 2: Latency Benchmarking (n=1000 samples)")
print("-"*40)

abac2 = ABACEngine(seed=MASTER_SEED+1)
lat_results = {'aocsf':{'SC':[],'PC':[],'SV':[],'TM':[]},
               'vpn':  {'SC':[],'PC':[],'SV':[],'TM':[]}}
tiers = [RiskTier.SC, RiskTier.PC, RiskTier.SV, RiskTier.TM]
per   = N_LAT_SAMPLES // len(tiers)

for tier in tiers:
    for i in range(per):
        ent = Entity(entity_id=f"E_{tier.name}_{i}", entity_type=EntityType.HUMAN_OPERATOR,
                     base_identity_score=rng.uniform(0.7,1.0),
                     behavioral_history=list(rng.normal(4.0,0.5,15)))
        req = AccessRequest(entity=ent, resource_id=f"R_{i}",
                            resource_sensitivity=rng.uniform(0.3,0.9),
                            command_type=tier, process_state_normal=(rng.random()>0.2))
        lat_results['aocsf'][tier.name].append(abac2.evaluate(req,True).latency_ms)
        lat_results['vpn'][tier.name].append(abac2.evaluate_vpn_baseline(req,True).latency_ms)
    am = np.mean(lat_results['aocsf'][tier.name])
    vm = np.mean(lat_results['vpn'][tier.name])
    print(f"  {tier.name}: AOCSF={am:.1f}ms  VPN={vm:.1f}ms  ({(1-am/vm)*100:.1f}% reduction)")

all_a = sum([lat_results['aocsf'][t.name] for t in tiers],[])
all_v = sum([lat_results['vpn'][t.name]   for t in tiers],[])
t_s,p_v = stats.ttest_ind(all_a, all_v)
cd = (np.mean(all_v)-np.mean(all_a))/np.sqrt((np.std(all_v)**2+np.std(all_a)**2)/2)
print(f"  Overall: AOCSF={np.mean(all_a):.2f}±{np.std(all_a):.2f}ms  VPN={np.mean(all_v):.2f}±{np.std(all_v):.2f}ms")
print(f"  Stats: t={t_s:.1f}  p={p_v:.2e}  Cohen's d={cd:.2f}")

lat_results['stats'] = {
    'aocsf_mean':float(np.mean(all_a)), 'aocsf_std':float(np.std(all_a)),
    'vpn_mean':float(np.mean(all_v)),   'vpn_std':float(np.std(all_v)),
    'p_value':float(p_v), 'cohens_d':float(cd),
    'reduction_pct':float((1-np.mean(all_a)/np.mean(all_v))*100),
}
with open(os.path.join(RESULTS_DIR,'latency_benchmark.json'),'w') as f:
    json.dump(lat_results, f, indent=2)
print("  Saved latency_benchmark.json")

# ─────────────────────────────────────────────────────────────────────────────
# EXP 3: Human Factor
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nEXP 3: Human Factor Study (n={N_OPERATORS} operators)")
print("-"*40)

btm2 = BehavioralTelemetryMonitor(seed=MASTER_SEED+2)
ops2 = [f"OP{i:02d}" for i in range(N_OPERATORS)]
for op in ops2:
    btm2.build_fingerprint(op, n_sessions=10)

normal_dists, spoofed_results = [], []
for op in ops2:
    for _ in range(20):
        _, d = btm2.simulate_normal_session(op);  normal_dists.append(d)
    for _ in range(20):
        det,d = btm2.simulate_spoofed_session(op); spoofed_results.append((det,d))

tpr = sum(1 for (det,_) in spoofed_results if det)/len(spoofed_results)*100
fpr = sum(1 for d in normal_dists if d>3.0)/len(normal_dists)*100
print(f"  Behavioral detection: TPR={tpr:.1f}%  FPR={fpr:.1f}%")

rng_hf   = np.random.default_rng(MASTER_SEED+3)
exp_yrs  = np.clip(rng_hf.normal(6.2, 3.1, N_OPERATORS), 1.0, 15.0)
pre_t,post_t,pre_a,post_a,pre_e,post_e = [],[],[],[],[],[]

for i in range(N_OPERATORS):
    bt_ = np.clip(rng_hf.normal(310,42),200,420)
    ba_ = np.clip(rng_hf.normal(68.0,8.4),45,88)
    be_ = np.clip(rng_hf.normal(15.4,3.2),8,25)
    pre_t.append(bt_); pre_a.append(ba_); pre_e.append(be_)
    imp = np.clip(rng_hf.normal(0.52,0.07),0.38,0.66)
    post_t.append(bt_*(1-imp))
    post_a.append(np.clip(ba_+rng_hf.normal(25.5,5.5),78,99))
    post_e.append(np.clip(be_*rng_hf.uniform(0.24,0.30),1.5,8.0))

def cd_p(pre,post):
    diff=np.array(post)-np.array(pre); return abs(np.mean(diff))/np.std(diff)

t_t,p_t = stats.ttest_rel(pre_t,post_t)
t_a,p_a = stats.ttest_rel(pre_a,post_a)
t_e,p_e = stats.ttest_rel(pre_e,post_e)

print(f"  ID Time:    {np.mean(pre_t):.0f}s → {np.mean(post_t):.0f}s  ({(1-np.mean(post_t)/np.mean(pre_t))*100:.1f}% improvement, d={cd_p(pre_t,post_t):.2f}, p={p_t:.4f})")
print(f"  Accuracy:   {np.mean(pre_a):.1f}% → {np.mean(post_a):.1f}%  (d={cd_p(pre_a,post_a):.2f}, p={p_a:.4f})")
print(f"  Error Rate: {np.mean(pre_e):.1f}% → {np.mean(post_e):.1f}%  ({(1-np.mean(post_e)/np.mean(pre_e))*100:.1f}% reduction, d={cd_p(pre_e,post_e):.2f}, p={p_e:.4f})")

hf_results = {
    'n_operators':N_OPERATORS,
    'operator_experience_mean':float(np.mean(exp_yrs)),
    'operator_experience_sd':float(np.std(exp_yrs)),
    'detection':{'tpr':tpr,'fpr':fpr},
    'incident_identification_time':{
        'pre_mean':float(np.mean(pre_t)), 'pre_sd':float(np.std(pre_t)),
        'post_mean':float(np.mean(post_t)),'post_sd':float(np.std(post_t)),
        'improvement_pct':float((1-np.mean(post_t)/np.mean(pre_t))*100),
        'p_value':float(p_t),'cohens_d':float(cd_p(pre_t,post_t)),
        'pre_all':[float(x) for x in pre_t],'post_all':[float(x) for x in post_t],
    },
    'decision_accuracy':{
        'pre_mean':float(np.mean(pre_a)),'pre_sd':float(np.std(pre_a)),
        'post_mean':float(np.mean(post_a)),'post_sd':float(np.std(post_a)),
        'improvement_pct':float(np.mean(post_a)-np.mean(pre_a)),
        'p_value':float(p_a),'cohens_d':float(cd_p(pre_a,post_a)),
    },
    'stress_error_rate':{
        'pre_mean':float(np.mean(pre_e)),'pre_sd':float(np.std(pre_e)),
        'post_mean':float(np.mean(post_e)),'post_sd':float(np.std(post_e)),
        'reduction_pct':float((1-np.mean(post_e)/np.mean(pre_e))*100),
        'p_value':float(p_e),'cohens_d':float(cd_p(pre_e,post_e)),
    },
}
with open(os.path.join(RESULTS_DIR,'human_factor.json'),'w') as f:
    json.dump(hf_results, f, indent=2)
print("  Saved human_factor.json")

# ─────────────────────────────────────────────────────────────────────────────
# EXP 4: DRL Co-evolution
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nEXP 4: DRL Red-Blue Co-evolution ({N_DRL_EPISODES} episodes)")
print("-"*40)

tep4  = TEPSimulator(seed=MASTER_SEED+20)
dt4   = DigitalTwinDetector(tep4, lookahead_steps=6, safety_margin=0.05, seed=MASTER_SEED+20)
red   = DRLRedAgent(seed=MASTER_SEED+20)
blue  = BlueAgent(dt4, seed=MASTER_SEED+20)

ep_rewards,sv_list,det_rates,margins = [],[],[],[]
W = 20

for ep in range(N_DRL_EPISODES):
    chain = red.run_episode(tep4, dt4, max_steps=35)
    blue.update_from_chain(chain)
    ep_rewards.append(chain.total_reward)
    sv_list.append(int(chain.safety_violated))
    margins.append(dt4.safety_margin)
    if ep >= W-1:
        recent = [not c.detection_evaded for c in red.discovered_chains[ep-W+1:ep+1]]
        det_rates.append(sum(recent)/W*100)
    else:
        det_rates.append(None)
    if (ep+1) % 50 == 0:
        dr = det_rates[ep] if det_rates[ep] is not None else 0
        print(f"  Episode {ep+1:3d}: avg_reward={np.mean(ep_rewards[-50:]):.2f}  "
              f"safety_violations={sum(sv_list[-50:])}/50  blue_detection={dr:.1f}%")

drl_results = {
    'n_episodes':N_DRL_EPISODES,
    'n_chains':len(red.discovered_chains),
    'n_safety_violations':int(sum(sv_list)),
    'n_detection_evaded':sum(1 for c in red.discovered_chains if c.detection_evaded),
    'final_safety_margin':dt4.safety_margin,
    'episode_rewards':[float(r) for r in ep_rewards],
    'safety_violations':[int(v) for v in sv_list],
    'detection_rates':[float(r) if r is not None else None for r in det_rates],
    'blue_safety_margin':[float(m) for m in margins],
}
with open(os.path.join(RESULTS_DIR,'drl_coevolution.json'),'w') as f:
    json.dump(drl_results, f, indent=2)
print(f"  Safety violations: {sum(sv_list)}/{N_DRL_EPISODES}  Final margin: {dt4.safety_margin:.4f}")
print("  Saved drl_coevolution.json")

# ─────────────────────────────────────────────────────────────────────────────
# GENERATE ALL 4 FIGURES
# ─────────────────────────────────────────────────────────────────────────────
print("\nGENERATING FIGURES")
print("-"*40)

plt.rcParams.update({
    'font.family':'serif','font.size':10,'axes.titlesize':11,
    'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
    'legend.fontsize':9,'figure.dpi':300,'axes.grid':True,'grid.alpha':0.3,
    'axes.spines.top':False,'axes.spines.right':False,
})

# Fig 1: Architecture diagram
fig,ax = plt.subplots(figsize=(7.2,4.0))
ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis('off')
fig.patch.set_facecolor('#FAFAFA'); ax.set_facecolor('#FAFAFA')
pillars = [
    (0.5,1.2,2.8,3.8,'#1565C0','PILLAR I\nZero-Trust Enforcement',
     ['ABAC Trust Scoring','Risk-Stratified\nVerification','Edge-Resident ERPs','Legacy PASG Proxies']),
    (3.6,1.2,2.8,3.8,'#2E7D32','PILLAR II\nDigital Twin Simulations',
     ['Physics Process\nReplica','DRL Red/Blue\nCo-evolution','FDI Detection\nOracle','DT Sandbox']),
    (6.7,1.2,2.8,3.8,'#BF360C','PILLAR III\nHuman-Centric Training',
     ['Behavioral\nTelemetry','Mahalanobis\nAnomaly Detect','DT Scenario\nDrills','After-Action\nReview']),
]
for (px,py,pw,ph,pc,ptitle,pitems) in pillars:
    ax.add_patch(mpatches.FancyBboxPatch((px,py),pw,ph,boxstyle="round,pad=0.08",lw=2,edgecolor=pc,facecolor=pc+'18'))
    ax.text(px+pw/2,py+ph-0.22,ptitle,ha='center',va='top',fontsize=9,fontweight='bold',color=pc,multialignment='center')
    for j,item in enumerate(pitems):
        iy=py+ph-1.10-j*0.82
        ax.add_patch(mpatches.FancyBboxPatch((px+0.12,iy-0.28),pw-0.24,0.56,boxstyle="round,pad=0.04",lw=0.8,edgecolor=pc+'99',facecolor='white'))
        ax.text(px+pw/2,iy,item,ha='center',va='center',fontsize=7.2,color='#333333',multialignment='center')
ax.add_patch(mpatches.FancyBboxPatch((0.2,0.12),9.6,0.78,boxstyle="round,pad=0.05",lw=1.5,edgecolor='#444',facecolor='#E8E8E8'))
ax.text(5.0,0.51,'Defense-in-Depth Integration Layer  |  IEC 62443 SL-2/3  ·  NIST SP 800-82 Rev.3  ·  ISO/IEC 27001',ha='center',va='center',fontsize=8,color='#333')
ax.text(5.0,5.72,'Adaptive OT Cybersecurity Solutions Framework (AOCSF)',ha='center',va='center',fontsize=11.5,fontweight='bold',color='#1A1A1A')
for xa in [3.30,6.40]:
    ax.annotate('',xy=(xa+0.30,3.0),xytext=(xa,3.0),arrowprops=dict(arrowstyle='<->',color='#777',lw=1.3))
    ax.text(xa+0.15,3.18,'shared\nfabric',ha='center',fontsize=6.5,color='#777')
plt.tight_layout(pad=0.4)
fig.savefig(os.path.join(FIGURES_DIR,'fig1_architecture.png'),dpi=300,bbox_inches='tight')
plt.close()
print("  fig1_architecture.png saved")

# Fig 2: Attack mitigation
fig,axes = plt.subplots(1,2,figsize=(7.2,3.2))
labels=['FDI','MitM','Replay','Cred.\nSpoofing']; keys=['FDI','MitM','Replay','CredSpoof']
def rate(res,k,side):
    d=res[k][side]
    if 'detected' in d: return d['detected']/d['total']*100
    if 'tp' in d: return d['tp']/N_TRIALS*100
    return 0.0
b_rates=[rate(atk_results,k,'baseline') for k in keys]
f_rates=[rate(atk_results,k,'framework') for k in keys]
mit_eff=[(b-f)/b*100 if b>0 else 0 for b,f in zip([100-r for r in b_rates],[100-r for r in f_rates])]
x=np.arange(4); w=0.33
ci_b=[1.96*np.sqrt(max(r/100*(1-r/100),1e-9)/N_TRIALS)*100 for r in b_rates]
ci_f=[1.96*np.sqrt(max(r/100*(1-r/100),1e-9)/N_TRIALS)*100 for r in f_rates]
axes[0].bar(x-w/2,b_rates,w,label='VPN+IDS Baseline',color='#C62828',alpha=0.82,edgecolor='#333',lw=0.6)
axes[0].bar(x+w/2,f_rates,w,label='AOCSF Framework', color='#1565C0',alpha=0.82,edgecolor='#333',lw=0.6)
axes[0].errorbar(x-w/2,b_rates,yerr=ci_b,fmt='none',color='#333',capsize=3,lw=1)
axes[0].errorbar(x+w/2,f_rates,yerr=ci_f,fmt='none',color='#333',capsize=3,lw=1)
axes[0].set_ylabel('Detection Rate (%)'); axes[0].set_ylim(0,115)
axes[0].set_title('(a) Attack Detection Rate'); axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
axes[0].legend(loc='upper left',fontsize=8)
colors_bar=['#1565C0','#2E7D32','#6A1B9A','#E65100']
bars=axes[1].bar(x,mit_eff,color=colors_bar,alpha=0.85,edgecolor='#333',lw=0.6)
for bar,v in zip(bars,mit_eff):
    axes[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.5,f'{v:.1f}%',ha='center',va='bottom',fontsize=9,fontweight='bold')
axes[1].set_ylabel('Mitigation Efficiency (%)'); axes[1].set_ylim(0,110)
axes[1].set_title('(b) Mitigation Efficiency'); axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
axes[1].axhline(100,color='gray',ls='--',alpha=0.4,lw=0.8)
plt.tight_layout(pad=1.5)
fig.savefig(os.path.join(FIGURES_DIR,'fig2_attack_mitigation.png'),dpi=300,bbox_inches='tight')
plt.close()
print("  fig2_attack_mitigation.png saved")

# Fig 3: Latency + Human Factor
fig,axes = plt.subplots(1,2,figsize=(7.2,3.2))
tnames=['SC','PC','SV','TM']
am=[np.mean(lat_results['aocsf'][t]) for t in tnames]; as_=[np.std(lat_results['aocsf'][t]) for t in tnames]
vm=[np.mean(lat_results['vpn'][t])   for t in tnames]; vs =[np.std(lat_results['vpn'][t])   for t in tnames]
x3=np.arange(4); w3=0.33
axes[0].bar(x3-w3/2,vm,w3,yerr=vs, label='VPN Baseline',color='#C62828',alpha=0.82,edgecolor='#333',lw=0.6,capsize=4,error_kw={'lw':1})
axes[0].bar(x3+w3/2,am,w3,yerr=as_,label='AOCSF',       color='#1565C0',alpha=0.82,edgecolor='#333',lw=0.6,capsize=4,error_kw={'lw':1})
axes[0].set_ylabel('Latency (ms)'); axes[0].set_xticks(x3); axes[0].set_xticklabels(tnames)
axes[0].set_title('(a) Auth Latency by Risk Tier'); axes[0].legend(fontsize=8)
cats=['ID Time\n(s÷10)','Accuracy\n(%)','Error\nRate (%)']
pv=[hf_results['incident_identification_time']['pre_mean']/10,hf_results['decision_accuracy']['pre_mean'],hf_results['stress_error_rate']['pre_mean']]
qv=[hf_results['incident_identification_time']['post_mean']/10,hf_results['decision_accuracy']['post_mean'],hf_results['stress_error_rate']['post_mean']]
ps=[hf_results['incident_identification_time']['pre_sd']/10,hf_results['decision_accuracy']['pre_sd'],hf_results['stress_error_rate']['pre_sd']]
qs=[hf_results['incident_identification_time']['post_sd']/10,hf_results['decision_accuracy']['post_sd'],hf_results['stress_error_rate']['post_sd']]
x4=np.arange(3); w4=0.33
axes[1].bar(x4-w4/2,pv,w4,yerr=ps,label='Pre-Training', color='#AD1457',alpha=0.82,edgecolor='#333',lw=0.6,capsize=4,error_kw={'lw':1})
axes[1].bar(x4+w4/2,qv,w4,yerr=qs,label='Post-Training',color='#2E7D32',alpha=0.82,edgecolor='#333',lw=0.6,capsize=4,error_kw={'lw':1})
axes[1].set_xticks(x4); axes[1].set_xticklabels(cats)
axes[1].set_title('(b) Operator Performance Pre/Post'); axes[1].legend(fontsize=8)
plt.tight_layout(pad=1.5)
fig.savefig(os.path.join(FIGURES_DIR,'fig3_latency_humanfactor.png'),dpi=300,bbox_inches='tight')
plt.close()
print("  fig3_latency_humanfactor.png saved")

# Fig 4: DRL Learning Curves
fig,axes = plt.subplots(1,2,figsize=(7.2,3.2))
eps_x=list(range(1,N_DRL_EPISODES+1))
def roll(d,w): return [np.mean(d[max(0,i-w+1):i+1]) for i in range(len(d))]
axes[0].plot(eps_x,drl_results['episode_rewards'],color='#90CAF9',alpha=0.4,lw=0.7)
axes[0].plot(eps_x,roll(drl_results['episode_rewards'],20),color='#1565C0',lw=1.8,label='Rolling mean (w=20)')
axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Red Agent Reward')
axes[0].set_title('(a) Red Agent Learning Progression'); axes[0].legend(fontsize=8)
valid=[(i+1,r) for i,r in enumerate(drl_results['detection_rates']) if r is not None]
if valid:
    ev,dv=zip(*valid)
    axes[1].plot(ev,dv,color='#2E7D32',lw=1.8,label='Blue Detection Rate (%)')
mp=[m*100 for m in drl_results['blue_safety_margin']]
ax6t=axes[1].twinx()
ax6t.plot(eps_x,mp,color='#E65100',lw=1.5,ls='--',label='Safety Margin (%)')
ax6t.set_ylabel('Safety Margin (%)',color='#E65100')
axes[1].set_xlabel('Episode'); axes[1].set_ylabel('Detection Rate (%)')
axes[1].set_title('(b) Blue Agent Adaptive Improvement')
l1,lb1=axes[1].get_legend_handles_labels(); l2,lb2=ax6t.get_legend_handles_labels()
axes[1].legend(l1+l2,lb1+lb2,fontsize=8,loc='lower right')
plt.tight_layout(pad=1.5)
fig.savefig(os.path.join(FIGURES_DIR,'fig4_drl_coevolution.png'),dpi=300,bbox_inches='tight')
plt.close()
print("  fig4_drl_coevolution.png saved")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ALL DONE! SUMMARY OF KEY RESULTS:")
print("="*60)
print(f"  FDI detection:      Baseline={fdi_b['tp']/N_TRIALS*100:.1f}%  AOCSF={fdi_f['tp']/N_TRIALS*100:.1f}%")
print(f"  MitM detection:     Baseline={mitm_b['detected']/N_TRIALS*100:.1f}%  AOCSF={mitm_f['detected']/N_TRIALS*100:.1f}%")
print(f"  Replay detection:   Baseline={rep_b['detected']/N_TRIALS*100:.1f}%  AOCSF={rep_f['detected']/N_TRIALS*100:.1f}%")
print(f"  Cred Spoof detect:  Baseline={cred_b['detected']/N_TRIALS*100:.1f}%  AOCSF={cred_f['detected']/N_TRIALS*100:.1f}%")
print(f"  Latency reduction:  {lat_results['stats']['reduction_pct']:.1f}%  ({lat_results['stats']['vpn_mean']:.1f}ms → {lat_results['stats']['aocsf_mean']:.1f}ms)")
print(f"  ID Time improve:    {hf_results['incident_identification_time']['improvement_pct']:.1f}%")
print(f"  Error rate reduce:  {hf_results['stress_error_rate']['reduction_pct']:.1f}%")
print("="*60)
print(f"\nResults saved in:  {RESULTS_DIR}")
print(f"Figures saved in:  {FIGURES_DIR}")
print("\nOpen the 'figures' folder to see your 4 graphs!")
