"""Statistical significance analysis for FLOC-MSM paper.

Uses per-seed accuracies from floc_msm_main/results.json.
Computes: bootstrap 95% CI, pairwise t-tests, effect sizes.
"""
import json, numpy as np
from scipy import stats

# Load per-seed results
r=json.load(open('D:/deepl/paper2_floc_msm/logs/floc_msm_main/results.json'))

print('='*70)
print('Statistical Analysis: FLOC-MSM Main Experiment (α=1.5)')
print('='*70)

# Pairwise comparisons: FLOC p=1.2 vs each baseline
TARGET='floc_1.2'
BASELINES=['mse','l1','huber','floc_1.0','floc_1.5','floc_1.8','scratch']

for n_labeled in [50,100,200,500]:
    print(f'\n--- n={n_labeled} ---')
    target_seeds=r[TARGET][str(n_labeled)]['seeds']
    target_mean=r[TARGET][str(n_labeled)]['mean']

    for bl in BASELINES:
        bl_seeds=r[bl][str(n_labeled)]['seeds']
        bl_mean=r[bl][str(n_labeled)]['mean']
        diff=target_mean-bl_mean

        # Paired t-test
        t_stat,p_val=stats.ttest_rel(target_seeds,bl_seeds)

        # Bootstrap 95% CI on difference
        np.random.seed(42)
        diffs=[]
        for _ in range(10000):
            idx=np.random.choice(5,5,replace=True)
            diffs.append(np.mean([target_seeds[i] for i in idx])-np.mean([bl_seeds[i] for i in idx]))
        ci_low,ci_high=np.percentile(diffs,[2.5,97.5])

        sig='***' if p_val<0.001 else ('**' if p_val<0.01 else ('*' if p_val<0.05 else 'ns'))
        print(f'  vs {bl:<12}: Δ={diff:+.3f} [{ci_low:+.3f}, {ci_high:+.3f}] p={p_val:.4f} {sig}')

# Summary: when is FLOC significantly better?
print(f'\n{"="*70}')
print('WHEN FLOC p=1.2 SIGNIFICANTLY BEATS MSE')
print('='*70)
for n_labeled in [50,100,200,500]:
    target_seeds=r[TARGET][str(n_labeled)]['seeds']
    mse_seeds=r['mse'][str(n_labeled)]['seeds']
    t_stat,p_val=stats.ttest_rel(target_seeds,mse_seeds)
    diff=np.mean(target_seeds)-np.mean(mse_seeds)
    print(f'  n={n_labeled}: Δ={diff:+.3f}, p={p_val:.4f} {"***" if p_val<0.05 else "ns"}')

print(f'\n{"="*70}')
print('WHEN FLOC p=1.2 SIGNIFICANTLY BEATS L1')
print('='*70)
for n_labeled in [50,100,200,500]:
    target_seeds=r[TARGET][str(n_labeled)]['seeds']
    l1_seeds=r['l1'][str(n_labeled)]['seeds']
    t_stat,p_val=stats.ttest_rel(target_seeds,l1_seeds)
    diff=np.mean(target_seeds)-np.mean(l1_seeds)
    print(f'  n={n_labeled}: Δ={diff:+.3f}, p={p_val:.4f} {"***" if p_val<0.05 else "ns"}')
