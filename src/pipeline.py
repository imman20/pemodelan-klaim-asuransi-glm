"""Pipeline GLM dengan sampel deterministik polis dan audit data sumber penuh."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance
import statsmodels.api as sm

from src.data.download import load
from src.data.prepare import prepare, split_policies
from src.models import CAT, NUM, design, fit_models, predict, relativities


def deciles(frame, order, actual, predictions):
    ordered = frame.sort_values(order, kind='stable').copy()
    # Bin boundaries are cumulative exposure, not equal policy counts.
    midpoint = ordered.Exposure.cumsum() - ordered.Exposure/2
    ordered['decile'] = np.minimum((10*midpoint/ordered.Exposure.sum()).astype(int)+1, 10)
    rows = []
    portfolio_rate = ordered[actual].sum()/ordered.Exposure.sum()
    for decile, group in ordered.groupby('decile'):
        exp = group.Exposure.sum()
        row = {'decile': int(decile), 'policies': len(group), 'exposure': exp,
               'actual_annual_cost': group[actual].sum()/exp,
               'actual_lift': (group[actual].sum()/exp)/portfolio_rate}
        for pred in predictions:
            row[pred] = group[pred].sum()/exp
        rows.append(row)
    return pd.DataFrame(rows)


def run(freq=None, sev=None, output='.', max_policies=100000):
    root = Path(output)
    for folder in ['reports', 'assets']:
        (root/folder).mkdir(parents=True, exist_ok=True)
    if freq is None:
        freq, sev = load()
    policies, claims, audit = prepare(freq, sev)
    train, test = split_policies(policies, max_policies=max_policies)
    train_claims = claims.loc[claims.IDpol.isin(train.IDpol)]
    test_claims = claims.loc[claims.IDpol.isin(test.IDpol)]
    models = fit_models(train, train_claims)
    pred = predict(models, test)
    cap = models['cap']
    limited_agg = claims.assign(limited_amount=claims.ClaimAmount.clip(upper=cap)).groupby('IDpol').limited_amount.sum()
    observed = test[['IDpol', 'Exposure', 'ClaimNb', 'severity_count', 'total_amount']].merge(pred, on='IDpol', validate='one_to_one')
    observed['limited_amount'] = observed.IDpol.map(limited_agg).fillna(0)
    observed['double_lift_ratio'] = observed.boost_period_cost/observed.period_cost
    rows = []
    for name, rate in [('Poisson GLM', pred.annual_frequency),
                       ('NB2 GLM', pred.nb_annual_frequency),
                       ('Gradient boosting', pred.boost_annual_frequency)]:
        count = rate.to_numpy()*observed.Exposure.to_numpy()
        rows.append({'model': name, 'test_count_poisson_deviance': mean_poisson_deviance(observed.ClaimNb, count),
                     'test_nb2_deviance_at_train_alpha': float(sm.families.NegativeBinomial(
                         alpha=models['alpha']).deviance(observed.ClaimNb.to_numpy(), count)/len(count)),
                     'test_exposure_weighted_rate_deviance': mean_poisson_deviance(
                         observed.ClaimNb/observed.Exposure, rate, sample_weight=observed.Exposure),
                     'actual_claims': int(observed.ClaimNb.sum()), 'predicted_claims': float(count.sum()),
                     'count_calibration_ratio': count.sum()/observed.ClaimNb.sum()})
    evaluation = pd.DataFrame(rows)
    evaluation.to_csv(root/'reports/evaluation.csv', index=False)
    test_claim_design = test_claims.merge(test.drop(columns=['severity_count', 'total_amount']),
                                         on='IDpol', validate='many_to_one')
    claim_x = design(models['prep'], test_claim_design)
    severity_rows = []
    for label, target in [('limited', test_claim_design.ClaimAmount.clip(upper=cap)),
                           ('unlimited', test_claim_design.ClaimAmount)]:
        fitted = models['gamma_'+label]
        pp = fitted.predict(claim_x)
        severity_rows.append({'target': label, 'test_claim_rows': len(target),
                              'gamma_deviance': mean_gamma_deviance(target, pp),
                              'actual_mean': target.mean(), 'predicted_mean': pp.mean(),
                              'train_aic': fitted.aic})
    severity_evaluation = pd.DataFrame(severity_rows)
    severity_evaluation.to_csv(root/'reports/severity_evaluation.csv', index=False)
    rel = relativities(models)
    rel.to_csv(root/'reports/relativities.csv', index=False)
    prep = models['prep']
    references = dict(zip(CAT, [str(c[0]) for c in prep.named_transformers_['category'].named_steps['onehot'].categories_]))
    scale = dict(zip(NUM, prep.named_transformers_['numeric'].named_steps['scale'].scale_.tolist()))
    lift = deciles(observed, 'annual_pure_premium', 'limited_amount', ['period_cost', 'boost_period_cost'])
    double = deciles(observed, 'double_lift_ratio', 'limited_amount', ['period_cost', 'boost_period_cost'])
    lift.to_csv(root/'reports/lift.csv', index=False)
    double.to_csv(root/'reports/double_lift.csv', index=False)
    observed.to_csv(root/'reports/test_predictions.csv', index=False)
    pd.concat([train[['IDpol']].assign(split='train'), test[['IDpol']].assign(split='test')]).to_csv(root/'reports/split.csv', index=False)
    factors = []
    for factor in CAT:
        table = train.groupby(factor, dropna=False).agg(policies=('IDpol', 'size'),
                    claims=('ClaimNb', 'sum'), exposure=('Exposure', 'sum')).reset_index()
        table['annual_frequency'] = table.claims/table.exposure
        table['factor'] = factor
        table = table.rename(columns={factor: 'category'})
        factors.append(table)
    eda = pd.concat(factors, ignore_index=True)
    eda.to_csv(root/'reports/eda_rates.csv', index=False)
    train_amounts = train_claims.ClaimAmount
    sensitivity = []
    for quantile in [.95, .99, 1.0]:
        threshold = float(train_amounts.quantile(quantile))
        sensitivity.append({'train_quantile': quantile, 'train_limit': threshold,
                            'train_claims_above': int((train_amounts > threshold).sum()),
                            'test_claims_above': int((test_claims.ClaimAmount > threshold).sum()),
                            'test_total_limited_cost': float(test_claims.ClaimAmount.clip(upper=threshold).sum())})
    # q=1 means the train maximum, not an unlimited target; report uncapped explicitly.
    sensitivity.append({'train_quantile': 'unlimited', 'train_limit': np.nan,
                        'train_claims_above': 0, 'test_claims_above': 0,
                        'test_total_limited_cost': float(test_claims.ClaimAmount.sum())})
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(root/'reports/tail_sensitivity.csv', index=False)
    result = {'seed': 2024, 'sampling_rule': 'simple random sample of policy IDs before 80:20 split',
              'requested_max_policies': max_policies, 'sample_policies': len(train)+len(test),
              'train_policies': len(train), 'test_policies': len(test),
              'train_claim_rows': len(train_claims), 'test_claim_rows': len(test_claims),
              'overlap_policy_ids': len(set(train.IDpol) & set(test.IDpol)),
              'audit': audit, 'train_cap_q99': cap,
              'train_pearson_dispersion': float(models['poisson'].pearson_chi2/models['poisson'].df_resid),
              'nb2_alpha_train_moments': models['alpha'],
              'poisson_train_aic': float(models['poisson'].aic),
              'nb_train_aic_fixed_alpha': float(models['nb'].aic),
              'nb_train_aic_plus_alpha_parameter': float(models['nb'].aic+2),
              'references': references, 'numeric_sd_train': scale,
              'frequency_evaluation': rows, 'severity_evaluation': severity_rows,
              'test_exposure': float(test.Exposure.sum()),
              'test_actual_limited_cost': float(observed.limited_amount.sum()),
              'test_predicted_limited_cost': float(observed.period_cost.sum()),
              'test_limited_cost_calibration_ratio': float(observed.period_cost.sum()/observed.limited_amount.sum()),
              'test_boost_predicted_limited_cost': float(observed.boost_period_cost.sum()),
              'test_actual_unlimited_cost': float(observed.total_amount.sum()),
              'test_predicted_unlimited_cost': float(observed.unlimited_period_cost.sum()),
              'train_claim_cost_removed_by_cap': float((train_claims.ClaimAmount-train_claims.ClaimAmount.clip(upper=cap)).sum()),
              'test_claim_cost_removed_by_cap': float((test_claims.ClaimAmount-test_claims.ClaimAmount.clip(upper=cap)).sum()),
              'all_glm_converged': True,
              'gamma_optimization': {name: models[name].optimization_diagnostics for name in
                                     ['gamma_limited', 'gamma_unlimited']},
              'glm_iterations': {name: models[name].fit_history['iteration'] for name in
                                 ['poisson', 'nb', 'gamma_limited', 'gamma_unlimited']}}
    (root/'reports/hasil.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, table, title in [(axes[0], lift, 'Lift: desil premi GLM'),
                              (axes[1], double, 'Double lift: desil rasio boosting / GLM')]:
        ax.plot(table.decile, table.actual_annual_cost, 'o-', label='Aktual terbatas')
        ax.plot(table.decile, table.period_cost, 's-', label='Prediksi GLM')
        ax.plot(table.decile, table.boost_period_cost, '^-', label='Prediksi boosting')
        ax.set(title=title, xlabel='Desil berbobot exposure', ylabel='Biaya tahunan per exposure (EUR)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root/'assets/lift.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    area = eda.query("factor == 'Area'")
    ax.bar(area.category, area.annual_frequency, color='#087e8b')
    ax.set(title='Frekuensi klaim tahunan per area, data latih', ylabel='Total klaim / total exposure', xlabel='Area')
    fig.tight_layout()
    fig.savefig(root/'assets/frequency_area.png', dpi=160)
    plt.close(fig)
    from src.reporting import write_reports
    write_reports(root, result, evaluation, severity_evaluation, lift, double, sensitivity)
    from src.artifacts import normalize_text
    normalize_text([*(root/'reports').glob('*'), root/'README.md'])
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-policies', type=int, default=100000)
    args = parser.parse_args()
    if args.max_policies <= 0:
        parser.error('--max-policies harus positif')
    run(max_policies=args.max_policies)
