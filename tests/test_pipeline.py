import numpy as np
import pandas as pd
import pytest

from src.data.prepare import prepare, pure_premium, rate_to_count, split_policies
from src.models import design, fit_gamma, fit_models, predict
from src.pipeline import deciles, run


@pytest.fixture
def portfolio():
    # Fixture sintetis hanya untuk pengujian, bukan angka laporan portofolio.
    rng = np.random.default_rng(12)
    n = 700
    freq = pd.DataFrame({'IDpol': np.arange(n), 'Exposure': rng.uniform(.3, 1, n),
        'Area': rng.choice(['A', 'B'], n), 'VehGas': rng.choice(['Diesel', 'Regular'], n),
        'Region': rng.choice(['R11', 'R12'], n), 'VehBrand': rng.choice(['B1', 'B2'], n),
        'VehPower': rng.integers(4, 10, n), 'VehAge': rng.integers(0, 20, n),
        'DrivAge': rng.integers(18, 80, n), 'BonusMalus': rng.integers(50, 150, n),
        'Density': rng.integers(10, 1000, n)})
    freq['ClaimNb'] = rng.poisson(freq.Exposure*.7)
    ids = np.repeat(freq.IDpol, freq.ClaimNb)
    sev = pd.DataFrame({'IDpol': ids, 'ClaimAmount': rng.gamma(2, 300, len(ids))})
    return freq, sev


def test_claim_aggregation_does_not_duplicate_exposure():
    freq = pd.DataFrame({'IDpol': [1, 2], 'Exposure': [.5, 1.], 'ClaimNb': [2, 0]})
    sev = pd.DataFrame({'IDpol': [1, 1], 'ClaimAmount': [100., 300.]})
    joined, _, audit = prepare(freq, sev)
    assert len(joined) == 2
    assert joined.Exposure.sum() == 1.5
    assert joined.loc[joined.IDpol == 1, 'total_amount'].item() == 400
    assert joined.loc[joined.IDpol == 1, 'severity_count'].item() == 2
    assert audit['count_mismatch_policies'] == 0


def test_split_keeps_all_claims_in_one_partition(portfolio):
    freq, sev = portfolio
    policies, claims, _ = prepare(freq, sev)
    train, test = split_policies(policies, max_policies=600)
    assert set(train.IDpol).isdisjoint(test.IDpol)
    assert len(train)+len(test) == 600
    train_claims = claims.loc[claims.IDpol.isin(train.IDpol)]
    test_claims = claims.loc[claims.IDpol.isin(test.IDpol)]
    assert set(train_claims.IDpol).isdisjoint(test_claims.IDpol)
    assert len(train_claims) == train.severity_count.sum()


def test_invalid_exposure_and_duplicate_ids(portfolio):
    freq, sev = portfolio
    freq.loc[0, 'Exposure'] = 0
    freq.loc[1, 'Exposure'] = -1
    policies, _, audit = prepare(freq, sev)
    assert audit['invalid_exposure_rows'] == 2
    assert (policies.Exposure > 0).all()
    with pytest.raises(ValueError):
        prepare(pd.concat([freq, freq.iloc[:1]]), sev)
    freq.loc[2, 'ClaimNb'] = .5
    with pytest.raises(ValueError):
        prepare(freq, sev)


def test_unmatched_claim_audit(portfolio):
    freq, sev = portfolio
    sev = pd.concat([sev, pd.DataFrame({'IDpol': [9999, 9999], 'ClaimAmount': [10., 20.]})])
    _, _, audit = prepare(freq, sev)
    assert audit['severity_ids_absent_from_frequency'] == 2
    assert audit['severity_unique_ids_absent'] == 1


def test_rate_count_premium_and_exposure():
    np.testing.assert_allclose(rate_to_count([.2, .4], [.5, 2]), [.1, .8])
    np.testing.assert_allclose(pure_premium([.2, .4], [100, 200]), [20, 80])
    np.testing.assert_allclose(pure_premium([.2, .4], [100, 200], [.5, 2]), [10, 160])
    with pytest.raises(ValueError):
        rate_to_count([1], [0])


def test_model_offset_preprocessing_and_train_cap(portfolio):
    freq, sev = portfolio
    policies, claims, _ = prepare(freq, sev)
    train, test = split_policies(policies)
    train_claims = claims.loc[claims.IDpol.isin(train.IDpol)]
    model = fit_models(train, train_claims)
    assert model['cap'] == pytest.approx(train_claims.ClaimAmount.quantile(.99))
    training_means = model['prep'].named_transformers_['numeric'].named_steps['scale'].mean_.copy()
    changed = test.copy()
    changed['Density'] = 1e12
    predict(model, changed)
    np.testing.assert_array_equal(training_means, model['prep'].named_transformers_['numeric'].named_steps['scale'].mean_)
    pred = predict(model, test)
    np.testing.assert_allclose(pred.predicted_count, pred.annual_frequency*test.Exposure.to_numpy())
    np.testing.assert_allclose(model['poisson'].predict(design(model['prep'], test), offset=np.log(test.Exposure)), pred.predicted_count)
    np.testing.assert_allclose(pred.period_cost, pred.annual_pure_premium*test.Exposure.to_numpy())
    doubled = test.copy()
    doubled.Exposure *= 2
    doubled_prediction = predict(model, doubled)
    np.testing.assert_allclose(doubled_prediction.period_cost, 2*pred.period_cost)


def test_exposure_weighted_deciles():
    frame = pd.DataFrame({'order': range(20), 'Exposure': [1]*10+[2]*10,
                          'actual': [2]*10+[8]*10, 'prediction': [2]*10+[8]*10})
    table = deciles(frame, 'order', 'actual', ['prediction'])
    assert table.exposure.sum() == 30
    actual = np.average(table.actual_annual_cost, weights=table.exposure)
    assert actual == pytest.approx(frame.actual.sum()/frame.Exposure.sum())
    assert np.average(table.actual_lift, weights=table.exposure) == pytest.approx(1)


def test_gamma_optimizer_is_stationary_and_invariant_to_currency_scale():
    rng = np.random.default_rng(80)
    x = np.column_stack([np.ones(120), rng.normal(size=120)])
    y = rng.gamma(2, np.exp(6+.3*x[:, 1])/2)
    original = fit_gamma(y, x)
    rescaled = fit_gamma(y*1e6, x)
    assert original.converged and rescaled.converged
    assert original.optimization_diagnostics['max_absolute_gradient_after_irls'] < 1e-5
    np.testing.assert_allclose(original.predict(x)*1e6, rescaled.predict(x), rtol=1e-6)


def test_pipeline_smoke_with_fixture(portfolio, tmp_path):
    freq, sev = portfolio
    result = run(freq, sev, output=tmp_path, max_policies=600)
    assert result['overlap_policy_ids'] == 0
    assert result['all_glm_converged']
    assert result['sample_policies'] == 600
    assert (tmp_path/'reports/lift.csv').exists()
    assert (tmp_path/'assets/lift.png').stat().st_size > 1000
