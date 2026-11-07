"""GLM frekuensi dengan offset dan Gamma pada baris klaim positif."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from src.data.prepare import rate_to_count

CAT = ['Area', 'VehGas', 'Region', 'VehBrand']
NUM = ['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'log_density']


def features(frame):
    result = frame.copy()
    result['log_density'] = np.log1p(result.Density.clip(lower=0))
    return result[CAT+NUM]


def preprocessor():
    numeric = Pipeline([('median', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    categorical = Pipeline([('mode', SimpleImputer(strategy='most_frequent')),
                            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer([('category', categorical, CAT), ('numeric', numeric, NUM)])


def design(prep, frame):
    return sm.add_constant(prep.transform(features(frame)), has_constant='add')


def fit_gamma(target, x):
    """Optimize Gamma log-link coefficients before statsmodels IRLS inference."""
    y = np.asarray(target, dtype=float)
    scale = float(y.mean())
    z = y/scale
    start = np.zeros(x.shape[1])

    def objective(beta):
        eta = x@beta
        return float(np.mean(eta + z*np.exp(-eta)))

    def gradient(beta):
        return x.T@(1-z*np.exp(-(x@beta)))/len(y)

    def hessian(beta):
        weight = z*np.exp(-(x@beta))
        return (x.T*weight)@x/len(y)

    optimum = minimize(objective, start, jac=gradient, hess=hessian, method='trust-exact',
                       options={'gtol': 1e-8, 'maxiter': 500})
    max_gradient = float(np.max(np.abs(gradient(optimum.x))))
    if max_gradient > 1e-6:
        raise RuntimeError(f'Optimasi Gamma belum stasioner: {optimum.message}; grad={max_gradient}')
    beta = optimum.x.copy()
    beta[0] += np.log(scale)
    result = sm.GLM(y, x, family=sm.families.Gamma(sm.families.links.Log())).fit(
        start_params=beta, maxiter=50, tol=1e-6, rtol=1e-7, tol_criterion='params')
    final_beta = result.params.copy()
    final_beta[0] -= np.log(scale)
    final_gradient = float(np.max(np.abs(gradient(final_beta))))
    if final_gradient > 1e-5:
        raise RuntimeError(f'IRLS menggeser Gamma dari optimum: grad={final_gradient}')
    result.optimization_diagnostics = {'method': 'trust-exact on mean(eta + y_scaled*exp(-eta)), then IRLS',
        'optimizer_success': bool(optimum.success), 'optimizer_message': str(optimum.message),
        'optimizer_iterations': int(optimum.nit), 'max_absolute_gradient': max_gradient,
        'max_absolute_gradient_after_irls': final_gradient}
    return result


def fit_models(train, train_claims):
    prep = preprocessor()
    prep.fit(features(train))
    x = design(prep, train)
    offset = np.log(train.Exposure.to_numpy())
    poisson = sm.GLM(train.ClaimNb.to_numpy(), x, offset=offset,
                     family=sm.families.Poisson()).fit(maxiter=100)
    mu = poisson.predict(x, offset=offset)
    # NB2 method-of-moments from TRAIN residuals. Fixed plug-in alpha, not joint MLE.
    alpha = max(1e-6, float(np.sum((train.ClaimNb.to_numpy()-mu)**2-mu)/np.sum(mu**2)))
    nb = sm.GLM(train.ClaimNb.to_numpy(), x, offset=offset,
                family=sm.families.NegativeBinomial(alpha=alpha)).fit(maxiter=100)
    merged = train_claims.merge(train.drop(columns=['severity_count', 'total_amount'], errors='ignore'),
                               on='IDpol', how='inner', validate='many_to_one')
    if len(merged) < x.shape[1] + 5:
        raise ValueError('Terlalu sedikit klaim positif untuk fit Gamma.')
    cap = float(merged.ClaimAmount.quantile(.99))
    claim_x = design(prep, merged)
    gamma_limited = fit_gamma(merged.ClaimAmount.clip(upper=cap), claim_x)
    gamma_unlimited = fit_gamma(merged.ClaimAmount, claim_x)
    for name, model in [('Poisson', poisson), ('NB2', nb), ('Gamma limited', gamma_limited), ('Gamma unlimited', gamma_unlimited)]:
        if not model.converged or not np.isfinite(model.params).all():
            raise RuntimeError(f'{name} tidak konvergen.')
    boost_prep = preprocessor()
    bg = Pipeline([('prep', boost_prep), ('model', HistGradientBoostingRegressor(
        loss='poisson', max_iter=100, max_leaf_nodes=15, learning_rate=.08, random_state=2024,
        early_stopping=False))])
    with threadpool_limits(limits=2):
        bg.fit(features(train), train.ClaimNb/train.Exposure, model__sample_weight=train.Exposure)
    return {'prep': prep, 'poisson': poisson, 'nb': nb, 'alpha': alpha,
            'gamma_limited': gamma_limited, 'gamma_unlimited': gamma_unlimited,
            'boosting': bg, 'cap': cap, 'training_claims': merged}


def predict(models, frame):
    x = design(models['prep'], frame)
    # offset=0 gives annual rate; observed-period count then multiplies exposure.
    rate = models['poisson'].predict(x, offset=np.zeros(len(frame)))
    nb_rate = models['nb'].predict(x, offset=np.zeros(len(frame)))
    with threadpool_limits(limits=2):
        boost_rate = models['boosting'].predict(features(frame))
    severity = models['gamma_limited'].predict(x)
    unlimited = models['gamma_unlimited'].predict(x)
    return pd.DataFrame({'IDpol': frame.IDpol.to_numpy(), 'annual_frequency': rate,
                         'predicted_count': rate_to_count(rate, frame.Exposure),
                         'nb_annual_frequency': nb_rate, 'boost_annual_frequency': boost_rate,
                         'limited_severity': severity, 'unlimited_severity': unlimited,
                         'annual_pure_premium': rate*severity,
                         'period_cost': rate*severity*frame.Exposure.to_numpy(),
                         'boost_period_cost': boost_rate*severity*frame.Exposure.to_numpy(),
                         'unlimited_period_cost': rate*unlimited*frame.Exposure.to_numpy()})


def relativities(models):
    names = ['intercept'] + models['prep'].get_feature_names_out().tolist()
    rows = []
    for name in ['poisson', 'nb', 'gamma_limited', 'gamma_unlimited']:
        result = models[name]
        for term, beta, interval in zip(names, result.params, result.conf_int()):
            rows.append({'model': name, 'term': term, 'coefficient': beta,
                         'relativity': np.exp(beta), 'ci_low': np.exp(interval[0]),
                         'ci_high': np.exp(interval[1])})
    return pd.DataFrame(rows)
