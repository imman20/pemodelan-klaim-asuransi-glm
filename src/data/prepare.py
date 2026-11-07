"""Validasi dan agregasi yang mempertahankan satu baris per polis."""
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 2024


def prepare(freq, sev):
    freq, sev = freq.copy(), sev.copy()
    if freq.IDpol.duplicated().any():
        raise ValueError('IDpol frekuensi berulang; tidak boleh menjumlah exposure otomatis.')
    if freq.IDpol.isna().any() or sev.IDpol.isna().any():
        raise ValueError('ID polis hilang.')
    for col in ['Exposure', 'ClaimNb']:
        if freq[col].isna().any() or not np.isfinite(freq[col]).all():
            raise ValueError(f'{col} tidak hingga atau hilang.')
    if (freq.ClaimNb < 0).any() or (freq.ClaimNb % 1 != 0).any():
        raise ValueError('ClaimNb harus bilangan bulat nonnegatif.')
    if sev.ClaimAmount.isna().any() or not np.isfinite(sev.ClaimAmount).all():
        raise ValueError('Nilai klaim tidak hingga atau hilang.')
    invalid = freq.Exposure <= 0
    nonpositive = sev.ClaimAmount <= 0
    audit = {'raw_policies': len(freq), 'raw_claim_rows': len(sev),
             'duplicate_policy_ids': int(freq.IDpol.duplicated().sum()),
             'exact_duplicate_claim_rows': int(sev.duplicated().sum()),
             'invalid_exposure_rows': int(invalid.sum()),
             'exposure_above_one_rows': int((freq.Exposure > 1).sum()),
             'nonpositive_claim_rows': int(nonpositive.sum()),
             'missing_cells_freq': int(freq.isna().sum().sum()),
             'missing_cells_sev': int(sev.isna().sum().sum()),
             'severity_ids_absent_from_frequency': int((~sev.IDpol.isin(freq.IDpol)).sum()),
             'severity_unique_ids_absent': int(sev.loc[~sev.IDpol.isin(freq.IDpol), 'IDpol'].nunique())}
    # Identical claim rows may be distinct real claims; no claim ID exists to deduplicate safely.
    freq = freq.loc[~invalid].copy()
    positive = sev.loc[~nonpositive].copy()
    aggregate = positive.groupby('IDpol').ClaimAmount.agg(severity_count='size', total_amount='sum')
    policies = freq.merge(aggregate, on='IDpol', how='left', validate='one_to_one')
    policies[['severity_count', 'total_amount']] = policies[['severity_count', 'total_amount']].fillna(0)
    if len(policies) != len(freq) or not np.isclose(policies.Exposure.sum(), freq.Exposure.sum()):
        raise AssertionError('Join menggandakan polis atau exposure.')
    audit['count_mismatch_policies'] = int((policies.ClaimNb != policies.severity_count).sum())
    audit['frequency_claim_total'] = int(policies.ClaimNb.sum())
    audit['matched_positive_claim_total'] = int(policies.severity_count.sum())
    audit['positive_claim_frequency_zero_policies'] = int(((policies.ClaimNb == 0) & (policies.severity_count > 0)).sum())
    return policies, positive, audit


def split_policies(policies, max_policies=100000, seed=SEED):
    sample = policies.sample(n=min(max_policies, len(policies)), random_state=seed).sort_values('IDpol')
    train_ids, test_ids = train_test_split(sample.IDpol, test_size=.2, random_state=seed)
    train = sample.loc[sample.IDpol.isin(train_ids)].copy()
    test = sample.loc[sample.IDpol.isin(test_ids)].copy()
    return train, test


def rate_to_count(rate, exposure):
    exposure = np.asarray(exposure)
    if (exposure <= 0).any() or not np.isfinite(exposure).all():
        raise ValueError('Exposure harus positif dan hingga.')
    return np.asarray(rate)*exposure


def pure_premium(annual_frequency, severity, exposure=1):
    return np.asarray(annual_frequency)*np.asarray(severity)*np.asarray(exposure)
