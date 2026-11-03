"""Multiclass history classifier, all-pairs refinement and fixed fusion rule."""
import itertools
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
POLICY = {"gamma": .25, "beta": 1., "rho": .5}
PRIOR_SETTING = {"name": "full_history__leaves7_min60", "view": "full_history", "leaves": 7, "minimum": 60}
PAIR_SETTING = {"name": "network_with_physical_changes__min60", "view": "network_with_physical_changes", "minimum": 60}
def features(arr, schema, setting):
    w, n = schema['total_width'], schema['network_width']
    assert arr['history'].shape[1] == w*4+3
    if setting['view'] == 'network_history':
        ids = np.r_[np.concatenate([np.arange(n)+w*k for k in range(4)]), np.arange(w*4, w*4+3)]
    elif setting['view'] == 'network_with_physical_changes':
        ids = np.r_[np.arange(n), np.arange(w, w*4+3)]
    else:
        ids = np.arange(w*4+3)
    return arr['history'][:, ids].astype(np.float64)


def fit_prior(arr, schema, setting):
    x = features(arr, schema, setting)
    scaler = StandardScaler().fit(x)
    y = arr['labels']
    weights = np.empty(len(y), dtype=np.float64)
    for label in range(schema['classes']):
        ids = np.flatnonzero(y == label)
        ids = ids[np.argsort(arr['raw_indices'][ids], kind='stable')]
        weights[ids] = len(y) / (schema['classes']*len(ids)) * np.linspace(1, 2, len(ids))
    weights /= weights.mean()
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, max_leaf_nodes=setting['leaves'],
        min_samples_leaf=setting['minimum'], l2_regularization=5, max_bins=128, early_stopping=False, random_state=0)
    model.fit(np.clip(scaler.transform(x), -10, 10), y, sample_weight=weights)
    np.testing.assert_array_equal(model.classes_, np.arange(schema['classes']))
    return {'setting': setting, 'schema': schema, 'scaler': scaler, 'model': model}


def probabilities(head, arr):
    return head['model'].predict_proba(np.clip(head['scaler'].transform(features(arr, head['schema'], head['setting'])), -10, 10))


def fit_pairs(arr, schema, setting, linear_source=None):
    x = features(arr, schema, setting)
    output = {'schema': schema, 'setting': setting, 'pairs': {}}
    for a, b in itertools.combinations(range(schema['classes']), 2):
        mask = np.isin(arr['labels'], [a, b])
        y = arr['labels'][mask]
        raw = arr['raw_indices'][mask]
        scaler = StandardScaler().fit(x[mask])
        values = np.clip(scaler.transform(x[mask]), -10, 10)
        weights = np.empty(len(y), dtype=np.float64)
        for label in [a, b]:
            ids = np.flatnonzero(y == label)
            ids = ids[np.argsort(raw[ids], kind='stable')]
            weights[ids] = len(y)/(2*len(ids))*np.linspace(1, 2, len(ids))
        weights /= weights.mean()
        if linear_source is None:
            linear = LogisticRegression(C=1, solver='lbfgs', tol=1e-8, max_iter=3000, random_state=0)
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                linear.fit(values, y, sample_weight=weights)
        else:
            linear = linear_source['pairs'][(a, b)]['linear']
            np.testing.assert_array_equal(scaler.mean_, linear_source['pairs'][(a, b)]['scaler'].mean_)
        tree = HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, max_leaf_nodes=15,
            min_samples_leaf=setting['minimum'], l2_regularization=5, max_bins=128, early_stopping=False, random_state=0)
        tree.fit(values, y, sample_weight=weights)
        for classifier in [linear, tree]:
            np.testing.assert_array_equal(classifier.classes_, [a, b])
        output['pairs'][(a, b)] = {'scaler': scaler, 'linear': linear, 'tree': tree}
    return output


def pair_probabilities(head, arr):
    x = features(arr, head['schema'], head['setting'])
    return {pair: np.stack([m.predict_proba(np.clip(items['scaler'].transform(x), -10, 10))[:, 1]
                           for m in [items['linear'], items['tree']]], axis=-1)
            for pair, items in head['pairs'].items()}


def decision(base, prior, pairs, setting):
    p = (1-setting['gamma'])*base+setting['gamma']*prior
    output = p.copy()
    top2 = np.sort(np.argsort(p, axis=1, kind='stable')[:, -2:], axis=1)
    for (a, b), probs in pairs.items():
        mask = (top2[:, 0] == a) & (top2[:, 1] == b)
        mass = p[mask, a]+p[mask, b]
        conditional = (1-setting['beta'])*probs[mask, 0]+setting['beta']*probs[mask, 1]
        refined = (1-setting['rho'])*p[mask, b]/np.maximum(mass, 1e-15)+setting['rho']*conditional
        output[mask, b] = mass*refined
        output[mask, a] = mass*(1-refined)
    np.testing.assert_allclose(output.sum(1), 1, atol=1e-6)
    return output
