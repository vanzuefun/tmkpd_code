import numpy as np
from scipy.spatial import cKDTree


def standardize(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size < 5 or (not np.isfinite(x).all()):
        raise ValueError('A finite one-dimensional series with >=5 samples is required')
    sd = float(x.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError('The input series has zero or invalid variance')
    return np.ascontiguousarray((x - x.mean()) / sd)


def delay_vectors(x, m, tau):
    if m < 1 or tau < 1:
        raise ValueError('m and tau must be positive integers')
    n = len(x) - (m - 1) * tau
    if n < 2:
        raise ValueError('Insufficient samples for the embedding')
    return np.ascontiguousarray(x[np.arange(n)[:, None] + tau * np.arange(m)])


def tmkpd(x, m=2, tau=1, ell=1, k=0.3, min_count=50):
    """Return tmKPD and ordered-pair matching counts; x is a 1-D NumPy array."""
    if ell < 1 or k <= 0 or min_count < 1:
        raise ValueError('Invalid parameters')
    v = delay_vectors(x, m, tau)
    n = len(v) - ell
    if n < 2:
        raise ValueError('Insufficient transitions')
    a, b = (v[:n], v[ell:ell + n])
    ar, br = (a[:, ::-1], b[:, ::-1])
    radius = k * float(np.std(x, ddof=1))
    theiler = (m - 1) * tau + ell

    def count(left, right):
        total = int(cKDTree(left).count_neighbors(cKDTree(right), radius, p=np.inf))
        excluded = 0
        for offset in range(-min(theiler, n - 1), min(theiler, n - 1) + 1):
            start = max(0, -offset)
            stop = min(n, n - offset)
            distance = np.max(np.abs(
                left[start:stop] - right[start + offset:stop + offset]
            ), axis=1)
            excluded += int(np.count_nonzero(distance <= radius))
        return total - excluded
    joint = np.concatenate([a, b], axis=1)
    reverse_joint = np.concatenate([br, ar], axis=1)
    nf = count(a, a)
    nb = count(a, br)
    sf = count(joint, joint)
    sb = count(joint, reverse_joint)
    pf, pb = (sf / nf if nf else np.nan, sb / nb if nb else np.nan)
    valid = bool(min(nf, nb, sf, sb) >= min_count and np.isfinite(pf + pb) and (pf + pb > 0))
    delta = (pf - pb) / (pf + pb) if valid else np.nan
    return dict(
        value=float(abs(delta)), valid=valid,
        reason='' if valid else 'insufficient_matching_counts',
        delta=float(delta), Pf=float(pf), Pb=float(pb),
        Nf=nf, Nb=nb, Sf=sf, Sb=sb, radius=radius,
        theiler=theiler, vectors=n, state_dimension=m,
    )
