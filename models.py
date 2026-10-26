import numpy as np
from scipy.integrate import solve_ivp
MODEL_NAMES = ('IID-Gaussian', 'IID-Exponential', 'GaussianAR03', 'GaussianAR08', 'MonotoneAR', 'SkewAR03', 'SkewAR08', 'QuadraticMA', 'Logistic', 'Henon', 'Lorenz')
FAMILIES = ('GaussianAR', 'MixedInnovationAR', 'ConditionalMatching')
_COMPARISON_SEED = 20261002
_COMPARISON_STAGE = 901
_MATRICES = [{'model': 'M0',
  'matrix': [[0.32765939064145183,
              0.20298151077359414,
              0.06354795701194496,
              0.4058111415730091],
             [0.1140821383514797,
              0.16621489816488935,
              0.6561549476142028,
              0.06354801586942815],
             [0.21528563202479953,
              0.4155178535082566,
              0.16621485860282623,
              0.20298165586411765],
             [0.34297283898226893,
              0.21528573755325997,
              0.11408223677102602,
              0.3276591866934453]],
  'stationary': [0.25, 0.25, 0.25, 0.25]},
 {'model': 'M1',
  'matrix': [[0.34042079673286246,
              0.1946529333664103,
              0.05533965942970872,
              0.4095866104710185],
             [0.13448855460640682,
              0.17453021474963568,
              0.6085940763697786,
              0.08238715427417898],
             [0.26407666951165426,
              0.42882461286989265,
              0.21138616205668215,
              0.09571255556177094],
             [0.2610139791490764,
              0.20199223901406138,
              0.12468010214383052,
              0.4123136796930318]],
  'stationary': [0.25, 0.25, 0.25, 0.25]},
 {'model': 'M2',
  'matrix': [[0.32446194179672283,
              0.17844128651706562,
              0.048106234110047365,
              0.44899053757616425],
             [0.2477381717397803,
              0.2160498402397228,
              0.46388555908764173,
              0.07232642893285512],
             [0.19923679627524846,
              0.4115842442914284,
              0.3109503629025121,
              0.07822859653081105],
             [0.22856309018824839,
              0.19392462895178308,
              0.17705784389979884,
              0.4004544369601697]],
  'stationary': [0.25, 0.25, 0.25, 0.25]},
 {'model': 'M3',
  'matrix': [[0.34786705968830706,
              0.2254537050835064,
              0.039371960979026364,
              0.3873072742491601],
             [0.30495118671374055,
              0.26889404490448665,
              0.30827540350998256,
              0.11787936487179018],
             [0.1757459905826887,
              0.29695407161021237,
              0.46447432669693145,
              0.06282561111016749],
             [0.17143576301526375,
              0.2086981784017946,
              0.18787830881405965,
              0.431987749768882]],
  'stationary': [0.25, 0.25, 0.25, 0.25]},
 {'model': 'M4',
  'matrix': [[0.3749860177479917,
              0.3011853298965564,
              0.08922004697799603,
              0.23460860537745587],
             [0.30480269413312655,
              0.291595554710112,
              0.17417504530246078,
              0.2294267058543007],
             [0.22971578505223933,
              0.23675185533145593,
              0.46793819430041,
              0.06559416531589468],
             [0.09049550306664245,
              0.17046726006187563,
              0.2686667134191332,
              0.470370523452349]],
  'stationary': [0.25, 0.25, 0.25, 0.25]}]

def _generate_map(model, n, rng, burn=1200, lorenz_dt=0.05):
    total = n + burn
    if model == 'Logistic':
        x = np.empty(total)
        x[0] = rng.uniform(0.1, 0.9)
        for i in range(1, total):
            x[i] = 4 * x[i - 1] * (1 - x[i - 1])
        x = x[burn:]
    elif model == 'Henon':
        x = np.empty(total)
        y = np.empty(total)
        x[0], y[0] = rng.uniform(-0.05, 0.05, size=2)
        for i in range(1, total):
            x[i] = 1 - 1.4 * x[i - 1] ** 2 + y[i - 1]
            y[i] = 0.3 * x[i - 1]
        x = x[burn:]
    elif model == 'Lorenz':

        def rhs(t, z):
            a, b, c = z
            return [10 * (b - a), a * (28 - c) - b, a * b - 8 / 3 * c]
        times = 20 + lorenz_dt * np.arange(n)
        initial = np.array([1.0, 1.0, 1.0]) + rng.uniform(-0.01, 0.01, size=3)
        solution = solve_ivp(rhs, (0, float(times[-1])), initial, t_eval=times, rtol=1e-08, atol=1e-10, max_step=0.02)
        if not solution.success or solution.y.shape[1] != n:
            raise ValueError(f'Lorenz integration failed: {solution.message}')
        x = solution.y[0]
    else:
        raise ValueError(f'Unknown model {model}')
    if len(x) != n or not np.isfinite(x).all() or np.std(x) <= 0:
        raise ValueError(f'Invalid generated trajectory for {model}')
    return np.ascontiguousarray(x)

def ar_process(innovations, phi):
    x = np.empty(len(innovations))
    x[0] = 0.0
    for i in range(1, len(x)):
        x[i] = phi * x[i - 1] + innovations[i]
    return x

def _validation_series(model, n, rng, lorenz_dt=0.05):
    burn = 1200
    if model in ('Logistic', 'Henon', 'Lorenz'):
        return _generate_map(model, n, rng, burn=burn, lorenz_dt=lorenz_dt)
    if model == 'IID-Gaussian':
        return rng.normal(size=n)
    if model == 'IID-Exponential':
        return rng.exponential(size=n) - 1
    if model in ('GaussianAR03', 'GaussianAR08', 'MonotoneAR'):
        phi = 0.3 if model == 'GaussianAR03' else 0.8
        x = ar_process(rng.normal(size=n + burn), phi)[burn:]
        if model == 'MonotoneAR':
            x = np.exp(0.7 * x * np.sqrt(1 - phi * phi))
        return x
    if model in ('SkewAR03', 'SkewAR08'):
        phi = 0.3 if model == 'SkewAR03' else 0.8
        return ar_process(rng.exponential(size=n + burn) - 1, phi)[burn:]
    if model == 'QuadraticMA':
        e = rng.normal(size=n + burn + 1)
        return (e[1:] + 0.8 * (e[:-1] ** 2 - 1))[burn:]
    raise ValueError(f'Unknown self-validation model: {model}')

def _controlled_series(s, rep, n):
    f = s['family']
    z = s['coefficient']
    rng = np.random.default_rng(np.random.SeedSequence([_COMPARISON_SEED, _COMPARISON_STAGE, FAMILIES.index(f), rep]))
    states = None
    if f == 'GaussianAR':
        x = np.empty(n)
        x[0] = rng.normal()
        innov = rng.normal(size=n - 1)
        for i in range(1, n):
            x[i] = z * x[i - 1] + np.sqrt(1 - z * z) * innov[i - 1]
    elif f == 'MixedInnovationAR':
        burn = 4000
        g = rng.normal(size=n + burn)
        e = rng.exponential(size=n + burn) - 1
        innov = ((1 - z) * g + z * e) / np.sqrt((1 - z) ** 2 + z * z)
        last = 0.0
        v = np.empty(n + burn)
        for i, a in enumerate(innov):
            last = 0.6 * last + a
            v[i] = last
        x = v[burn:]
    else:
        P = np.array(_MATRICES[int(z)]['matrix'])
        cum = P.cumsum(axis=1)
        states = np.empty(n, dtype=np.int8)
        states[0] = rng.integers(4)
        uniform = rng.random(n - 1)
        for i, u in enumerate(uniform, 1):
            states[i] = np.searchsorted(cum[states[i - 1]], u)
        x = states + rng.uniform(-0.05, 0.05, n)
        assert 0.1 < 0.3 * np.std(x, ddof=1) < 0.9
    return (x, states)

def generate_model(name, n=2000, rep=0):
    """Generate a validation series; rep is zero-based."""
    index = MODEL_NAMES.index(name)
    rng = np.random.default_rng(np.random.SeedSequence([20260930, 100, index, rep]))
    return _validation_series(name, n, rng)

def generate_controlled(family, coefficient, n=4000, rep=0):
    """Generate a controlled series; coefficient 0--4 selects M0--M4."""
    if family not in FAMILIES:
        raise ValueError(f'Unknown family: {family}')
    if family == 'GaussianAR' and (not abs(coefficient) < 1):
        raise ValueError('GaussianAR requires abs(coefficient) < 1')
    if family == 'MixedInnovationAR' and (not 0 <= coefficient <= 1):
        raise ValueError('MixedInnovationAR requires coefficient in [0, 1]')
    if family == 'ConditionalMatching' and (int(coefficient) != coefficient or not 0 <= coefficient <= 4):
        raise ValueError('ConditionalMatching requires an integer from 0 to 4')
    x, _ = _controlled_series(dict(family=family, coefficient=coefficient), rep, n)
    return x
