import json, math, os, random

root = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
out = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')
with open(os.path.join(root, 'task.json'), encoding='utf-8') as f:
    task = json.load(f)
names = [u['symbol'] for u in task['universe']]
edges = task['bin_edges']

def normal_cdf(x, mean, sd):
    return 0.5 * math.erfc(-(x - mean) / (sd * math.sqrt(2)))

def observations(item):
    with open(os.path.join(root, item['file']), encoding='utf-8') as f:
        bars = sorted((json.loads(line) for line in f if line.strip()), key=lambda b: b['t'])
    hourly = [math.log(float(b['c']) / float(a['c'])) for a, b in zip(bars, bars[1:])
              if b['t'] - a['t'] == 3600000 and float(a['c']) > 0 and float(b['c']) > 0]
    return bars, hourly

observed = [observations(u) for u in task['universe']]
priors = {'AAPLX': .021, 'AMZNX': .026, 'GOOGLX': .025, 'METAX': .029,
          'MSFTX': .021, 'MSTRX': .055, 'NVDAX': .036, 'SPCXX': .042, 'TSLAX': .045}
vols = []
for name, (_, rs) in zip(names, observed):
    prior = priors.get(name, .03)
    # A single day's hourly bars are noisy: shrink realized variance toward a prior.
    rv = 24 * sum(r * r for r in rs) / len(rs) if rs else prior * prior
    weight = len(rs) / (len(rs) + 30)
    vols.append(min(.12, max(.012, math.sqrt(weight * rv + (1-weight) * prior * prior))))

def first_known_return(bars):
    # The final input contains two hours after the previous 16:00 NY close.
    # Approximate that close with the bar ending at the previous day's TWAP end.
    close = task['window']['days'][0].get('previous')
    if not close:
        return 0.0
    from datetime import datetime
    day_end = datetime.fromisoformat(task['window']['days'][0]['twap_until'].replace('Z', '+00:00'))
    cutoff = int(datetime.fromisoformat(close + 'T' + day_end.strftime('%H:%M:%S') + '+00:00').timestamp() * 1000)
    matching = [b for b in bars if b['t'] == cutoff]
    if not matching or not bars or bars[-1]['t'] <= cutoff:
        return 0.0
    return max(-.15, min(.15, float(bars[-1]['c']) / float(matching[-1]['c']) - 1))

known = [first_known_return(b) for b, _ in observed]

def build_day(day_index):
    means = known if day_index == 0 else [0.0] * len(names)
    # Mixture adds a tail component to avoid assigning vanishing probability to shocks.
    sizes = []
    for name, mean, sd in zip(names, means, vols):
        for i in range(len(edges)+1):
            lo = edges[i-1] if i else -math.inf
            hi = edges[i] if i < len(edges) else math.inf
            p = .88*(normal_cdf(hi, mean, sd)-normal_cdf(lo, mean, sd))
            p += .12*(normal_cdf(hi, mean, 2.2*sd)-normal_cdf(lo, mean, 2.2*sd))
            sizes.append({'symbol': name, 'bin': i, 'p': math.floor(max(0,p)*1e8)/1e8})
    rng = random.Random(20261109 + day_index)
    count = [[0]*len(names) for _ in range(task['places'])]
    draws = 30000
    for _ in range(draws):
        common = rng.gauss(0, 1)
        tail = 2.2 if rng.random() < .12 else 1.0
        vals = [means[i] + tail*vols[i]*(.55*common + math.sqrt(1-.55**2)*rng.gauss(0,1)) for i in range(len(names))]
        ranking = sorted(range(len(names)), key=lambda i: (-vals[i], names[i]))
        for k, i in enumerate(ranking[:task['places']]):
            count[k][i] += 1
    probabilities = [{'symbol': n, 'place': k+1, 'p': math.floor((count[k][i]+1)/(draws+len(names))*1e8)/1e8}
                     for k in range(task['places']) for i, n in enumerate(names)]
    remaining = set(range(len(names)))
    order = []
    for k in range(task['places']):
        pick = min(remaining, key=lambda i: (-count[k][i], names[i]))
        remaining.remove(pick)
        order.append(names[pick])
    return {'date': task['window']['days'][day_index]['date'], 'probabilities': probabilities, 'order': order, 'sizes': sizes}

os.makedirs(out, exist_ok=True)
with open(os.path.join(out, task['output']), 'w', encoding='utf-8') as f:
    json.dump({'days': [build_day(i) for i in range(len(task['window']['days']))]}, f, separators=(',', ':'))
    f.write('\n')
