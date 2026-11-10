import json, math, os, random

root = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
out = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')
with open(os.path.join(root, 'task.json'), encoding='utf-8') as f:
    task = json.load(f)
names = [u['symbol'] for u in task['universe']]
edges = task['bin_edges']

def normal_cdf(x, mean, sd):
    return .5 * math.erfc((mean - x) / (sd * math.sqrt(2)))

def read_history(item):
    path = os.path.join(root, item['file'])
    with open(path, encoding='utf-8') as source:
        bars = sorted((json.loads(line) for line in source if line.strip()), key=lambda b: b['t'])
    returns = []
    for a, b in zip(bars, bars[1:]):
        if b['t'] - a['t'] == 3600000 and float(a['c']) > 0 and float(b['c']) > 0:
            returns.append(math.log(float(b['c']) / float(a['c'])))
    return bars, returns

observed = [read_history(item) for item in task['universe']]
priors = {'AAPLX': .021, 'AMZNX': .026, 'GOOGLX': .025, 'METAX': .029,
          'MSFTX': .021, 'MSTRX': .055, 'NVDAX': .036, 'SPCXX': .042, 'TSLAX': .045}

def daily_volatility(name, returns):
    prior = priors.get(name, .03)
    realized = 24 * sum(r*r for r in returns) / len(returns) if returns else prior**2
    weight = len(returns) / (len(returns) + 30)
    variance = weight * realized + (1-weight) * prior**2
    return max(.012, min(.12, math.sqrt(variance)))

vols = [daily_volatility(name, series[1]) for name, series in zip(names, observed)]

def first_known_return(bars):
    # Use the close-hour bar as an approximation to the previous close TWAP.
    close_date = task['window']['days'][0].get('previous')
    if not close_date or not bars:
        return 0.0
    from datetime import datetime
    end = datetime.fromisoformat(task['window']['days'][0]['twap_until'].replace('Z', '+00:00'))
    previous_close = datetime.fromisoformat(close_date + 'T' + end.strftime('%H:%M:%S') + '+00:00')
    cutoff = int(previous_close.timestamp() * 1000)
    matching = [bar for bar in bars if bar['t'] == cutoff]
    if not matching or bars[-1]['t'] <= cutoff:
        return 0.0
    change = float(bars[-1]['c']) / float(matching[-1]['c']) - 1
    return max(-.15, min(.15, change))

known = [first_known_return(bars) for bars, _ in observed]

def build_day(day_index):
    means = known if day_index == 0 else [0.0 for _ in names]
    sizes = []
    for name, mean, sd in zip(names, means, vols):
        for i in range(len(edges) + 1):
            lower = edges[i-1] if i else -math.inf
            upper = edges[i] if i < len(edges) else math.inf
            central = normal_cdf(upper, mean, sd) - normal_cdf(lower, mean, sd)
            tail = normal_cdf(upper, mean, 2.2*sd) - normal_cdf(lower, mean, 2.2*sd)
            probability = max(0.0, .88 * central + .12 * tail)
            sizes.append({'symbol': name, 'bin': i, 'p': math.floor(probability*1e8)/1e8})
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
