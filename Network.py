import numpy as np
import os
import networkx as nx
from glob import glob
from scipy.signal import correlate

# ==========================
# USER PARAMETERS
# ==========================
DATA_DIR = "deltaE"       # directory containing residue ΔE files
MAX_LAG = 200             # frames (choose based on sampling)
CORR_THRESHOLD = 0.03     # minimum correlation to keep an edge
TOP_PATHS = 5             # number of dominant pathways to report

# ==========================
# LOAD ΔE TIME SERIES
# ==========================
files = sorted(glob(os.path.join(DATA_DIR, "*.dat")))

print(f"\nFound {len(files)} deltaE files\n")

deltaE = {}

for file in files:
    resname = os.path.basename(file).replace(".dat", "")
    try:
        data = np.loadtxt(file, comments=('#','@'))

        # Handle 2-column (time, energy) or 1-column files
        if data.ndim == 2:
            ts = data[:,1]
        else:
            ts = data

        if len(ts) < 100:
            print(f"Skipping {resname}: too short")
            continue

        deltaE[resname] = ts - ts.mean()
        print(f"Loaded {resname}: {len(ts)} frames")

    except Exception as e:
        print(f"Failed loading {resname}: {e}")

# ============================
# RESIDUE LIST
# ============================
residues = list(deltaE.keys())

print(f"\nUsing {len(residues)} residues for network construction\n")
#======
def cross_corr(x, y, max_lag):
    """
    Time-lagged normalized cross-correlation.
    Returns lags and correlation values.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    x = x - x.mean()
    y = y - y.mean()

    corr = []
    lags = np.arange(-max_lag, max_lag + 1)

    denom = np.sqrt(np.sum(x**2) * np.sum(y**2))
    if denom == 0:
        return lags, np.zeros(len(lags))

    for lag in lags:
        if lag < 0:
            c = np.sum(x[:lag] * y[-lag:])
        elif lag > 0:
            c = np.sum(x[lag:] * y[:-lag])
        else:
            c = np.sum(x * y)

        corr.append(c / denom)

    return lags, np.array(corr)

# ============================
# DIAGNOSTIC: CORRELATION RANGE
# ============================

max_corr = 0.0
mean_corr = 0.0
count = 0

for i in range(len(residues)):
    for j in range(i+1, len(residues)):
        lags, corr = cross_corr(deltaE[residues[i]],
                                deltaE[residues[j]],
                                MAX_LAG)
        peak = np.max(np.abs(corr))
        max_corr = max(max_corr, peak)
        mean_corr += peak
        count += 1

mean_corr /= count

print("\n=== ΔE CORRELATION DIAGNOSTICS ===")
print(f"Max |C_ij| observed : {max_corr:.3f}")
print(f"Mean |C_ij| observed: {mean_corr:.3f}")
print("================================\n")


# ==========================
# TIME-LAGGED CROSS-CORRELATION
# ==========================
def max_lagged_corr(x, y, max_lag):
    corr = correlate(x, y, mode='full')
    lags = np.arange(-len(x)+1, len(x))
    valid = np.where(np.abs(lags) <= max_lag)[0]
    corr = corr[valid]
    lags = lags[valid]
    corr /= (np.std(x) * np.std(y) * len(x))
    idx = np.argmax(np.abs(corr))
    return corr[idx], lags[idx]

# ==========================
# BUILD DIRECTED NETWORK
# ==========================
G = nx.DiGraph()

for i in residues:
    for j in residues:
        if i == j:
            continue
        c, lag = max_lagged_corr(deltaE[i], deltaE[j], MAX_LAG)
        if abs(c) >= CORR_THRESHOLD:
            if lag > 0:
                G.add_edge(i, j, weight=abs(c), corr=c, lag=lag)

print(f"Network constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# ==========================
# PATHWAY EXTRACTION
# ==========================
def pathway_score(path, graph):
    weights = []
    for u, v in zip(path[:-1], path[1:]):
        weights.append(graph[u][v]['weight'])
    return np.prod(weights)

paths = []

for source in G.nodes():
    for target in G.nodes():
        if source != target:
            for path in nx.all_simple_paths(G, source, target, cutoff=6):
                score = pathway_score(path, G)
                paths.append((path, score))

paths = sorted(paths, key=lambda x: x[1], reverse=True)

# ==========================
# REPORT TOP PATHWAYS
# ==========================
print("\nTop Energy Transfer Pathways:\n")
for i, (path, score) in enumerate(paths[:TOP_PATHS]):
    print(f"P{i+1}: {' -> '.join(path)} | Score = {score:.4e}")

