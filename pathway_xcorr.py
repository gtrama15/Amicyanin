import numpy as np
from scipy.signal import correlate
import os

# ===============================
# USER INPUT SECTION
# ===============================

DT = 0.001          # ps (1 fs)
MAX_LAG_PS = 2.0    # correlation window
ALPHA = 2.0         # pathway lag penalty

DELTAE_DIR = "deltaE/"
ENERGY_DIR = "energy2/"

# Define pathways (residue labels must match filenames)
PATHWAYS = {
    "P1": ["MET28", "LYS29", "GLU31"],
    "P2": ["MET98", "TYR30", "GLU31"],
    "P3": ["MET98", "PHE97", "CYS92", "THR32", "PRO33"],
    "P4": ["HIS95", "PRO94", "MET71", "MET72", "ILE25", "ALA26", "LYS29"],
#    "P5": ["ARG48", "GLU49"],
    "P5": ["GLU49", "ARG48"],
    "P6": ["MET51", "ALA50", "LYS74", "GLU75", "ARG48"],
    "P7": ["HIS95", "MET28", "LYS29", "GLU31"],
    "P8": ["PRO52", "ASN47", "ARG48"],
    "P9": ["HIS53", "ASN54", "VAL55", "TRP45", "VAL43", "LEU35", "HIS36"]
}

# ===============================
# DATA LOADING
# ===============================

def load_timeseries(filepath):
    data = np.loadtxt(filepath)
    return data[:, -1] if data.ndim > 1 else data

def load_data(residues, directory):
    data = {}
    for res in residues:
        file = os.path.join(directory, f"{res}.dat")
        if not os.path.isfile(file):
            raise FileNotFoundError(f"Missing file: {file}")
        data[res] = load_timeseries(file)
    return data

# Collect all residues involved
all_residues = sorted(set(r for p in PATHWAYS.values() for r in p))

deltaE = load_data(all_residues, DELTAE_DIR)

# Only first-shell residues need raw energy
first_shell = sorted(set(p[0] for p in PATHWAYS.values()))
energy = load_data(first_shell, ENERGY_DIR)

print("\nDEBUG: FIRST-SHELL RESIDUES LOADED")
print(sorted(energy.keys()))

# ===============================
# CROSS-CORRELATION FUNCTIONS
# ===============================

def cross_corr(source, target):
    source = source - source.mean()
    target = target - target.mean()

    corr = correlate(target, source, mode="full")
    corr /= (np.std(source) * np.std(target) * len(source))

    lags = np.arange(-len(source)+1, len(source)) * DT
    mask = (lags >= 0) & (lags <= MAX_LAG_PS)

    return lags[mask], corr[mask]

def analyze_link(src, tgt):
    lags, corr = cross_corr(energy[src], deltaE[tgt])
    idx = np.argmax(np.abs(corr))

    return {
        "src": src,
        "tgt": tgt,
        "Cmax": corr[idx],
        "tau": lags[idx]
    }

# ===============================
# PATHWAY ANALYSIS
# ===============================

#FIRST_SHELL = {"MET28", "MET98", "HIS95", "ARG48", "MET51", "HIS95", "PRO52", "HIS53" }
FIRST_SHELL = {"MET28", "MET98", "HIS95", "GLU49", "MET51", "HIS95", "PRO52", "HIS53" }

def analyze_pathway(path):
    """
    Analyze a single pathway.
    Only the first-shell -> second-shell link is directional.
    """
    results = {}

    src = path[0]
    tgt = path[1]

    if src not in FIRST_SHELL:
        raise ValueError(f"{src} is not a first-shell residue")

    # Directional coupling
    lags, corr = cross_corr(energy[src], deltaE[tgt])
    peak = np.max(np.abs(corr))

    results["directional"] = {
        "src": src,
        "tgt": tgt,
        "peak_corr": peak
    }

      # Supportive ΔE–ΔE correlations (optional)
    supportive = []
    for i in range(1, len(path) - 1):
        r1, r2 = path[i], path[i+1]
        lags, corr = cross_corr(deltaE[r1], deltaE[r2])
        supportive.append({
            "pair": f"{r1}–{r2}",
            "peak_corr": np.max(np.abs(corr))
        })

    results["supportive"] = supportive
    return results

# ===============================
# RUN ANALYSIS
# ===============================

print("\n=== ENERGY TRANSFER PATHWAY ANALYSIS ===\n")

for pid, path in PATHWAYS.items():
    print(f"Pathway {pid}: {' → '.join(path)}")

    results = analyze_pathway(path)

    # Directional link
    d = results["directional"]
    print(
        f"  Directional: {d['src']} → {d['tgt']} | "
        f"peak corr = {d['peak_corr']:.3f}"
    )

    # Supportive links
    for s in results["supportive"]:
        print(
            f"  Supportive:  {s['pair']} | "
            f"peak corr = {s['peak_corr']:.3f}"
        )

    # Optional pathway score (heuristic)
    pathway_score = d["peak_corr"]
    if results["supportive"]:
        pathway_score *= np.mean(
            [s["peak_corr"] for s in results["supportive"]]
        )

    print(f"  Pathway score = {pathway_score:.3f}\n")
summary = []

for pid, path in PATHWAYS.items():
    results = analyze_pathway(path)

    d = results["directional"]
    supportive = results["supportive"]

    score = d["peak_corr"]
    if supportive:
        score *= np.mean([s["peak_corr"] for s in supportive])

    summary.append((pid, score))

print("\n=== PATHWAY RANKING ===")
for pid, score in sorted(summary, key=lambda x: x[1], reverse=True):
    print(f"{pid}: {score:.4f}")




