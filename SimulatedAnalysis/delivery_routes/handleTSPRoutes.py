from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Dict, Tuple, Optional, Callable, Any

def parse_line(line: str) -> Optional[dict]:
    line = line.strip()
    if not line:
        return None

    try:
        path_part, cost_part = line.rsplit(":", 1)
    except ValueError:
        return None

    try:
        cost = float(cost_part.strip())
    except ValueError:
        return None

    stores = [s.strip() for s in path_part.split(",") if s.strip()]
    if not stores:
        return None

    return {
        "path_str": path_part.strip(),
        "cost": cost,
        "stores": stores,
    }


def load_path_files(file: str) -> List[dict]:
    """
    Load and parse path file.
    Returns a list of parsed dicts from parse_line().
    """
    parsed_rows: List[dict] = []

    with open(file, "r") as f:
        for raw_line in f:
            parsed = parse_line(raw_line)
            if parsed is not None:
                parsed_rows.append(parsed)

    return parsed_rows

import glob
import os

def combine_path_files(base_out_path: str):
    """
    Combine all files matching f"{base_out_path}_*.txt" into one file.
    """
    combined_path = f"{base_out_path}_all.txt"

    part_files = sorted(glob.glob(f"{base_out_path}_*.txt"))

    if not part_files:
        print(f"No files found matching {base_out_path}_*.txt")
        return

    with open(combined_path, "w") as f_out:
        for part_path in part_files:
            with open(part_path, "r") as f_in:
                f_out.writelines(f_in)

    print(f"Combined {len(part_files)} files into: {combined_path}")


def reverse_path_line(line: str):
    line = line.strip()
    if not line:
        return None
    try:
        path_part, cost_part = line.rsplit(":", 1)
    except ValueError:
        return None
    stores = [s.strip() for s in path_part.split(",") if s.strip()]
    if len(stores) < 2:
        return None
    return f"{', '.join(stores[::-1])} : {cost_part.strip()}"

def features_ngrams(cand, k=2):
    """
    Extracts k-gram tuples from cand["stores"].

    Examples:
      k=2 → (A,B), (B,C)
      k=3 → (A,B,C), (B,C,D)

    Returns: set[tuple[str, ...]]
    """
    stores = cand["stores"]
    if k <= 0:
        raise ValueError("k must be >= 1")

    return {
        tuple(stores[i:i+k])
        for i in range(len(stores) - k + 1)
    }

PAD = "<PAD>"

def features_token_sequence(cand, L=None):
    """
    Returns a tuple of length L for stable Hamming distance.
    If L is None, uses the raw length (NOT recommended for hamming across routes).
    """
    stores = cand["stores"]
    if L is None:
        return tuple(stores)
    if len(stores) >= L:
        return tuple(stores[:L])
    return tuple(stores + [PAD] * (L - len(stores)))
def hamming(a, b):
    # a, b are tuples (or strings) of equal length
    return sum(x != y for x, y in zip(a, b))

class GreedyConstraint:
    def prepare(self, candidates):
        pass
    def start(self, start_idx):
        pass
    def can_add(self, idx):
        raise NotImplementedError
    def add(self, idx):
        pass

def greedy_pack(candidates, start_idx: int, constraint: GreedyConstraint):
    constraint.prepare(candidates)
    constraint.start(start_idx)

    selected = [start_idx]
    for j in range(len(candidates)):
        if j == start_idx:
            continue
        if constraint.can_add(j):
            constraint.add(j)
            selected.append(j)
    return selected
class TransitionNoOverlapConstraint(GreedyConstraint):
    def __init__(self, feature_fn):
        self.feature_fn = feature_fn

    def prepare(self, candidates):
        self.feats = [self.feature_fn(c) for c in candidates]

    def start(self, start_idx):
        self.used = set(self.feats[start_idx])

    def can_add(self, idx):
        f = self.feats[idx]
        return not (f & self.used)

    def add(self, idx):
        self.used |= self.feats[idx]
import numpy as np

class HammingThresholdConstraintFast(GreedyConstraint):
    """
    Precompute pairwise compatibility once:
      ok[i, j] = True iff hamming(seq[i], seq[j]) >= min_dist
    Then greedy uses fast boolean masking.
    """
    def __init__(self, seq_fn, min_dist: int, dtype=np.int32):
        self.seq_fn = seq_fn
        self.min_dist = int(min_dist)
        self.dtype = dtype

    def prepare(self, candidates):
        # 1) build fixed-length sequences once
        seqs = [self.seq_fn(c) for c in candidates]

        # Ensure they're tuples/lists of tokens -> map to ints for fast compare
        # If tokens are already ints, this is cheap.
        # Flatten token vocabulary:
        tok2id = {}
        seq_ids = []
        next_id = 0
        for s in seqs:
            row = []
            for tok in s:
                if tok not in tok2id:
                    tok2id[tok] = next_id
                    next_id += 1
                row.append(tok2id[tok])
            seq_ids.append(row)

        self.seqs = np.asarray(seq_ids, dtype=self.dtype)  # shape (n, L)
        n, L = self.seqs.shape

        # 2) Precompute ok matrix in chunks to avoid huge temporary RAM
        # ok[i,j] = (L - matches) >= min_dist  <=>  matches <= L - min_dist
        max_matches = L - self.min_dist

        ok = np.ones((n, n), dtype=bool)
        block = 512  # tune if needed

        for i0 in range(0, n, block):
            i1 = min(n, i0 + block)
            A = self.seqs[i0:i1]          # (bi, L)

            # Compare A vs all seqs: (bi, 1, L) == (1, n, L) -> (bi, n, L)
            # Then sum matches across L -> (bi, n)
            matches = (A[:, None, :] == self.seqs[None, :, :]).sum(axis=2)
            ok[i0:i1, :] = (matches <= max_matches)

        # Symmetry + self-compatibility doesn't matter but keep True
        np.fill_diagonal(ok, True)
        self.ok = ok

    def start(self, start_idx):
        self.selected = [start_idx]
        # allowed[j] means "still compatible with all selected so far"
        self.allowed = self.ok[start_idx].copy()
        self.allowed[start_idx] = False  # don't re-add

    def can_add(self, idx):
        return bool(self.allowed[idx])

    def add(self, idx):
        self.selected.append(idx)
        # Intersect allowed set with compatibility row for idx
        self.allowed &= self.ok[idx]
        self.allowed[idx] = False

def incompat_pairs_from_transitions(candidates, feature_fn):
    feats = [feature_fn(c) for c in candidates]
    trans_to_paths = {}
    for i, f in enumerate(feats):
        for t in f:
            trans_to_paths.setdefault(t, []).append(i)

    pairs = set()
    for t, idxs in trans_to_paths.items():
        if len(idxs) > 1:
            # all pairs in this bucket conflict
            idxs = sorted(idxs)
            for a_i in range(len(idxs)):
                for b_i in range(a_i + 1, len(idxs)):
                    pairs.add((idxs[a_i], idxs[b_i]))
    return pairs
def incompat_pairs_from_hamming(candidates, seq_fn, min_dist: int):
    seqs = [seq_fn(c) for c in candidates]
    pairs = set()
    n = len(candidates)
    for i in range(n):
        for j in range(i + 1, n):
            if hamming(seqs[i], seqs[j]) < min_dist:
                pairs.add((i, j))
    return pairs

from ortools.sat.python import cp_model

def optimal_pack(
    candidates,
    incompat_pairs,
    must_include_idx=None,
    time_limit_s=1000,
    num_workers=8,
    minimize_cost_tiebreak=True,
):
    model = cp_model.CpModel()
    n = len(candidates)
    x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

    if must_include_idx is not None:
        model.Add(x[must_include_idx] == 1)

    # Incompatibility constraints: not both
    for i, j in incompat_pairs:
        model.Add(x[i] + x[j] <= 1)

    total_count = sum(x)

    if minimize_cost_tiebreak:
        SCALE = 1000
        costs_int = [int(round(candidates[i]["cost"] * SCALE)) for i in range(n)]
        total_cost_int = sum(costs_int[i] * x[i] for i in range(n))
        max_cost_int = max(costs_int) if costs_int else 0
        BIG = (max_cost_int * n + 1)
        model.Maximize(BIG * total_count - total_cost_int)
    else:
        model.Maximize(total_count)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = int(num_workers)

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [], {"status": solver.StatusName(status)}

    selected = [i for i in range(n) if solver.Value(x[i]) == 1]
    stats = {
        "status": solver.StatusName(status),
        "n_selected": len(selected),
        "sum_cost": sum(candidates[i]["cost"] for i in selected),
        "objective_value": solver.ObjectiveValue(),
        "wall_time_s": solver.WallTime(),
    }
    return selected, stats

def greedy_pack_hamming(candidates, start_idx, seq_fn, min_dist):
    seqs = [seq_fn(c) for c in candidates]

    selected = [start_idx]
    for j in range(len(candidates)):
        if j == start_idx:
            continue

        ok = True
        for i in selected:
            if hamming(seqs[i], seqs[j]) < min_dist:
                ok = False
                break

        if ok:
            selected.append(j)

    return selected