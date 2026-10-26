"""Grid-search RANSAC threshold/iterations for flow.estimate_velocity.

Each (threshold, iterations) pair is scored by RMSE against mocap velocity on
both datasets, relative to the least-squares (ransac=False) baseline. Lower
score is better; 2.0 means "same as least squares" on both linear and angular.

    python tune_ransac.py
    python tune_ransac.py --thresholds 0.1 0.2 0.3 --iterations 10 20 --repeats 3
"""

import argparse
import itertools
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import flow
from pose import estimate_pose
from support import load_dataset

DATASETS = (1, 4)
_data = {}


def _load():
    """Load each dataset once per process and cache the previous-frame poses."""
    if not _data:
        for number in DATASETS:
            dataset = load_dataset(number)
            poses = [estimate_pose(frame) for frame in dataset.frames[:-1]]
            _data[number] = (dataset, poses)
    return _data


def run(number, ransac, threshold=None, iterations=None, seed=0):
    """Return (time, values) like main.run_estimation, with RANSAC params forced."""
    dataset, poses = _load()[number]
    original = flow.velocity_ransac
    if ransac:
        flow.velocity_ransac = lambda *a, **k: original(
            *a, **{**k, "threshold": threshold, "iterations": iterations})
    random.seed(seed)
    frames = dataset.frames
    time = np.array([f.time for f in frames])
    time = (time[:-1] + time[1:]) / 2
    values = np.full((6, len(frames) - 1), np.nan)
    failures = 0
    try:
        for index, (previous, current) in enumerate(zip(frames[:-1], frames[1:])):
            if poses[index] is None:
                continue
            try:
                velocity = flow.estimate_velocity(previous, current, poses[index], ransac=ransac)
            except np.linalg.LinAlgError:
                failures += 1
                continue
            if velocity is not None:
                values[:, index] = velocity
    finally:
        flow.velocity_ransac = original
    return time, values, failures


def rmse(number, time, values):
    """(linear, angular) RMSE against mocap interpolated to the estimate times."""
    dataset, _ = _load()[number]
    reference = np.array([np.interp(time, dataset.reference_time, row)
                          for row in dataset.vicon[6:12]])
    valid = np.all(np.isfinite(values), axis=0)
    error = values[:, valid] - reference[:, valid]
    return np.sqrt(np.mean(error[:3] ** 2)), np.sqrt(np.mean(error[3:] ** 2))


def evaluate(config):
    """Mean over repeats of per-dataset (linear, angular) RMSE, plus failure count."""
    threshold, iterations, repeats = config
    result, failures = {}, 0
    for number in DATASETS:
        scores = []
        for seed in range(repeats):
            time, values, failed = run(number, True, threshold, iterations, seed)
            scores.append(rmse(number, time, values))
            failures += failed
        result[number] = np.mean(scores, axis=0)
    return threshold, iterations, result, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    parser.add_argument("--iterations", type=int, nargs="+", default=[5, 10, 20, 40, 80])
    parser.add_argument("--repeats", type=int, default=1,
                        help="random seeds averaged per config")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    try:
        baseline = {n: rmse(n, *run(n, False)[:2]) for n in DATASETS}
        print("Least squares baseline (linear m/s, angular rad/s RMSE):")
        for n in DATASETS:
            print(f"  dataset {n}: {baseline[n][0]:.4f}  {baseline[n][1]:.4f}")
    except Exception as error:  # ransac=False path broken: score raw RMSE instead
        print(f"Least squares baseline failed ({type(error).__name__}: {error});"
              " scoring by raw lin + ang RMSE instead")
        baseline = {n: (1.0, 1.0) for n in DATASETS}

    configs = [(t, i, args.repeats) for t, i in itertools.product(args.thresholds, args.iterations)]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for threshold, iterations, result, failures in pool.map(evaluate, configs):
            score = np.mean([result[n][0] / baseline[n][0] + result[n][1] / baseline[n][1]
                             for n in DATASETS])
            rows.append((score, threshold, iterations, result, failures))
            print(f"  done threshold={threshold:<5} iterations={iterations:<4} score={score:.3f}",
                  flush=True)

    rows.sort(key=lambda row: row[0])
    if all(baseline[n] == (1.0, 1.0) for n in DATASETS):
        print("\nscore = mean over datasets of (lin RMSE + ang RMSE)")
    else:
        print("\nscore = mean over datasets of (lin / lin_LS + ang / ang_LS); LS baseline = 2.000")
    header = "  ".join(f"d{n} lin    d{n} ang " for n in DATASETS)
    print(f"{'score':>6}  {'thresh':>6}  {'iters':>5}  {header} fails")
    for score, threshold, iterations, result, failures in rows:
        cells = "  ".join(f"{result[n][0]:.4f}  {result[n][1]:.4f}  " for n in DATASETS)
        print(f"{score:6.3f}  {threshold:6.3f}  {iterations:5d}  {cells}{failures:5d}")
    best = rows[0]
    print(f"\nBest: threshold = {best[1]}, iterations = {best[2]}")


if __name__ == "__main__":
    main()
