# all_folds_eval.py

"""
5 fold cross-validation evaluation with:
 - McNemar's test
 - Paired t-test
 - Summary statistics and tables along with thesis summary

"""

import json
import subprocess
import sys
import os

# Adding project root to path so that evaluation scripts be ran from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from statistics import mean, stdev
from scipy import stats
import numpy as np
import evaluation.single_fold_eval as evaluate_fold

"""
    Configurations
"""
NUM_FOLDS = 5
METHODS = ["rule", "presidio", "customdeberta"]
OUTPUT_FILE = "cross_validation_results.json"

def banner(text):
    print("\n" + "=" * 90)
    print(text.center(90))
    print("=" * 90 + "\n")


""" 
    STEP 1: Run single_fold_eval.py on all folds
"""
def evaluate_single_fold(fold_idx):
    banner(f"STARTING FOLD {fold_idx}")

    val_file = f"data/fold_{fold_idx}_val.json"

    if not os.path.exists(val_file):
        print(f"Missing validation file: {val_file}")
        return False

    print(f"Using validation file: {val_file} ({os.path.getsize(val_file)} bytes)\n")
    print("▶ Running single_fold_eval.py...\n")

    script_path = os.path.join("evaluation", "single_fold_eval.py")

    process = subprocess.Popen(
        [sys.executable, script_path, str(fold_idx)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode != 0:
        print(f"Fold {fold_idx} failed")
        return False

    print(f"\nFINISHED FOLD {fold_idx}")
    print("-" * 90 + "\n")
    return True


def evaluate_all_folds():
    banner("STEP 1: RUNNING ALL FOLDS")

    for fold_idx in range(NUM_FOLDS):
        print(f"Progress: Fold {fold_idx+1}/{NUM_FOLDS}")
        print("-" * 40)

        if not evaluate_single_fold(fold_idx):
            print(f"Error in fold {fold_idx}")
            sys.exit(1)

        time.sleep(0.3)

    print("\nAll folds evaluated!\n")


"""
    STEP 2: Load all fold results
"""

def load_fold_results():
    banner("STEP 2: LOADING RESULTS")

    results = {
        m: {"precision": [], "recall": [], "f1": [], "support": []}
        for m in METHODS
    }

    for fold_idx in range(NUM_FOLDS):
        fname = f"fold_{fold_idx}_comparison.json"

        print(f"Loading {fname}")

        if not os.path.exists(fname):
            print(f"Missing: {fname}")
            sys.exit(1)

        with open(fname, "r", encoding="utf-8") as f:
            data = json.load(f)

        for method in METHODS:
            d = data["methods"].get(method)
            if d:
                results[method]["precision"].append(d["precision"])
                results[method]["recall"].append(d["recall"])
                results[method]["f1"].append(d["f1"])
                results[method]["support"].append(d["support"])

        print("   ✔ Loaded.")

    return results


def compute_statistics(results):
    summary = {}

    for method, m in results.items():
        summary[method] = {
            "precision": {
                "mean": mean(m["precision"]),
                "std": stdev(m["precision"]) if len(m["precision"]) > 1 else 0,
                "min": min(m["precision"]),
                "max": max(m["precision"]),
                "values": m["precision"]
            },
            "recall": {
                "mean": mean(m["recall"]),
                "std": stdev(m["recall"]) if len(m["recall"]) > 1 else 0,
                "min": min(m["recall"]),
                "max": max(m["recall"]),
                "values": m["recall"]
            },
            "f1": {
                "mean": mean(m["f1"]),
                "std": stdev(m["f1"]) if len(m["f1"]) > 1 else 0,
                "min": min(m["f1"]),
                "max": max(m["f1"]),
                "values": m["f1"]
            },
            "total_support": sum(m["support"])
        }

    return summary


"""
    STEP 3A: McNemar's Test
"""
def load_predictions_for_mcnemar(fold_idx, method):
    from pipeline.anonymizer import run_method

    val_file = f"data/fold_{fold_idx}_val.json"

    true_labels, pred_labels, _, _ = run_method(method, val_file)

    true_flat = []
    pred_flat = []

    for tdoc, pdoc in zip(true_labels, pred_labels):
        for t, p in zip(tdoc, pdoc):
            if t.startswith("B-") or p.startswith("B-"):
                true_flat.append(t)
                pred_flat.append(p)

    return true_flat, pred_flat


def mcnemar_test(method1, method2, fold_idx):
    true, pred1 = load_predictions_for_mcnemar(fold_idx, method1)
    _, pred2 = load_predictions_for_mcnemar(fold_idx, method2)

    a = b = c = d = 0

    for t, p1, p2 in zip(true, pred1, pred2):
        c1 = (p1 == t)
        c2 = (p2 == t)

        if c1 and c2:
            a += 1
        elif c1 and not c2:
            b += 1
        elif not c1 and c2:
            c += 1
        else:
            d += 1

    if b + c < 25:
        if b + c == 0:
            return None, 1.0, a, b, c, d
        p = stats.binom_test(b, b + c, 0.5)
        return None, p, a, b, c, d

    chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, 1)

    return chi2, p_value, a, b, c, d


def perform_mcnemar_tests(summary):
    banner("STEP 3A: MCNEMAR'S TEST")

    methods = list(summary.keys())
    results = []

    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            m1 = methods[i]
            m2 = methods[j]

            print(f"\nComparing {m1} vs {m2}...")

            pvals = []
            for fold in range(NUM_FOLDS):
                chi2, p, *_ = mcnemar_test(m1, m2, fold)
                pvals.append(p)
                print(f"  Fold {fold}: p={p:.4f}")

            combined_stat = -2 * sum(np.log(p) for p in pvals if p > 0)
            combined_p = 1 - stats.chi2.cdf(combined_stat, df=2 * NUM_FOLDS)

            print(f"Combined p-value: {combined_p:.6f}")

            results.append({
                "method1": m1,
                "method2": m2,
                "p_values": pvals,
                "combined_p": combined_p,
                "significant_005": combined_p < 0.05,
                "significant_001": combined_p < 0.01
            })

    return results


"""
    STEP 3B: Paired T-Test"""
def paired_t_test(a, b):
    return stats.ttest_rel(a, b)


def wilcoxon_test(a, b):
    try:
        return stats.wilcoxon(a, b)
    except ValueError:
        return 0, 1.0


def cohens_d(a, b):
    diff = np.array(a) - np.array(b)
    return diff.mean() / diff.std() if diff.std() != 0 else 0


def perform_ttest_comparisons(summary):
    banner("STEP 3B: T-TESTS")

    methods = list(summary.keys())
    out = []

    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            m1 = methods[i]
            m2 = methods[j]

            s1 = summary[m1]["f1"]["values"]
            s2 = summary[m2]["f1"]["values"]

            t, p = paired_t_test(s1, s2)
            w, pw = wilcoxon_test(s1, s2)
            d = cohens_d(s1, s2)

            print(f"\n{m1} vs {m2}: p={p:.4f}")

            out.append({
                "method1": m1,
                "method2": m2,
                "t_stat": float(t),
                "p_value": float(p),
                "wilcoxon_p": float(pw),
                "cohens_d": float(d),
                "significant_005": p < 0.05,
                "significant_001": p < 0.01
            })

    return out


"""
    Print fold-by-fold results and summary tables,
    along with thesis summary.
"""
def print_fold_by_fold(results):
    banner("F1 SCORES BY FOLD")

    print(f"{'Fold':<8} {'Rule':<12} {'Presidio':<12} {'DeBERTa':<12}")
    print("-" * 50)

    for i in range(NUM_FOLDS):
        print(f"{i:<8} "
              f"{results['rule']['f1'][i]:<12.4f} "
              f"{results['presidio']['f1'][i]:<12.4f} "
              f"{results['customdeberta']['f1'][i]:<12.4f}")

    print("-" * 50)


def print_summary_table(summary):
    banner("SUMMARY TABLES")

    print(f"{'Method':<15} {'Mean F1':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 60)

    for m, s in summary.items():
        f = s["f1"]
        print(f"{m:<15} {f['mean']:<10.4f} {f['std']:<10.4f} {f['min']:<10.4f} {f['max']:<10.4f}")

    print("-" * 60)


def print_mcnemar_table(results):
    banner("MCNEMAR RESULTS")

    print(f"{'Comparison':<25} {'p-value':<15} {'Sig':<8}")
    print("-" * 60)

    for r in results:
        sig = "**" if r["significant_001"] else ("*" if r["significant_005"] else "ns")
        name = f"{r['method1']} vs {r['method2']}"
        print(f"{name:<25} {r['combined_p']:<15.6f} {sig:<8}")

    print("-" * 60)


def print_ttest_table(results):
    banner("T-TEST RESULTS")

    print(f"{'Comparison':<25} {'p-value':<15} {'Sig':<8} {'d':<8}")
    print("-" * 60)

    for r in results:
        sig = "**" if r["significant_001"] else ("*" if r["significant_005"] else "ns")
        name = f"{r['method1']} vs {r['method2']}"
        print(f"{name:<25} {r['p_value']:<15.6f} {sig:<8} {r['cohens_d']:<8.3f}")

    print("-" * 60)


def print_thesis_summary(summary, mcnemar_results, ttest_results):
    banner("Thesis Summary :)")

    best = max(summary, key=lambda m: summary[m]["f1"]["mean"])
    print(f"Best model: {best.upper()} (F1={summary[best]['f1']['mean']:.4f})\n")

    print("Statistical Significance:")
    for r in mcnemar_results:
        print(f" - McNemar {r['method1']} vs {r['method2']}: p={r['combined_p']:.6f}")

    for r in ttest_results:
        print(f" - T-test {r['method1']} vs {r['method2']}: p={r['p_value']:.6f}")


"""
    Save evaluation results to a JSON file.
"""
def save_results(results, summary, mcnemar_results, ttest_results):
    data = {
        "fold_results": results,
        "summary": summary,
        "mcnemar": mcnemar_results,
        "ttests": ttest_results
    }
    
    # Convert all numpy types to native Python types
    data = evaluate_fold.convert_to_serializable(data)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved results to {OUTPUT_FILE}\n") 



""" 
    Run full evaluation process 
"""
def run_full_evaluation():
    banner("5-FOLD EVALUATION STARTED")

    evaluate_all_folds()
    results = load_fold_results()
    summary = compute_statistics(results)

    mcnemar_results = perform_mcnemar_tests(summary)
    ttest_results = perform_ttest_comparisons(summary)

    print_fold_by_fold(results)
    print_summary_table(summary)
    print_mcnemar_table(mcnemar_results)
    print_ttest_table(ttest_results)
    print_thesis_summary(summary, mcnemar_results, ttest_results)

    save_results(results, summary, mcnemar_results, ttest_results)

    banner("5-FOLD EVALUATION COMPLETE")


if __name__ == "__main__":
    run_full_evaluation()
