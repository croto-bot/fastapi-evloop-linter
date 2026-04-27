#!/usr/bin/env python3
"""Benchmark script for unseen-module adversarial cases.

Mirrors the structure of benchmark.py but runs only the cases defined in
tests/adversarial/unseen_modules.py — modules that are NOT in BLOCKING_PATTERNS.

A correct generic detector must:
  - Flag all positive cases (expected_violations > 0)
  - Leave all negative cases clean (expected_violations == 0)

Outputs METRIC lines compatible with autoresearch.
"""

import sys
import time
from pathlib import Path

# Add src to path (same as benchmark.py)
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi_evloop_linter.checker import EventLoopChecker
from tests.adversarial.unseen_modules import generate_unseen_cases, TestCase


def run_benchmark() -> dict:
    """Run the unseen-module adversarial benchmark suite."""
    cases = generate_unseen_cases()

    checker = EventLoopChecker(max_depth=20)

    total_cases = len(cases)
    total_expected = 0
    total_found = 0
    total_false_negatives = 0
    total_false_positives = 0
    cases_fully_detected = 0
    cases_partially_detected = 0
    cases_missed = 0
    # Negative-case tracking
    negative_total = 0
    negative_false_positives = 0
    max_depth_found = 0
    detection_by_category: dict[str, dict] = {}

    start_time = time.perf_counter()

    for case in cases:
        result = checker.check_source(case.source, filepath=f"<{case.name}>")

        found = len(result.violations)
        expected = case.expected_violations

        # Negative-case accounting
        if expected == 0:
            negative_total += 1
            if found > 0:
                negative_false_positives += found
                total_false_positives += found
            # Negative cases don't contribute to detection metrics
            # but we still track them for the full picture
            cat = case.category
            if cat not in detection_by_category:
                detection_by_category[cat] = {
                    "expected": 0, "found": 0, "cases": 0,
                    "detected_cases": 0, "false_positives": 0,
                }
            detection_by_category[cat]["cases"] += 1
            detection_by_category[cat]["false_positives"] += found
            if found == 0:
                detection_by_category[cat]["detected_cases"] += 1
            continue

        total_expected += expected
        total_found += min(found, expected)

        if found >= expected:
            cases_fully_detected += 1
        elif found > 0:
            cases_partially_detected += 1
        else:
            cases_missed += 1

        if found < expected:
            total_false_negatives += (expected - found)
        if found > expected:
            total_false_positives += (found - expected)

        if result.violations:
            case_max_depth = max(v.depth for v in result.violations)
            max_depth_found = max(max_depth_found, case_max_depth)

        cat = case.category
        if cat not in detection_by_category:
            detection_by_category[cat] = {
                "expected": 0, "found": 0, "cases": 0,
                "detected_cases": 0, "false_positives": 0,
            }
        detection_by_category[cat]["expected"] += expected
        detection_by_category[cat]["found"] += min(found, expected)
        detection_by_category[cat]["cases"] += 1
        if found >= expected:
            detection_by_category[cat]["detected_cases"] += 1

    elapsed = time.perf_counter() - start_time

    # Positive cases only for primary detection metrics
    positive_cases = total_cases - negative_total
    detection_rate = (total_found / total_expected * 100) if total_expected > 0 else 0
    case_detection_rate = (cases_fully_detected / positive_cases * 100) if positive_cases > 0 else 0
    missed = total_expected - total_found

    return {
        "total_cases": total_cases,
        "positive_cases": positive_cases,
        "negative_cases": negative_total,
        "total_expected_violations": total_expected,
        "total_found": total_found,
        "missed_violations": missed,
        "false_negatives": total_false_negatives,
        "false_positives": total_false_positives,
        "negative_false_positives": negative_false_positives,
        "cases_fully_detected": cases_fully_detected,
        "cases_partially_detected": cases_partially_detected,
        "cases_missed": cases_missed,
        "detection_rate": round(detection_rate, 1),
        "case_detection_rate": round(case_detection_rate, 1),
        "max_depth_found": max_depth_found,
        "elapsed_ms": round(elapsed * 1000, 1),
        "by_category": detection_by_category,
    }


def main() -> int:
    results = run_benchmark()

    # METRIC lines for autoresearch (same format as benchmark.py)
    print(f"METRIC missed={results['missed_violations']}")
    print(f"METRIC detection_rate={results['detection_rate']}")
    print(f"METRIC cases_missed={results['cases_missed']}")
    print(f"METRIC cases_fully_detected={results['cases_fully_detected']}")
    print(f"METRIC false_positives={results['false_positives']}")
    print(f"METRIC negative_false_positives={results['negative_false_positives']}")
    print(f"METRIC elapsed_ms={results['elapsed_ms']}")

    # Human-readable summary
    print(f"\n--- Unseen-Module Benchmark Results ---")
    print(f"Total test cases:        {results['total_cases']}")
    print(f"  Positive (must flag):  {results['positive_cases']}")
    print(f"  Negative (must pass):  {results['negative_cases']}")
    print(f"Expected violations:     {results['total_expected_violations']}")
    print(f"Found violations:        {results['total_found']}")
    print(f"Missed violations:       {results['missed_violations']}")
    print(f"False positives:         {results['false_positives']}")
    print(f"  On negative cases:     {results['negative_false_positives']}")
    print(f"Detection rate:          {results['detection_rate']}%")
    print(f"Case detection rate:     {results['case_detection_rate']}%")
    print(f"  Fully detected:        {results['cases_fully_detected']}")
    print(f"  Partially detected:    {results['cases_partially_detected']}")
    print(f"  Completely missed:     {results['cases_missed']}")
    print(f"Max depth found:         {results['max_depth_found']}")
    print(f"Elapsed:                 {results['elapsed_ms']}ms")

    print(f"\n--- By Category ---")
    for cat in sorted(results["by_category"].keys()):
        info = results["by_category"][cat]
        if cat == "negative":
            fp = info.get("false_positives", 0)
            ok = info["detected_cases"]
            print(f"  {cat}: {ok}/{info['cases']} clean (false_positives={fp})")
        else:
            exp = info["expected"]
            fnd = info["found"]
            rate = (fnd / exp * 100) if exp > 0 else 0
            case_rate = (info["detected_cases"] / info["cases"] * 100) if info["cases"] > 0 else 0
            print(
                f"  {cat}: {fnd}/{exp} violations ({rate:.0f}%), "
                f"{info['detected_cases']}/{info['cases']} cases ({case_rate:.0f}%)"
            )

    # Exit 0 only when all positive cases pass AND no negative false-positives
    success = (results["missed_violations"] == 0 and results["negative_false_positives"] == 0)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
