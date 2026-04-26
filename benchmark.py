#!/usr/bin/env python3
"""Benchmark script for fastapi-evloop-linter.

Runs the linter against adversarial test cases and measures:
- Detection rate (how many expected violations were found)
- False negative rate (missed violations)
- Total violations found

Outputs METRIC lines for autoresearch.
"""

import json
import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi_evloop_linter.checker import EventLoopChecker
from fastapi_evloop_linter.callgraph import analyze_source
from tests.adversarial.generator import generate_all_cases, generate_random_cases, TestCase


def run_benchmark() -> dict:
    """Run the full adversarial benchmark suite."""
    # Generate all test cases
    cases = generate_all_cases()
    cases.extend(generate_random_cases(10, seed=42))

    checker = EventLoopChecker(max_depth=20)

    total_cases = len(cases)
    total_expected = 0
    total_found = 0
    total_false_negatives = 0
    total_false_positives = 0
    cases_fully_detected = 0
    cases_partially_detected = 0
    cases_missed = 0
    max_depth_found = 0
    detection_by_difficulty = {}  # difficulty -> (expected, found)

    start_time = time.perf_counter()

    for case in cases:
        result = checker.check_source(case.source, filepath=f"<{case.name}>")

        # Count how many violations the linter found
        found = len(result.violations)
        expected = case.expected_violations

        total_expected += expected
        total_found += min(found, expected)  # Don't count extras as true positives

        if found >= expected:
            cases_fully_detected += 1
            total_found = total_found  # Already counted
        elif found > 0:
            cases_partially_detected += 1
        else:
            cases_missed += 1

        # Track false positives/negatives
        if found < expected:
            total_false_negatives += (expected - found)
        if found > expected:
            total_false_positives += (found - expected)

        # Track max depth found
        if result.violations:
            case_max_depth = max(v.depth for v in result.violations)
            max_depth_found = max(max_depth_found, case_max_depth)

        # Track by difficulty
        if case.difficulty not in detection_by_difficulty:
            detection_by_difficulty[case.difficulty] = {"expected": 0, "found": 0, "cases": 0, "detected_cases": 0}
        detection_by_difficulty[case.difficulty]["expected"] += expected
        detection_by_difficulty[case.difficulty]["found"] += min(found, expected)
        detection_by_difficulty[case.difficulty]["cases"] += 1
        if found >= expected:
            detection_by_difficulty[case.difficulty]["detected_cases"] += 1

    elapsed = time.perf_counter() - start_time

    # Calculate metrics
    detection_rate = (total_found / total_expected * 100) if total_expected > 0 else 0
    case_detection_rate = (cases_fully_detected / total_cases * 100) if total_cases > 0 else 0

    # Missed score (lower is better) - this is our primary metric
    missed = total_expected - total_found

    return {
        "total_cases": total_cases,
        "total_expected_violations": total_expected,
        "total_found": total_found,
        "missed_violations": missed,
        "false_negatives": total_false_negatives,
        "false_positives": total_false_positives,
        "cases_fully_detected": cases_fully_detected,
        "cases_partially_detected": cases_partially_detected,
        "cases_missed": cases_missed,
        "detection_rate": round(detection_rate, 1),
        "case_detection_rate": round(case_detection_rate, 1),
        "max_depth_found": max_depth_found,
        "elapsed_ms": round(elapsed * 1000, 1),
        "by_difficulty": detection_by_difficulty,
    }


def main():
    results = run_benchmark()

    # Output structured metrics for autoresearch
    print(f"METRIC missed={results['missed_violations']}")
    print(f"METRIC detection_rate={results['detection_rate']}")
    print(f"METRIC cases_missed={results['cases_missed']}")
    print(f"METRIC cases_fully_detected={results['cases_fully_detected']}")
    print(f"METRIC false_positives={results['false_positives']}")
    print(f"METRIC elapsed_ms={results['elapsed_ms']}")

    # Print summary
    print(f"\n--- Benchmark Results ---")
    print(f"Total test cases:    {results['total_cases']}")
    print(f"Expected violations:  {results['total_expected_violations']}")
    print(f"Found violations:     {results['total_found']}")
    print(f"Missed violations:    {results['missed_violations']}")
    print(f"False positives:      {results['false_positives']}")
    print(f"Detection rate:       {results['detection_rate']}%")
    print(f"Case detection rate:  {results['case_detection_rate']}%")
    print(f"  Fully detected:     {results['cases_fully_detected']}")
    print(f"  Partially detected: {results['cases_partially_detected']}")
    print(f"  Completely missed:  {results['cases_missed']}")
    print(f"Max depth found:      {results['max_depth_found']}")
    print(f"Elapsed:              {results['elapsed_ms']}ms")

    # Print by difficulty
    print(f"\n--- By Difficulty ---")
    for diff in sorted(results['by_difficulty'].keys()):
        info = results['by_difficulty'][diff]
        rate = (info['found'] / info['expected'] * 100) if info['expected'] > 0 else 0
        case_rate = (info['detected_cases'] / info['cases'] * 100) if info['cases'] > 0 else 0
        print(f"  Difficulty {diff}: {info['found']}/{info['expected']} violations ({rate:.0f}%), "
              f"{info['detected_cases']}/{info['cases']} cases ({case_rate:.0f}%)")

    return 0 if results['missed_violations'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
