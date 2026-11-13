"""Prints what the evidence files in tests/evidence contain, as one table.

    python tests/summarise_evidence.py
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
total = total_failed = 0
print("| evidence file | checks | failed | selftest max abs diff |")
print("| --- | --- | --- | --- |")
for path in sorted(glob.glob(os.path.join(HERE, "evidence", "*.json"))):
    data = json.load(open(path, encoding="utf-8"))
    checks = len(data["checks"])
    failed = len(data.get("failed", []))
    total += checks
    total_failed += failed
    diff = ""
    if "selftest" in data and isinstance(data["selftest"], dict):
        diff = "%.2e" % data["selftest"].get("maxAbsDiff", float("nan"))
    print("| %s | %d | %d | %s |" % (os.path.basename(path), checks, failed, diff))
print("\n%d checks in total, %d failed" % (total, total_failed))
