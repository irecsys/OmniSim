"""
CI helper: fail if any test was skipped. Reads the output of `pytest -rs` (file given as the argument), so that
no test can fail or be skipped silently on CI.

Usage:
    python -m pytest -q -rs | tee pytest.txt; python scripts/ci_check_skips.py pytest.txt
"""
import re
import sys

EXPECTED = 0                      # no test may be skipped on CI


def main(path):
    text = open(path, encoding='utf-8').read()
    skips = re.findall(r'^SKIPPED \[(\d+)\] (\S+?):\d+: (.*)$', text, flags=re.M)
    n = sum(int(s[0]) for s in skips)
    for count, f, reason in skips:
        print(f"skipped {count} in {f}: {reason}")
    if n != EXPECTED:
        print(f"FAIL: expected {EXPECTED} skipped tests, found {n}")
        return 1
    print(f"OK: {EXPECTED} skipped tests")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
