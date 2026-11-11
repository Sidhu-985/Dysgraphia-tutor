#!/usr/bin/env bash
# Runs every test suite available in this environment and writes logs + JUnit
# files to the directory given (default ../evidence/alphabet_dqn/test_logs).
# Flutter tests are attempted only if `flutter` is on PATH; otherwise that is
# recorded as NOT RUN (never as passed).
set -u
OUT=${1:-../evidence/alphabet_dqn/test_logs}
mkdir -p "$OUT"
cd "$(dirname "$0")/.."                       # backend/
python3 -m pytest -q -p no:cacheprovider -rsx --junitxml="$OUT/backend_junit.xml" > "$OUT/backend_pytest.log" 2>&1
echo "backend exit $?" | tee "$OUT/summary.txt"
tail -1 "$OUT/backend_pytest.log" | tee -a "$OUT/summary.txt"
(cd ../dysgraphia_rl && python3 -m pytest -q -p no:cacheprovider -rsx --junitxml="$OUT/rl_junit.xml" > "$OUT/rl_pytest.log" 2>&1)
echo "dysgraphia_rl exit $?" | tee -a "$OUT/summary.txt"
tail -1 "$OUT/rl_pytest.log" | tee -a "$OUT/summary.txt"
(cd ../tablet_app && python3 tool/dart_syntax_check.py . > "$OUT/dart_syntax_check.log" 2>&1)
echo "dart syntax check exit $? (syntax only; NOT flutter analyze)" | tee -a "$OUT/summary.txt"
tail -1 "$OUT/dart_syntax_check.log" | tee -a "$OUT/summary.txt"
if command -v flutter >/dev/null 2>&1; then
  (cd ../tablet_app && flutter pub get && flutter analyze && flutter test) > "$OUT/flutter.log" 2>&1
  echo "flutter exit $?" | tee -a "$OUT/summary.txt"
else
  echo "flutter: NOT RUN (Flutter SDK not installed in this environment)" | tee -a "$OUT/summary.txt"
fi
