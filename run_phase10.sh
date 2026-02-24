#!/bin/bash
cd "$(dirname "$0")"
echo "=== Running main.py ==="
python main.py
echo ""
echo "=== Phase 10 Verification ==="
python -c "
import json
r = json.load(open('results.json'))
rf = r['models']['rf']
ens = r['models']['ensemble']
print('RF  opt_thresh:', rf['optimal_threshold'])
print('RF  SPE@thresh:', rf['specificity_at_threshold'], '>= 0.70?', rf['specificity_at_threshold'] >= 0.70)
print('RF  SEN@thresh:', rf['sensitivity_at_threshold'], '>= 0.75?', rf['sensitivity_at_threshold'] >= 0.75)
print('ENS AUC:       ', ens['auc'], '>= 0.800?', ens['auc'] >= 0.800)
thr2 = rf['specificity_at_threshold'] >= 0.70 and rf['sensitivity_at_threshold'] >= 0.75
ens2 = ens['auc'] >= 0.800
print()
print('THR-02:', thr2, '| ENS-02:', ens2)
print('PASS' if thr2 and ens2 else 'FAIL')
"
