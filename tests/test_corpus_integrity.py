import json
from pathlib import Path

def test_baseline_results_schema():
    p = Path(__file__).resolve().parent.parent / 'corpus' / 'baseline_py312_results.json'
    data = json.loads(p.read_text(encoding='utf-8'))
    assert data['track_a_historical_regressions']['total_historical_cases'] == 18
    assert data['track_a_historical_regressions']['sciverify_caught'] == 14
