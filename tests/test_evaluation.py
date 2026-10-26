"""Offline end-to-end held-out, robustness, SHAP and Grad-CAM validation."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from tests.test_merge import REPO
from crossskinnet.eval.metrics import bootstrap_f1, classification_metrics
from crossskinnet.io import config, read_csv


def test_metrics_and_group_bootstrap():
    """Perfect predictions give perfect discrimination and small known calibration error."""
    y = np.array([0,0,1,1])
    p = np.array([[.9,.1],[.9,.1],[.1,.9],[.1,.9]])
    result = classification_metrics(y,p,['a','b'])
    assert result['accuracy'] == result['macro_f1'] == result['macro_auc_roc'] == 1
    assert abs(result['ece'] - .1) < 1e-6
    assert bootstrap_f1(y,p,['a','b','c','d'],42,20) == bootstrap_f1(y,p,['a','b','c','d'],42,20)
    assert classification_metrics(np.array([],dtype=int), np.zeros((0,2)), ['a','b'])['macro_f1'] is None


def test_evaluate_cli(trained_pipeline: tuple[Path, Path]):
    """The complete synthetic pipeline produces scores, ablations and real XAI artifacts."""
    cfg, checkpoint = trained_pipeline
    result = subprocess.run([sys.executable,str(REPO / 'scripts/07_evaluate.py'),
        '--config',str(cfg),'--checkpoint',str(checkpoint)], capture_output=True,text=True)
    assert result.returncode == 0,result.stderr
    output = config(cfg)['output'] / 'results/tiny_finetuned'
    metrics = json.loads((output / 'metrics.json').read_text())
    assert 'test_full' in metrics and 'zero_shot_all_missing' in metrics
    assert len(read_csv(output / 'robustness.csv')) == 18
    assert (output / 'test_confusion.png').exists()
    assert list((output / 'explanations').glob('*_gradcam.png'))
    shap = read_csv(output / 'explanations/metadata_shap.csv')
    assert shap and all(float(r['reconstruction_error']) < 1e-4 for r in shap)
    assert (output / 'explanations/metadata_shap_global.png').exists()
    assert len(read_csv(output.parent / 'backbone_comparison.csv')) == 2


def test_external_class_scope_keeps_off_scope_predictions_wrong():
    """No PH2 BCC support: evaluate supported labels without restricting model argmax."""
    y = np.array([1,1,2,2])
    probability = np.array([[.01,.98,.01],[.01,.98,.01],[.01,.01,.98],[.01,.01,.98]])
    result = classification_metrics(y,probability,['bcc','mel','nevus'],evaluation_labels=[1,2])
    assert result['macro_f1']==1
    assert abs(result['macro_f1_all_training_classes']-2/3)<1e-6
    assert result['evaluated_classes']=='mel|nevus'
    probability[0]=[.98,.01,.01]
    assert classification_metrics(y,probability,['bcc','mel','nevus'],evaluation_labels=[1,2])['accuracy']==.75
    # Fixed supported labels remain in bootstrap scope even when a resample misses one.
    low,high = bootstrap_f1(y,probability,['a','b','c','d'],42,20,evaluation_labels=[1,2])
    assert 0<=low<=high<=1
