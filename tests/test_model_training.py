"""Offline tiny-backbone routing, learnability and CLI training smoke test."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from tests.test_merge import REPO
from crossskinnet.models.network import CrossSkinNet
from crossskinnet.data.metadata import WIDTHS, FIELDS
from crossskinnet.train.engine import FocalLoss, seed_all


def test_routing_and_loss_decrease():
    """Fallback ignores metadata fillers; full fusion learns a tiny synthetic task."""
    seed_all(42)
    torch.set_num_threads(2)
    model = CrossSkinNet(2, 'tiny', {'pretrained':False}, dimension=16, heads=2)
    image = torch.randn(8,3,16,16)
    values = torch.randn(8,sum(WIDTHS))
    masks = torch.ones(8,len(FIELDS))
    masks[0] = 0
    masks[1,::2] = 0
    output = model(image,values,masks)
    assert output['logits'].shape == (8,2)
    assert torch.allclose(output['logits'][0],output['image_logits'][0])
    altered = values.clone()
    altered[0] = 1000
    assert torch.allclose(model(image,altered,masks)['logits'][0],output['logits'][0])
    labels = torch.arange(8) % 2
    optimizer = torch.optim.Adam(model.parameters(),lr=.02)
    criterion = FocalLoss(torch.ones(2))
    before = float(criterion(output['logits'],labels).detach())
    for _ in range(40):
        optimizer.zero_grad()
        loss = criterion(model(image,values,masks)['logits'],labels)
        loss.backward()
        optimizer.step()
    assert float(criterion(model(image,values,masks)['logits'],labels).detach()) < before * .3


def test_train_cli_and_resume(prepared_pipeline: Path):
    """The actual CLI saves an epoch checkpoint and calibrates only on validation."""
    args = [sys.executable,str(REPO / 'scripts/06_train.py'),'--config',str(prepared_pipeline),
            '--backbone','tiny','--epochs','1']
    result = subprocess.run(args,capture_output=True,text=True)
    assert result.returncode == 0,result.stderr
    from crossskinnet.io import config, read_csv
    run = config(prepared_pipeline)['output'] / 'runs/tiny_finetuned'
    assert (run / 'epoch_001.pt').exists()
    checkpoint = torch.load(run / 'best_calibrated.pt', map_location='cpu', weights_only=False)
    assert np.isfinite(checkpoint['calibration']['temperature'])
    assert checkpoint['calibration']['validation_nll_after'] <= checkpoint['calibration']['validation_nll_before']
    assert len(read_csv(run / 'metrics.csv')) == 1
    result = subprocess.run(args + ['--resume',str(run / 'last.pt')], capture_output=True,text=True)
    assert result.returncode == 0,result.stderr
