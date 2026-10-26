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


def test_train_cli_and_resume(trained_pipeline: tuple[Path, Path]):
    """The actual CLI saves an epoch checkpoint and calibrates only on validation."""
    prepared_pipeline, _ = trained_pipeline
    args = [sys.executable,str(REPO / 'scripts/06_train.py'),'--config',str(prepared_pipeline),
            '--backbone','tiny','--epochs','1']
    from crossskinnet.io import config, read_csv
    run = config(prepared_pipeline)['output'] / 'runs/tiny_finetuned'
    assert (run / 'epoch_001.pt').exists()
    checkpoint = torch.load(run / 'best_calibrated.pt', map_location='cpu', weights_only=False)
    assert np.isfinite(checkpoint['calibration']['temperature'])
    assert checkpoint['calibration']['validation_nll_after'] <= checkpoint['calibration']['validation_nll_before']
    assert len(read_csv(run / 'metrics.csv')) == 1
    result = subprocess.run(args + ['--resume',str(run / 'last.pt')], capture_output=True,text=True)
    assert result.returncode == 0,result.stderr


def test_tiny_local_dino_adapters_and_cam(tmp_path: Path):
    """Tiny randomly initialized local DINO models exercise token reshaping offline."""
    from transformers import Dinov2Config, Dinov2Model, DINOv3ViTConfig, DINOv3ViTModel
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from crossskinnet.eval.explain import FixedMetadata, _reshape
    torch.set_num_threads(2)
    for name, config_class, model_class in [('dinov2', Dinov2Config, Dinov2Model),
                                           ('dinov3', DINOv3ViTConfig, DINOv3ViTModel)]:
        cfg = config_class(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                           intermediate_size=64, image_size=32, patch_size=8)
        if name == 'dinov3':
            cfg.num_register_tokens = 2
        folder = tmp_path / name
        model_class(cfg).save_pretrained(folder)
        model = CrossSkinNet(2, name, {'id':str(folder), 'local_files_only':True, 'frozen':True},16,2).eval()
        image = torch.randn(1,3,32,32,requires_grad=True)
        meta, mask = torch.ones(1,sum(WIDTHS)), torch.ones(1,len(FIELDS))
        assert model(image,meta,mask)['logits'].shape == (1,2)
        with GradCAM(model=FixedMetadata(model,meta,mask),target_layers=[model.image_encoder.cam_layer()],
                     reshape_transform=lambda value:_reshape(model,value)) as cam:
            heatmap = cam(image, targets=[ClassifierOutputTarget(0)])
            assert heatmap.shape == (1,32,32) and np.isfinite(heatmap).all()


def test_resume_continues_next_epoch(prepared_pipeline: Path, tmp_path: Path):
    """A time-budget stop resumes the saved optimizer/scheduler into the next epoch."""
    import yaml
    from crossskinnet.io import config, read_csv
    from crossskinnet.train.engine import run
    cfg = yaml.safe_load(prepared_pipeline.read_text())
    old_output = config(prepared_pipeline)['output']
    cfg['paths']['unified_root'] = str(old_output / 'CrossSkinNet_Unified_Dataset')
    cfg['paths']['output_root'] = str(tmp_path / 'outputs')
    cfg['training'].update(epochs=2, max_hours=0)
    path = tmp_path / 'resume.yaml'
    path.write_text(yaml.safe_dump(cfg))
    run(path)
    directory = tmp_path / 'outputs/runs/tiny_finetuned'
    assert len(read_csv(directory / 'metrics.csv')) == 1
    run(path, resume=directory / 'last.pt')
    assert len(read_csv(directory / 'metrics.csv')) == 2
    assert (directory / 'epoch_002.pt').exists()


def test_pending_dedup_blocks_training(prepared_pipeline: Path, tmp_path: Path):
    """An old built dataset cannot bypass an active unresolved duplicate review."""
    import json
    import yaml
    import pytest
    from crossskinnet.io import config
    from crossskinnet.train.engine import run
    cfg = yaml.safe_load(prepared_pipeline.read_text())
    cfg['paths']['unified_root'] = str(config(prepared_pipeline)['output'] / 'CrossSkinNet_Unified_Dataset')
    output = tmp_path / 'pending'
    output.mkdir()
    cfg['paths']['output_root'] = str(output)
    (output / 'dedup_status.json').write_text(json.dumps({'policy':'review','ready':False}))
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='Duplicate review incomplete'):
        run(path)
    assert not (output / 'runs').exists()


def test_controlled_baselines_and_image_only_invariance():
    """Image-only ignores metadata, while concat supports missing-field fallback."""
    from crossskinnet.models.network import CrossSkinNet
    seed_all(42)
    torch.set_num_threads(2)
    image=torch.randn(3,3,16,16)
    values=torch.randn(3,sum(WIDTHS))
    mask=torch.ones(3,len(FIELDS))
    for architecture in ('image_only','concat'):
        model=CrossSkinNet(3,'tiny',{},16,2,architecture).eval()
        result=model(image,values,mask)
        assert result['logits'].shape==(3,3)
        missing=model(image,torch.zeros_like(values),torch.zeros_like(mask))
        assert torch.allclose(missing['logits'],missing['image_logits'])
        if architecture=='image_only':
            assert torch.allclose(result['logits'],missing['logits'])
