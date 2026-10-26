"""One-entry preparation/training/evaluation with fake sources and no weight downloads."""
from pathlib import Path
import json
import shutil
import sys
import subprocess

import pytest
import yaml

from tests.test_merge import REPO
from crossskinnet.data.synthetic import make_synthetic_data
from crossskinnet.io import write_csv,read_csv
from crossskinnet.pipeline import prepare,train,evaluate,plan


@pytest.fixture(scope='module')
def paper_config(tmp_path_factory):
    """Give the complete paper controller an explicit synthetic input schema."""
    root=make_synthetic_data(tmp_path_factory.mktemp('paper')/'inputs')
    generated=yaml.safe_load((root/'config.yaml').read_text())
    cfg=yaml.safe_load((REPO/'configs/paper.yaml').read_text())
    for source,settings in cfg['datasets'].items():
        settings['path']=generated['datasets'][source]['path'] if settings.get('enabled',True) else str(root/'not_attached')
    cfg['paths']['output_root']=str(root.parent/'output')
    cfg['study']['target_labels']=[f'synthetic_class_{i}' for i in range(3)]
    cfg['training'].update(epochs=1,image_size=32,batch_size=16,device='cpu')
    cfg['evaluation'].update(bootstrap_repetitions=10,explanations_per_outcome=1,shap_samples=32)
    cfg['workflow'].update(dataset_schemas=str(root/'dataset_schemas.yaml'),label_mapping=str(root.parent/'labels.csv'),
                           site_mapping=str(root.parent/'sites.yaml'),hash_cache_candidates=[])
    cfg['dedup']['grouping_schemas']=str(root/'dataset_schemas.yaml')
    schemas=yaml.safe_load((root/'dataset_schemas.yaml').read_text())
    schemas['datasets']['PAD-UFES-20']['columns']['patient_id']='group_id'
    schemas['datasets']['PAD-UFES-20']['grouping']={'lesion_id_scope':'patient','required_group_field':'patient_id'}
    schemas['datasets']['HAM10000']['grouping']={'required_group_field':'lesion_id'}
    schemas['datasets']['PH2']['columns'].pop('age')
    schemas['datasets']['PH2']['columns'].pop('lesion_id')
    (root/'dataset_schemas.yaml').write_text(yaml.safe_dump(schemas))
    shutil.copyfile(root/'HAM10000/images/image_000.png',root/'PH2/images/image_000.png')
    mappings=[dict(source_dataset=source,original_label=f'synthetic_class_{i}',unified_label=f'synthetic_class_{i}',
              coarse_group='other',include='True',notes='Synthetic software test only')
              for source in ('PAD-UFES-20','HAM10000','PH2') for i in range(3)]
    write_csv(root.parent/'labels.csv',mappings)
    (root.parent/'sites.yaml').write_text('datasets: {}\n')
    path=root.parent/'paper.yaml'
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_prepare_repeatability_and_full_tiny_run(paper_config: Path):
    """Complete current preparation, same-split baselines and actual evaluation work."""
    dataset=prepare(paper_config)
    cfg=yaml.safe_load(paper_config.read_text())
    output=Path(cfg['paths']['output_root'])
    status=json.loads((output/'dedup_status.json').read_text())
    assert status['policy']=='quarantine' and status['ready']
    assert status['quarantined_training_rows']>0
    assert {r['source_dataset'] for r in read_csv(dataset/'metadata.csv')}=={'PAD-UFES-20','HAM10000'}
    assert {r['source_dataset'] for r in read_csv(dataset/'zero_shot.csv')}=={'PH2'}
    original=(dataset/'metadata.csv').read_bytes()
    prepare(paper_config)
    assert (dataset/'metadata.csv').read_bytes()==original
    assert json.loads((output/'preparation_status.json').read_text())['reused_image_export']
    for architecture in ('image_only','concat','crossskinnet'):
        checkpoint=train(paper_config,'tiny',architecture=architecture,epochs=1)
        assert checkpoint.exists()
        assert train(paper_config,'tiny',architecture=architecture,epochs=1)==checkpoint
    results=evaluate(paper_config,'tiny',architecture='crossskinnet')
    assert list((results/'explanations').glob('*_gradcam.png'))
    assert (results/'explanations/metadata_shap.csv').exists()
    assert not read_csv(output/'results/paper_results.csv')
    assert len(read_csv(output/'results/software_smoke_results.csv'))==2
    assert len(plan(paper_config))==30
    # Run the actual public entry point with the same prepared split/checkpoint.
    result=subprocess.run([sys.executable,str(REPO/'scripts/08_paper.py'),'run','--config',str(paper_config),
        '--backbone','tiny','--epochs','1'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    flow=read_csv(output/'cohort_flow.csv')
    assert all(int(row['selected_images'])==int(row['verified_duplicate_removed'])+
        int(row['uncertain_or_external_linked_group_excluded'])+int(row['retained_images']) for row in flow)
    with pytest.raises(ValueError,match='config differs'):
        train(paper_config,'tiny',epochs=2)


def test_anatomical_map_covers_observed_strings():
    """Known source terms map; unseen anatomy cannot silently become other."""
    from crossskinnet.data.metadata import _normalize
    mapping=yaml.safe_load((REPO/'configs/site_mapping.yaml').read_text())['datasets']
    schema={'columns':{'age':'age','sex':'sex','site':'site'}}
    for source,raw,expected in [('PAD-UFES-20','ARM','upper_limb'),('HAM10000','acral','hands_feet')]:
        result=_normalize(dict(source_dataset=source,image_id='test',age='40',sex='male',site=raw),schema,mapping[source])
        assert result['site']==expected
    with pytest.raises(ValueError,match='Map raw site'):
        _normalize(dict(source_dataset='PAD-UFES-20',image_id='test',site='UNSEEN_REGION'),schema,mapping['PAD-UFES-20'])
