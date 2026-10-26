"""One reproducible entry point for preparation, individual experiments and paper tables."""
from __future__ import annotations

from collections import defaultdict, Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import yaml

from crossskinnet.io import config, read_csv, write_csv
from crossskinnet.data.merge import merge_datasets
from crossskinnet.data.labels import apply as apply_labels
from crossskinnet.data.metadata import apply as apply_metadata
from crossskinnet.data.dedup import run as deduplicate
from crossskinnet.data.build_unified import run as build


def digest(path: Path) -> str:
    """Hash large files with bounded host-memory usage."""
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def resources(path: Path, cfg: dict[str, Any]) -> dict[str, Path]:
    """Resolve schema and mapping paths solely from workflow configuration."""
    result = {}
    for key in ('dataset_schemas','label_mapping','site_mapping'):
        value = cfg.get('workflow',{}).get(key)
        if not value:
            raise ValueError(f'Missing workflow.{key} in {path}')
        file = Path(value)
        result[key] = file if file.is_absolute() else path.resolve().parent / file
        if not result[key].is_file():
            raise ValueError(f'Missing configured {key}: {result[key]}')
    return result


def _write_config(path: Path, cfg: dict[str, Any]) -> None:
    """Persist a runtime config without derived Path objects or secrets."""
    serial = deepcopy(cfg)
    serial.pop('output',None)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(yaml.safe_dump(serial,sort_keys=False))


def cohort_report(output: Path, dataset: Path) -> None:
    """Account for every selected development image without calling quarantine a duplicate."""
    selected=read_csv(output/'metadata_train_pool.csv')
    removed=read_csv(output/'removed_training_duplicates.csv')
    excluded=read_csv(output/'quarantined_training.csv')
    retained=read_csv(dataset/'metadata.csv')
    counts=[Counter((row['source_dataset'],row['unified_label']) for row in table)
            for table in (selected,removed,excluded,retained)]
    rows=[]
    for source,label in sorted(counts[0]):
        key=(source,label)
        before,confirmed,quarantined,after=[count[key] for count in counts]
        assert before==confirmed+quarantined+after,'Cohort accounting mismatch'
        groups={row['group_id'] for row in retained if (row['source_dataset'],row['unified_label'])==key}
        rows.append(dict(source_dataset=source,unified_label=label,selected_images=before,
            verified_duplicate_removed=confirmed,uncertain_or_external_linked_group_excluded=quarantined,
            retained_images=after,recorded_split_groups=len(groups)))
    write_csv(output/'cohort_flow.csv',rows)
    summary=dict(selected_development_images=len(selected),verified_duplicate_removals=len(removed),
        group_quarantine_images=len(excluded),retained_development_images=len(retained),
        raw_external_reference_images=len(read_csv(output/'raw_test_pool.csv')),
        eligible_external_images=len(read_csv(dataset/'zero_shot.csv')),
        quarantine_is_duplicate_verdict=False)
    (output/'cohort_summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Prepared study cohort','',
        'These are data preparation counts, not model performance or a publication result.','',
        f"Selected development images: {len(selected)}.",
        f"Verified duplicate removals: {len(removed)}.",
        f"Group-quarantine exclusions: {len(excluded)} (not duplicate verdicts).",
        f"Retained development images: {len(retained)}.",
        f"Eligible external evaluation images: {summary['eligible_external_images']}.",
        f"Raw external overlap references: {summary['raw_external_reference_images']}.",'',
        'Input images are untouched. Quarantine excludes entire recorded development groups',
        'and verified-copy-linked groups before splitting. This conservative cohort selection',
        'can change class/source distributions and must be disclosed in the paper.',
        'PAD groups are recorded patients; HAM groups are lesions, not proven distinct patients.',
        'PH2 stays external, including raw reference images excluded from evaluation labels.',
        'See cohort_flow.csv, dataset_stats.json and configuration_snapshot for provenance.']
    (output/'COHORT_REPORT.md').write_text('\n'.join(lines)+'\n')


def prepare(path: Path) -> Path:
    """Run all preparation stages; reuse export only with matching prepared inputs."""
    cfg = config(path)
    files = resources(path,cfg)
    output = cfg['output']
    runtime = deepcopy(cfg)
    runtime['dedup']['grouping_schemas'] = str(files['dataset_schemas'])
    runtime_path = output / 'runtime_prepare.yaml'
    _write_config(runtime_path,runtime)
    status_path = output / 'preparation_status.json'
    status = dict(complete=False,stage='merge')
    status_path.write_text(json.dumps(status,indent=2))
    try:
        merge_datasets(runtime_path,files['dataset_schemas'])
        status['stage']='labels'
        status_path.write_text(json.dumps(status,indent=2))
        apply_labels(runtime_path,files['label_mapping'])
        status['stage']='metadata'
        status_path.write_text(json.dumps(status,indent=2))
        apply_metadata(runtime_path,files['dataset_schemas'],files['site_mapping'])
        cache = output / 'image_hashes.csv'
        if not cache.exists():
            for value in cfg.get('workflow',{}).get('hash_cache_candidates',[]):
                prior = Path(value)
                if prior.is_file():
                    shutil.copyfile(prior,cache)
                    break
        status['stage']='deduplication'
        status_path.write_text(json.dumps(status,indent=2))
        deduplicate(runtime_path)
        inputs = [output/'deduped_train_pool.csv',output/'metadata_test_pool.csv',output/'dedup_status.json',cache,*files.values()]
        fingerprint = hashlib.sha256(json.dumps(dict(version=1,seed=cfg['seed'],
            policy=cfg['dedup'],study=cfg.get('study',{}),metadata=cfg.get('metadata',{}),
            implementation={name:digest(Path(__file__).parent/'data'/f'{name}.py') for name in ('labels','metadata','dedup','quarantine','grouping','build_unified')},
            inputs={str(file):digest(file) for file in inputs}),sort_keys=True).encode()).hexdigest()
        target = output / 'CrossSkinNet_Unified_Dataset'
        old_file = output / 'prepared_dataset.json'
        old = json.loads(old_file.read_text()) if old_file.exists() else {}
        reusable = old.get('fingerprint')==fingerprint and (target/'dataset_integrity.json').exists()
        if reusable:
            from crossskinnet.train.engine import validate_integrity
            validate_integrity(target)
            rows = read_csv(target/'metadata.csv')+read_csv(target/'zero_shot.csv')
            reusable = all((target/r['image_path']).is_file() for r in rows)
        if not reusable:
            status['stage']='build'
            status_path.write_text(json.dumps(status,indent=2))
            target = build(runtime_path,files['label_mapping'],files['dataset_schemas'])
        cohort_report(output,target)
        record = dict(fingerprint=fingerprint,dataset_root=str(target),split_seed=cfg['seed'],
            dataset_metadata_sha256=digest(target/'metadata.csv'),
            external_metadata_sha256=digest(target/'zero_shot.csv'))
        old_file.write_text(json.dumps(record,indent=2))
        snapshots = output / 'configuration_snapshot'
        snapshots.mkdir(exist_ok=True)
        for key,file in files.items():
            shutil.copyfile(file,snapshots/f'{key}{file.suffix}')
        _write_config(snapshots/'paper.yaml',runtime)
        status.update(complete=True,stage='prepared',reused_image_export=reusable,**record)
        status_path.write_text(json.dumps(status,indent=2))
        print(f'PREPARATION COMPLETE: {target}',flush=True)
        return target
    except Exception as error:
        status['error']=str(error)
        status_path.write_text(json.dumps(status,indent=2))
        raise


def _prepared(cfg: dict[str, Any]) -> Path:
    """Require a completed current preparation, never an old partial directory."""
    path = cfg['output']/'preparation_status.json'
    if not path.exists() or not json.loads(path.read_text()).get('complete'):
        raise ValueError('Run 08_paper.py prepare successfully before experiments')
    state = json.loads(path.read_text())
    root = Path(state['dataset_root'])
    if digest(root/'metadata.csv')!=state['dataset_metadata_sha256'] or digest(root/'zero_shot.csv')!=state['external_metadata_sha256']:
        raise ValueError('Prepared dataset changed; rerun preparation')
    return root


def experiment_id(architecture: str, backbone: str, frozen: bool, seed: int) -> str:
    """Give every model/mode/seed its own retained output directory."""
    return f'{architecture}_{backbone}_{"frozen" if frozen else "finetuned"}_seed{seed}'


def train(path: Path, backbone: str, architecture: str='crossskinnet', seed: int | None=None,
          frozen: bool=False, epochs: int | None=None) -> Path:
    """Train or resume one experiment on the identical prepared dataset split."""
    from crossskinnet.train.engine import run as train_model

    cfg = config(path)
    root = _prepared(cfg)
    seed = int(cfg['seed'] if seed is None else seed)
    name = experiment_id(architecture,backbone,frozen,seed)
    directory = cfg['output']/'experiments'/name
    directory.mkdir(parents=True,exist_ok=True)
    runtime = deepcopy(cfg)
    runtime['seed']=seed
    runtime['paths'].update(output_root=str(directory),unified_root=str(root))
    runtime['model']['architecture']=architecture
    if epochs is not None:
        runtime['training']['epochs']=epochs
    runtime_path = directory/'run_config.yaml'
    previous = yaml.safe_load(runtime_path.read_text()) if runtime_path.exists() else None
    serial = deepcopy(runtime)
    serial.pop('output',None)
    if previous is not None and previous!=serial:
        raise ValueError(f'{name}: config differs from existing run; keep the original schedule or use a fresh output_root')
    _write_config(runtime_path,runtime)
    prefix = '' if architecture=='crossskinnet' else architecture+'_'
    run_dir = directory/'runs'/f'{prefix}{backbone}_{"frozen" if frozen else "finetuned"}'
    state_file = run_dir/'training_state.json'
    checkpoint = run_dir/'best_calibrated.pt'
    if state_file.exists():
        state = json.loads(state_file.read_text())
        if state.get('dataset_signature')!=digest(root/'metadata.csv'):
            raise ValueError('Existing experiment was trained on another dataset; use a fresh output_root')
        if state.get('complete') and state.get('calibration_complete') and checkpoint.exists():
            record = dict(experiment=name,experiment_role='software_smoke' if backbone=='tiny' else 'paper_experiment',architecture=architecture,backbone=backbone,frozen=frozen,seed=seed,
                checkpoint=str(checkpoint),dataset_signature=digest(root/'metadata.csv'),training=state)
            record_file = directory/'experiment.json'
            if not record_file.exists():
                record_file.write_text(json.dumps(record,indent=2))
            print(f'Completed experiment reused: {name}',flush=True)
            return checkpoint
    resume = run_dir/'last.pt'
    checkpoint = train_model(runtime_path,backbone,resume=resume if resume.exists() else None,frozen=frozen)
    record = dict(experiment=name,experiment_role='software_smoke' if backbone=='tiny' else 'paper_experiment',architecture=architecture,backbone=backbone,frozen=frozen,seed=seed,
        checkpoint=str(checkpoint),dataset_signature=digest(root/'metadata.csv'),
        training=json.loads(state_file.read_text()))
    (directory/'experiment.json').write_text(json.dumps(record,indent=2))
    print(f'Experiment state: {record["training"]}',flush=True)
    return checkpoint


def evaluate(path: Path, backbone: str, architecture: str='crossskinnet', seed: int | None=None,
             frozen: bool=False, explainability: bool=True) -> Path:
    """Evaluate a completed experiment; never rank time-budget-stopped partial runs."""
    from crossskinnet.eval.evaluate import run as evaluate_model

    cfg = config(path)
    _prepared(cfg)
    name = experiment_id(architecture,backbone,frozen,int(cfg['seed'] if seed is None else seed))
    directory = cfg['output']/'experiments'/name
    record_path = directory/'experiment.json'
    if not record_path.exists():
        raise ValueError(f'Train {name} first')
    record = json.loads(record_path.read_text())
    if not record['training']['complete']:
        raise ValueError('Training stopped at the time budget; resume it before final paper evaluation')
    destination = evaluate_model(directory/'run_config.yaml',Path(record['checkpoint']),explainability)
    record['results']=str(destination)
    record['evaluated']=True
    record['explainability']=explainability
    record_path.write_text(json.dumps(record,indent=2))
    summarize(path)
    return destination


def plan(path: Path) -> list[dict[str, Any]]:
    """List 4 backbones x 2 modes x configured seeds, plus controlled baselines."""
    cfg = config(path)
    options = cfg['workflow']
    rows = []
    for seed in options.get('seeds',[cfg['seed']]):
        specs = [('crossskinnet',backbone,frozen) for backbone in options['backbones'] for frozen in (True,False)]
        specs += [(architecture,options.get('baseline_backbone','efficientnet'),False)
                  for architecture in options.get('baselines',['image_only','concat'])]
        for architecture,backbone,frozen in specs:
            rows.append(dict(experiment=experiment_id(architecture,backbone,frozen,int(seed)),
                architecture=architecture,backbone=backbone,frozen=frozen,seed=int(seed)))
    write_csv(cfg['output']/'experiment_plan.csv',rows)
    print('Planned experiments:',len(rows),'; this command does not launch them.',flush=True)
    return rows


def summarize(path: Path) -> Path:
    """Collect only actual complete evaluations and report across-seed mean/std."""
    cfg = config(path)
    rows = []
    expected = _prepared(cfg)
    signature = digest(expected/'metadata.csv')
    for file in sorted((cfg['output']/'experiments').glob('*/experiment.json')):
        record = json.loads(file.read_text())
        if not record.get('evaluated') or not record['training']['complete']:
            continue
        if record['dataset_signature']!=signature:
            raise ValueError('Experiment datasets differ; do not mix these comparison rows')
        metrics = json.loads((Path(record['results'])/'metrics.json').read_text())
        for split in ('test','zero_shot'):
            metric = metrics[f'{split}_full']
            rows.append({key:record[key] for key in ('experiment','experiment_role','architecture','backbone','frozen','seed')} |
                dict(split=split,**{key:metric[key] for key in ('samples','accuracy','balanced_accuracy','macro_f1','macro_f1_all_training_classes','macro_auc_roc','ece','evaluated_classes')}))
    results = cfg['output']/'results'
    results.mkdir(exist_ok=True)
    fields = ['experiment','experiment_role','architecture','backbone','frozen','seed','split','samples','accuracy','balanced_accuracy',
              'macro_f1','macro_f1_all_training_classes','macro_auc_roc','ece','evaluated_classes']
    write_csv(results/'all_completed_runs.csv',rows,fields)
    write_csv(results/'software_smoke_results.csv',[row for row in rows if row['experiment_role']=='software_smoke'],fields)
    rows = [row for row in rows if row['experiment_role']=='paper_experiment']
    write_csv(results/'paper_results.csv',rows,fields)
    groups: dict[tuple[Any,...],list[dict[str,Any]]] = defaultdict(list)
    for row in rows:
        groups[(row['architecture'],row['backbone'],row['frozen'],row['split'])].append(row)
    summary = []
    for (architecture,backbone,frozen,split),members in groups.items():
        values = [row['macro_f1'] for row in members if row['macro_f1'] is not None]
        summary.append(dict(architecture=architecture,backbone=backbone,frozen=frozen,split=split,
            completed_seeds=len(values),seeds='|'.join(str(row['seed']) for row in members),
            mean_macro_f1=float(np.mean(values)) if values else None,
            std_macro_f1=float(np.std(values,ddof=1)) if len(values)>1 else None))
    write_csv(results/'seed_summary.csv',summary,['architecture','backbone','frozen','split','completed_seeds','seeds','mean_macro_f1','std_macro_f1'])
    print(f'Actual completed result rows: {len(rows)}; {results}',flush=True)
    return results
