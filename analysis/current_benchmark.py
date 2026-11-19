"""Primary new-experiment entry point: dense-era V1, V2, or causal rolling V3."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis.data_policy import (DENSE_V1, DENSE_V2_TWO_RANK, DENSE_V3,
                                  MIN_EVENT_START_MS, POLICY_ID)
from analysis.holdout_dataset import HoldoutDataset
from analysis.temporal_dataset import TemporalDataset

DEFAULTS = {'v1':DENSE_V1, 'v2':DENSE_V2_TWO_RANK, 'v3':DENSE_V3}


def load_dataset(protocol, path=None):
    if protocol not in DEFAULTS:
        raise ValueError('Unknown protocol')
    path=Path(path) if path is not None else DEFAULTS[protocol]
    if protocol=='v1':
        data=HoldoutDataset(path)
    elif protocol=='v2':
        data=TemporalDataset(path)
    else:
        from analysis.rolling_dataset import RollingDataset
        data=RollingDataset(path)
    spec=data.manifest['protocol']
    if (spec.get('minimum_event_start_ms') != MIN_EVENT_START_MS
            or spec.get('data_policy_id') != POLICY_ID):
        raise ValueError('Current experiments require the dense population policy; old datasets are legacy-only')
    expected={'v1':'holdout_v1_dense', 'v2':'temporal_holdout_v2_dense', 'v3':'rolling_holdout_v3'}[protocol]
    if not spec.get('version','').startswith(expected):
        raise ValueError('Dataset version does not match the requested protocol')
    parts=pd.read_csv(data.root/'partitions.csv')
    if parts.empty or parts.start_ms.isna().any() or (parts.start_ms<MIN_EVENT_START_MS).any():
        raise ValueError('Dataset population contains pre-dense activities')
    train_ids=set(data.manifest['splits']['train']['event_ids'])
    for curve in data.train_curves():
        if curve.start<MIN_EVENT_START_MS or curve.event not in train_ids:
            raise ValueError('Initial training history violates the population policy')
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',choices=list(DEFAULTS),required=True)
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    parser.add_argument('--max-epochs',type=int,default=80)
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    data=load_dataset(args.protocol,args.dataset)
    if args.protocol=='v1':
        if args.max_epochs<1 or args.batch_size<1:parser.error('epochs and batch size must be positive')
        from analysis.holdout_benchmark import run_benchmark
        args.dataset=data.root
        run_benchmark(args)
    elif args.protocol=='v2':
        if args.resume:parser.error('V2 training requires a new run; completed prediction scoring is cached separately')
        from analysis.temporal_train import run_training
        run_training(data,args.run)
    else:
        from analysis.rolling_benchmark import run_experiment as run_rolling
        run_rolling(data,args.run,resume=args.resume)


if __name__=='__main__':main()
