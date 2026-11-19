"""Complete training-only diagnostics from frozen V1 predictions and references."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.current_benchmark import load_dataset
from analysis.dense_v1_extensions import DATASET_ID, read_values, save_predictions
from analysis.holdout_dataset import sha256, write_json


def check_training_coverage(actual, reported, selected, unavailable=()):
    actual, reported, unavailable = map(set,[actual,reported,unavailable])
    allowed = {'template','same_type_template','shape_analog'}
    if not unavailable <= allowed or not unavailable <= reported:
        raise ValueError('Only declared causal-history baselines may be unavailable')
    if actual != reported-unavailable or selected not in actual:
        raise ValueError('Training method coverage differs or selected reference is unavailable')


def training_combinations(family, values, reference):
    values = dict(values)
    if family in ['relationship','structured']:
        values['reference'] = reference
        source, prefix = ('new_top3','blend_new_') if family=='relationship' else ('neural_top3','blend_neural_')
        for weight in [.25,.5,.75]:
            values[prefix+str(weight)] = weight*values[source]+(1-weight)*reference
    elif family in ['deadline','quality','quality_extended']:
        values['current_reference'] = reference
        if family.startswith('quality'):
            for name in list(values):
                if name!='current_reference' and '_seed' not in name:
                    values['blend25_'+name] = .25*values[name]+.75*reference
    return values


def replay_training(run, core_training, output):
    run, core_training, output = map(Path,[run,core_training,output])
    if output.exists():
        raise FileExistsError(output)
    data = load_dataset('v1')
    samples = data.inputs('train'); labels = data.labels('train')
    cache = {}; identities = {}

    def replay(directory):
        directory = directory.resolve()
        if directory in cache:
            return cache[directory]
        config = json.loads((directory/'run_manifest.json').read_text())
        seal = json.loads((directory/'blind_test_manifest.json').read_text())
        if seal['dataset_id'] != DATASET_ID:
            raise ValueError('Training replay cohort differs')
        for name,digest in seal['artifacts'].items():
            if sha256(directory/name) != digest:
                raise ValueError('Sealed artifact changed: '+name)
        selection = json.loads((directory/'selection.json').read_text())
        unavailable = []
        if 'family' not in config:
            receipt = json.loads((core_training/'replay.json').read_text())
            if receipt['run_id'] != config['run_id'] or receipt['dataset_id'] != DATASET_ID:
                raise ValueError('Core training receipt identity differs')
            if receipt['blind_seal_sha256'] != sha256(directory/'blind_test_manifest.json'):
                raise ValueError('Core blind seal identity differs')
            for name,digest in json.loads((core_training/'manifest.json').read_text())['artifacts'].items():
                if sha256(core_training/name) != digest:
                    raise ValueError('Core training replay changed: '+name)
            values = read_values(core_training/'train_predictions.csv',samples)
            unavailable = receipt.get('unavailable_training_methods',[])
        else:
            values = read_values(directory/'train_predictions.csv',samples)
            if config['reference'] is not None:
                ref_directory = Path(config['reference']['directory'])
                ref_values = replay(ref_directory)
                values = training_combinations(config['family'],values,ref_values[config['reference']['method']])
        check_training_coverage(values,selection['reported_methods'],selection['selected_recipe'],unavailable)
        identities[str(directory)] = {'run_id':config['run_id'],
                                     'blind_seal_sha256':sha256(directory/'blind_test_manifest.json'),
                                     'unavailable_training_methods':unavailable}
        cache[directory] = values
        return values

    values = replay(run)
    from analysis.holdout_evaluate import case_metadata, score_predictions
    from analysis.event_features import EventFeatures
    from analysis.relationship_benchmark import rows
    metadata = case_metadata(samples,EventFeatures.load(data.metadata_directory))
    metrics = {name:score_predictions(metadata,labels,rows(samples,name,pred)) for name,pred in values.items()}
    output.mkdir(parents=True)
    save_predictions(output/'train_predictions.csv',samples,values)
    pd.DataFrame([{'method':name,**score['overall']} for name,score in metrics.items()]).to_csv(output/'train_summary.csv',index=False)
    write_json(output/'metrics.json',metrics)
    write_json(output/'replay.json',{'dataset_id':DATASET_ID,'sources':identities,
               'test_labels_read':False,'refitted':False,'methods':len(values),
               'caveat':'Training errors are resubstitution fit, not out-of-sample accuracy.'})
    (output/'source.py').write_text(Path(__file__).read_text())
    write_json(output/'manifest.json',{'artifacts':{p.name:sha256(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({'run':str(run),'methods':len(values),'output':str(output)}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--core-training',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    replay_training(args.run,args.core_training,args.output)
