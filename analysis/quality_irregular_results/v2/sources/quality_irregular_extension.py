"""One explicitly registered 240-epoch sensitivity check; preserve the 80-epoch run."""
import argparse
from pathlib import Path
from analysis import quality_irregular_benchmark as b


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--reference',type=Path,default=b.ROOT/'analysis/holdout_results/structured_neural_v1')
    args=parser.parse_args()
    b.NEURAL_SETTINGS=b.NEURAL_SETTINGS|{'epochs':240}
    b.SOURCES=b.SOURCES+['analysis/quality_irregular_extension.py',
                        'docs/superpowers/plans/2026-11-11-irregular-convergence.md']
    b.develop(args.run,b.HoldoutDataset(),args.reference)
