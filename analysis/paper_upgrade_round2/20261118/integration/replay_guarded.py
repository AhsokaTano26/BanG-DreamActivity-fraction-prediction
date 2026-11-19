"""Run frozen B equations with physical-case guards and explicit source binding.

Use the original --inputs/--config/--output arguments. New directories only.
"""
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
B=HERE.parent/'workspaces/B'
sys.path.insert(0,str(B))
import replay
from sharing_input_guard import GuardedSharingBackend

def main():
    original=replay._source_fingerprints
    def fingerprints():
        return original()|{str(p.relative_to(HERE.parent)):replay.sha256(p) for p in
                           [HERE/'sharing_input_guard.py',Path(__file__).resolve()]}
    replay._source_fingerprints=fingerprints
    replay.model=GuardedSharingBackend(replay.model)
    replay.main()

if __name__=='__main__':main()
