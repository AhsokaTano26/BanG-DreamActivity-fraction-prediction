"""Pure input hardening around an injected, unchanged frozen sharing backend.

This adapter adds physical-case identity checks; all estimating equations stay
in the archived backend. It does not read files, labels or global state.
"""
import numpy as np
import pandas as pd

KEYS=('event','rank','cutoff')

def validate_physical_cases(frame):
    if not isinstance(frame,pd.DataFrame) or set(KEYS)-set(frame):
        raise ValueError('Physical event/rank/cutoff fields are required')
    normalized={}
    for key in KEYS:
        values=pd.to_numeric(frame[key],errors='raise').to_numpy(float)
        if not np.isfinite(values).all() or np.any(values!=np.floor(values)):
            raise ValueError('Finite integral physical case identifiers required')
        normalized[key]=values
    if pd.DataFrame(normalized).duplicated().any():
        raise ValueError('Repeated physical event/rank/cutoff despite case aliases')

class GuardedSharingBackend:
    def __init__(self,backend):self._backend=backend
    def __getattr__(self,name):return getattr(self._backend,name)
    def _validate(self,history,current):
        validate_physical_cases(history);validate_physical_cases(current)
    def issue_experts(self,history,blind_current):
        self._validate(history,blind_current)
        return self._backend.issue_experts(history,blind_current)
    def mix_experts(self,history,blind_current,configuration,experts):
        self._validate(history,blind_current)
        return self._backend.mix_experts(history,blind_current,configuration,experts)
    def predict(self,history,blind_current,configuration):
        self._validate(history,blind_current)
        return self._backend.predict(history,blind_current,configuration)
