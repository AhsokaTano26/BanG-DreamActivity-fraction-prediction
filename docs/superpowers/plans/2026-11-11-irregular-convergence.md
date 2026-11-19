# Neural optimization-budget check

This is an explicit amendment after the registered 80-epoch development study,
not a replacement of its preserved results. The first five completed neural
fits reached all 80 epochs; observed best epochs included 80, 78, 80, and 72.
Their late learning curves continued to improve. Calling this general failure
of irregular-time models would confuse optimization budget with model capacity.

Ruling: One additional fixed-budget experiment in a NEW `v2` directory is within
the user's request to continue neural exploration and AGENTS' validation-only
training-round selection. The original plan prohibited adaptive searching within
v1; v1 remains sealed. This separate convergence check is explicitly disclosed.

- Increase ONLY maximum epochs from 80 to 240 for BOTH networks and ALL seeds
  42/123/2026. Keep architecture, initialization, samples, features, optimizer,
  loss, early-stop patience20/min30 and selection formulas unchanged.
- Train fresh from the same seeds, not from a best checkpoint without its
  optimizer state. Verify every available first-80 trace matches the original.
- Re-run the small deterministic statistical controls identically; require
  identical predictions. This costs extra CPU but simplifies full run auditing.
- Preserve v1 and report both budgets. No further extension or parameter search
  in this task, even if a 240-epoch fit still reaches the ceiling.
- Never read test inputs/labels and never report a new fixed-test improvement.
- Use the same numerical solver; 2/4-step outputs are sensitivity diagnostics.

Complete when six extended fits, trace/control checks, metric/checkpoint audits,
independent review and the report are finished. Production default stays frozen
unless a later distinct final evaluation supports replacement.
