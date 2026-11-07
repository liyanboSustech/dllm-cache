# K4 correction study

Communicate in Chinese for this project. Read README.md and the frozen protocol
in data/k4_correction_20261107/protocol.json before running research.

- User authorized only the first-step K4-versus-random explanation experiment.
- The archived scripts/, vendor/, data/, historical reports/ and references
  listed in references/export_manifest.json are immutable evidence.
- Use `python run_k4.py ...` for portable execution. Do not invoke old g4 launchers.
- Do not tune JS, weights, checkpoint selection, layer sets, thresholds, or samples.
- No second-step selector or independent-sample experiment is authorized.
- Four idle GPUs, two per model; same question and all its conditions on the same
  physical GPU. Never stop another job. No hidden mixed-hardware historical comparison.
- Keep FP16, native sampler, original QK/topk/inf semantics, and efficient SDPA.
- Hardware/path adaptations are recorded in each run's execution.json; do not
  claim historical V100 bitwise equality is required on a different GPU type.
- Models/weights, credentials, local host settings and raw run outputs stay out
  of Git. Run artifacts go under runs/ and must be retained for audit.
- No task accuracy claim before condition/model-hidden semantic review and the
  frozen paired analysis. Strict Final-only numbers are not the primary endpoint.
- If support gates fail or evidence is insufficient, stop this direction. Do
  not retune or add samples to chase significance.
