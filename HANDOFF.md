# HANDOFF.md — state of ProofVoice for whoever picks this up next

Read PROGRESS.md (log + measured numbers), README.md (how to run), api/CONTRACT.md (API), RESULTS.md (auto-generated).

## Decisions needing the human

1. **Score orientation** — organizer scorer = higher score ⇒ bona fide; HEARSAY docs = cm-score is p(synthetic).
   We produce both `outputs/team_predictions.tsv` (p_synth, primary) and `outputs/team_predictions_bonafide_high.tsv`.
   Ask organizers which one they score before submitting.
2. **One-time NSA review** — not used. Draft is `outputs/team_predictions.tsv` (+ `.json` sidecar with model sha1).
   Nothing has been submitted anywhere.
3. **Model weights are not in git** (`models/selected/model.pt`, ~330 MB, gitignored). For a public repo, attach
   them to a GitHub release / Git LFS / HF, or judges can't build the Docker image. `docker build` works locally.
4. `api/.env`: `FORENSIC_WINDOW_MS` changed 2000 → 3000 (timeline windows must be ≥ held-out min clip 3.0 s).
5. Demo video + public repo push + Devpost text: not done (need the human).

## How to resume

```bash
# tests (84+)
cd api && ../.venv/Scripts/python -m pytest app/tests ../ml/tests -q --timeout 120
# app
.venv/Scripts/python -m uvicorn app.main:app --app-dir api --port 8000   # + cd web && npm run dev
# regenerate TSV from models/selected
.venv/Scripts/python ml/generate_hearsay_tsv.py --input data/hearsay/test --template data/hearsay/template/HearsayScoreKey4TeamX.tsv --output outputs/team_predictions.tsv
# selecting a different trained model
.venv/Scripts/python ml/calibrate.py --model-dir models/<run> --select --name "<name>"
# evaluation artefacts -> outputs/eval_summary.json + RESULTS.md
.venv/Scripts/python ml/robustness.py --model-dir models/selected --n 1400
.venv/Scripts/python ml/score_ood.py
.venv/Scripts/python ml/aggregation.py
.venv/Scripts/python ml/ablate.py --primary models/<selected run>
.venv/Scripts/python ml/build_eval_summary.py
```

GPU notes (RTX 4060 8 GB, WDDM): never exceed ~7 GB or it silently spills to shared memory (10× slower). Keep one
GPU job at a time. Training uses XLS-R truncated to 12 layers, bs 8 × accum 2, ~40 min / 3000 steps incl. evals.
Do not run heavy CPU jobs during training (augmentation is CPU-bound).

## Status at hand-off

- **Selected model: xlsr12_v3** — val minDCF 0.066, EER 1.5 %, AUC 0.999 (official scorer, group-disjoint val).
- `outputs/team_predictions.tsv` (+ `_bonafide_high.tsv`, `.json` sidecar) generated from v3 and validated.
- 86 tests pass. Web UI + API verified end-to-end in a browser (Forensic Lab, Red Team with live Grok Voice).
- Robustness, Grok-OOD (6/6 flagged), latency, fusion ablation, eval summary: done (RESULTS.md).
- QA audit: `outputs/qa_audit.md`; bugs B-1..B-5 fixed (stream stall, TSV dup checks, red-team player).

## Not done (in priority order)
1. Confirm score orientation with organizers; decide on the one-time NSA review (draft ready, nothing sent).
2. Publish `models/selected/model.pt` (LFS / release) so a fresh clone can `docker build`.
3. Record the 90–120 s demo video; push public repo; Devpost write-up.
4. Live Trust with a real microphone was not exercised in a browser (WebSocket path is covered by tests).
   **Try it before demoing**: on-device mic audio is out of domain and may read "inconclusive"/synthetic;
   the MP3 robustness result (bona fide FRR 2.6 %→14.7 %) suggests codec/channel shift raises false alarms.
5. Optional experiments with ready scripts: `ml/train_aasist.py` (family 2), unseen-generator retrain
   (`ml/train.py --exclude-generators elevenlabs xtts_v2 grad_tts` then `ml/unseen.py`), codec augmentation on
   bona fide to fix the MP3 false-alarm shift.
