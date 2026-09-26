# ProofVoice — Adversarial QA + Sponsor-Requirement Audit (Workstream H)

Read-only review. 84/84 tests pass (`cd api && ../.venv/Scripts/python -m pytest app/tests ../ml/tests -q --timeout 120`,
49.3 s). Docker build was running during this audit (`outputs/docker_build.log`, mid `pip install` at time of review) —
not evaluated live; static review of `Dockerfile` only.

---

## 1. Requirement checklist

| Requirement (source) | Status | Evidence |
|---|---|---|
| Common audio input (WAV/MP3/M4A/OGG/MP4/ffmpeg-decodable) | **MET** | `api/app/audio/ingest.py:57-72` `load_audio` (soundfile → ffmpeg fallback); tested with 48 kHz stereo WAV and MP3 (`api/app/tests/test_api.py:21-46`) |
| Multiple forensic techniques | **MET** | 7 distinct families: primary SSL detector, metadata, spectral, prosody, compression, splice, quality (`api/app/forensics/*.py`, `README.md:81-95`) |
| Synthetic probability 0.0–1.0 | **MET** | `orchestration/pipeline.py:189-190` `cm_score = round(p,6)`; `ml/validate_hearsay_tsv.py:78-81` enforces `[0,1]` |
| Explainability | **MET** | Every report carries `evidence`, `summary`, `flags`, `used_in_score` per technique (`api/CONTRACT.md:38-46`) |
| Source code | **MET** | Full repo present, no obfuscation |
| Docker | **PARTIAL** | `Dockerfile` builds an offline image (`HF_HUB_OFFLINE=1`, local backbone config load in `ssl_model.py:98-111`, no network needed at inference), but `COPY models/selected/ models/selected/` requires `model.pt` (~330 MB) which is **gitignored** (`.gitignore:11`, confirmed absent from git). A judge doing a fresh `git clone` + `docker build` will fail at that COPY step unless the weights are separately published (Git LFS / release / HF) — **not yet done** (`HANDOFF.md` item 3, still open). See Bug D-1. |
| Prediction TSV | **MET** | `ml/generate_hearsay_tsv.py` writes `outputs/team_predictions.tsv`, 1671 rows, validated in-script (`generate_hearsay_tsv.py:119-125`) |
| Required tests: audio decode | **MET** | `test_api.py:21,35` |
| 16 kHz normalization | **MET** | `test_api.py:21-29` (48 kHz→16 kHz), `ingest.py:34-39` shared `soxr_hq` resampler with training (`ml/audio_cache.py`) |
| Score range | **MET** | `test_api.py:71` `test_score_range_and_reload_determinism` |
| Official minDCF parity | **MET** | `ml/tests/test_mindcf_parity.py` — fixture regression + subprocess parity vs organizer `evaluation.py` + orientation guard + cost-model guard (4 tests) |
| Model reload | **MET** | same test as score range (reload determinism) |
| Temporal aggregation | **MET** | `test_api.py:86` `test_windows_cover_clip`, `test_api.py:94` `test_regions_and_ttc` |
| TSV validation | **MET** | `ml/tests/` doesn't have a direct validator unit test, but `ml/validate_hearsay_tsv.py` is exercised end-to-end by `generate_hearsay_tsv.py` on every TSV run; no standalone pytest for the validator itself — **PARTIAL** (no dedicated `test_validate_hearsay_tsv.py` asserting duplicate/missing/OOB/placeholder detection in isolation) |
| WebSocket | **MET** | `test_api.py:158` `test_websocket_stream` |
| File-upload integration | **MET** | `test_api.py:146` `test_file_upload_e2e`, `:153` rejects garbage |
| Grok fallback | **MET** | `test_grok.py` 8 tests: no-key fallback, connection-failure fallback, chunk ordering, key-redaction (×2) |
| **Missing from required-tests list**: silent-audio at pipeline/API level, stereo at API/pipeline level (only decode-level), corrupt/very-short at WS level | **PARTIAL** | see §2 |
| TSV generator behaviours (frozen model, preserve filenames, `[0,1]`, no dup/missing rows, never overwrite template) | **MET** | `ml/generate_hearsay_tsv.py:79-99` (refuses to overwrite template; `assert len(scores)==len(names)==len(set(names))`) |
| TSV validator (fails loudly) | **MET** | `ml/validate_hearsay_tsv.py`; exits 1 with itemized errors; see loophole in §2 (B-1) |
| Docker offline (no xAI/internet/browser for core HEARSAY path) | **MET** (mechanism) / see D-1 for build reproducibility | `Dockerfile` CMD only invokes `ml/generate_hearsay_tsv.py`; xAI import is lazy inside `/redteam` routes only, never imported by the TSV path |
| `techniques_run` / `why_run` / `evidence` JSON shape | **MET** | `orchestration/pipeline.py:186-207` returns exactly this shape plus `branch_log` |
| No fake metrics | **MET** | `RESULTS.md` header states auto-generated from result files; `EvalDrawer.tsx` renders "not computed yet" on 404 (`EvalDrawer.tsx:200-202`); `/eval/summary` 404s if `outputs/eval_summary.json` absent (`main.py:210-215`) |
| Sponsor — Oracle (minDCF, ablation, robustness, temporal, time-to-confidence) | **MET** | `RESULTS.md` has model-selection table, per-generator minDCF, fusion ablation; `ml/robustness.py`, `ml/ablate.py`, `ml/aggregation.py` exist. **Caveat**: `outputs/results/` on disk only contains `baseline_aasist_zeroshot.json` and `fusion.json` — no `robustness.json` or `ablation.json` artifact was found on disk (see Honesty §3, H-1) even though RESULTS.md/README reference robustness numbers |
| Sponsor — Meta (human-trust narrative, not a companion/social app) | **MET** | `README.md:127-138`, `CONTEXT.md` — framing is consistent, no companion features in `web/` |
| Sponsor — SpaceXAI (Grok structurally meaningful, key server-side, Grok never sets score) | **MET** | `api/app/services/grok.py` key never leaves server (`test_grok.py:98,111,131` explicitly test key-leak prevention); `main.py:160-167` Grok output only feeds `analyze()`, doesn't set score directly |
| Final demo steps (human→Grok→timeline→robustness→TSV/Docker) | **PARTIAL** | See Demo Punch List §4 — mechanically all pieces exist, but no fully rehearsed timing/script confirmation is recorded in PROGRESS.md beyond "verified in the browser end-to-end" for Grok (PROGRESS.md:107-108) |

---

## 2. Bugs

### B-1 (Medium) — Live Trust silently freezes after ~120 s of continuous streaming
`api/app/services/stream.py:17,40,54`. `MAX_BUFFER_S = 120.0`; `push()` truncates the ring buffer to the last
120 s (`self.buf = np.concatenate([...])[-int(MAX_BUFFER_S*SR):]`). `step()` sets `self.next_at = n + self.hop`
where `n = len(self.buf)`. Once the buffer saturates at the 120 s cap, `len(self.buf)` stops growing, but each
`step()` still advances `next_at` past that cap (`next_at = cap + hop`). Since `len(self.buf)` can never exceed
the cap again, `ready()` (`len(self.buf) >= self.next_at`) becomes permanently false — `analysis.window` events
stop firing with no error sent to the client. **Repro**: open Live Trust, speak continuously for >2 minutes; the
rolling probability chart will stop updating with no error message (the WS stays open, `onerror`/`onclose` never
fire). Low risk for a 90–120 s demo, but a judge who lets the mic run during Q&A, or a long red-team stream, will
see this. **Fix**: track a monotonically increasing sample counter separate from the truncated buffer length (e.g.
`self.total_samples_pushed`), and compute `next_at`/window indices relative to that counter, converting to
buffer-relative offsets only when reading `self.buf`.

### B-2 (Low) — `speech_ratio` denominator bug for the same long-session case
`api/app/services/stream.py:59-60`: `speech_ratio = min(1.0, self.speech_samples / max(1, n))` where
`self.speech_samples` accumulates for the whole session but `n = len(self.buf)` is capped at 120 s worth of
samples. Once the session exceeds 120 s, `speech_samples` can exceed `n`, so `speech_ratio` saturates at 1.0
regardless of whether the most recent audio is silent — `analysis_confidence` and `status_for()` would then treat
a currently-silent stream as fully-evidenced speech. Same root cause and fix as B-1 (use a rolling/local speech
estimate, not a lifetime counter divided by a capped buffer length).

### B-3 (Low) — Duplicate-basename collision in the TSV generator is silent
`ml/generate_hearsay_tsv.py:47-51` `find_wavs()`: `if p.suffix.lower() in {...} and p.name not in found: found[p.name] = p`.
If the held-out `--input` directory (or an extracted zip) ever contains two files with the same basename in
different subdirectories, the first one encountered (by `sorted(inp.rglob("*"))` path order) silently wins with no
warning — the other file's audio is never scored, but its filename still gets a `cm-score` value copied from the
wrong file. Not observed in the current flat 1,671-file held-out set, but there is no guard if organizers ship a
nested structure or a re-run adds a second archive on top. **Fix**: raise/log on a basename collision instead of
silently keeping the first match.

### B-4 (Low) — TSV validator disables duplicate-filename detection above 5,000 rows
`ml/validate_hearsay_tsv.py:57`: `dups = {...} if len(pnames) < 5000 else set()`. This is a deliberate perf guard,
but it means the validator's "no duplicates" guarantee silently stops applying if a future test set (or a
concatenated multi-file run) exceeds 5,000 rows — the docstring/requirement ("no duplicates") no longer holds for
large inputs without any error or warning. Not a risk for the current 1,671-row HEARSAY set, but worth a `O(n log n)`
dedup (sort + adjacent-compare) instead of the `O(n^2)`-avoidance cutoff so the check never has to be skipped.

### B-5 (Low) — `RedTeam.tsx` never disposes the PCM player between runs
`web/components/RedTeam.tsx:74-79`: `playerRef.current` is created once per `runStream()` call but is only ever
set, never explicitly stopped/nulled on `redteam.done`, `onclose`, or when the user clicks "Generate" a second
time. Running the red-team stream repeatedly during a demo could leave prior `AudioContext`/player instances
alive (audio overlap or a slow memory/handle leak). Low severity for a single demo run, worth a `player.stop()` /
`playerRef.current = null` in `reset()` and `ws.onclose`.

### B-6 (Informational — orientation ambiguity, not a code bug but a submission risk)
`PROGRESS.md:5-14`, `HANDOFF.md:6-9`: the organizer's own `HackGTMinDCF.zip` scorer treats a **higher** score as
more bona fide, while the HEARSAY instructions describe `cm-score` as **p(synthetic)** (higher = more synthetic).
The team correctly produces both orientations (`team_predictions.tsv` and `team_predictions_bonafide_high.tsv`)
and flags this as an open question for the human, but **if the wrong file is submitted, minDCF would be
catastrophic** (near 1.0, i.e., worse than random) rather than merely degraded. This is the single highest-leverage
unresolved item before any submission — confirm with organizers or use the one-time NSA review before the final
push. (Already surfaced by the team; repeating here because it's the top demo/submission risk.)

### B-7 (Low) — no dedicated unit test for `ml/validate_hearsay_tsv.py`
Confirmed: `ml/tests/` contains only `test_mindcf_parity.py`; there is no `test_validate_hearsay_tsv.py` asserting
each failure mode (dup names, OOB scores, wrong row count, all-identical placeholder, BOM/CRLF) in isolation —
today those code paths are only exercised indirectly whenever `generate_hearsay_tsv.py` is run against a real
1,671-row dataset. A future edit to the validator could regress one of these checks without any test catching it.

### Not a bug, but worth flagging: secret handling looks correct
`api/.env` variable **names** were checked structurally (contents not read/printed per instructions): `config.py`
loads `api/.env` via `dotenv` and never logs values; `.gitignore:2-6` and `.dockerignore:14-15` both exclude
`.env`/`.env.*`; `Dockerfile` does `COPY api/ api/` but `.dockerignore` applies to the whole build context so
`api/.env` is excluded from the image regardless of subdirectory. `test_grok.py:98,111,131` explicitly test that
error paths and repr() never leak the API key or raw PCM. No leakage risk found.

---

## 3. Honesty audit

- **H-1 (worth resolving before demo)** — `RESULTS.md`, `README.md`, and `HANDOFF.md` all reference robustness
  numbers ("Robustness under laundering (MP3/Opus/AAC/μ-law/noise/gain/clipping)" — `README.md:132`) and an
  ablation table (present, `RESULTS.md:45-56`, backed by `outputs/results/fusion.json`), but **no
  `outputs/results/robustness.json` (or equivalent) was found on disk** at audit time — only
  `baseline_aasist_zeroshot.json` and `fusion.json` exist in `outputs/results/`. `outputs/robustness_v1.log` exists
  (a run log) but its structured result file wasn't located. If `outputs/eval_summary.json` (241 lines, present)
  embeds the robustness table already, this is fine — **verify `ml/build_eval_summary.py`'s robustness section is
  populated from a real `ml/robustness.py` output before the demo**, not from the log file alone, so the
  Evaluation drawer doesn't show stale/partial data during the live Oracle segment.
- **H-2** — No other unbacked claims found. `README.md`'s "Honest limitations" section (lines 119-125) is
  consistent with the measured numbers in `RESULTS.md` (e.g., explicitly calling out the 44-sample LJ validation
  set as noisy, and citing unseen-generator numbers as the better open-world estimate). This is good practice and
  matches CONTEXT.md's truthfulness rules (no "universal deepfake detection", no "no watermark = human", etc.).
- **H-3** — `PROGRESS.md`'s "held-out sanity" paragraph (line 92-95) is a good example of honest reporting: it
  states the model flags 45% of held-out audio vs. an organizer-stated ~30% base rate, and explicitly says this
  means "the val-optimal threshold transfers imperfectly" rather than hiding the discrepancy. No issues.
- **H-4** — UI: confirmed no component fabricates a number when data is absent — `Gauge.tsx:19,60` renders `—` for
  `null`; `FileMetadataPanel.tsx:3-9` renders `—` for null/empty; `EvalDrawer.tsx:200-202` shows "Evaluation not
  computed yet." on 404. Consistent with `api/CONTRACT.md`'s "No field may be filled with a made-up value" rule.

---

## 4. Demo punch list (90–120 s: human mic → Grok red team → evidence timeline → robustness/ablation → TSV + Docker)

1. **Before anything**: confirm which score orientation the organizers actually want (B-6) — this doesn't affect
   the live demo narrative but must be locked down before any TSV is submitted.
2. **Model file for Docker**: publish `models/selected/model.pt` (Git LFS / GitHub release / HF) so a judge's
   `git clone && docker build` actually succeeds — currently it will fail on `COPY models/selected/` (D-1/checklist
   row "Docker"). Do this before the submission deadline, not just before the demo.
3. **Pre-flight the live app** 10–15 min before demo time: `GET /health` shows `detector.device` and a loaded
   model (not `degraded`); confirm `XAI_API_KEY` is set so Grok Voice realtime (not the cached fallback) fires —
   the cached-fixture fallback is honestly labeled but is a materially weaker demo moment.
4. **Human sample (Live Trust)**: speak for well under 120 s continuously (B-1) — if you need to demonstrate a
   longer live session for Q&A, restart the stream rather than letting one session run past 2 minutes.
5. **Grok Red Team**: run once before the real demo to warm caches (HF model already loaded via `/health` warm-up
   in `main.py:41-42`, but the Grok connection itself has its own latency) — verify `source` returned is
   `"grok_voice"` (realtime), not `"grok_tts"`/`"cached_fixture"`, on the venue Wi-Fi.
6. **Evidence timeline**: open the Forensic Lab report for a file with a clear synthetic/human split so
   `techniques_run`/`why_run`/`evidence`/`branch_log` all show non-trivial content (e.g., a file that triggers the
   `weak` branch — spectral+prosody — rather than one so confident that only `primary_detector` and `quality` run).
7. **Robustness/ablation (Oracle segment)**: open the Evaluation drawer once beforehand to confirm it renders
   `outputs/eval_summary.json` fully (no missing robustness section — see H-1) instead of discovering a 404 or an
   empty section live.
8. **TSV + Docker**: have `outputs/team_predictions.tsv` and its `.json` sidecar (model sha1, calibration) already
   generated and `ml/validate_hearsay_tsv.py` passing, shown on-screen rather than run live (regeneration takes
   real GPU/CPU time); mention the Docker command but don't attempt a live `docker build` during the 90–120 s
   window — reserve that for the Q&A/judging-table walkthrough, using a pre-built image if possible.
9. **Closing line** ready verbatim per PHASES.md: *"Generation is becoming real-time. Verification has to become
   real-time too."*
10. **Fallback plan**: if venue Wi-Fi drops mid-demo, the Grok fallback path (`cached_fixture`) is tested
    (`test_grok.py:44-84`) and honestly labeled in the UI (`RedTeam.tsx:16-20` `SOURCE_LABEL`) — rehearse narrating
    that fallback once so it doesn't look like an unplanned failure if it triggers live.

---

## Top 5 issues (summary)

1. **Docker reproducibility for judges (HIGH, checklist)** — `model.pt` (~330 MB) is gitignored and not yet
   published anywhere; a fresh clone + `docker build` will fail at `COPY models/selected/`. Must be fixed before
   submission, independent of the demo.
2. **Score orientation ambiguity (HIGH, submission risk, already known to the team)** — organizer scorer expects
   higher=bona fide, HEARSAY docs say higher=synthetic; submitting the wrong file yields catastrophic (not just
   degraded) minDCF. Confirm with organizers or use the one-time review before final submission.
3. **Live Trust silently stalls after ~120 s (MEDIUM, real bug)** — ring-buffer truncation in
   `api/app/services/stream.py` desyncs `next_at` from the capped buffer length; no error is surfaced to the
   client, the probability chart just stops updating. Low risk for a 90–120 s scripted demo, real risk for anything
   longer (extended Q&A demo, long red-team stream).
4. **Possible missing robustness result artifact (MEDIUM, honesty/demo risk)** — `outputs/results/` only has
   `baseline_aasist_zeroshot.json` and `fusion.json` on disk; verify `outputs/eval_summary.json`'s robustness
   section is genuinely populated (not stale) before showing the Oracle segment live.
5. **TSV validator has two silent loopholes at scale (LOW)** — duplicate-basename collision in
   `generate_hearsay_tsv.py`'s file finder, and the validator's own duplicate-check disables itself above 5,000
   rows. Neither affects the current 1,671-file held-out set, but both remove a safety net without any warning if
   inputs change.
