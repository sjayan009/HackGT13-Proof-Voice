# ProofVoice API contract (v1)

Single source of truth shared by backend (`api/`), frontend (`web/`) and ML (`ml/`).
All probabilities are **p_synthetic in [0,1]** (1.0 = synthetic). No field may be filled with a
made-up value: if something was not computed it is `null` / omitted and the UI says so.

Base URL: `http://127.0.0.1:8000` (see `web/.env.local`: `NEXT_PUBLIC_API_BASE_URL`, `NEXT_PUBLIC_WS_BASE_URL`).

## Status vocabulary

`insufficient_evidence` | `likely_human` | `inconclusive` | `likely_synthetic`

## AnalysisReport (returned by `POST /analyze/file`, embedded elsewhere)

```jsonc
{
  "id": "uuid",
  "file": {                       // from ffprobe / soundfile; null fields allowed
    "name": "clip.wav", "container": "wav", "codec": "pcm_s16le",
    "sample_rate": 16000, "channels": 1, "bit_depth": 16,
    "duration_s": 3.41, "bitrate": null, "metadata": {"encoder": "Lavf60"}
  },
  "synthetic_probability": 0.87,  // clip-level, calibrated for display
  "cm_score": 0.87,               // the exact value that would go into the HEARSAY TSV
  "analysis_confidence": 0.74,    // evidence sufficiency 0..1 (duration, detector agreement, SNR)
  "status": "likely_synthetic",
  "decision_threshold": 0.62,     // p_synthetic threshold selected on validation minDCF
  "detector": {"name": "…", "version": "…", "raw_score": 3.1, "latency_ms": 41.0},
  "timeline": [                   // sliding windows (default 2.0 s window, 0.5 s hop)
    {"start_ms": 0, "end_ms": 2000, "synthetic_probability": 0.8, "analysis_confidence": 0.6}
  ],
  "suspicious_regions": [
    {"start_ms": 500, "end_ms": 2500, "peak_probability": 0.93, "reason": "detector window score above threshold"}
  ],
  "time_to_confidence_ms": 1500,  // first time the rolling decision became stable & confident; null if never
  "techniques_run": ["metadata", "primary_detector", "spectral"],
  "why_run": {"spectral": "primary detector confidence weak (p=0.55)"},
  "evidence": {
    "<technique>": {
      "summary": "one plain-English sentence",
      "suspicion": 0.3,           // 0..1 branch-level suspicion or null if not a scoring branch
      "features": {"spectral_flatness_mean": 0.12},
      "flags": ["band-limited at 4 kHz"],
      "used_in_score": false      // true only if the branch feeds the cm_score
    }
  },
  "branch_log": [
    {"technique": "compression", "ran": false, "reason": "lossless PCM input", "runtime_ms": 0.0}
  ],
  "processing_ms": 210.0,
  "disclaimer": "Probabilistic evidence, not proof. …"
}
```

## Endpoints

### `GET /health`
`{"status":"ok","version":"0.1.0","detector":{"name":"…","version":"…","device":"cuda"},"grok_available":true}`

### `POST /analyze/file`
multipart form field `file` (any ffmpeg-decodable audio/video). Returns `AnalysisReport`.
Errors: 400 `{"detail":"…"}` for undecodable / empty / too long (> MAX_UPLOAD_MB).

### `WS /analyze/stream`
1. Client → text JSON: `{"type":"start","sample_rate":48000,"encoding":"f32le"|"s16le","source":"mic"|"grok"|"file"}`
2. Client → binary frames: mono PCM in the declared encoding (any chunk size, e.g. 20–100 ms).
3. Server → text JSON events:
   - `{"type":"ready"}`
   - `{"type":"analysis.window","t_ms":2500,"start_ms":500,"end_ms":2500,"synthetic_probability":0.81,
      "rolling_probability":0.77,"analysis_confidence":0.55,"status":"inconclusive",
      "time_to_confidence_ms":null,"level_dbfs":-23.1,"speech_ratio":0.8}`
   - `{"type":"analysis.final","report":AnalysisReport}` after client sends `{"type":"stop"}`
   - `{"type":"error","detail":"…"}`

### `POST /redteam/grok`
JSON `{"text": "optional line for Grok to speak", "voice": "optional"}`.
Server generates speech with xAI (key stays server-side), runs the **same** pipeline, returns
`{"source":"grok_voice"|"grok_tts"|"cached_fixture","text":"…","audio_wav_b64":"…","sample_rate":24000,"report":AnalysisReport,"note":"…"}`.
`source` must honestly say which path produced the audio.

### `WS /redteam/stream`
1. Client → `{"type":"start","text":"optional"}`
2. Server → interleaved
   - `{"type":"audio.chunk","pcm16_b64":"…","sample_rate":24000,"seq":0}` (the actual generated audio, for playback)
   - `analysis.window` events (same schema as `/analyze/stream`) computed on those same samples
   - `{"type":"redteam.done","source":"grok_voice"|"grok_tts"|"cached_fixture","report":AnalysisReport}`

### `GET /eval/summary`
Returns `outputs/eval_summary.json` verbatim (minDCF/EER/AUC tables, ablation, robustness, score histograms,
model selection). 404 if not yet generated — UI must show "not computed yet", never placeholder numbers.

### `POST /explain`
`{"report": AnalysisReport}` → `{"text":"…","model":"grok-…","grounded":true}`; the text may only restate
fields present in the report. 503 if no xAI key.
