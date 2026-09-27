# ProofVoice — web

Next.js (App Router, TypeScript) frontend for ProofVoice. Implements Forensic
Lab, Live Trust, Red Team (Grok), and an Evaluation drawer against the API
contract in `../api/CONTRACT.md`.

## Run

```bash
npm install
npm run dev
```

The app runs on http://localhost:3000.

It talks to the backend at the URLs in `.env.local`:

```
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
NEXT_PUBLIC_WS_BASE_URL=ws://127.0.0.1:8000
```

If the real backend (in `../api`) isn't running yet, errors are shown inline
("Could not reach the ProofVoice API…") rather than fake data — the app never
fabricates numbers.

## Optional: local mock backend

For frontend-only development before the real backend is ready, a small mock
server is included. It is **not** imported by the app and is only meant for
manual local testing:

```bash
npm run mock
```

This starts a fake backend on `:8000` implementing every contract endpoint
with randomly generated (clearly fake) data. Set `MOCK_EVAL=404 npm run mock`
to exercise the "Evaluation not computed yet" UI path.

## UX / production notes

- Tabs are an ARIA tablist (arrow keys, Home/End), deep-linkable via `#live` / `#redteam`; panels stay
  mounted so a running session or a finished report survives switching tabs.
- The Evaluation drawer is a modal dialog: Esc closes, focus is trapped and restored, background scroll locked.
- In-flight uploads are cancellable and superseded by newer ones; stale responses are ignored.
- Dark by default, light follows the OS; honors `prefers-reduced-motion`, `prefers-reduced-transparency`,
  `prefers-contrast`, forced colors, and prints cleanly. Works down to 375 px wide.
- `next.config.mjs` sends `nosniff`, `Referrer-Policy`, `X-Frame-Options: DENY` and a `Permissions-Policy`
  that allows the microphone for this origin only.

## Build

```bash
npm run build
```

## Structure

- `app/` — App Router shell (`layout.tsx`, `page.tsx` with the tab switcher, `globals.css`)
- `app/error.tsx`, `app/not-found.tsx`, `app/icon.svg` — error boundary, 404 and favicon
- `components/` — screens (`ForensicLab`, `LiveTrust`, `RedTeam`, `EvalDrawer`) and shared report UI (`ReportView`, `Gauge`, `StatusChip`, `TrustTimelineChart` (with the waveform on the same time axis), `RollingProbabilityChart`, `EvidenceCard`, `BranchLogTable`, `FileMetadataPanel`, `LevelMeter`, `Callout`, `Icon`)
- `lib/types.ts` — TypeScript types mirroring `api/CONTRACT.md` exactly
- `lib/api.ts` — REST client (`/health`, `/analyze/file`, `/redteam/grok`, `/eval/summary`, `/explain`) with timeouts/abort, and WS URL helpers
- `lib/useHealth.ts` — single shared `/health` poller behind the header status pill
- `lib/format.ts` — display-only formatting (never changes values)
- `lib/mic.ts` — microphone capture (AudioWorklet, ScriptProcessorNode fallback)
- `lib/audio.ts` — PCM encode/decode helpers, local waveform decoding, PCM16 playback queue
- `public/pcm-worklet-processor.js` — AudioWorklet processor used by Live Trust
- `scripts/mock-server.mjs` — dev-only mock backend, never imported by the app

## Contract ambiguities encountered

- `EvalSummary` (`GET /eval/summary`) has no fixed schema in the contract
  beyond "verbatim `outputs/eval_summary.json`". The drawer renders it
  generically: histogram-shaped objects (`{bins, bonafide, spoof}`) become
  overlaid bar charts, arrays of flat objects become tables, flat objects
  become key/value tables, and anything else falls back to indented JSON.
- `WS /analyze/stream` doesn't specify a `sample_rate` unit beyond "AudioContext
  sample rate" for mic capture — the client sends the live `AudioContext.sampleRate`
  (browser-dependent, commonly 48000) rather than resampling to 16 kHz client-side,
  since the contract shows the server accepting `sample_rate` explicitly.
  Resampling to the model's expected rate is assumed to happen server-side.
  The mic AudioWorklet emits chunks of 2048 samples; the contract allows "any
  chunk size (e.g. 20-100ms)" so this was chosen as a reasonable default.
  Frame chunking is client discretion.
- `POST /redteam/grok` and `WS /redteam/stream`'s `audio_wav_b64` /
  `pcm16_b64` payloads' exact WAV/PCM framing (e.g. header presence for the
  base64 WAV blob) isn't specified beyond field names; the fallback path
  doesn't currently decode/play `audio_wav_b64` (only the streaming path
  plays audio, since it's explicitly PCM16 chunks) — this could be added once
  the real backend confirms the base64 payload is header-less PCM16 vs. a
  full WAV container.
- `analysis_confidence` in `analysis.window` events is rendered as a soft
  shaded uncertainty band around the rolling-probability line (width shrinks
  as confidence rises) since the contract doesn't prescribe a specific visual
  encoding.
- No field in the contract specifies MIME/type constraints beyond "any
  ffmpeg-decodable audio/video" for `/analyze/file`; the file input accepts a
  broad `audio/*,video/mp4,.wav,.mp3,.m4a,.ogg,.mp4` filter but the server is
  the source of truth for actual support (400 on undecodable input is
  surfaced verbatim).
