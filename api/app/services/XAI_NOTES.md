# xAI API notes (verified 2026-09-26 against docs.x.ai)

These are the exact endpoints/events `grok.py` targets. Verified via WebFetch against the
official docs.x.ai pages listed below. If xAI changes these, update here first.

## 1. Grok Voice realtime (primary path)

Source: https://docs.x.ai/developers/model-capabilities/audio/voice-agent
(OpenAI-realtime-compatible transport.)

- WebSocket URL: `wss://api.x.ai/v1/realtime?model=grok-voice-latest`
- Auth: `Authorization: Bearer {XAI_API_KEY}` header on the websocket handshake (server-side only,
  never sent to a client).
- Model: `grok-voice-latest` (alias of `grok-voice-think-fast-2.0`), read from
  `XAI_REALTIME_MODEL` env var with `grok-voice-latest` as fallback.

Client -> server events used:
- `session.update` — sets `session.voice`, `session.instructions`, `session.turn_detection`
  (we set to `null`/manual since we are driving a scripted line, not a live mic conversation),
  and `session.audio.output.format` (`{"type": "audio/pcm", "rate": 24000}`).
  **Verified live 2026-09-26**: the format `type` discriminator is `"audio/pcm"` (not
  `"pcm16"` as some third-party docs summaries imply) — a real connection returned a pydantic
  `invalid_event` error listing the exact literal values accepted: `audio/pcm`, `audio/pcmu`,
  `audio/pcma`, `audio/opus`. Fixed in `grok.py` after hitting this against the live API.
- `conversation.item.create` — creates a `message` item with `role: "user"` and content that
  asks Grok to speak the given line verbatim (a text content part).
- `response.create` — triggers generation of the spoken response.

Server -> client events consumed:
- `conversation.created` — session/conversation ack.
- `response.created` — response started.
- `response.output_audio.delta` — base64 PCM16 audio chunk (the field name in current docs is
  `response.output_audio.delta`; some older examples call this `response.audio.delta` — we
  accept both keys defensively since it is a fast-moving realtime spec).
- `response.done` — response complete, stop reading.
- `error` — surfaced as `GrokUnavailable`.

Audio format: PCM16 little-endian, sample rate configurable (8/16/22.05/24/32/44.1/48 kHz); we
request/assume 24000 Hz (the realtime default) to match `api/CONTRACT.md`'s
`"sample_rate":24000`.

## 2. xAI TTS (fallback 1)

Source: https://docs.x.ai/developers/model-capabilities/audio/text-to-speech

- REST: `POST https://api.x.ai/v1/tts`
- Auth: `Authorization: Bearer {XAI_API_KEY}`
- Body: `{"text": "...", "language": "en", "voice_id": "eve", "output_format": {"codec": "pcm",
  "sample_rate": 24000}}`
- Response: raw audio bytes (Content-Type varies by codec; with `codec: "pcm"` we get raw
  16-bit PCM samples directly, no container to parse).
- A websocket streaming variant exists (`wss://api.x.ai/v1/tts`) with `text.delta` /
  `audio.delta` events, but we use the simple REST call for the fallback since it is a single
  short utterance, not incremental text.

## 3. Chat completions (optional grounded explanation)

Source: https://docs.x.ai/developers/model-capabilities/legacy/chat-completions

- REST: `POST https://api.x.ai/v1/chat/completions`
- Auth: `Authorization: Bearer {XAI_API_KEY}`
- Body: `{"model": "<model>", "messages": [{"role":"system","content":"..."},
  {"role":"user","content":"..."}], "stream": false}`
- Model: read from `XAI_EXPLAIN_MODEL` env var; falls back to `grok-4-fast` style small model
  name if unset — we default to `"grok-4"` since `api/.env`'s `XAI_EXPLAIN_MODEL` is blank in
  this repo. (Docs mention `grok-4.7` as current flagship; either works, `grok-4` is used as the
  safe/cheap default here.)
- Chat Completions is flagged "legacy" in current docs (xAI is steering new integrations to a
  Responses API) but remains supported and is sufficient for a single grounded-explanation call.

## Fallback 3: cached fixture

`api/app/fixtures/grok_fixture.wav` + `.json` sidecar, generated once from a real successful
Grok Voice call (see JSON sidecar for the exact text/model/sample_rate/timestamp used). Used
when there is no `XAI_API_KEY`, DNS/connection failure, timeout, or any non-2xx/error event from
both live paths above. `source="cached_fixture"` in that case — never claim `grok_voice` or
`grok_tts` when the audio actually came from disk.
