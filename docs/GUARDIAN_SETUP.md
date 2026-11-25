# ProofVoice Guardian — first real, consented phone call

This takes one real call from a phone, through Twilio, into ProofVoice, and shows the result on the Guardian
dashboard. It needs about 15 minutes. Everything runs on your machine except Twilio and Supabase.

```
caller ──► your Twilio number ──► /twilio/voice  (disclosure: "press 1 to allow the check")
                                   /twilio/consent ──► <Start><Stream> caller audio ──► wss://…/twilio/media ──► Aegis (GPU)
                                                   └─► <Dial> your real phone (the conversation itself never waits on ProofVoice)
Guardian dashboard ◄── WS /guardian/live ◄── risk engine     SMS alert ──► your phone (optional)
```

## 1. Secrets in `api/.env` (never commit this file)

The public Supabase values are already filled in. Add these:

| variable | where to find it |
|---|---|
| `SUPABASE_SECRET_KEY` | Supabase dashboard → project **ProofVoice Guardian** → Project Settings → API Keys → *secret* key (`sb_secret_…`). The backend uses it to look up your Guardian line and write call events. |
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` | Twilio Console → Account info |
| `TWILIO_PHONE_NUMBER` | your Twilio number in E.164, e.g. `+14045550123` |
| `PUBLIC_BASE_URL` | the `https://…ngrok…` URL from step 2 (no trailing slash). Webhook signatures are checked against this exact URL. |

## 2. Start the backend and ngrok

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir api --host 127.0.0.1 --port 8000
```

```powershell
ngrok http 8000
```

No ngrok account? A Cloudflare quick tunnel needs no account or token (`winget install Cloudflare.cloudflared`):

```powershell
& "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://127.0.0.1:8000
```

Either tunnel gets a new random URL every time it restarts; update `PUBLIC_BASE_URL` and the Twilio webhook when it does.

Put the ngrok `https://` URL into `PUBLIC_BASE_URL` and restart the backend. `GET /health` should show
`"guardian": {"dev_mode": false, "store": "supabase", "twilio": true, "public_base_url": true}`.

## 3. Point your Twilio number at ProofVoice

Twilio Console → Phone Numbers → your number → **Voice configuration** → *A call comes in*: **Webhook**,
`https://<your-ngrok>/twilio/voice`, **HTTP POST**. Save.

## 4. Set up Guardian in the web app

```powershell
npm --prefix web run dev
```

Open http://localhost:3000/guardian, sign up with email + password (Supabase Auth; confirm the email if
Supabase asks — in Supabase → Authentication → URL Configuration set the Site URL to `http://localhost:3000`),
then:

1. **Guardian line**: Twilio number = your Twilio number; recipient phone = the phone that should ring.
2. **Trusted contacts**: the people you would call back to check (name + their known number).
3. Click **Enable desktop alerts**.

## 5. Make the call

**Only one phone?** Call your Twilio number from the phone you set as the recipient. ProofVoice detects that the protected person is calling their own line, says "Self test mode", dials nobody, and analyses what it hears until you hang up.

With two phones: from a *different* phone, call your Twilio number. You hear the disclosure; press **1**. Your recipient phone
rings; talk normally. The dashboard shows the live call card within ~3 s of speech. To see a warning, play
synthetic speech into the calling phone (for example the Grok samples in `data/grok_samples/`), only with
everyone on the call aware that it is a test.

Pressing **2** (or nothing) connects the call without analysis; the dashboard shows "analysis unavailable".

## Known limits

- **Twilio trial accounts** prepend a trial message and can only dial **verified** numbers: verify the recipient
  phone under Verified Caller IDs.
- **SMS alerts** from US long codes need A2P 10DLC registration; without it Twilio may reject them (the dashboard
  alert still works). Set `GUARDIAN_SMS_ALERTS=0` to turn SMS off.
- The model's measured weaknesses apply: noisy lines and some codecs raise false alarms (see
  `docs/research/telephony_eval.md` and `RESULTS.md`). The call-level policy abstains rather than guessing
  when the audio is too poor.
- Call monitoring and consent laws differ by jurisdiction; get legal advice before any real deployment.

## Without a phone

`POST /guardian/simulate {"scenario": "synthetic"|"genuine"|"mixed"}` (or the buttons on the Guardian page)
streams a recording through the identical Twilio media path. To drive it over a real WebSocket:

```powershell
cd api; ..\.venv\Scripts\python.exe -m app.telephony.sim ws://127.0.0.1:8000/twilio/media --scenario mixed --token <token from POST /guardian/simulate with "transport":"external">
```
