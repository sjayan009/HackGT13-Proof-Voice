-- ProofVoice Guardian: minimal event storage. No audio, no transcripts, no voiceprints.
-- Applied to Supabase project "ProofVoice Guardian" (rsvsqojfdfhfbrifblnj) as migration `guardian_schema`.
create table public.guardian_lines (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null default auth.uid() references auth.users(id) on delete cascade,
  twilio_number text not null unique,          -- dedicated ProofVoice number (E.164)
  recipient_phone text not null,               -- protected person's real phone (E.164)
  display_name text,
  created_at timestamptz not null default now()
);

create table public.trusted_contacts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null default auth.uid() references auth.users(id) on delete cascade,
  name text not null,
  relationship text,
  phone text not null,
  created_at timestamptz not null default now()
);

create table public.call_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  line_id uuid references public.guardian_lines(id) on delete set null,
  call_sid text not null unique,
  caller_number text,
  consent text not null check (consent in ('granted','declined','simulated')),
  started_at timestamptz not null default now(),
  ended_at timestamptz,
  final_state text,
  peak_evidence real,
  alerted boolean not null default false
);

create table public.risk_events (
  id bigint generated always as identity primary key,
  session_id uuid not null references public.call_sessions(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  t_ms integer,
  kind text not null check (kind in ('state','alert','verification','unavailable')),
  state text,
  evidence real,
  detail jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index on public.guardian_lines (user_id);
create index on public.trusted_contacts (user_id);
create index on public.call_sessions (user_id, started_at desc);
create index on public.call_sessions (line_id);
create index on public.risk_events (session_id);
create index on public.risk_events (user_id);

alter table public.guardian_lines enable row level security;
alter table public.trusted_contacts enable row level security;
alter table public.call_sessions enable row level security;
alter table public.risk_events enable row level security;

-- Owners manage their own lines and contacts.
create policy "own lines" on public.guardian_lines for all to authenticated
  using (user_id = (select auth.uid())) with check (user_id = (select auth.uid()));
create policy "own contacts" on public.trusted_contacts for all to authenticated
  using (user_id = (select auth.uid())) with check (user_id = (select auth.uid()));
-- Sessions/events are written by the backend (service role); owners may read and delete them.
create policy "read own sessions" on public.call_sessions for select to authenticated
  using (user_id = (select auth.uid()));
create policy "delete own sessions" on public.call_sessions for delete to authenticated
  using (user_id = (select auth.uid()));
create policy "read own events" on public.risk_events for select to authenticated
  using (user_id = (select auth.uid()));
-- Owners record the outcome of their own independent verification.
create policy "insert own verification" on public.risk_events for insert to authenticated
  with check (user_id = (select auth.uid()) and kind = 'verification'
              and exists (select 1 from public.call_sessions s where s.id = session_id and s.user_id = (select auth.uid())));
