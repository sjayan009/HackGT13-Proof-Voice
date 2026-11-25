// Browser Supabase client (publishable key only; RLS limits every query to the signed-in owner).
// null when the env is not configured: Guardian then runs in backend "dev mode" with no login.

import { createClient, type SupabaseClient } from "@supabase/supabase-js";

const URL = process.env.NEXT_PUBLIC_SUPABASE_URL;
const KEY = process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY;

export const supabase: SupabaseClient | null = URL && KEY ? createClient(URL, KEY) : null;
