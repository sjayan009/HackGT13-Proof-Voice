"use client";

import { useEffect, useState } from "react";
import { getHealth } from "./api";
import type { HealthResponse } from "./types";

export type HealthState =
  | { kind: "checking" }
  | { kind: "ok"; data: HealthResponse }
  | { kind: "down"; message: string };

const POLL_OK_MS = 30_000;
const POLL_DOWN_MS = 5_000;

// One poller shared by every subscriber (header pill, red team hints, …).
let current: HealthState = { kind: "checking" };
const listeners = new Set<(s: HealthState) => void>();
let timer: ReturnType<typeof setTimeout> | undefined;
let inFlight = false;

async function check() {
  if (inFlight) return;
  inFlight = true;
  clearTimeout(timer);
  let next: HealthState;
  try {
    next = { kind: "ok", data: await getHealth() };
  } catch (err) {
    next = { kind: "down", message: err instanceof Error ? err.message : "unreachable" };
  }
  inFlight = false;
  current = next;
  listeners.forEach((l) => l(next));
  if (listeners.size > 0) timer = setTimeout(check, next.kind === "ok" ? POLL_OK_MS : POLL_DOWN_MS);
}

function onVisible() {
  if (document.visibilityState === "visible") check();
}

/** Polls GET /health; backs off when healthy, retries quickly when the API is unreachable. */
export function useHealth(): HealthState {
  const [state, setState] = useState<HealthState>(current);

  useEffect(() => {
    listeners.add(setState);
    if (listeners.size === 1) {
      document.addEventListener("visibilitychange", onVisible);
      check();
    } else {
      setState(current);
    }
    return () => {
      listeners.delete(setState);
      if (listeners.size === 0) {
        clearTimeout(timer);
        document.removeEventListener("visibilitychange", onVisible);
      }
    };
  }, []);

  return state;
}
