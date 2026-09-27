"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { claimAudioFocus, onAudioFocus } from "./audioFocus";

/**
 * Plays an audio URL and reports its position every animation frame, so a playhead can track the audio
 * continuously (timeupdate alone fires ~4×/s and looks steppy).
 */
export function useAudioPlayhead(url: string | null | undefined, owner: string) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [positionMs, setPositionMs] = useState<number | null>(null);
  const [durationMs, setDurationMs] = useState<number | null>(null);

  useEffect(() => {
    setPlaying(false);
    setPositionMs(null);
    setDurationMs(null);
    if (!url) return;
    const a = new Audio(url);
    a.preload = "metadata";
    audioRef.current = a;
    let raf = 0;
    const tick = () => {
      setPositionMs(a.currentTime * 1000);
      raf = requestAnimationFrame(tick);
    };
    const onPlay = () => {
      setPlaying(true);
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(tick);
    };
    const onPause = () => {
      setPlaying(false);
      cancelAnimationFrame(raf);
      setPositionMs(a.currentTime * 1000);
    };
    const onEnded = () => {
      onPause();
      setPositionMs(null);
    };
    const onMeta = () => Number.isFinite(a.duration) && setDurationMs(a.duration * 1000);
    a.addEventListener("play", onPlay);
    a.addEventListener("pause", onPause);
    a.addEventListener("ended", onEnded);
    a.addEventListener("loadedmetadata", onMeta);
    const off = onAudioFocus(owner, () => a.pause());
    return () => {
      off();
      cancelAnimationFrame(raf);
      a.pause();
      a.removeAttribute("src");
      audioRef.current = null;
    };
  }, [url, owner]);

  const toggle = useCallback(() => {
    const a = audioRef.current;
    if (!a) return;
    if (a.paused) {
      claimAudioFocus(owner);
      a.play().catch(() => setPlaying(false));
    } else {
      a.pause();
    }
  }, [owner]);

  const seek = useCallback(
    (ms: number) => {
      const a = audioRef.current;
      if (!a) return;
      a.currentTime = Math.max(0, ms / 1000);
      setPositionMs(ms);
      if (a.paused) {
        claimAudioFocus(owner);
        a.play().catch(() => setPlaying(false));
      }
    },
    [owner]
  );

  return { playing, positionMs, durationMs, toggle, seek, available: !!url };
}
