"use client";

import { useEffect, useRef, useState } from "react";
import { claimAudioFocus, onAudioFocus } from "@/lib/audioFocus";
import { groupSamples, loadSampleManifest, type Sample, type SampleManifest } from "@/lib/samples";
import Icon from "./Icon";

const OWNER = "sample-preview";

function groupTone(group: string, samples: Sample[]) {
  if (/hard/i.test(group)) return "var(--amber)";
  return samples.every((s) => s.truth === "bonafide") ? "var(--green)" : "var(--red)";
}

function fmtDur(s: number | null) {
  if (s == null) return "";
  return `0:${String(Math.round(s)).padStart(2, "0")}`;
}

/** Shared preview player: one <audio> element for the whole library. */
function usePreview() {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const idRef = useRef<string | null>(null);
  const [playingId, setPlayingId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  // UI state follows the element's own events, so it can never drift from what is audible.
  useEffect(() => {
    const a = new Audio();
    a.preload = "none";
    audioRef.current = a;
    const onTime = () => setProgress(a.duration ? a.currentTime / a.duration : 0);
    const onPlay = () => setPlayingId(idRef.current);
    const onPause = () => setPlayingId(null);
    const onEnd = () => {
      setPlayingId(null);
      setProgress(0);
    };
    a.addEventListener("timeupdate", onTime);
    a.addEventListener("play", onPlay);
    a.addEventListener("pause", onPause);
    a.addEventListener("ended", onEnd);
    const off = onAudioFocus(OWNER, () => a.pause());
    return () => {
      off();
      a.pause();
      a.removeAttribute("src");
    };
  }, []);

  const toggle = (s: Sample) => {
    const a = audioRef.current;
    if (!a) return;
    if (playingId === s.id) {
      a.pause();
      return;
    }
    claimAudioFocus(OWNER);
    if (idRef.current !== s.id) {
      idRef.current = s.id;
      a.src = s.file;
      setProgress(0);
    }
    a.play().catch(() => setPlayingId(null));
  };

  const stop = () => audioRef.current?.pause();

  return { playingId, progress, toggle, stop };
}

function PlayButton({
  sample,
  playing,
  progress,
  onToggle,
  size = 40,
}: {
  sample: Sample;
  playing: boolean;
  progress: number;
  onToggle: () => void;
  size?: number;
}) {
  const r = size / 2 - 2;
  const c = 2 * Math.PI * r;
  return (
    <button
      type="button"
      className="play-btn"
      style={{ width: size, height: size }}
      onClick={(e) => {
        e.stopPropagation();
        onToggle();
      }}
      aria-label={`${playing ? "Pause" : "Play"} ${sample.title}`}
      aria-pressed={playing}
    >
      <svg className="play-ring" width={size} height={size} aria-hidden>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
          strokeDasharray={`${playing ? progress * c : 0} ${c}`}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
      </svg>
      <Icon name={playing ? "pause" : "play"} size={size * 0.4} />
    </button>
  );
}

export default function SampleLibrary({
  onAnalyze,
  activeId,
  busy,
  variant = "grid",
}: {
  onAnalyze: (s: Sample) => void;
  activeId: string | null;
  busy: boolean;
  variant?: "grid" | "strip";
}) {
  const [manifest, setManifest] = useState<SampleManifest | null | undefined>(undefined);
  const { playingId, progress, toggle, stop } = usePreview();

  const stripRef = useRef<HTMLDivElement>(null);
  const [stripEnd, setStripEnd] = useState(false);
  const updateStripEnd = () => {
    const el = stripRef.current;
    if (el) setStripEnd(el.scrollLeft + el.clientWidth >= el.scrollWidth - 4);
  };

  useEffect(() => {
    loadSampleManifest().then(setManifest);
  }, []);

  // Mouse wheels scroll vertically; translate that into horizontal movement over the strip.
  useEffect(() => {
    const el = stripRef.current;
    if (!el) return;
    updateStripEnd();
    const onWheel = (e: WheelEvent) => {
      if (Math.abs(e.deltaY) <= Math.abs(e.deltaX) || el.scrollWidth <= el.clientWidth) return;
      const atStart = el.scrollLeft <= 0 && e.deltaY < 0;
      const atEnd = el.scrollLeft + el.clientWidth >= el.scrollWidth - 1 && e.deltaY > 0;
      if (atStart || atEnd) return; // let the page scroll once the strip is exhausted
      e.preventDefault();
      el.scrollLeft += e.deltaY;
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [variant, manifest]);

  if (manifest === undefined) {
    return variant === "grid" ? <div className="skeleton" style={{ height: 180 }} /> : null;
  }
  if (!manifest || manifest.samples.length === 0) return null;

  const analyze = (s: Sample) => {
    stop();
    onAnalyze(s);
  };

  if (variant === "strip") {
    return (
      <div
        className={`sample-strip ${stripEnd ? "at-end" : ""}`}
        role="list"
        aria-label="Demo samples"
        ref={stripRef}
        onScroll={updateStripEnd}
      >
        {manifest.samples.map((s) => (
          <div
            role="listitem"
            key={s.id}
            className={`sample-chip ${activeId === s.id ? "active" : ""}`}
            data-truth={s.truth}
          >
            <PlayButton
              sample={s}
              size={30}
              playing={playingId === s.id}
              progress={progress}
              onToggle={() => toggle(s)}
            />
            <button
              type="button"
              className="sample-chip-label"
              onClick={() => analyze(s)}
              disabled={busy}
              title={`Analyze ${s.title}`}
            >
              {s.title}
            </button>
          </div>
        ))}
      </div>
    );
  }

  return (
    <section className="sample-library" aria-labelledby="samples-title">
      <div className="panel-head" style={{ marginBottom: 12 }}>
        <div>
          <h3 id="samples-title" style={{ fontSize: "1rem" }}>
            No file handy? Try a sample
          </h3>
          <p className="panel-sub" style={{ margin: "2px 0 0" }}>
            Press play to listen first, then analyze. Every clip is held out from training, and the true
            label is shown after the result.
          </p>
        </div>
      </div>
      <div className="sample-groups">
        {groupSamples(manifest.samples).map(([group, items]) => (
          <div key={group} className="sample-group">
            <div className="sample-group-title">
              <span className="dot" style={{ background: groupTone(group, items) }} aria-hidden />
              {group}
            </div>
            <div className="sample-grid">
              {items.map((s) => (
                <article
                  key={s.id}
                  className={`sample-card ${activeId === s.id ? "active" : ""} ${
                    playingId === s.id ? "playing" : ""
                  }`}
                >
                  <PlayButton
                    sample={s}
                    playing={playingId === s.id}
                    progress={progress}
                    onToggle={() => toggle(s)}
                  />
                  <div className="sample-body">
                    <div className="sample-title-row">
                      <h4 className="sample-title">{s.title}</h4>
                      <span className="sample-dur num">{fmtDur(s.duration_s)}</span>
                    </div>
                    <div className="sample-note">{s.note}</div>
                    {s.transcript && <p className="sample-transcript">“{s.transcript}”</p>}
                  </div>
                  <button
                    type="button"
                    className="btn btn-sm sample-analyze"
                    onClick={() => analyze(s)}
                    disabled={busy}
                    aria-label={`Analyze ${s.title}`}
                  >
                    Analyze
                    <Icon name="arrow" size={13} />
                  </button>
                </article>
              ))}
            </div>
          </div>
        ))}
      </div>
      <p className="panel-sub" style={{ margin: "14px 0 0" }}>
        Sources: {Array.from(new Set(manifest.samples.map((s) => s.credit))).join(" · ")}.
      </p>
    </section>
  );
}
