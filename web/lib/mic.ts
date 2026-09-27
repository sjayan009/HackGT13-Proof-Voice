// Audio capture for live streaming: microphone or a played-back clip, both through the same PCM tap.
// AudioWorklet preferred, ScriptProcessorNode fallback. Emits mono Float32 PCM chunks at the
// AudioContext's native sample rate.

export interface MicCaptureHandle {
  sampleRate: number;
  stop: () => void;
}

export interface MicCaptureOptions {
  onChunk: (chunk: Float32Array) => void;
  onLevel?: (dbfs: number) => void;
}

export interface ClipCaptureOptions extends MicCaptureOptions {
  /** Fired when the clip finishes playing (not when stopped manually). */
  onEnded?: () => void;
  /** Called with playback position (seconds) while playing. */
  onProgress?: (seconds: number, duration: number) => void;
}

function computeDbfsLocal(samples: Float32Array): number {
  if (samples.length === 0) return -100;
  let sumSq = 0;
  for (let i = 0; i < samples.length; i++) sumSq += samples[i] * samples[i];
  const rms = Math.sqrt(sumSq / samples.length);
  if (rms <= 0) return -100;
  return Math.max(-100, 20 * Math.log10(rms));
}

function newContext(): AudioContext {
  const Ctor = window.AudioContext || (window as any).webkitAudioContext;
  return new Ctor();
}

/** Connects `source` to a PCM tap that forwards chunks to opts.onChunk. Returns cleanup functions. */
async function attachTap(
  ctx: AudioContext,
  source: AudioNode,
  opts: MicCaptureOptions
): Promise<Array<() => void>> {
  const cleanup: Array<() => void> = [];
  const emit = (chunk: Float32Array) => {
    opts.onChunk(chunk);
    opts.onLevel?.(computeDbfsLocal(chunk));
  };

  // The tap's output isn't for listening; route it through a muted gain to keep the graph alive.
  const silentGain = ctx.createGain();
  silentGain.gain.value = 0;
  silentGain.connect(ctx.destination);
  cleanup.push(() => {
    try {
      silentGain.disconnect();
    } catch {}
  });

  try {
    if (ctx.audioWorklet) {
      await ctx.audioWorklet.addModule("/pcm-worklet-processor.js");
      const node = new AudioWorkletNode(ctx, "pcm-worklet-processor");
      node.port.onmessage = (ev: MessageEvent<Float32Array>) => emit(ev.data);
      source.connect(node);
      node.connect(silentGain);
      cleanup.push(() => {
        try {
          node.disconnect();
        } catch {}
      });
      return cleanup;
    }
  } catch {
    /* fall back below */
  }

  const processor = (ctx as any).createScriptProcessor ? ctx.createScriptProcessor(4096, 1, 1) : null;
  if (!processor) {
    throw new Error("Neither AudioWorklet nor ScriptProcessorNode is available in this browser.");
  }
  processor.onaudioprocess = (e: AudioProcessingEvent) => emit(new Float32Array(e.inputBuffer.getChannelData(0)));
  source.connect(processor);
  processor.connect(silentGain);
  cleanup.push(() => {
    try {
      processor.disconnect();
    } catch {}
  });
  return cleanup;
}

export async function startMicCapture(opts: MicCaptureOptions): Promise<MicCaptureHandle> {
  // Forensic capture: disable the browser's call-processing chain. Echo cancellation, noise suppression and
  // auto-gain reshape the spectrum the detector relies on, and the model was never trained on that processing.
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  });

  const ctx = newContext();
  const source = ctx.createMediaStreamSource(stream);
  const cleanup: Array<() => void> = [() => stream.getTracks().forEach((t) => t.stop())];
  try {
    cleanup.push(...(await attachTap(ctx, source, opts)));
  } catch (err) {
    cleanup.forEach((fn) => fn());
    ctx.close().catch(() => {});
    throw err;
  }
  cleanup.push(() => {
    try {
      source.disconnect();
    } catch {}
    ctx.close().catch(() => {});
  });

  return { sampleRate: ctx.sampleRate, stop: () => cleanup.forEach((fn) => fn()) };
}

/**
 * Plays an audio clip through the speakers and streams the exact same samples to the detector in real time.
 * This is the digital path: no loudspeaker → room → microphone channel in between.
 */
export async function startClipCapture(clip: Blob, opts: ClipCaptureOptions): Promise<MicCaptureHandle> {
  const ctx = newContext();
  await ctx.resume().catch(() => {});
  let buffer: AudioBuffer;
  try {
    buffer = await ctx.decodeAudioData(await clip.arrayBuffer());
  } catch {
    ctx.close().catch(() => {});
    throw new Error("This browser could not decode that audio file for live streaming.");
  }

  const source = ctx.createBufferSource();
  source.buffer = buffer;
  // Downmix to mono for the tap (the wire format is mono); play the original to the speakers.
  const mono = ctx.createGain();
  mono.channelCount = 1;
  mono.channelCountMode = "explicit";
  mono.channelInterpretation = "speakers";
  source.connect(mono);
  source.connect(ctx.destination);

  let cleanup: Array<() => void>;
  try {
    cleanup = await attachTap(ctx, mono, opts);
  } catch (err) {
    ctx.close().catch(() => {});
    throw err;
  }

  let stopped = false;
  let raf = 0;
  const t0 = ctx.currentTime;
  const tick = () => {
    opts.onProgress?.(Math.min(buffer.duration, ctx.currentTime - t0), buffer.duration);
    raf = requestAnimationFrame(tick);
  };
  source.onended = () => {
    cancelAnimationFrame(raf);
    // Let the last worklet chunk flush before reporting completion.
    setTimeout(() => {
      if (!stopped) opts.onEnded?.();
    }, 150);
  };
  source.start();
  raf = requestAnimationFrame(tick);

  return {
    sampleRate: ctx.sampleRate,
    stop: () => {
      stopped = true;
      cancelAnimationFrame(raf);
      try {
        source.onended = null;
        source.stop();
      } catch {}
      cleanup.forEach((fn) => fn());
      try {
        source.disconnect();
        mono.disconnect();
      } catch {}
      ctx.close().catch(() => {});
    },
  };
}
