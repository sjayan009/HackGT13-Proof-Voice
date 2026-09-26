// Microphone capture: AudioWorklet preferred, ScriptProcessorNode fallback.
// Emits mono Float32 PCM chunks at the AudioContext's native sample rate.

export interface MicCaptureHandle {
  sampleRate: number;
  stop: () => void;
}

export interface MicCaptureOptions {
  onChunk: (chunk: Float32Array) => void;
  onLevel?: (dbfs: number) => void;
}

function computeDbfsLocal(samples: Float32Array): number {
  if (samples.length === 0) return -100;
  let sumSq = 0;
  for (let i = 0; i < samples.length; i++) sumSq += samples[i] * samples[i];
  const rms = Math.sqrt(sumSq / samples.length);
  if (rms <= 0) return -100;
  return Math.max(-100, 20 * Math.log10(rms));
}

export async function startMicCapture(
  opts: MicCaptureOptions
): Promise<MicCaptureHandle> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  const AudioContextCtor =
    window.AudioContext || (window as any).webkitAudioContext;
  const ctx: AudioContext = new AudioContextCtor();
  const source = ctx.createMediaStreamSource(stream);

  const cleanupFns: Array<() => void> = [];
  cleanupFns.push(() => stream.getTracks().forEach((t) => t.stop()));

  let usedWorklet = false;
  try {
    if (ctx.audioWorklet) {
      await ctx.audioWorklet.addModule("/pcm-worklet-processor.js");
      const node = new AudioWorkletNode(ctx, "pcm-worklet-processor");
      node.port.onmessage = (ev: MessageEvent<Float32Array>) => {
        const chunk = ev.data;
        opts.onChunk(chunk);
        if (opts.onLevel) opts.onLevel(computeDbfsLocal(chunk));
      };
      source.connect(node);
      // Worklet output isn't meant for playback; connect to a muted gain to
      // keep the graph alive in some browsers without audible feedback.
      const silentGain = ctx.createGain();
      silentGain.gain.value = 0;
      node.connect(silentGain);
      silentGain.connect(ctx.destination);
      usedWorklet = true;
      cleanupFns.push(() => {
        try {
          node.disconnect();
          silentGain.disconnect();
        } catch {}
      });
    }
  } catch {
    usedWorklet = false;
  }

  if (!usedWorklet) {
    const bufferSize = 4096;
    const processor = (ctx as any).createScriptProcessor
      ? ctx.createScriptProcessor(bufferSize, 1, 1)
      : null;
    if (!processor) {
      throw new Error(
        "Neither AudioWorklet nor ScriptProcessorNode is available in this browser."
      );
    }
    processor.onaudioprocess = (e: AudioProcessingEvent) => {
      const input = e.inputBuffer.getChannelData(0);
      const copy = new Float32Array(input);
      opts.onChunk(copy);
      if (opts.onLevel) opts.onLevel(computeDbfsLocal(copy));
    };
    source.connect(processor);
    const silentGain = ctx.createGain();
    silentGain.gain.value = 0;
    processor.connect(silentGain);
    silentGain.connect(ctx.destination);
    cleanupFns.push(() => {
      try {
        processor.disconnect();
        silentGain.disconnect();
      } catch {}
    });
  }

  cleanupFns.push(() => {
    try {
      source.disconnect();
    } catch {}
  });
  cleanupFns.push(() => {
    ctx.close().catch(() => {});
  });

  return {
    sampleRate: ctx.sampleRate,
    stop: () => cleanupFns.forEach((fn) => fn()),
  };
}
