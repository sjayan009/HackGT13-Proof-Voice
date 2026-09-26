// Audio helpers shared by Live Trust and Red Team screens.
// Only used to encode/decode PCM for the wire protocol and to draw local
// waveform previews client-side. Never used to compute detection results.

export function floatTo16BitPCM(input: Float32Array): Int16Array {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

export function base64ToInt16Array(b64: string): Int16Array {
  const binary = atob(b64);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
  // Ensure even length for Int16
  const usable = bytes.buffer.slice(0, len - (len % 2));
  return new Int16Array(usable);
}

export function int16ToFloat32(input: Int16Array): Float32Array {
  const out = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    out[i] = input[i] / (input[i] < 0 ? 0x8000 : 0x7fff);
  }
  return out;
}

/** RMS-based dBFS estimate for a Float32 PCM buffer, for a simple level meter. */
export function computeDbfs(samples: Float32Array): number {
  if (samples.length === 0) return -Infinity;
  let sumSq = 0;
  for (let i = 0; i < samples.length; i++) sumSq += samples[i] * samples[i];
  const rms = Math.sqrt(sumSq / samples.length);
  if (rms <= 0) return -100;
  return Math.max(-100, 20 * Math.log10(rms));
}

/** Decode an arbitrary audio File into raw samples for local waveform drawing only. */
export async function decodeAudioFileForWaveform(
  file: File
): Promise<{ peaks: number[]; duration: number } | null> {
  try {
    const arrayBuffer = await file.arrayBuffer();
    const AudioContextCtor =
      window.AudioContext || (window as any).webkitAudioContext;
    if (!AudioContextCtor) return null;
    const ctx: AudioContext = new AudioContextCtor();
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
    const channel = audioBuffer.getChannelData(0);
    const buckets = 400;
    const blockSize = Math.max(1, Math.floor(channel.length / buckets));
    const peaks: number[] = [];
    for (let i = 0; i < buckets; i++) {
      const start = i * blockSize;
      let max = 0;
      for (let j = start; j < start + blockSize && j < channel.length; j++) {
        const v = Math.abs(channel[j]);
        if (v > max) max = v;
      }
      peaks.push(max);
    }
    const duration = audioBuffer.duration;
    ctx.close().catch(() => {});
    return { peaks, duration };
  } catch {
    return null;
  }
}

/** Simple PCM16 mono player that queues chunks in order via Web Audio. */
export class PcmStreamPlayer {
  private ctx: AudioContext;
  private nextStartTime = 0;
  private sampleRate: number;

  constructor(sampleRate: number) {
    const AudioContextCtor =
      window.AudioContext || (window as any).webkitAudioContext;
    this.ctx = new AudioContextCtor({ sampleRate });
    this.sampleRate = sampleRate;
    this.nextStartTime = this.ctx.currentTime;
  }

  enqueue(int16: Int16Array) {
    const float32 = int16ToFloat32(int16);
    const buffer = this.ctx.createBuffer(1, float32.length, this.sampleRate);
    buffer.copyToChannel(float32, 0);
    const source = this.ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.ctx.destination);
    const startAt = Math.max(this.nextStartTime, this.ctx.currentTime);
    source.start(startAt);
    this.nextStartTime = startAt + buffer.duration;
  }

  close() {
    this.ctx.close().catch(() => {});
  }
}
