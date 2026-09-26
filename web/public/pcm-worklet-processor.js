// AudioWorklet processor that forwards mono Float32 PCM frames to the main
// thread in fixed-size blocks. Used by Live Trust to stream mic audio over
// WS /analyze/stream (encoding "f32le").
class PcmWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = [];
    this._chunkSize = 2048; // ~42ms at 48kHz, ~128ms at 16kHz
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      const channelData = input[0];
      for (let i = 0; i < channelData.length; i++) {
        this._buffer.push(channelData[i]);
      }
      while (this._buffer.length >= this._chunkSize) {
        const chunk = this._buffer.splice(0, this._chunkSize);
        const out = new Float32Array(chunk);
        this.port.postMessage(out, [out.buffer]);
      }
    }
    return true;
  }
}

registerProcessor("pcm-worklet-processor", PcmWorkletProcessor);
