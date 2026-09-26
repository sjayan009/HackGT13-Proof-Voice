import type { FileMeta } from "@/lib/types";

function Row({ k, v }: { k: string; v: string | number | null }) {
  return (
    <>
      <div className="k">{k}</div>
      <div className="v">{v == null || v === "" ? "—" : v}</div>
    </>
  );
}

export default function FileMetadataPanel({ file }: { file: FileMeta }) {
  return (
    <div className="kv-list">
      <Row k="Name" v={file.name} />
      <Row k="Container" v={file.container} />
      <Row k="Codec" v={file.codec} />
      <Row k="Sample rate" v={file.sample_rate ? `${file.sample_rate} Hz` : null} />
      <Row k="Channels" v={file.channels} />
      <Row k="Bit depth" v={file.bit_depth ? `${file.bit_depth}-bit` : null} />
      <Row k="Duration" v={file.duration_s != null ? `${file.duration_s.toFixed(2)} s` : null} />
      <Row k="Bitrate" v={file.bitrate != null ? `${file.bitrate} bps` : null} />
      {file.metadata &&
        Object.entries(file.metadata).map(([k, v]) => (
          <Row key={k} k={k} v={typeof v === "object" ? JSON.stringify(v) : String(v)} />
        ))}
    </div>
  );
}
