import type { FileMeta } from "@/lib/types";
import { num } from "@/lib/format";

function Row({ k, v }: { k: string; v: string | number | null | undefined }) {
  return (
    <>
      <dt className="k">{k}</dt>
      <dd className="v" style={{ margin: 0 }}>
        {v == null || v === "" ? "—" : v}
      </dd>
    </>
  );
}

export default function FileMetadataPanel({ file }: { file: FileMeta | null | undefined }) {
  if (!file) return <div className="empty-state">No file metadata (live stream).</div>;
  return (
    <dl className="kv-list" style={{ margin: 0 }}>
      <Row k="Name" v={file.name} />
      <Row k="Container" v={file.container} />
      <Row k="Codec" v={file.codec} />
      <Row k="Sample rate" v={file.sample_rate ? `${file.sample_rate.toLocaleString()} Hz` : null} />
      <Row k="Channels" v={file.channels} />
      <Row k="Bit depth" v={file.bit_depth ? `${file.bit_depth}-bit` : null} />
      <Row k="Duration" v={file.duration_s != null ? `${file.duration_s.toFixed(2)} s` : null} />
      <Row k="Bitrate" v={file.bitrate != null ? `${Math.round(file.bitrate / 1000)} kbps` : null} />
      {file.metadata && Object.entries(file.metadata).map(([k, v]) => <Row key={k} k={k} v={num(v)} />)}
    </dl>
  );
}
