export default function Waveform({ peaks }: { peaks: number[] | null }) {
  if (!peaks || peaks.length === 0) {
    return <div className="empty-state">Waveform unavailable for this file.</div>;
  }
  const width = 800;
  const height = 80;
  const mid = height / 2;
  const barW = width / peaks.length;
  const maxPeak = Math.max(...peaks, 0.001);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height}>
      {peaks.map((p, i) => {
        const h = Math.max(1, (p / maxPeak) * (height - 4));
        return (
          <rect
            key={i}
            x={i * barW}
            y={mid - h / 2}
            width={Math.max(1, barW - 0.5)}
            height={h}
            fill="var(--accent)"
            opacity={0.7}
          />
        );
      })}
    </svg>
  );
}
