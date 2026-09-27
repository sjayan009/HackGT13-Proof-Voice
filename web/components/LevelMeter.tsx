export default function LevelMeter({ dbfs }: { dbfs: number | null }) {
  const value = dbfs == null || !Number.isFinite(dbfs) ? -100 : dbfs;
  const frac = Math.max(0, Math.min(1, (value + 60) / 60));
  const quiet = dbfs != null && value < -50;
  return (
    <div>
      <div
        className="level-meter"
        role="meter"
        aria-label="Microphone input level"
        aria-valuemin={-60}
        aria-valuemax={0}
        aria-valuenow={Math.round(Math.max(-60, value))}
      >
        <div className="level-meter-fill" style={{ transform: `scaleX(${frac})` }} />
      </div>
      <div className="level-caption">
        <span>{quiet ? "Very quiet: move closer to the microphone" : "Input level"}</span>
        <span className="num">{dbfs == null ? "—" : `${value.toFixed(1)} dBFS`}</span>
      </div>
    </div>
  );
}
