export default function LevelMeter({ dbfs }: { dbfs: number | null }) {
  const value = dbfs == null ? -100 : dbfs;
  const pct = Math.max(0, Math.min(100, ((value + 60) / 60) * 100));
  return (
    <div>
      <div className="level-meter">
        <div className="level-meter-fill" style={{ width: `${pct}%` }} />
      </div>
      <div style={{ fontSize: 11, color: "var(--text-faint)", marginTop: 4 }}>
        Input level: {dbfs == null ? "—" : `${dbfs.toFixed(1)} dBFS`}
      </div>
    </div>
  );
}
