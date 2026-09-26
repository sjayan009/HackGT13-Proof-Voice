"use client";

export default function Header({ onOpenEval }: { onOpenEval: () => void }) {
  return (
    <header className="app-header">
      <div className="brand">
        <h1>ProofVoice</h1>
        <div className="tagline">
          A real-time evidence layer for synthetic speech.
        </div>
      </div>
      <div className="header-actions">
        <button className="btn" onClick={onOpenEval}>
          Evaluation
        </button>
      </div>
    </header>
  );
}
