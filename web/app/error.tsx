"use client";

import { useEffect } from "react";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <main className="main-content" id="main">
      <div className="panel" role="alert">
        <div className="page-intro">
          <h2>Something went wrong</h2>
          <p>
            The interface hit an unexpected error while rendering. No analysis result was altered — reload
            to start again.
          </p>
        </div>
        <pre className="json">{error.message}</pre>
        <div style={{ marginTop: 16, display: "flex", gap: 8 }}>
          <button className="btn btn-primary" onClick={reset}>
            Try again
          </button>
          <button className="btn" onClick={() => window.location.reload()}>
            Reload page
          </button>
        </div>
      </div>
    </main>
  );
}
