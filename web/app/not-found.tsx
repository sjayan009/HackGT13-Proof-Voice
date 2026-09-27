import Link from "next/link";

export default function NotFound() {
  return (
    <main className="main-content" id="main">
      <div className="panel">
        <div className="page-intro">
          <h2>Page not found</h2>
          <p>ProofVoice is a single-page app.</p>
        </div>
        <Link className="btn btn-primary" href="/">
          Back to ProofVoice
        </Link>
      </div>
    </main>
  );
}
