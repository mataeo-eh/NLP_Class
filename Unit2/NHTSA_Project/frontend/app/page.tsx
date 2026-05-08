const backendBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "https://nlp-class.onrender.com";

export default function HomePage() {
  return (
    <main className="page-shell">
      <section className="card">
        <p className="eyebrow">NHTSA Project</p>
        <h1>Frontend placeholder</h1>
        <p className="body-copy">
          This deploy exists so the frontend team can start building on a Vercel
          recognized app structure without waiting on backend or product logic.
        </p>
        <p className="meta-copy">
          Backend base URL:{" "}
          <a href={backendBaseUrl} target="_blank" rel="noreferrer">
            {backendBaseUrl}
          </a>
        </p>
      </section>
    </main>
  );
}
