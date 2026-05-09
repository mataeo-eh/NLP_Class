"use client";

// Minimal smoke-test UI for POST /run on the FastAPI backend.
//
// Why this is so bare-bones
// -------------------------
// The real frontend is owned by another contributor. This page exists only to
// give the backend author a way to confirm, in a real browser, that the SSE
// pipeline works end-to-end (CORS, streaming, event parsing). Anything beyond
// that — styling, error UX, retries, audio in/out — is intentionally out of
// scope so it doesn't conflict with the real frontend work.

import { useRef, useState } from "react";

const backendBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "https://nlp-class.onrender.com";

export default function HomePage() {
  const [request, setRequest] = useState(
    "Give me 2 complaints from the database with their CDESCR text",
  );
  const [running, setRunning] = useState(false);
  const [log, setLog] = useState("");
  // Hold an AbortController so a second click can cancel an in-flight stream
  // rather than fire a parallel one.
  const abortRef = useRef<AbortController | null>(null);

  // Append a line to the log area without losing existing content.
  function appendLog(line: string) {
    setLog((prev) => (prev ? prev + "\n" + line : line));
  }

  // Parse a chunk of the SSE byte stream into discrete event frames.
  //
  // SSE wire format (per WHATWG): events are separated by a blank line, and
  // within each event the lines look like `field: value`. We only care about
  // `event:` (name) and `data:` (JSON payload). The buffer carries any
  // partial trailing event across chunks.
  function consumeBuffer(buffer: string): { events: Array<{ event: string; data: string }>; rest: string } {
    const events: Array<{ event: string; data: string }> = [];
    let rest = buffer;
    while (true) {
      const sep = rest.indexOf("\n\n");
      if (sep === -1) break;
      const frame = rest.slice(0, sep);
      rest = rest.slice(sep + 2);
      let eventName = "message";
      const dataLines: string[] = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) eventName = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      events.push({ event: eventName, data: dataLines.join("\n") });
    }
    return { events, rest };
  }

  async function handleRun() {
    // Cancel any in-flight stream from a previous click.
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setRunning(true);
    setLog("");
    appendLog(`POST ${backendBaseUrl}/run`);

    try {
      const response = await fetch(`${backendBaseUrl}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_request: request }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        appendLog(`HTTP ${response.status} — ${await response.text()}`);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const { events, rest } = consumeBuffer(buffer);
        buffer = rest;
        for (const ev of events) {
          appendLog(`[${ev.event}] ${ev.data}`);
        }
      }
    } catch (err) {
      // Aborts are user-initiated, not real errors — surface as a hint.
      const message = err instanceof Error ? err.message : String(err);
      appendLog(`stream ended: ${message}`);
    } finally {
      setRunning(false);
    }
  }

  return (
    <main className="page-shell">
      <section className="card">
        <p className="eyebrow">NHTSA Project</p>
        <h1>Pipeline smoke test</h1>
        <p className="body-copy">
          Sends one POST to <code>/run</code> on the FastAPI backend and prints
          each Server-Sent Event as it arrives. Used to confirm the deployed
          pipeline is alive — not a real UI.
        </p>
        <p className="meta-copy">
          Backend base URL:{" "}
          <a href={backendBaseUrl} target="_blank" rel="noreferrer">
            {backendBaseUrl}
          </a>
        </p>

        <textarea
          value={request}
          onChange={(e) => setRequest(e.target.value)}
          rows={3}
          style={{
            width: "100%",
            marginTop: "1.25rem",
            padding: "0.75rem",
            fontFamily: "inherit",
            fontSize: "0.95rem",
            border: "1px solid rgba(23, 32, 51, 0.15)",
            borderRadius: "0.5rem",
            resize: "vertical",
          }}
        />
        <button
          type="button"
          onClick={handleRun}
          disabled={running || request.trim().length === 0}
          style={{
            marginTop: "0.75rem",
            padding: "0.6rem 1.1rem",
            fontFamily: "inherit",
            fontSize: "0.95rem",
            border: "1px solid #172033",
            borderRadius: "0.5rem",
            background: running ? "#dde3ee" : "#172033",
            color: running ? "#5c6b86" : "white",
            cursor: running ? "default" : "pointer",
          }}
        >
          {running ? "Running…" : "Run pipeline"}
        </button>

        <pre
          style={{
            marginTop: "1.25rem",
            padding: "0.9rem",
            background: "#0f1628",
            color: "#d6e1f4",
            borderRadius: "0.5rem",
            fontSize: "0.78rem",
            lineHeight: 1.45,
            maxHeight: "24rem",
            overflow: "auto",
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
          }}
        >
          {log || "(no events yet)"}
        </pre>
      </section>
    </main>
  );
}
