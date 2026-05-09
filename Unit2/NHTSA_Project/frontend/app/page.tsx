"use client";

// Minimal smoke-test UI for POST /run on the FastAPI backend.
//
// Why this is so bare-bones
// -------------------------
// The real frontend is owned by another contributor. This page exists only to
// give the backend author a way to confirm, in a real browser, that the SSE
// pipeline works end-to-end (CORS, streaming, event parsing). The audio control
// is intentionally small: it only proves that the browser can call the hosted
// /audio/speech endpoint after /run returns a final response.

import { useRef, useState } from "react";

const configuredBackendBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
const backendBaseUrl = configuredBackendBaseUrl ?? "";

type SseEvent = {
  event: string;
  data: string;
};

export default function HomePage() {
  const [request, setRequest] = useState(
    "Give me 2 complaints from the database with their CDESCR text",
  );
  const [running, setRunning] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [log, setLog] = useState("");
  const [finalResponse, setFinalResponse] = useState("");
  const [audioUrl, setAudioUrl] = useState("");
  // Hold an AbortController so a second click can cancel an in-flight stream
  // rather than fire a parallel one.
  const abortRef = useRef<AbortController | null>(null);
  const audioUrlRef = useRef("");

  // Append a line to the log area without losing existing content.
  function appendLog(line: string) {
    setLog((prev) => (prev ? prev + "\n" + line : line));
  }

  function replaceAudioUrl(nextUrl: string) {
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
    }
    audioUrlRef.current = nextUrl;
    setAudioUrl(nextUrl);
  }

  // Parse a chunk of the SSE byte stream into discrete event frames.
  //
  // SSE wire format (per WHATWG): events are separated by a blank line, and
  // within each event the lines look like `field: value`. We only care about
  // `event:` (name) and `data:` (JSON payload). The buffer carries any
  // partial trailing event across chunks.
  function consumeBuffer(buffer: string): { events: SseEvent[]; rest: string } {
    const events: SseEvent[] = [];
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

  function extractFinalResponse(eventData: string): string {
    try {
      const payload = JSON.parse(eventData);
      const response = payload.response;
      return typeof response === "string" ? response.trim() : "";
    } catch {
      return "";
    }
  }

  async function speakText(text: string) {
    const spokenText = text.trim();
    if (!backendBaseUrl || !spokenText) return;

    setSpeaking(true);
    appendLog(`POST ${backendBaseUrl}/audio/speech`);
    try {
      const response = await fetch(`${backendBaseUrl}/audio/speech`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: spokenText }),
      });

      if (!response.ok) {
        appendLog(`audio HTTP ${response.status} — ${await response.text()}`);
        return;
      }

      const blob = await response.blob();
      const nextUrl = URL.createObjectURL(blob);
      replaceAudioUrl(nextUrl);

      const audio = new Audio(nextUrl);
      try {
        await audio.play();
        appendLog("[audio] playback started");
      } catch {
        appendLog("[audio] browser blocked autoplay; use the audio controls below");
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      appendLog(`audio failed: ${message}`);
    } finally {
      setSpeaking(false);
    }
  }

  async function handleRun() {
    if (!backendBaseUrl) {
      appendLog("NEXT_PUBLIC_API_BASE_URL is not configured, so the frontend does not know which backend to call.");
      return;
    }

    // Cancel any in-flight stream from a previous click.
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setRunning(true);
    setLog("");
    setFinalResponse("");
    replaceAudioUrl("");
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
          if (ev.event === "completed") {
            const responseText = extractFinalResponse(ev.data);
            setFinalResponse(responseText);
            if (responseText) {
              await speakText(responseText);
            } else {
              appendLog("[audio] completed event did not include a response string");
            }
          }
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
          each Server-Sent Event as it arrives. When the pipeline completes, it
          sends the final response to <code>/audio/speech</code> and plays the
          returned WAV.
        </p>
        <p className="meta-copy">
          Backend base URL:{" "}
          {backendBaseUrl ? (
            <a href={backendBaseUrl} target="_blank" rel="noreferrer">
              {backendBaseUrl}
            </a>
          ) : (
            <strong>missing NEXT_PUBLIC_API_BASE_URL</strong>
          )}
        </p>

        <textarea
          value={request}
          onChange={(e) => setRequest(e.target.value)}
          rows={3}
          className="request-input"
        />
        <div className="action-row">
          <button
            type="button"
            onClick={handleRun}
            disabled={running || speaking || request.trim().length === 0 || !backendBaseUrl}
          >
            {running ? "Running..." : "Run pipeline"}
          </button>
          <button
            type="button"
            onClick={() => speakText(finalResponse)}
            disabled={running || speaking || !finalResponse || !backendBaseUrl}
          >
            {speaking ? "Speaking..." : "Replay audio"}
          </button>
        </div>

        {audioUrl ? (
          <audio className="audio-player" src={audioUrl} controls />
        ) : null}

        <pre className="event-log">
          {log || "(no events yet)"}
        </pre>
      </section>
    </main>
  );
}
