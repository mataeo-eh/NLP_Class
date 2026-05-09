"use client";

// Minimal hosted-audio smoke-test UI for the FastAPI backend.
//
// This page now exercises the full hosted browser contract:
//   1. warm up with the CLI greeting text
//   2. accept either typed input or push-to-talk audio upload
//   3. stream per-node narration audio plus the final answer

import { useRef, useState } from "react";

const configuredBackendBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
const backendBaseUrl = configuredBackendBaseUrl ?? "";

type SseEvent = {
  event: string;
  data: string;
};

type WarmupResponse = {
  welcome_text?: unknown;
  ready_prompt?: unknown;
};

type WindowWithWebkitAudioContext = Window &
  typeof globalThis & {
    webkitAudioContext?: typeof AudioContext;
  };

function pcmChunkToAudioBuffer(
  audioContext: AudioContext,
  pcmBytes: Uint8Array,
  sampleRate: number,
): AudioBuffer {
  const sampleCount = pcmBytes.byteLength / 2;
  const channelData = new Float32Array(sampleCount);
  const view = new DataView(
    pcmBytes.buffer,
    pcmBytes.byteOffset,
    pcmBytes.byteLength,
  );

  for (let index = 0; index < sampleCount; index += 1) {
    const pcmValue = view.getInt16(index * 2, true);
    channelData[index] = pcmValue / 32768;
  }

  const audioBuffer = audioContext.createBuffer(1, sampleCount, sampleRate);
  audioBuffer.copyToChannel(channelData, 0, 0);
  return audioBuffer;
}

function buildWaveBlobFromPcmChunks(
  pcmChunks: Uint8Array[],
  sampleRate: number,
): Blob {
  const pcmByteLength = pcmChunks.reduce(
    (total, chunk) => total + chunk.byteLength,
    0,
  );
  const wavBuffer = new ArrayBuffer(44 + pcmByteLength);
  const view = new DataView(wavBuffer);
  let offset = 0;

  function writeAscii(value: string) {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset, value.charCodeAt(index));
      offset += 1;
    }
  }

  writeAscii("RIFF");
  view.setUint32(offset, 36 + pcmByteLength, true);
  offset += 4;
  writeAscii("WAVE");
  writeAscii("fmt ");
  view.setUint32(offset, 16, true);
  offset += 4;
  view.setUint16(offset, 1, true);
  offset += 2;
  view.setUint16(offset, 1, true);
  offset += 2;
  view.setUint32(offset, sampleRate, true);
  offset += 4;
  view.setUint32(offset, sampleRate * 2, true);
  offset += 4;
  view.setUint16(offset, 2, true);
  offset += 2;
  view.setUint16(offset, 16, true);
  offset += 2;
  writeAscii("data");
  view.setUint32(offset, pcmByteLength, true);
  offset += 4;

  const body = new Uint8Array(wavBuffer, offset);
  let bodyOffset = 0;
  for (const chunk of pcmChunks) {
    body.set(chunk, bodyOffset);
    bodyOffset += chunk.byteLength;
  }

  return new Blob([wavBuffer], { type: "audio/wav" });
}

function guessAudioFilename(mimeType: string): string {
  const normalized = mimeType.toLowerCase();
  if (normalized.includes("wav")) return "speech.wav";
  if (normalized.includes("ogg")) return "speech.ogg";
  if (normalized.includes("mp4")) return "speech.m4a";
  if (normalized.includes("mpeg")) return "speech.mp3";
  return "speech.webm";
}

export default function HomePage() {
  const [request, setRequest] = useState("");
  const [warmingUp, setWarmingUp] = useState(false);
  const [warmedUp, setWarmedUp] = useState(false);
  const [running, setRunning] = useState(false);
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [assistantStatus, setAssistantStatus] = useState(
    "Press Run pipeline to hear the greeting and unlock the hosted audio flow.",
  );
  const [log, setLog] = useState("");
  const [finalResponse, setFinalResponse] = useState("");
  const [audioUrl, setAudioUrl] = useState("");

  const abortRef = useRef<AbortController | null>(null);
  const speechAbortRef = useRef<AbortController | null>(null);
  const audioUrlRef = useRef("");
  const audioElementRef = useRef<HTMLAudioElement | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const playbackCursorRef = useRef(0);
  const activeSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const speechQueueRef = useRef<string[]>([]);
  const drainingSpeechQueueRef = useRef(false);

  function appendLog(line: string) {
    setLog((prev) => (prev ? prev + "\n" + line : line));
  }

  function replaceAudioUrl(nextUrl: string) {
    if (audioElementRef.current) {
      audioElementRef.current.pause();
      audioElementRef.current.currentTime = 0;
    }
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
    }
    audioUrlRef.current = nextUrl;
    setAudioUrl(nextUrl);
  }

  function stopScheduledPlayback() {
    for (const source of activeSourcesRef.current) {
      try {
        source.stop();
      } catch {
        // AudioBufferSourceNode.stop() throws if the source already ended.
      }
    }
    activeSourcesRef.current.clear();
    const audioContext = audioContextRef.current;
    playbackCursorRef.current = audioContext ? audioContext.currentTime : 0;
  }

  function stopMicrophoneStream() {
    const stream = mediaStreamRef.current;
    if (!stream) return;
    for (const track of stream.getTracks()) {
      track.stop();
    }
    mediaStreamRef.current = null;
  }

  function interruptSpeechPlayback() {
    speechQueueRef.current = [];
    speechAbortRef.current?.abort();
    stopScheduledPlayback();
    if (audioElementRef.current) {
      audioElementRef.current.pause();
      audioElementRef.current.currentTime = 0;
    }
  }

  async function ensureAudioContextReady(): Promise<AudioContext | null> {
    const AudioContextCtor =
      window.AudioContext ??
      (window as WindowWithWebkitAudioContext).webkitAudioContext;
    if (!AudioContextCtor) {
      return null;
    }

    let audioContext = audioContextRef.current;
    if (!audioContext || audioContext.state === "closed") {
      audioContext = new AudioContextCtor();
      audioContextRef.current = audioContext;
    }

    if (audioContext.state === "suspended") {
      await audioContext.resume();
    }

    return audioContext;
  }

  function schedulePcmChunk(
    audioContext: AudioContext,
    pcmBytes: Uint8Array,
    sampleRate: number,
  ) {
    const audioBuffer = pcmChunkToAudioBuffer(audioContext, pcmBytes, sampleRate);
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);

    const startAt = Math.max(
      playbackCursorRef.current,
      audioContext.currentTime + 0.12,
    );
    source.addEventListener("ended", () => {
      activeSourcesRef.current.delete(source);
    });
    activeSourcesRef.current.add(source);
    source.start(startAt);
    playbackCursorRef.current = startAt + audioBuffer.duration;
  }

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
      const payload = JSON.parse(eventData) as { response?: unknown };
      return typeof payload.response === "string" ? payload.response.trim() : "";
    } catch {
      return "";
    }
  }

  function extractNarrationText(eventData: string): string {
    try {
      const payload = JSON.parse(eventData) as { text?: unknown };
      return typeof payload.text === "string" ? payload.text.trim() : "";
    } catch {
      return "";
    }
  }

  function extractMessage(eventData: string): string {
    try {
      const payload = JSON.parse(eventData) as { message?: unknown };
      return typeof payload.message === "string" ? payload.message.trim() : "";
    } catch {
      return "";
    }
  }

  async function speakBufferedText(text: string) {
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

      const audio = audioElementRef.current ?? new Audio(nextUrl);
      try {
        await audio.play();
        appendLog("[audio] playback started");
      } catch {
        appendLog("[audio] browser blocked autoplay; use the audio controls below");
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        appendLog("[audio] stream cancelled");
      } else {
        const message = err instanceof Error ? err.message : String(err);
        appendLog(`audio failed: ${message}`);
      }
    } finally {
      setSpeaking(false);
    }
  }

  async function speakStreamingText(
    text: string,
    options?: { interruptCurrent?: boolean },
  ) {
    const spokenText = text.trim();
    if (!backendBaseUrl || !spokenText) return;

    const audioContext = await ensureAudioContextReady();
    if (!audioContext) {
      appendLog(
        "[audio] browser has no AudioContext support; falling back to buffered WAV",
      );
      await speakBufferedText(spokenText);
      return;
    }

    const interruptCurrent = options?.interruptCurrent ?? true;
    if (interruptCurrent) {
      speechAbortRef.current?.abort();
      stopScheduledPlayback();
      if (audioElementRef.current) {
        audioElementRef.current.pause();
        audioElementRef.current.currentTime = 0;
      }
    }

    const controller = new AbortController();
    speechAbortRef.current = controller;
    setSpeaking(true);
    appendLog(`POST ${backendBaseUrl}/audio/speech/stream`);

    try {
      const response = await fetch(`${backendBaseUrl}/audio/speech/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: spokenText }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        appendLog(`audio HTTP ${response.status} — ${await response.text()}`);
        return;
      }

      const codec = response.headers.get("x-audio-codec")?.trim().toLowerCase();
      const sampleRateHeader = response.headers.get("x-audio-sample-rate");
      const channelsHeader = response.headers.get("x-audio-channels");
      const sampleRate = Number(sampleRateHeader);
      const channels = Number(channelsHeader);

      if (
        codec !== "linear16" ||
        !Number.isFinite(sampleRate) ||
        sampleRate <= 0 ||
        channels !== 1
      ) {
        appendLog(
          `[audio] unsupported stream contract codec=${codec ?? "missing"} sampleRate=${sampleRateHeader ?? "missing"} channels=${channelsHeader ?? "missing"}; falling back to buffered WAV`,
        );
        await speakBufferedText(spokenText);
        return;
      }

      const reader = response.body.getReader();
      const pcmChunks: Uint8Array[] = [];
      let carry = new Uint8Array(0);
      let loggedPlaybackStart = false;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        if (!value || value.byteLength === 0) continue;

        let combined = value.slice();
        if (carry.byteLength > 0) {
          const merged = new Uint8Array(carry.byteLength + combined.byteLength);
          merged.set(carry, 0);
          merged.set(combined, carry.byteLength);
          combined = merged;
          carry = new Uint8Array(0);
        }

        const remainder = combined.byteLength % 2;
        if (remainder !== 0) {
          carry = combined.slice(combined.byteLength - remainder);
          combined = combined.slice(0, combined.byteLength - remainder);
        }

        if (combined.byteLength === 0) continue;

        pcmChunks.push(combined);
        schedulePcmChunk(audioContext, combined, sampleRate);
        if (!loggedPlaybackStart) {
          appendLog("[audio] streaming playback scheduled");
          loggedPlaybackStart = true;
        }
      }

      if (carry.byteLength > 0) {
        appendLog(
          `[audio] ignored ${carry.byteLength} trailing byte(s) from the PCM stream because a 16-bit sample must be 2 bytes`,
        );
      }

      if (pcmChunks.length === 0) {
        appendLog("[audio] stream completed without audio bytes");
        return;
      }

      const nextUrl = URL.createObjectURL(
        buildWaveBlobFromPcmChunks(pcmChunks, sampleRate),
      );
      replaceAudioUrl(nextUrl);
      appendLog("[audio] replay controls are ready");
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        appendLog("[audio] stream cancelled");
      } else {
        const message = err instanceof Error ? err.message : String(err);
        appendLog(`audio failed: ${message}`);
      }
    } finally {
      if (speechAbortRef.current === controller) {
        speechAbortRef.current = null;
      }
      setSpeaking(false);
    }
  }

  async function drainSpeechQueue() {
    if (drainingSpeechQueueRef.current) {
      return;
    }
    drainingSpeechQueueRef.current = true;
    try {
      while (speechQueueRef.current.length > 0) {
        const nextText = speechQueueRef.current.shift();
        if (!nextText) continue;
        await speakStreamingText(nextText, { interruptCurrent: false });
      }
    } finally {
      drainingSpeechQueueRef.current = false;
    }
  }

  function queueSpeechText(text: string) {
    const spokenText = text.trim();
    if (!spokenText) return;
    speechQueueRef.current.push(spokenText);
    void drainSpeechQueue();
  }

  async function replayAudio() {
    if (audioElementRef.current && audioUrl) {
      interruptSpeechPlayback();
      try {
        await audioElementRef.current.play();
        appendLog("[audio] replay started");
      } catch {
        appendLog("[audio] replay was blocked; use the audio controls below");
      }
      return;
    }

    await speakStreamingText(finalResponse, { interruptCurrent: true });
  }

  async function uploadRecordedAudio(blob: Blob) {
    if (!backendBaseUrl) {
      return;
    }

    setTranscribing(true);
    appendLog(`POST ${backendBaseUrl}/audio/transcribe`);
    try {
      const formData = new FormData();
      const mimeType = blob.type || "audio/webm";
      formData.append("audio", blob, guessAudioFilename(mimeType));

      const response = await fetch(`${backendBaseUrl}/audio/transcribe`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        appendLog(`transcribe HTTP ${response.status} — ${await response.text()}`);
        setAssistantStatus("Transcription failed. Try again or type your request.");
        return;
      }

      const payload = (await response.json()) as { text?: unknown };
      const transcript =
        typeof payload.text === "string" ? payload.text.trim() : "";

      if (!transcript) {
        appendLog("[transcript] no text returned");
        setAssistantStatus(
          "I could not hear a clear transcript. Try again or type your request.",
        );
        return;
      }

      setRequest(transcript);
      appendLog(`[transcript] ${transcript}`);
      setAssistantStatus("Transcript ready. Edit it or send it.");
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      appendLog(`transcribe failed: ${message}`);
      setAssistantStatus("Transcription failed. Try again or type your request.");
    } finally {
      setTranscribing(false);
    }
  }

  async function startRecording() {
    if (!warmedUp) {
      setAssistantStatus("Run the warmup greeting first so the audio flow is ready.");
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setAssistantStatus("This browser does not support microphone capture.");
      return;
    }
    if (typeof MediaRecorder === "undefined") {
      setAssistantStatus("This browser does not support MediaRecorder.");
      return;
    }

    interruptSpeechPlayback();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      mediaStreamRef.current = stream;
      mediaRecorderRef.current = recorder;
      recordedChunksRef.current = [];

      recorder.addEventListener("dataavailable", (event) => {
        if (event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      });

      recorder.addEventListener("stop", () => {
        const mimeType =
          recorder.mimeType ||
          recordedChunksRef.current[0]?.type ||
          "audio/webm";
        const audioBlob = new Blob(recordedChunksRef.current, { type: mimeType });
        recordedChunksRef.current = [];
        mediaRecorderRef.current = null;
        stopMicrophoneStream();
        if (audioBlob.size === 0) {
          appendLog("[audio] no microphone audio was captured");
          setAssistantStatus("No audio was captured. Try again or type your request.");
          return;
        }
        void uploadRecordedAudio(audioBlob);
      });

      recorder.addEventListener("error", () => {
        appendLog("[audio] media recorder failed");
        setRecording(false);
        setTranscribing(false);
        stopMicrophoneStream();
      });

      recorder.start();
      setRecording(true);
      setAssistantStatus("Recording now. Press again when you are finished talking.");
      appendLog("[audio] microphone recording started");
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      appendLog(`microphone failed: ${message}`);
      setAssistantStatus("Microphone access failed. Type your request instead.");
      stopMicrophoneStream();
    }
  }

  async function toggleRecording() {
    if (recording) {
      const recorder = mediaRecorderRef.current;
      if (!recorder || recorder.state === "inactive") {
        setRecording(false);
        stopMicrophoneStream();
        return;
      }
      setRecording(false);
      setAssistantStatus("Transcribing audio...");
      recorder.stop();
      appendLog("[audio] microphone recording stopped");
      return;
    }

    await startRecording();
  }

  async function handleWarmup() {
    if (!backendBaseUrl) {
      appendLog(
        "NEXT_PUBLIC_API_BASE_URL is not configured, so the frontend does not know which backend to call.",
      );
      return;
    }

    abortRef.current?.abort();
    interruptSpeechPlayback();
    stopMicrophoneStream();
    setWarmingUp(true);
    setWarmedUp(false);
    setLog("");
    setRequest("");
    setFinalResponse("");
    replaceAudioUrl("");
    setAssistantStatus("Playing the welcome message...");
    appendLog(`POST ${backendBaseUrl}/session/warmup`);

    try {
      const response = await fetch(`${backendBaseUrl}/session/warmup`, {
        method: "POST",
      });
      if (!response.ok) {
        appendLog(`warmup HTTP ${response.status} — ${await response.text()}`);
        setAssistantStatus("Warmup failed. Check the backend and try again.");
        return;
      }

      const payload = (await response.json()) as WarmupResponse;
      const welcomeText =
        typeof payload.welcome_text === "string" ? payload.welcome_text.trim() : "";
      const readyPrompt =
        typeof payload.ready_prompt === "string"
          ? payload.ready_prompt.trim()
          : "Speak now.";

      setWarmedUp(true);
      if (welcomeText) {
        await speakStreamingText(welcomeText, { interruptCurrent: true });
      }
      setAssistantStatus(readyPrompt);
      appendLog(`[ready] ${readyPrompt}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      appendLog(`warmup failed: ${message}`);
      setAssistantStatus("Warmup failed. Check the backend and try again.");
    } finally {
      setWarmingUp(false);
    }
  }

  async function handleSendMessage() {
    if (!backendBaseUrl) {
      appendLog(
        "NEXT_PUBLIC_API_BASE_URL is not configured, so the frontend does not know which backend to call.",
      );
      return;
    }
    if (!warmedUp) {
      setAssistantStatus("Run the warmup greeting first.");
      return;
    }
    if (request.trim().length === 0) {
      setAssistantStatus("Type a message or record one before sending.");
      return;
    }

    abortRef.current?.abort();
    interruptSpeechPlayback();
    void ensureAudioContextReady();
    stopMicrophoneStream();
    const controller = new AbortController();
    abortRef.current = controller;

    setRunning(true);
    setLog("");
    setFinalResponse("");
    replaceAudioUrl("");
    setAssistantStatus("Pipeline running...");
    appendLog(`POST ${backendBaseUrl}/run`);

    try {
      const response = await fetch(`${backendBaseUrl}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_request: request.trim() }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        appendLog(`HTTP ${response.status} — ${await response.text()}`);
        setAssistantStatus("Pipeline request failed.");
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

          if (ev.event === "busy") {
            setAssistantStatus(
              extractMessage(ev.data) || "The backend is busy. Try again shortly.",
            );
            continue;
          }

          if (ev.event === "error") {
            setAssistantStatus(
              extractMessage(ev.data) || "The pipeline returned an error.",
            );
            continue;
          }

          if (ev.event === "narration") {
            const narrationText = extractNarrationText(ev.data);
            if (narrationText) {
              queueSpeechText(narrationText);
            }
            continue;
          }

          if (ev.event === "completed") {
            const responseText = extractFinalResponse(ev.data);
            setFinalResponse(responseText);
            if (responseText) {
              queueSpeechText(responseText);
              setAssistantStatus("Final response ready.");
            } else {
              setAssistantStatus("Pipeline completed without a final response string.");
              appendLog("[audio] completed event did not include a response string");
            }
          }
        }
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      appendLog(`stream ended: ${message}`);
      setAssistantStatus("The pipeline stream ended early.");
    } finally {
      setRunning(false);
    }
  }

  const controlsDisabled =
    warmingUp || running || recording || transcribing || !backendBaseUrl;

  return (
    <main className="page-shell">
      <section className="card">
        <p className="eyebrow">NHTSA Project</p>
        <h1>Hosted audio pipeline</h1>
        <p className="body-copy">
          Run pipeline now warms up the hosted voice flow with the same greeting
          the CLI uses. After that, you can either type a request or record one,
          then send it into <code>/run</code> and hear both intermediate
          narration and the final response through streamed TTS.
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

        <div
          className="status-box"
          data-state={
            recording
              ? "recording"
              : running
                ? "running"
                : warmedUp
                  ? "ready"
                  : "idle"
          }
        >
          {assistantStatus}
        </div>

        <textarea
          value={request}
          onChange={(e) => setRequest(e.target.value)}
          rows={4}
          className="request-input"
          placeholder="Type what you want to say, or use the push-to-talk button below."
          disabled={!warmedUp || warmingUp || running || recording || transcribing}
        />

        <div className="action-row">
          <button
            type="button"
            onClick={handleWarmup}
            disabled={controlsDisabled || speaking}
          >
            {warmingUp ? "Warming up..." : "Run pipeline"}
          </button>
          <button
            type="button"
            onClick={handleSendMessage}
            disabled={
              !warmedUp ||
              warmingUp ||
              running ||
              recording ||
              transcribing ||
              request.trim().length === 0 ||
              !backendBaseUrl
            }
          >
            {running ? "Sending..." : "Send message"}
          </button>
          <button
            type="button"
            onClick={toggleRecording}
            disabled={warmingUp || running || transcribing || !backendBaseUrl}
            className={recording ? "recording-button" : undefined}
          >
            {recording
              ? "Press when finished talking"
              : transcribing
                ? "Transcribing..."
                : "Press to begin talking"}
          </button>
          <button
            type="button"
            onClick={replayAudio}
            disabled={warmingUp || running || speaking || !finalResponse || !backendBaseUrl}
          >
            {speaking ? "Speaking..." : "Replay audio"}
          </button>
        </div>

        {audioUrl ? (
          <audio
            ref={audioElementRef}
            className="audio-player"
            src={audioUrl}
            controls
            preload="metadata"
          />
        ) : null}

        <pre className="event-log">{log || "(no events yet)"}</pre>
      </section>
    </main>
  );
}
