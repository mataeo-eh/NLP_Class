"use client";

// Minimal hosted-audio smoke-test UI for the FastAPI backend.
//
// This page now exercises the full hosted browser contract:
//   1. warm up with the CLI greeting text
//   2. accept either typed input or push-to-talk audio upload
//   3. stream per-node narration audio plus the final answer

import { useEffect, useRef, useState } from "react";

const configuredBackendBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
const backendBaseUrl = configuredBackendBaseUrl ?? "";

type SseEvent = {
  event: string;
  data: string;
};

type ChatTurn = {
  role: "user" | "assistant";
  text: string;
};

type WarmupResponse = {
  welcome_text?: unknown;
  ready_prompt?: unknown;
};

type TtsVoiceOption = {
  id: string;
  label: string;
  description: string;
};

type TtsProviderOption = {
  id: string;
  label: string;
  description: string;
  supports_streaming: boolean;
  default_voice_preset: string;
  voices: TtsVoiceOption[];
};

type TtsOptionsResponse = {
  default_tts_provider: string;
  default_voice_preset: string;
  default_preview_text: string;
  providers: TtsProviderOption[];
};

type PromptTemplate = {
  id: string;
  route: string;
  route_label: string;
  label: string;
  description: string;
  prompt_text: string;
  is_followup: boolean;
  followup_for: string | null;
};

type WindowWithWebkitAudioContext = Window &
  typeof globalThis & {
    webkitAudioContext?: typeof AudioContext;
  };

const fallbackVoicePreviewText =
  "Hello, how are you doing on this fine day. " +
  "Were you able to find everything you were looking for?";

function createConversationId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function pcmChunkToAudioBuffer(
  audioContext: AudioContext,
  pcmBytes: Uint8Array,
  codec: string,
  sampleRate: number,
): AudioBuffer {
  if (codec === "pcm_f32le") {
    const alignedBuffer = new ArrayBuffer(pcmBytes.byteLength);
    new Uint8Array(alignedBuffer).set(pcmBytes);
    const floatSamples = new Float32Array(alignedBuffer);
    const audioBuffer = audioContext.createBuffer(1, floatSamples.length, sampleRate);
    audioBuffer.copyToChannel(floatSamples, 0, 0);
    return audioBuffer;
  }

  if (codec === "linear16") {
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

  if (codec === "pcm16") {
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

  throw new Error(`Unsupported streamed audio codec: ${codec}`);
}

function buildWaveBlobFromPcmChunks(
  pcmChunks: Uint8Array[],
  codec: string,
  sampleRate: number,
): Blob {
  const pcmByteLength = pcmChunks.reduce(
    (total, chunk) => total + chunk.byteLength,
    0,
  );
  const bytesPerSample = bytesPerSampleForCodec(codec);
  if (!bytesPerSample) {
    throw new Error(`Unsupported streamed audio codec: ${codec}`);
  }
  const bitsPerSample = bytesPerSample * 8;
  const blockAlign = bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const wavFormatTag = codec === "pcm_f32le" ? 3 : 1;
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
  view.setUint16(offset, wavFormatTag, true);
  offset += 2;
  view.setUint16(offset, 1, true);
  offset += 2;
  view.setUint32(offset, sampleRate, true);
  offset += 4;
  view.setUint32(offset, byteRate, true);
  offset += 4;
  view.setUint16(offset, blockAlign, true);
  offset += 2;
  view.setUint16(offset, bitsPerSample, true);
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

function bytesPerSampleForCodec(codec: string): number | null {
  if (codec === "linear16") {
    return 2;
  }
  if (codec === "pcm16") {
    return 2;
  }
  if (codec === "pcm_f32le") {
    return 4;
  }
  return null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function parseTtsOptionsResponse(payload: unknown): TtsOptionsResponse | null {
  if (!isRecord(payload)) {
    return null;
  }

  const defaultProvider = payload.default_tts_provider;
  const defaultVoicePreset = payload.default_voice_preset;
  const defaultPreviewText = payload.default_preview_text;
  const rawProviders = payload.providers;

  if (
    typeof defaultProvider !== "string" ||
    typeof defaultVoicePreset !== "string" ||
    typeof defaultPreviewText !== "string" ||
    !Array.isArray(rawProviders)
  ) {
    return null;
  }

  const providers: TtsProviderOption[] = [];
  for (const rawProvider of rawProviders) {
    if (!isRecord(rawProvider) || !Array.isArray(rawProvider.voices)) {
      return null;
    }
    if (
      typeof rawProvider.id !== "string" ||
      typeof rawProvider.label !== "string" ||
      typeof rawProvider.description !== "string" ||
      typeof rawProvider.supports_streaming !== "boolean" ||
      typeof rawProvider.default_voice_preset !== "string"
    ) {
      return null;
    }

    const voices: TtsVoiceOption[] = [];
    for (const rawVoice of rawProvider.voices) {
      if (
        !isRecord(rawVoice) ||
        typeof rawVoice.id !== "string" ||
        typeof rawVoice.label !== "string" ||
        typeof rawVoice.description !== "string"
      ) {
        return null;
      }
      voices.push({
        id: rawVoice.id,
        label: rawVoice.label,
        description: rawVoice.description,
      });
    }

    providers.push({
      id: rawProvider.id,
      label: rawProvider.label,
      description: rawProvider.description,
      supports_streaming: rawProvider.supports_streaming,
      default_voice_preset: rawProvider.default_voice_preset,
      voices,
    });
  }

  return {
    default_tts_provider: defaultProvider,
    default_voice_preset: defaultVoicePreset,
    default_preview_text: defaultPreviewText,
    providers,
  };
}

export default function HomePage() {
  const [request, setRequest] = useState("");
  const [demoPrompts, setDemoPrompts] = useState<PromptTemplate[]>([]);
  const [promptPanelOpen, setPromptPanelOpen] = useState(false);
  const [conversationId, setConversationId] = useState(() => createConversationId());
  const [canContinueSession, setCanContinueSession] = useState(false);
  const [chatTurns, setChatTurns] = useState<ChatTurn[]>([]);
  const [voiceSettingsOpen, setVoiceSettingsOpen] = useState(false);
  const [ttsOptionsLoading, setTtsOptionsLoading] = useState(false);
  const [ttsProviders, setTtsProviders] = useState<TtsProviderOption[]>([]);
  const [selectedTtsProvider, setSelectedTtsProvider] = useState("deepgram");
  const [selectedVoicePreset, setSelectedVoicePreset] = useState(
    "aura-2-hyperion-en",
  );
  const [voicePreviewText, setVoicePreviewText] = useState(fallbackVoicePreviewText);
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
  const ttsSelectionRef = useRef({
    ttsProvider: "deepgram",
    voicePreset: "aura-2-hyperion-en",
  });

  function appendLog(line: string) {
    const cleanedLine = line.trim();
    if (!cleanedLine) return;
    setLog((prev) => (prev ? `${prev}\n\n${cleanedLine}` : cleanedLine));
  }

  function appendChatTurn(turn: ChatTurn) {
    const cleanedText = turn.text.trim();
    if (!cleanedText) return;
    setChatTurns((prev) => [...prev, { role: turn.role, text: cleanedText }]);
  }

  function findProviderOption(providerId: string): TtsProviderOption | null {
    return ttsProviders.find((provider) => provider.id === providerId) ?? null;
  }

  function applyTtsSelection(providerId: string, voicePreset: string) {
    ttsSelectionRef.current = {
      ttsProvider: providerId,
      voicePreset,
    };
    setSelectedTtsProvider(providerId);
    setSelectedVoicePreset(voicePreset);
  }

  function buildSpeechRequestBody(
    text: string,
    selection?: { ttsProvider: string; voicePreset: string },
  ) {
    const activeSelection = selection ?? ttsSelectionRef.current;
    return {
      text,
      tts_provider: activeSelection.ttsProvider,
      voice_preset: activeSelection.voicePreset,
    };
  }

  useEffect(() => {
    if (!backendBaseUrl) return;
    let cancelled = false;
    async function loadPrompts() {
      try {
        const response = await fetch(`${backendBaseUrl}/prompts`);
        if (!response.ok || cancelled) return;
        const data = (await response.json()) as unknown;
        if (!cancelled && Array.isArray(data)) {
          setDemoPrompts(data as PromptTemplate[]);
        }
      } catch {
        // prompts are non-critical; silently ignore fetch errors
      }
    }
    void loadPrompts();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!backendBaseUrl) {
      return;
    }

    let cancelled = false;

    async function loadTtsOptions() {
      setTtsOptionsLoading(true);
      try {
        const response = await fetch(`${backendBaseUrl}/audio/tts/options`, {
          method: "GET",
        });
        if (!response.ok) {
          const detail = await response.text();
          appendLog(`tts options HTTP ${response.status} - ${detail}`);
          return;
        }

        const payload = parseTtsOptionsResponse(await response.json());
        if (!payload) {
          appendLog("tts options failed validation on the frontend");
          return;
        }

        if (cancelled) {
          return;
        }

        setTtsProviders(payload.providers);
        setVoicePreviewText(payload.default_preview_text || fallbackVoicePreviewText);

        const preferredProvider =
          payload.providers.find(
            (provider) => provider.id === payload.default_tts_provider,
          ) ?? payload.providers[0];
        if (!preferredProvider) {
          appendLog("tts options response did not include any providers");
          return;
        }

        const preferredVoice =
          preferredProvider.voices.find(
            (voice) => voice.id === payload.default_voice_preset,
          )?.id ??
          preferredProvider.default_voice_preset ??
          preferredProvider.voices[0]?.id;
        if (!preferredVoice) {
          appendLog(
            `tts provider ${preferredProvider.id} did not include any selectable voices`,
          );
          return;
        }

        applyTtsSelection(preferredProvider.id, preferredVoice);
      } catch (err) {
        if (cancelled) {
          return;
        }
        const message = err instanceof Error ? err.message : String(err);
        appendLog(`tts options failed: ${message}`);
      } finally {
        if (!cancelled) {
          setTtsOptionsLoading(false);
        }
      }
    }

    void loadTtsOptions();

    return () => {
      cancelled = true;
    };
  }, []);

  function tryParseJson(rawText: string): unknown | null {
    try {
      return JSON.parse(rawText);
    } catch {
      return null;
    }
  }

  function formatLogValue(value: unknown, indent = 0): string[] {
    const prefix = "  ".repeat(indent);

    if (value === null) {
      return [`${prefix}null`];
    }
    if (typeof value === "boolean" || typeof value === "number") {
      return [`${prefix}${String(value)}`];
    }
    if (typeof value === "string") {
      const normalized = value.trim().replace(/\r\n/g, "\n");
      if (!normalized) {
        return [`${prefix}""`];
      }
      return normalized
        .split("\n")
        .map((line) => `${prefix}${line}`);
    }
    if (Array.isArray(value)) {
      if (value.length === 0) {
        return [`${prefix}[]`];
      }
      return value.flatMap((item) => {
        const itemLines = formatLogValue(item, indent + 1);
        if (itemLines.length === 1) {
          return [`${prefix}- ${itemLines[0].trimStart()}`];
        }
        return [`${prefix}- ${itemLines[0].trimStart()}`, ...itemLines.slice(1)];
      });
    }
    if (typeof value === "object") {
      const entries = Object.entries(value);
      if (entries.length === 0) {
        return [`${prefix}{}`];
      }
      return entries.flatMap(([key, entryValue]) => {
        const valueLines = formatLogValue(entryValue, indent + 1);
        if (valueLines.length === 1) {
          return [`${prefix}${key}: ${valueLines[0].trimStart()}`];
        }
        return [`${prefix}${key}:`, ...valueLines];
      });
    }
    return [`${prefix}${String(value)}`];
  }

  function formatSseEventForLog(event: SseEvent): string {
    const parsedPayload = tryParseJson(event.data);
    if (parsedPayload === null) {
      return `[${event.event}] ${event.data.trim()}`;
    }
    // Render structured event data into a terminal-style outline rather than
    // JSON.stringify output so embedded newlines display as real whitespace
    // instead of literal `\n` escape sequences.
    return [`[${event.event}]`, ...formatLogValue(parsedPayload, 1)].join("\n");
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

  async function unlockAudioPlaybackFromUserGesture(): Promise<void> {
    // MDN's Web Audio autoplay guidance is explicit here: create or resume the
    // AudioContext from inside a user gesture. If we wait until after an async
    // fetch resolves, many browsers keep the context suspended and all later
    // streamed TTS scheduling is silent even though the backend keeps sending
    // valid PCM bytes.
    const audioContext = await ensureAudioContextReady();
    if (!audioContext) {
      appendLog(
        "[audio] browser has no AudioContext support; streamed autoplay is unavailable",
      );
      return;
    }

    if (audioContext.state !== "running") {
      appendLog(
        `[audio] context remained ${audioContext.state} after the user gesture; browser autoplay may still block playback`,
      );
    }
  }

  function schedulePcmChunk(
    audioContext: AudioContext,
    pcmBytes: Uint8Array,
    codec: string,
    sampleRate: number,
  ) {
    const audioBuffer = pcmChunkToAudioBuffer(
      audioContext,
      pcmBytes,
      codec,
      sampleRate,
    );
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

  function extractSessionId(eventData: string): string {
    try {
      const payload = JSON.parse(eventData) as { session_id?: unknown };
      return typeof payload.session_id === "string" ? payload.session_id.trim() : "";
    } catch {
      return "";
    }
  }

  function extractTaskType(eventData: string): string {
    try {
      const payload = JSON.parse(eventData) as { task_type?: unknown };
      return typeof payload.task_type === "string" ? payload.task_type.trim() : "";
    } catch {
      return "";
    }
  }

  async function speakBufferedText(text: string) {
    const spokenText = text.trim();
    if (!backendBaseUrl || !spokenText) return;

    const activeSelection = { ...ttsSelectionRef.current };
    appendLog(
      `POST ${backendBaseUrl}/audio/speech (${activeSelection.ttsProvider}:${activeSelection.voicePreset})`,
    );
    setSpeaking(true);
    try {
      const response = await fetch(`${backendBaseUrl}/audio/speech`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildSpeechRequestBody(spokenText, activeSelection)),
      });

      if (!response.ok) {
        appendLog(`audio HTTP ${response.status} - ${await response.text()}`);
        return;
      }

      const blob = await response.blob();
      const nextUrl = URL.createObjectURL(blob);
      replaceAudioUrl(nextUrl);

      const audio = audioElementRef.current ?? new Audio(nextUrl);
      try {
        await audio.play();
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

    const activeSelection = { ...ttsSelectionRef.current };
    const providerOption = findProviderOption(activeSelection.ttsProvider);
    if (!providerOption?.supports_streaming) {
      await speakBufferedText(spokenText);
      return;
    }

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

    try {
      appendLog(
        `POST ${backendBaseUrl}/audio/speech/stream (${activeSelection.ttsProvider}:${activeSelection.voicePreset})`,
      );
      const response = await fetch(`${backendBaseUrl}/audio/speech/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildSpeechRequestBody(spokenText, activeSelection)),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        appendLog(`audio HTTP ${response.status} - ${await response.text()}`);
        return;
      }

      const codec = response.headers.get("x-audio-codec")?.trim().toLowerCase();
      const sampleRateHeader = response.headers.get("x-audio-sample-rate");
      const channelsHeader = response.headers.get("x-audio-channels");
      const sampleRate = Number(sampleRateHeader);
      const channels = Number(channelsHeader);
      const bytesPerSample = codec ? bytesPerSampleForCodec(codec) : null;

      if (
        !codec ||
        !bytesPerSample ||
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

        const remainder = combined.byteLength % bytesPerSample;
        if (remainder !== 0) {
          carry = combined.slice(combined.byteLength - remainder);
          combined = combined.slice(0, combined.byteLength - remainder);
        }

        if (combined.byteLength === 0) continue;

        pcmChunks.push(combined);
        schedulePcmChunk(audioContext, combined, codec, sampleRate);
      }

      if (carry.byteLength > 0) {
        appendLog(
          `[audio] ignored ${carry.byteLength} trailing byte(s) from the raw audio stream because a complete ${codec} sample was not available yet`,
        );
      }

      if (pcmChunks.length === 0) {
        appendLog("[audio] stream completed without audio bytes");
        return;
      }

      const nextUrl = URL.createObjectURL(
        buildWaveBlobFromPcmChunks(pcmChunks, codec, sampleRate),
      );
      replaceAudioUrl(nextUrl);
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

  function handleTtsProviderChange(nextProviderId: string) {
    const providerOption = findProviderOption(nextProviderId);
    const nextVoicePreset =
      providerOption?.default_voice_preset ?? providerOption?.voices[0]?.id ?? "";
    if (!providerOption || !nextVoicePreset) {
      return;
    }
    applyTtsSelection(nextProviderId, nextVoicePreset);
  }

  function handleVoicePresetChange(nextVoicePreset: string) {
    if (!nextVoicePreset.trim()) {
      return;
    }
    applyTtsSelection(selectedTtsProvider, nextVoicePreset);
  }

  async function previewSelectedVoice() {
    const previewText = voicePreviewText.trim();
    if (!previewText) {
      setAssistantStatus("Enter preview text before playing a voice sample.");
      return;
    }

    await unlockAudioPlaybackFromUserGesture();
    interruptSpeechPlayback();
    await speakBufferedText(previewText);
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
        appendLog(`transcribe HTTP ${response.status} - ${await response.text()}`);
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
      appendLog("[audio] sending transcript directly to /run");
      setAssistantStatus("Transcript captured. Sending it to the pipeline...");
      await runPipelineRequest(transcript);
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

    await unlockAudioPlaybackFromUserGesture();

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
        appendLog(`warmup HTTP ${response.status} - ${await response.text()}`);
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
    await runPipelineRequest(request);
  }

  function startNewConversation() {
    abortRef.current?.abort();
    interruptSpeechPlayback();
    stopMicrophoneStream();
    setConversationId(createConversationId());
    setCanContinueSession(false);
    setChatTurns([]);
    setLog("");
    setFinalResponse("");
    replaceAudioUrl("");
    setRequest("");
    setAssistantStatus(
      warmedUp
        ? "Started a new conversation. Send a fresh request when ready."
        : "Press Run pipeline to hear the greeting and unlock the hosted audio flow.",
    );
  }

  async function runPipelineRequest(userRequest: string) {
    const trimmedRequest = userRequest.trim();
    const continuingThisTurn = canContinueSession;

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
    if (trimmedRequest.length === 0) {
      setAssistantStatus("Type a message or record one before sending.");
      return;
    }

    abortRef.current?.abort();
    interruptSpeechPlayback();
    await unlockAudioPlaybackFromUserGesture();
    stopMicrophoneStream();
    appendChatTurn({ role: "user", text: trimmedRequest });
    setRequest("");
    const controller = new AbortController();
    abortRef.current = controller;

    setRunning(true);
    setFinalResponse("");
    replaceAudioUrl("");
    setAssistantStatus(
      continuingThisTurn
        ? "Continuing the active agentic conversation..."
        : "Pipeline running...",
    );
    if (!continuingThisTurn) {
      setLog("");
    } else {
      appendLog("----- follow-up -----");
    }
    appendLog(
      `POST ${backendBaseUrl}/run (${continuingThisTurn ? "continue" : "new"} conversation ${conversationId})`,
    );

    try {
      const response = await fetch(`${backendBaseUrl}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_request: trimmedRequest,
          session_id: conversationId,
          continue_session: continuingThisTurn,
        }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        if (response.status === 409) {
          setCanContinueSession(false);
        }
        appendLog(`HTTP ${response.status} - ${await response.text()}`);
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
          appendLog(formatSseEventForLog(ev));

          if (ev.event === "busy") {
            setCanContinueSession(continuingThisTurn);
            setAssistantStatus(
              extractMessage(ev.data) || "The backend is busy. Try again shortly.",
            );
            continue;
          }

          if (ev.event === "error") {
            setCanContinueSession(continuingThisTurn);
            setAssistantStatus(
              extractMessage(ev.data) || "The pipeline returned an error.",
            );
            continue;
          }

          if (ev.event === "started") {
            const startedSessionId = extractSessionId(ev.data);
            if (startedSessionId) {
              setConversationId(startedSessionId);
            }
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
            const taskType = extractTaskType(ev.data);
            const resumableAgenticConversation =
              taskType === "agentic_retrieve_and_analyze";
            setCanContinueSession(resumableAgenticConversation);
            setFinalResponse(responseText);
            if (responseText) {
              appendChatTurn({ role: "assistant", text: responseText });
              queueSpeechText(responseText);
              setAssistantStatus(
                resumableAgenticConversation
                  ? "Response ready. You can ask a follow-up in the same agentic conversation."
                  : "Final response ready.",
              );
            } else {
              setAssistantStatus("Pipeline completed without a final response string.");
              appendLog("[audio] completed event did not include a response string");
            }
          }
        }
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        appendLog("stream cancelled");
      } else {
        const message = err instanceof Error ? err.message : String(err);
        appendLog(`stream ended: ${message}`);
        setAssistantStatus("The pipeline stream ended early.");
      }
    } finally {
      setRunning(false);
    }
  }

  const currentTtsProviderOption = findProviderOption(selectedTtsProvider);
  const currentVoiceOptions = currentTtsProviderOption?.voices ?? [];
  const controlsDisabled =
    warmingUp || running || recording || transcribing || !backendBaseUrl;
  const ttsSettingsDisabled =
    warmingUp || recording || transcribing || !backendBaseUrl || ttsOptionsLoading;
  const statusState = recording
    ? "recording"
    : running
      ? "running"
      : warmedUp
        ? "ready"
        : "idle";
  const currentPhase = recording
    ? "Recording"
    : transcribing
      ? "Transcribing"
      : running
        ? "Analyzing"
        : warmingUp
          ? "Warming up"
          : warmedUp
            ? "Ready"
            : "Idle";
  const backendStateLabel = backendBaseUrl ? "Connected" : "Missing";
  const audioStateLabel = speaking ? "Speaking" : audioUrl ? "Audio ready" : "Silent";
  const sessionModeLabel = canContinueSession ? "Follow-up mode" : "New request mode";

  return (
    <main className="page-shell">
      <section className="top-bar" aria-label="Project header">
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true">
            N
          </span>
          <div>
            <p className="eyebrow">NHTSA Project</p>
            <h1>Voice analysis console</h1>
          </div>
        </div>
        <div className="backend-pill" data-connected={backendBaseUrl ? "true" : "false"}>
          <span className="status-dot" aria-hidden="true" />
          {backendStateLabel}
        </div>
      </section>

      <section className="hero-band" aria-label="Pipeline overview">
        <div>
          <p className="section-kicker">Hosted audio pipeline</p>
          <p className="hero-copy">
            Warm up the voice flow, send typed or recorded requests, and review
            streamed narration from the NHTSA analysis backend.
          </p>
        </div>
        <div className="signal-strip" aria-hidden="true">
          <span />
          <span />
          <span />
          <span />
        </div>
      </section>

      <section className="dashboard-layout">
        <aside className="side-panel" aria-label="Session status">
          <div className="status-box" data-state={statusState}>
            <p className="status-label">{currentPhase}</p>
            <p className="status-message">{assistantStatus}</p>
          </div>

          <div className="metric-grid">
            <div className="metric-tile">
              <span>Backend</span>
              <strong>{backendStateLabel}</strong>
            </div>
            <div className="metric-tile">
              <span>Audio</span>
              <strong>{audioStateLabel}</strong>
            </div>
            <div className="metric-tile metric-wide">
              <span>Session</span>
              <strong>{sessionModeLabel}</strong>
            </div>
          </div>

          <div className="session-panel">
            <p className="panel-title">Conversation ID</p>
            <code>{conversationId}</code>
          </div>

          <div className="backend-panel">
            <p className="panel-title">Backend base URL</p>
            {backendBaseUrl ? (
              <a href={backendBaseUrl} target="_blank" rel="noreferrer">
                {backendBaseUrl}
              </a>
            ) : (
              <strong>missing NEXT_PUBLIC_API_BASE_URL</strong>
            )}
          </div>
        </aside>

        <section className="workspace-panel" aria-label="Conversation workspace">
          <div className="panel-heading">
            <div>
              <p className="section-kicker">Transcript</p>
              <h2>Conversation</h2>
            </div>
            {canContinueSession ? (
              <span className="mode-badge">Continuing session</span>
            ) : (
              <span className="mode-badge">Fresh request</span>
            )}
          </div>

          <div className="conversation-panel">
            {chatTurns.length === 0 ? (
              <div className="conversation-empty">
                <p>No transcript yet.</p>
                <span>
                  Warm up the pipeline, then type or record a request to start
                  the analysis.
                </span>
              </div>
            ) : (
              <div className="conversation-list">
                {chatTurns.map((turn, index) => (
                  <article
                    key={`${turn.role}-${index}`}
                    className="conversation-turn"
                    data-role={turn.role}
                  >
                    <p className="conversation-role">
                      {turn.role === "user" ? "You" : "Assistant"}
                    </p>
                    <p className="conversation-text">{turn.text}</p>
                  </article>
                ))}
              </div>
            )}
          </div>

          {demoPrompts.length > 0 ? (
            <section className="voice-settings-panel" aria-label="Demo prompt templates">
              <div className="panel-heading compact">
                <div>
                  <p className="section-kicker">Templates</p>
                  <h2>Demo prompts</h2>
                </div>
                <button
                  type="button"
                  className="button button-ghost button-inline"
                  onClick={() => setPromptPanelOpen((prev) => !prev)}
                >
                  {promptPanelOpen ? "Hide templates" : "Choose template"}
                </button>
              </div>

              {promptPanelOpen ? (
                <div style={{ display: "flex", flexDirection: "column", gap: "10px", marginTop: "8px" }}>
                  {demoPrompts.map((pt) => (
                    <article
                      key={pt.id}
                      style={{
                        padding: "12px 14px",
                        border: "1px solid rgba(16,24,32,0.12)",
                        borderRadius: "var(--radius)",
                        background: "var(--panel-tint)",
                      }}
                    >
                      <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "4px", flexWrap: "wrap" }}>
                        <span
                          className="mode-badge"
                          style={{ fontSize: "0.74rem", minHeight: "24px", padding: "0 8px" }}
                        >
                          {pt.route_label}
                        </span>
                        {pt.is_followup ? (
                          <span
                            className="mode-badge"
                            style={{ fontSize: "0.74rem", minHeight: "24px", padding: "0 8px", background: "rgba(198,122,22,0.12)", color: "var(--amber)" }}
                          >
                            Follow-up
                          </span>
                        ) : null}
                        <strong style={{ fontSize: "0.92rem", color: "var(--navy)" }}>{pt.label}</strong>
                      </div>
                      <p style={{ margin: "0 0 10px", fontSize: "0.85rem", color: "var(--muted-ink)", lineHeight: 1.45 }}>
                        {pt.description}
                      </p>
                      <button
                        type="button"
                        className="button button-secondary"
                        style={{ fontSize: "0.82rem", padding: "6px 14px" }}
                        onClick={() => {
                          setRequest(pt.prompt_text);
                          setPromptPanelOpen(false);
                        }}
                      >
                        Load prompt
                      </button>
                    </article>
                  ))}
                </div>
              ) : null}
            </section>
          ) : null}

          <label className="composer-label" htmlFor="request-input">
            Message
          </label>
          <textarea
            id="request-input"
            value={request}
            onChange={(e) => setRequest(e.target.value)}
            rows={4}
            className="request-input"
            placeholder="Ask about vehicle safety, recalls, or crash-analysis context."
            disabled={!warmedUp || warmingUp || running || recording || transcribing}
          />

          <section className="voice-settings-panel" aria-label="Voice settings">
            <div className="panel-heading compact">
              <div>
                <p className="section-kicker">Speech output</p>
                <h2>Voice preset</h2>
              </div>
              <button
                type="button"
                className="button button-ghost button-inline"
                onClick={() => setVoiceSettingsOpen((prev) => !prev)}
                disabled={!backendBaseUrl || ttsOptionsLoading}
              >
                {voiceSettingsOpen ? "Hide voice options" : "Choose voice"}
              </button>
            </div>

            {voiceSettingsOpen ? (
              <>
                <div className="tts-grid">
                  <label className="tts-control" htmlFor="tts-provider-select">
                    <span>TTS provider</span>
                    <select
                      id="tts-provider-select"
                      className="tts-select"
                      value={selectedTtsProvider}
                      onChange={(event) => handleTtsProviderChange(event.target.value)}
                      disabled={ttsSettingsDisabled}
                    >
                      {ttsProviders.map((provider) => (
                        <option key={provider.id} value={provider.id}>
                          {provider.label}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="tts-control" htmlFor="tts-voice-select">
                    <span>Voice preset</span>
                    <select
                      id="tts-voice-select"
                      className="tts-select"
                      value={selectedVoicePreset}
                      onChange={(event) => handleVoicePresetChange(event.target.value)}
                      disabled={ttsSettingsDisabled || currentVoiceOptions.length === 0}
                    >
                      {currentVoiceOptions.map((voice) => (
                        <option key={voice.id} value={voice.id}>
                          {voice.label}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>

                {currentTtsProviderOption ? (
                  <p className="tts-helper">
                    {currentTtsProviderOption.description}
                    {currentTtsProviderOption.supports_streaming
                      ? " Live streamed playback is available for this provider."
                      : " This provider currently uses buffered WAV playback in the hosted UI."}
                  </p>
                ) : null}

                <label className="composer-label" htmlFor="voice-preview-input">
                  Preview text
                </label>
                <textarea
                  id="voice-preview-input"
                  value={voicePreviewText}
                  onChange={(event) => setVoicePreviewText(event.target.value)}
                  rows={3}
                  className="request-input request-input-compact"
                  placeholder={fallbackVoicePreviewText}
                  disabled={ttsSettingsDisabled}
                />

                <div className="action-row action-row-compact">
                  <button
                    type="button"
                    className="button button-secondary"
                    onClick={previewSelectedVoice}
                    disabled={
                      ttsSettingsDisabled ||
                      speaking ||
                      running ||
                      voicePreviewText.trim().length === 0
                    }
                  >
                    {speaking ? "Playing preview..." : "Play preview"}
                  </button>
                </div>
              </>
            ) : null}
          </section>

          <div className="action-row">
            <button
              type="button"
              className="button button-primary"
              onClick={handleWarmup}
              disabled={controlsDisabled || speaking}
            >
              {warmingUp ? "Warming up..." : "Run pipeline"}
            </button>
            <button
              type="button"
              className="button button-primary button-accent"
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
              className="button button-secondary"
              onClick={toggleRecording}
              disabled={warmingUp || running || transcribing || !backendBaseUrl}
              data-recording={recording ? "true" : "false"}
            >
              {recording
                ? "Finish recording"
                : transcribing
                  ? "Transcribing..."
                  : "Record voice"}
            </button>
            <button
              type="button"
              className="button button-secondary"
              onClick={replayAudio}
              disabled={warmingUp || running || speaking || !finalResponse || !backendBaseUrl}
            >
              {speaking ? "Speaking..." : "Replay audio"}
            </button>
            <button
              type="button"
              className="button button-ghost"
              onClick={startNewConversation}
              disabled={warmingUp || running || recording || transcribing}
            >
              New conversation
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
        </section>
      </section>

      <section className="log-panel" aria-label="Backend event stream">
        <div className="panel-heading compact">
          <div>
            <p className="section-kicker">Live stream</p>
            <h2>Event log</h2>
          </div>
          <span className="log-count">{log ? "Events received" : "Waiting"}</span>
        </div>
        <pre className="event-log">{log || "(no events yet)"}</pre>
      </section>
    </main>
  );
}
