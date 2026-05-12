"use client";

// Minimal hosted-audio smoke-test UI for the FastAPI backend.
//
// This page now exercises the full hosted browser contract:
//   1. warm up with the CLI greeting text
//   2. accept either typed input or push-to-talk audio upload
//   3. stream per-node narration audio plus the final answer

import { useEffect, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const configuredBackendBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
const backendBaseUrl = configuredBackendBaseUrl ?? "";

type SseEvent = {
  event: string;
  data: string;
};

type ChatTurn = {
  role: "user" | "assistant" | "narrator";
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

type ActiveNarrationPlayback = {
  controller: AbortController | null;
  cleanup: (() => void) | null;
  audioElement: HTMLAudioElement | null;
  sources: Set<AudioBufferSourceNode>;
  resolveFinished: (() => void) | null;
  scheduleComplete: boolean;
  settled: boolean;
};

// ---------------------------------------------------------------------------
// Chart panel types and component
// ---------------------------------------------------------------------------

type ChartData = {
  chart_name: string;
  summary: Record<string, unknown>;
};

const CHART_LABELS: Record<string, string> = {
  create_bar_chart:                        "LLM Subsystem Labels",
  create_model_year_chart:                 "Top Vehicles by Complaints",
  create_human_subsystem_frequency_chart:  "Component Frequency",
  compare_LLM_to_NHTSA:                   "LLM vs NHTSA Agreement",
  create_subsystem_frequency_by_make_chart:"Components by Make",
  create_subsystem_safety_signal_chart:    "Safety Signals by Subsystem",
  create_model_year_trend_chart:           "Complaints by Model Year",
};

const CHART_COLORS = ["#0f6b52", "#275c7d", "#c67a16", "#b5333f", "#172b3a"];

function ChartViewer({
  charts,
  index,
  onIndexChange,
}: {
  charts: ChartData[];
  index: number;
  onIndexChange: (i: number) => void;
}) {
  const active = charts[index];
  if (!active) return null;

  const { chart_name, summary } = active;
  const label = CHART_LABELS[chart_name] ?? chart_name;

  function renderChart() {
    // ---- create_bar_chart ------------------------------------------------
    if (chart_name === "create_bar_chart") {
      const valueCounts = (summary.value_counts ?? {}) as Record<string, Record<string, number>>;
      const colName = Object.keys(valueCounts)[0] ?? "";
      const counts = valueCounts[colName] ?? {};
      const data = Object.entries(counts)
        .map(([name, value]) => ({ name, value }))
        .sort((a, b) => b.value - a.value);
      return (
        <ResponsiveContainer width="100%" height={Math.max(200, data.length * 28)}>
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24, top: 4, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(16,24,32,0.08)" />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="name" width={140} tick={{ fontSize: 10 }} />
            <Tooltip formatter={(v) => [typeof v === "number" ? v.toLocaleString() : String(v ?? ""), "Count"]} />
            <Bar dataKey="value" radius={[0, 3, 3, 0]}>
              {data.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      );
    }

    // ---- create_model_year_chart -----------------------------------------
    if (chart_name === "create_model_year_chart") {
      const raw = (summary.top_5_make_model ?? []) as { vehicle: string; count: number }[];
      const data = [...raw].reverse();
      return (
        <ResponsiveContainer width="100%" height={Math.max(200, data.length * 40)}>
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24, top: 4, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(16,24,32,0.08)" />
            <XAxis type="number" tick={{ fontSize: 11 }} tickFormatter={(v) => typeof v === "number" ? v.toLocaleString() : String(v)} />
            <YAxis type="category" dataKey="vehicle" width={160} tick={{ fontSize: 9 }} />
            <Tooltip formatter={(v) => [typeof v === "number" ? v.toLocaleString() : String(v ?? ""), "Complaints"]} />
            <Bar dataKey="count" fill="#0f6b52" radius={[0, 3, 3, 0]} />
          </BarChart>
        </ResponsiveContainer>
      );
    }

    // ---- create_human_subsystem_frequency_chart --------------------------
    if (chart_name === "create_human_subsystem_frequency_chart") {
      const raw = (summary.component_counts ?? []) as { component: string; count: number }[];
      const data = raw.map(d => ({ name: d.component, value: d.count })).reverse();
      return (
        <ResponsiveContainer width="100%" height={Math.max(200, data.length * 26)}>
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24, top: 4, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(16,24,32,0.08)" />
            <XAxis type="number" tick={{ fontSize: 11 }} tickFormatter={(v) => typeof v === "number" ? v.toLocaleString() : String(v)} />
            <YAxis type="category" dataKey="name" width={160} tick={{ fontSize: 9 }} />
            <Tooltip formatter={(v) => [typeof v === "number" ? v.toLocaleString() : String(v ?? ""), "Complaints"]} />
            <Bar dataKey="value" fill="#275c7d" radius={[0, 3, 3, 0]} />
          </BarChart>
        </ResponsiveContainer>
      );
    }

    // ---- compare_LLM_to_NHTSA -------------------------------------------
    if (chart_name === "compare_LLM_to_NHTSA") {
      const ac = (summary.agreement_counts ?? { full: 0, partial: 0, none: 0 }) as {
        full: number; partial: number; none: number;
      };
      const pct = (summary.percentages ?? { full: 0, partial: 0, none: 0 }) as {
        full: number; partial: number; none: number;
      };
      const data = [
        { name: "Full", value: ac.full, pct: pct.full },
        { name: "Partial", value: ac.partial, pct: pct.partial },
        { name: "None", value: ac.none, pct: pct.none },
      ];
      const colors = ["#0f6b52", "#c67a16", "#b5333f"];
      return (
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={data} margin={{ left: 8, right: 24, top: 4, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(16,24,32,0.08)" />
            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip formatter={(v, _name, item) => {
              const num = typeof v === "number" ? v : 0;
              const pct = typeof item?.payload?.pct === "number" ? item.payload.pct : 0;
              return [`${num} (${pct.toFixed(1)}%)`, "Complaints"];
            }} />
            <Bar dataKey="value" radius={[3, 3, 0, 0]}>
              {data.map((_, i) => <Cell key={i} fill={colors[i]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      );
    }

    // ---- create_subsystem_frequency_by_make_chart -----------------------
    if (chart_name === "create_subsystem_frequency_by_make_chart") {
      type MakePlotted = { make: string; complaint_count: number; top_components: { component: string; count: number }[] };
      const makes = (summary.makes_plotted ?? []) as MakePlotted[];
      if (makes.length === 0) return <p className="chart-empty">No make data available.</p>;
      // Show first make; tabs handled by the index nav above
      const make = makes[0];
      const data = make.top_components.map(c => ({ name: c.component, value: c.count })).reverse();
      return (
        <>
          <p className="chart-make-label">{make.make} <span>({make.complaint_count.toLocaleString()} complaints)</span></p>
          <ResponsiveContainer width="100%" height={Math.max(180, data.length * 26)}>
            <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24, top: 4, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(16,24,32,0.08)" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 9 }} />
              <Tooltip formatter={(v) => [typeof v === "number" ? v.toLocaleString() : String(v ?? ""), "Complaints"]} />
              <Bar dataKey="value" fill="#172b3a" radius={[0, 3, 3, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </>
      );
    }

    // ---- create_subsystem_safety_signal_chart ---------------------------
    if (chart_name === "create_subsystem_safety_signal_chart") {
      type CompRow = { component: string; complaints: number; crash_rate_pct: number; fire_rate_pct: number };
      const raw = (summary.components ?? []) as CompRow[];
      const data = raw.map(c => ({ name: c.component, complaints: c.complaints, crash: c.crash_rate_pct, fire: c.fire_rate_pct })).reverse();
      return (
        <ResponsiveContainer width="100%" height={Math.max(200, data.length * 26)}>
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24, top: 4, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(16,24,32,0.08)" />
            <XAxis type="number" tick={{ fontSize: 11 }} tickFormatter={(v) => typeof v === "number" ? v.toLocaleString() : String(v)} />
            <YAxis type="category" dataKey="name" width={155} tick={{ fontSize: 9 }} />
            <Tooltip formatter={(v, name) => {
              const num = typeof v === "number" ? v : 0;
              const n = String(name);
              return [
                n === "complaints" ? num.toLocaleString() : `${num.toFixed(1)}%`,
                n === "complaints" ? "Complaints" : n === "crash" ? "Crash rate" : "Fire rate",
              ];
            }} />
            <Bar dataKey="complaints" fill="#275c7d" radius={[0, 3, 3, 0]} />
          </BarChart>
        </ResponsiveContainer>
      );
    }

    // ---- create_model_year_trend_chart ----------------------------------
    if (chart_name === "create_model_year_trend_chart") {
      const raw = (summary.top_5_years ?? []) as { year: number; count: number }[];
      const data = raw.map(y => ({ name: String(y.year), value: y.count }));
      return (
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data} margin={{ left: 8, right: 24, top: 4, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(16,24,32,0.08)" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => typeof v === "number" ? v.toLocaleString() : String(v)} />
            <Tooltip formatter={(v) => [typeof v === "number" ? v.toLocaleString() : String(v ?? ""), "Complaints"]} />
            <Bar dataKey="value" fill="#0f6b52" radius={[3, 3, 0, 0]}>
              {data.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      );
    }

    return <p className="chart-empty">No renderer for chart type: {chart_name}</p>;
  }

  return (
    <div className="chart-viewer">
      <div className="chart-viewer-header">
        <p className="section-kicker">Chart output</p>
        <h2>{label}</h2>
        {charts.length > 1 && (
          <div className="chart-tab-row">
            {charts.map((c, i) => (
              <button
                key={c.chart_name}
                type="button"
                className={`chart-tab${i === index ? " chart-tab-active" : ""}`}
                onClick={() => onIndexChange(i)}
              >
                {CHART_LABELS[c.chart_name] ?? c.chart_name}
              </button>
            ))}
          </div>
        )}
      </div>
      <div className="chart-viewport">
        {renderChart()}
      </div>
    </div>
  );
}

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
  const [awaitingClarification, setAwaitingClarification] = useState(false);
  const [pendingClarificationQuestion, setPendingClarificationQuestion] = useState("");
  const [running, setRunning] = useState(false);
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [narrationPaused, setNarrationPaused] = useState(false);
  const [assistantStatus, setAssistantStatus] = useState(
    "Press Start session to begin. The hosted controls stay idle until the welcome audio runs.",
  );
  const [log, setLog] = useState("");
  const [finalResponse, setFinalResponse] = useState("");
  const [audioUrl, setAudioUrl] = useState("");
  const [activeCharts, setActiveCharts] = useState<ChartData[]>([]);
  const [chartPanelIndex, setChartPanelIndex] = useState(0);

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
  const activeNarrationPlaybackRef = useRef<ActiveNarrationPlayback | null>(null);
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

  function createNarrationPlayback(
    controller: AbortController | null,
  ): { playback: ActiveNarrationPlayback; finished: Promise<void> } {
    let resolveFinished = () => {};
    const finished = new Promise<void>((resolve) => {
      resolveFinished = resolve;
    });
    const playback: ActiveNarrationPlayback = {
      controller,
      cleanup: null,
      audioElement: null,
      sources: new Set(),
      resolveFinished,
      scheduleComplete: false,
      settled: false,
    };
    activeNarrationPlaybackRef.current = playback;
    setSpeaking(true);
    setNarrationPaused(false);
    return { playback, finished };
  }

  function settleNarrationPlayback(playback: ActiveNarrationPlayback) {
    if (playback.settled) {
      return;
    }
    playback.settled = true;
    playback.cleanup?.();
    playback.cleanup = null;
    if (activeNarrationPlaybackRef.current === playback) {
      activeNarrationPlaybackRef.current = null;
    }
    setNarrationPaused(false);
    playback.resolveFinished?.();
    playback.resolveFinished = null;
  }

  function stopScheduledPlayback(playback?: ActiveNarrationPlayback | null) {
    const sources = playback?.sources ?? activeSourcesRef.current;
    for (const source of sources) {
      try {
        source.stop();
      } catch {
        // AudioBufferSourceNode.stop() throws if the source already ended.
      }
      activeSourcesRef.current.delete(source);
    }
    sources.clear();
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
    const playback = activeNarrationPlaybackRef.current;
    playback?.controller?.abort();
    speechAbortRef.current?.abort();
    stopScheduledPlayback(playback);
    if (playback?.audioElement) {
      playback.audioElement.pause();
      playback.audioElement.currentTime = 0;
    }
    if (audioElementRef.current) {
      audioElementRef.current.pause();
      audioElementRef.current.currentTime = 0;
    }
    if (playback) {
      settleNarrationPlayback(playback);
    } else {
      setNarrationPaused(false);
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
      setAssistantStatus(
        "This browser cannot unlock streamed audio automatically. Use the replay controls if speech does not start.",
      );
      return;
    }

    if (audioContext.state !== "running") {
      appendLog(
        `[audio] context remained ${audioContext.state} after the user gesture; browser autoplay may still block playback`,
      );
      setAssistantStatus(
        "Audio playback may still be blocked by the browser. If speech stays silent, use Replay audio or Play preview.",
      );
    }
  }

  function schedulePcmChunk(
    audioContext: AudioContext,
    pcmBytes: Uint8Array,
    codec: string,
    sampleRate: number,
    playback: ActiveNarrationPlayback,
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
      playback.sources.delete(source);
      if (playback.scheduleComplete && playback.sources.size === 0) {
        settleNarrationPlayback(playback);
      }
    });
    activeSourcesRef.current.add(source);
    playback.sources.add(source);
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

  function extractClarificationQuestion(eventData: string): string {
    try {
      const payload = JSON.parse(eventData) as { question?: unknown };
      return typeof payload.question === "string" ? payload.question.trim() : "";
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

  // Parse the `charts` field from a `completed` SSE event payload.
  // The backend sends charts as a dict keyed by chart_name, where each value is
  // the structured summary dict computed by the @tool wrapper.  We convert that
  // into a flat array so ChartViewer can iterate over it with a simple index.
  function extractCharts(eventData: string): ChartData[] {
    try {
      const payload = JSON.parse(eventData) as { charts?: unknown };
      const charts = payload.charts;
      if (!charts || typeof charts !== "object" || Array.isArray(charts)) return [];
      return Object.entries(charts as Record<string, unknown>)
        .filter(([, v]) => v !== null && typeof v === "object")
        .map(([name, summary]) => ({
          chart_name: name,
          summary: summary as Record<string, unknown>,
        }));
    } catch {
      return [];
    }
  }

  async function speakBufferedText(text: string) {
    const spokenText = text.trim();
    if (!backendBaseUrl || !spokenText) return;

    const activeSelection = { ...ttsSelectionRef.current };
    appendLog(
      `POST ${backendBaseUrl}/audio/speech (${activeSelection.ttsProvider}:${activeSelection.voicePreset})`,
    );
    const { playback, finished } = createNarrationPlayback(null);
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
      playback.audioElement = audio;
      playback.cleanup = () => {
        audio.removeEventListener("ended", handleEnded);
        audio.removeEventListener("error", handleError);
      };
      function handleEnded() {
        settleNarrationPlayback(playback);
      }
      function handleError() {
        settleNarrationPlayback(playback);
      }
      audio.addEventListener("ended", handleEnded);
      audio.addEventListener("error", handleError);
      playback.scheduleComplete = true;
      try {
        await audio.play();
      } catch {
        appendLog("[audio] browser blocked autoplay; use the audio controls below");
        setAssistantStatus(
          "Browser autoplay was blocked. Use the audio controls below to play the response.",
        );
        settleNarrationPlayback(playback);
      }
      await finished;
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        appendLog("[audio] stream cancelled");
      } else {
        const message = err instanceof Error ? err.message : String(err);
        appendLog(`audio failed: ${message}`);
      }
    } finally {
      settleNarrationPlayback(playback);
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
    const { playback, finished } = createNarrationPlayback(controller);

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
        schedulePcmChunk(audioContext, combined, codec, sampleRate, playback);
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
      playback.scheduleComplete = true;
      if (playback.sources.size === 0) {
        settleNarrationPlayback(playback);
      }
      await finished;
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        appendLog("[audio] stream cancelled");
      } else {
        const message = err instanceof Error ? err.message : String(err);
        appendLog(`audio failed: ${message}`);
      }
    } finally {
      playback.scheduleComplete = true;
      if (speechAbortRef.current === controller) {
        speechAbortRef.current = null;
      }
      settleNarrationPlayback(playback);
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

  async function toggleNarrationPause() {
    const playback = activeNarrationPlaybackRef.current;
    if (!playback) {
      return;
    }

    if (narrationPaused) {
      if (playback.audioElement) {
        try {
          await playback.audioElement.play();
        } catch {
          appendLog("[audio] resume was blocked; use Replay audio or the native audio controls");
          setAssistantStatus(
            "Browser playback blocked the resume request. Use Replay audio or the native audio controls below.",
          );
          return;
        }
      }
      const audioContext = audioContextRef.current;
      if (audioContext && audioContext.state === "suspended") {
        await audioContext.resume();
      }
      appendLog("[audio] narration resumed");
      setNarrationPaused(false);
      return;
    }

    if (playback.audioElement && !playback.audioElement.paused) {
      playback.audioElement.pause();
    }
    const audioContext = audioContextRef.current;
    if (audioContext && audioContext.state === "running") {
      await audioContext.suspend();
    }
    appendLog("[audio] narration paused");
    setNarrationPaused(true);
  }

  function stopNarrationPlayback() {
    const playback = activeNarrationPlaybackRef.current;
    if (!playback) {
      return;
    }
    playback.controller?.abort();
    if (speechAbortRef.current === playback.controller) {
      speechAbortRef.current = null;
    }
    stopScheduledPlayback(playback);
    if (playback.audioElement) {
      playback.audioElement.pause();
      playback.audioElement.currentTime = 0;
    }
    settleNarrationPlayback(playback);
    setSpeaking(false);
    appendLog("[audio] narration stopped");
  }

  async function replayAudio() {
    if (audioElementRef.current && audioUrl) {
      interruptSpeechPlayback();
      try {
        await audioElementRef.current.play();
        appendLog("[audio] replay started");
      } catch {
        appendLog("[audio] replay was blocked; use the audio controls below");
        setAssistantStatus(
          "Browser playback is still blocked. Use the native audio controls below to start the clip manually.",
        );
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
    // Route previews through the same streamed AudioContext path as narration
    // and final responses. This keeps preview playback inside the browser flow
    // that already survives autoplay restrictions after one user gesture.
    await speakStreamingText(previewText, { interruptCurrent: true });
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
      appendLog(
        awaitingClarification
          ? "[audio] sending clarification transcript directly to /run"
          : "[audio] sending transcript directly to /run",
      );
      setAssistantStatus(
        awaitingClarification
          ? "Clarification captured. Sending it back to the pipeline..."
          : "Transcript captured. Sending it to the pipeline...",
      );
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
    setAwaitingClarification(false);
    setPendingClarificationQuestion("");
    setLog("");
    setRequest("");
    setFinalResponse("");
    replaceAudioUrl("");
    setAssistantStatus("Starting session and playing the welcome message...");
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
    setAwaitingClarification(false);
    setPendingClarificationQuestion("");
    setChatTurns([]);
    setLog("");
    setFinalResponse("");
    replaceAudioUrl("");
    setRequest("");
    setAssistantStatus(
      warmedUp
        ? "Started a new conversation. Send a fresh request when ready."
        : "Press Start session to begin. The hosted controls stay idle until the welcome audio runs.",
    );
  }

  async function runPipelineRequest(userRequest: string) {
    const trimmedRequest = userRequest.trim();
    const continuingThisTurn = canContinueSession;
    const answeringClarification = awaitingClarification;

    if (!backendBaseUrl) {
      appendLog(
        "NEXT_PUBLIC_API_BASE_URL is not configured, so the frontend does not know which backend to call.",
      );
      return;
    }
    if (!warmedUp) {
      setAssistantStatus("Press Start session to begin.");
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
      answeringClarification
        ? "Sending the clarification answer back to the pipeline..."
        : continuingThisTurn
          ? "Continuing the active conversation..."
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
            setAwaitingClarification(answeringClarification);
            setAssistantStatus(
              extractMessage(ev.data) || "The backend is busy. Try again shortly.",
            );
            continue;
          }

          if (ev.event === "error") {
            setCanContinueSession(continuingThisTurn);
            setAwaitingClarification(answeringClarification);
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
              appendChatTurn({ role: "narrator", text: narrationText });
              queueSpeechText(narrationText);
            }
            continue;
          }

          if (ev.event === "clarification_required") {
            const question = extractClarificationQuestion(ev.data);
            setAwaitingClarification(true);
            setPendingClarificationQuestion(question);
            setCanContinueSession(true);
            if (question) {
              appendChatTurn({ role: "narrator", text: question });
              queueSpeechText(question);
            }
            setAssistantStatus(
              question
                ? "Clarification needed. Reply by text or voice to continue the same session."
                : "Clarification needed. Reply to continue the same session.",
            );
            continue;
          }

          if (ev.event === "completed") {
            const responseText = extractFinalResponse(ev.data);
            const taskType = extractTaskType(ev.data);
            const resumableAgenticConversation =
              taskType === "agentic_retrieve_and_analyze";
            setAwaitingClarification(false);
            setPendingClarificationQuestion("");
            setCanContinueSession(resumableAgenticConversation);
            setFinalResponse(responseText);
            // Surface any charts the pipeline produced during this run.
            const newCharts = extractCharts(ev.data);
            if (newCharts.length > 0) {
              setActiveCharts(newCharts);
              setChartPanelIndex(0);
            }
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
    : awaitingClarification
      ? "ready"
    : running
      ? "running"
      : warmedUp
        ? "ready"
        : "idle";
  const currentPhase = recording
    ? "Recording"
    : awaitingClarification
      ? "Awaiting reply"
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
  const audioStateLabel = narrationPaused
    ? "Paused"
    : speaking
      ? "Speaking"
      : audioUrl
        ? "Audio ready"
        : "Silent";
  const sessionModeLabel = awaitingClarification
    ? "Clarification pending"
    : canContinueSession
      ? "Follow-up mode"
      : "New request mode";

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

      <section
        className="dashboard-layout"
        data-chart-open={activeCharts.length > 0 ? "true" : "false"}
      >
        <aside className="side-panel" aria-label="Session status">
          <div className="status-box" data-state={statusState}>
            <p className="status-label">{currentPhase}</p>
            <p className="status-message">{assistantStatus}</p>
            {awaitingClarification && pendingClarificationQuestion ? (
              <p className="status-note">
                Waiting on your answer: {pendingClarificationQuestion}
              </p>
            ) : null}
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
                  Press Start session, then type or record a request to begin
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
                    {turn.role === "user" ? null : (
                      <p className="conversation-role">
                        {turn.role === "assistant" ? "NHTSA AI Agent" : "Narrator"}
                      </p>
                    )}
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
              {warmingUp ? "Starting..." : "Start session"}
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
              className="button button-secondary"
              onClick={() => {
                void toggleNarrationPause();
              }}
              disabled={!speaking}
            >
              {narrationPaused ? "Resume narration" : "Pause narration"}
            </button>
            <button
              type="button"
              className="button button-secondary"
              onClick={stopNarrationPlayback}
              disabled={!speaking}
            >
              Stop narration
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

          <p className="controls-note">
            Press Start session first. The hosted voice flow does not begin until the
            welcome audio has unlocked the browser playback path.
          </p>

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

        {/* Chart panel — collapsed by default, expands via CSS when activeCharts.length > 0 */}
        <aside className="chart-panel" aria-label="Chart viewer">
          <ChartViewer
            charts={activeCharts}
            index={chartPanelIndex}
            onIndexChange={setChartPanelIndex}
          />
        </aside>
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
