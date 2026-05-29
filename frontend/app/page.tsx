"use client";

import { useState, useEffect, useRef } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  PieChart,
  Pie,
} from "recharts";

// ==========================================
// 1. FALLBACK HEATMAP CONFUSION MATRIX
// ==========================================
function FallbackMatrix({ id, label }: { id: string; label: string }) {
  const classes = ["Support Gen", "Fileservice", "Software", "O365", "Active Dir", "Computer"];
  
  // Custom static values modeled after real classification accuracy signatures
  const getValue = (row: number, col: number) => {
    if (id === "lora_b") {
      // Failure run: chaotic distributed false predictions, weak diagonal
      if (row === col) return Math.floor(10 + Math.random() * 8);
      return Math.floor(15 + Math.random() * 15);
    }
    
    // Successful run: heavy weights along the diagonal, minimal off-diagonal noise
    if (row === col) {
      const base = 
        id === "embeddings" ? 86 : 
        id === "tfidf" ? 71 : 
        id === "synthetic" ? 79 : 
        id === "gpt_few" ? 78 : 
        id === "gpt_zero" ? 62 : 72;
      return Math.floor(base + Math.random() * 5);
    }
    return Math.floor(Math.random() * 6);
  };

  return (
    <div className="w-full max-w-lg mx-auto bg-black/40 border border-white/[0.08] rounded-2xl p-4 sm:p-6 shadow-2xl space-y-4">
      <div className="flex items-center justify-between text-xs text-zinc-500 font-mono">
        <span>[HEATMAP MATRIX MOCKUP]</span>
        <span className="text-amber-500/80 font-medium flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
          Flask Server Port 5000 Offline
        </span>
      </div>
      <div className="text-center pb-2 border-b border-white/[0.05]">
        <h4 className="text-sm font-bold text-zinc-200">{label}</h4>
      </div>
      <div className="grid grid-cols-7 gap-1 font-mono text-[9px] sm:text-[10px]">
        {/* Top Header Corner */}
        <div className="h-8 flex items-center justify-center text-zinc-600 text-[8px] text-center leading-none">
          Pred \ Act
        </div>
        {/* Column Labels */}
        {classes.map((cls) => (
          <div key={`h-${cls}`} className="h-8 flex items-center justify-center text-zinc-500 text-center font-semibold truncate px-0.5">
            {cls.split(" ")[0]}
          </div>
        ))}

        {/* Matrix Rows */}
        {classes.map((rowCls, rowIndex) => (
          <div key={`row-group-${rowIndex}`} className="contents">
            {/* Row Label */}
            <div className="h-9 sm:h-10 flex items-center justify-start text-zinc-500 font-semibold truncate pr-1">
              {rowCls.split(" ")[0]}
            </div>
            {/* Data Cells */}
            {classes.map((colCls, colIndex) => {
              const val = getValue(rowIndex, colIndex);
              const isDiagonal = rowIndex === colIndex;
              let styleClass = "bg-white/[0.02] text-zinc-500 border-white/[0.03]";
              
              if (isDiagonal) {
                if (id === "lora_b") {
                  styleClass = "bg-red-500/20 text-red-400 border-red-500/30 font-bold shadow-[inset_0_0_8px_rgba(239,68,68,0.2)]";
                } else {
                  styleClass = "bg-orange-500/30 text-orange-200 border-orange-500/40 font-bold shadow-[inset_0_0_10px_rgba(249,115,22,0.25)]";
                }
              } else if (val > 15 && id === "lora_b") {
                styleClass = "bg-white/[0.06] text-zinc-400 border-white/[0.06]";
              }
              
              return (
                <div
                  key={`cell-${rowIndex}-${colIndex}`}
                  className={`h-9 sm:h-10 flex flex-col items-center justify-center rounded border ${styleClass} transition-all duration-300`}
                >
                  <span className="text-[10px] sm:text-xs">{val}%</span>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

// ==========================================
// 2. FALLBACK LOSS CURVES CHART
// ==========================================
function FallbackCurve({ model }: { model: "A" | "B" }) {
  const isModelA = model === "A";
  const width = 320;
  const height = 180;
  const padding = 30;
  
  // Pre-calculate line points based on actual training history
  const trainPoints: string[] = [];
  const valPoints: string[] = [];
  const epochs = 10;
  
  for (let i = 0; i <= epochs; i++) {
    const x = padding + (i / epochs) * (width - padding * 2);
    let trainVal = 0;
    let valVal = 0;
    
    if (isModelA) {
      // Model A: Standard exponential convergence
      trainVal = 0.15 + 1.6 * Math.exp(-i / 2.2);
      valVal = 0.25 + 1.2 * Math.exp(-i / 2.8);
    } else {
      // Model B: Spike divergence after epoch 3
      if (i < 3) {
        trainVal = 1.8 - 0.22 * i;
        valVal = 1.5 - 0.12 * i;
      } else {
        trainVal = 1.1 + 0.08 * (i - 3) + Math.sin(i * 1.5) * 0.1;
        valVal = 1.14 + 0.45 * Math.pow(i - 3, 1.45);
      }
    }
    
    const maxVal = isModelA ? 2.0 : 5.0;
    const yTrain = height - padding - (trainVal / maxVal) * (height - padding * 2);
    const yVal = height - padding - (valVal / maxVal) * (height - padding * 2);
    
    trainPoints.push(`${x},${yTrain}`);
    valPoints.push(`${x},${yVal}`);
  }

  return (
    <div className="w-full bg-black/40 border border-white/[0.08] rounded-2xl p-4 sm:p-6 shadow-2xl space-y-4">
      <div className="flex items-center justify-between text-xs text-zinc-500 font-mono">
        <span>[LOSS CURVES MOCKUP]</span>
        <span className={`${isModelA ? 'text-green-500/80' : 'text-red-500/80'} font-medium flex items-center gap-1.5`}>
          <span className={`w-1.5 h-1.5 rounded-full ${isModelA ? 'bg-green-500' : 'bg-red-500'} animate-pulse`} />
          {isModelA ? "Loss Converged" : "Loss Diverged"}
        </span>
      </div>
      <div className="text-center pb-2 border-b border-white/[0.05]">
        <h4 className="text-sm font-bold text-zinc-200">
          LoRA Model {model} Training History
        </h4>
      </div>
      <div className="flex justify-center py-2">
        <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} className="overflow-visible font-mono">
          {/* Background Grid Lines */}
          <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#27272a" strokeWidth={1} />
          <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#27272a" strokeWidth={1} />
          <line x1={padding} y1={padding + (height - padding * 2) / 2} x2={width - padding} y2={padding + (height - padding * 2) / 2} stroke="#18181b" strokeWidth={1} strokeDasharray="3,3" />
          
          {/* Axis Labels */}
          <text x={padding - 6} y={padding + 4} fill="#71717a" fontSize={8} textAnchor="end">
            {isModelA ? "2.0" : "5.0"}
          </text>
          <text x={padding - 6} y={height - padding + 3} fill="#71717a" fontSize={8} textAnchor="end">
            0.0
          </text>
          <text x={padding} y={height - padding + 12} fill="#71717a" fontSize={8} textAnchor="start">
            Ep 0
          </text>
          <text x={width - padding} y={height - padding + 12} fill="#71717a" fontSize={8} textAnchor="end">
            Ep 10
          </text>

          {/* Line Plots */}
          <path d={`M ${trainPoints.join(" L ")}`} fill="none" stroke="#f97316" strokeWidth={2} />
          <path d={`M ${valPoints.join(" L ")}`} fill="none" stroke={isModelA ? "#22c55e" : "#ef4444"} strokeWidth={2} strokeDasharray={isModelA ? "" : "3,3"} />
        </svg>
      </div>
      {/* Legend Block */}
      <div className="flex items-center justify-center gap-6 text-[10px] text-zinc-400 font-medium">
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-0.5 bg-orange-500 inline-block" />
          <span>Training Loss</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className={`w-3 h-0.5 ${isModelA ? 'bg-green-500' : 'bg-red-500 border-dashed'} inline-block`} />
          <span>Validation Loss</span>
        </div>
      </div>
    </div>
  );
}

// ==========================================
// 3. MAIN COMPONENT
// ==========================================
export default function Home() {
  const [activeTab, setActiveTab] = useState<"report" | "demo">("demo");
  const [ticketText, setTicketText] = useState("");
  const [mounted, setMounted] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea to fit content up to 200px max
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [ticketText]);

  // States for dynamic fallbacks
  const [imageErrors, setImageErrors] = useState<Record<string, boolean>>({});
  const [loraAError, setLoraAError] = useState(false);
  const [loraBError, setLoraBError] = useState(false);

  // Default confusion matrix tab
  const [activeMatrix, setActiveMatrix] = useState("embeddings");

  // Demo Classification states
  const [isLoading, setIsLoading] = useState(false);
  const [predictionResults, setPredictionResults] = useState<any>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Read URL hash on page load
  useEffect(() => {
    setMounted(true);
    const hash = window.location.hash;
    if (hash === "#report") {
      setActiveTab("report");
    } else if (hash === "#demo") {
      setActiveTab("demo");
    } else {
      setActiveTab("demo");
      window.location.hash = "demo";
    }
  }, []);

  // Sync activeTab to window.location.hash on change
  useEffect(() => {
    if (mounted) {
      window.location.hash = activeTab;
    }
  }, [activeTab, mounted]);

  const handleClassify = async () => {
    if (!ticketText.trim()) return;
    setIsLoading(true);
    setErrorMessage(null);
    setPredictionResults(null);

    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: ticketText }),
      });

      if (!response.ok) {
        throw new Error("Failed to reach classifier");
      }

      const data = await response.json();
      setPredictionResults(data);
    } catch (err) {
      setErrorMessage("Could not reach the classifier — make sure the Flask server is running.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setTicketText("");
    setPredictionResults(null);
    setErrorMessage(null);
  };

  const getClassTextColorOnly = (label: string) => {
    switch (label) {
      case "Support general":
        return "text-blue-400";
      case "Fileservice":
        return "text-cyan-400";
      case "Software":
        return "text-purple-400";
      case "O365":
        return "text-green-400";
      case "Active Directory":
        return "text-amber-400";
      case "Computer-Services":
        return "text-red-400";
      default:
        return "text-zinc-300";
    }
  };

  const getClassProgressBgOnly = (label: string) => {
    switch (label) {
      case "Support general":
        return "bg-blue-500";
      case "Fileservice":
        return "bg-cyan-500";
      case "Software":
        return "bg-purple-500";
      case "O365":
        return "bg-green-500";
      case "Active Directory":
        return "bg-amber-500";
      case "Computer-Services":
        return "bg-red-500";
      default:
        return "bg-zinc-500";
    }
  };

  const modelsConfig = [
    { key: "tfidf", label: "TF-IDF + Logistic Regression" },
    { key: "embeddings", label: "Sentence Embeddings + LR" },
    { key: "gpt_fewshot", label: "GPT-4o mini (few-shot)" },
    { key: "lora", label: "DistilBERT + LoRA" },
    { key: "embeddings_augmented", label: "Embeddings + LR (Augmented)" },
    { key: "gpt_zeroshot", label: "GPT-4o mini (Zero-shot)" },
  ];

  // Class Distribution Data (Horizontal Bar Chart)
  const classDistribution = [
    { name: "Support general", value: 336 },
    { name: "Fileservice", value: 203 },
    { name: "Software", value: 125 },
    { name: "O365", value: 124 },
    { name: "Active Directory", value: 78 },
    { name: "Computer-Services", value: 62 },
  ];

  // Language Distribution Data (Donut Chart)
  const languageDistribution = [
    { name: "English", value: 721, percentage: "77.0%" },
    { name: "Portuguese", value: 94, percentage: "10.0%" },
    { name: "German", value: 60, percentage: "6.4%" },
    { name: "Unknown", value: 50, percentage: "5.3%" },
    { name: "Spanish", value: 11, percentage: "1.2%" },
  ];
  const LANG_COLORS = ["#f97316", "#f59e0b", "#d97706", "#71717a", "#3f3f46"];

  // Model Evaluation Results Data
  const modelsResults = [
    { name: "Original Keras dirty", value: 0.72 },
    { name: "TF-IDF + LR", value: 0.69 },
    { name: "Embeddings + LR", value: 0.74, isWinner: true },
    { name: "GPT zero-shot", value: 0.64 },
    { name: "GPT few-shot", value: 0.71 },
    { name: "LoRA A", value: 0.70 },
    { name: "LoRA B", value: 0.11, isFailure: true },
    { name: "Embeddings + synthetic", value: 0.72 },
  ];

  // Confusion Matrices Definitions
  const matrices = [
    { id: "embeddings", label: "Embeddings + LR", filename: "emb_confusion_matrix.png" },
    { id: "tfidf", label: "TF-IDF + LR", filename: "tfidf_confusion_matrix.png" },
    { id: "synthetic", label: "Embeddings + Synthetic", filename: "emb_augmented_confusion_matrix.png" },
    { id: "gpt_zero", label: "GPT Zero-shot", filename: "gpt_zeroshot_confusion_matrix.png" },
    { id: "gpt_few", label: "GPT Few-shot", filename: "gpt_fewshot_confusion_matrix.png" },
    { id: "lora_a", label: "LoRA Model A", filename: "lora_model_a_confusion_matrix.png" },
    { id: "lora_b", label: "LoRA Model B", filename: "lora_model_b_confusion_matrix.png" },
  ];

  const activeMatrixObj = matrices.find(m => m.id === activeMatrix) || matrices[0];

  // Custom Bar Chart Tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-[#0f0f15]/95 border border-white/[0.08] backdrop-blur-md p-3 rounded-xl shadow-xl">
          <p className="text-xs font-semibold text-zinc-400">{label}</p>
          <p className="text-sm font-bold text-orange-400 mt-1">
            {payload[0].name}: <span className="text-zinc-100">{payload[0].value}</span>
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#0a0a0f] text-zinc-100 selection:bg-orange-500 selection:text-black">
      {/* Sticky Navigation Bar */}
      <nav className="sticky top-0 z-50 w-full border-b border-white/[0.06] bg-[#0a0a0f]/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          {/* Logo & Name */}
          <div className="flex items-center gap-3 group cursor-pointer">
            <div className="w-6 h-6 bg-gradient-to-br from-orange-400 to-orange-600 rounded-md shadow-[0_0_12px_rgba(249,115,22,0.4)] transition-all duration-300 group-hover:scale-105" />
            <span className="font-bold text-lg tracking-tight bg-gradient-to-r from-white via-zinc-200 to-zinc-400 bg-clip-text text-transparent">
              Ticket Classifier
            </span>
          </div>

          {/* Pill Tab Buttons */}
          <div className="flex items-center gap-2 p-1 bg-white/[0.02] border border-white/[0.05] rounded-full">
            <button
              onClick={() => setActiveTab("report")}
              className={`px-5 py-1.5 rounded-full text-sm font-medium transition-all duration-300 cursor-pointer ${
                activeTab === "report"
                  ? "bg-gradient-to-r from-orange-500 to-amber-500 text-neutral-950 shadow-[0_2px_10px_rgba(249,115,22,0.3)]"
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.04]"
              }`}
            >
              Report
            </button>
            <button
              onClick={() => setActiveTab("demo")}
              className={`px-5 py-1.5 rounded-full text-sm font-medium transition-all duration-300 cursor-pointer ${
                activeTab === "demo"
                  ? "bg-gradient-to-r from-orange-500 to-amber-500 text-neutral-950 shadow-[0_2px_10px_rgba(249,115,22,0.3)]"
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.04]"
              }`}
            >
              Demo
            </button>
          </div>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-10 transition-all duration-500">
        {activeTab === "report" ? (
          /* ==============================================================
             REPORT VIEW
             ============================================================== */
          <div className="space-y-16 animate-in fade-in slide-in-from-bottom-4 duration-500">
            
            {/* Hero Section */}
            <div className="relative overflow-hidden rounded-3xl border border-white/[0.04] bg-gradient-to-b from-white/[0.02] to-transparent p-8 sm:p-12">
              <div className="absolute -top-24 -right-24 w-96 h-96 bg-orange-500/10 rounded-full blur-[100px] pointer-events-none" />
              <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-amber-500/5 rounded-full blur-[120px] pointer-events-none" />

              <div className="relative z-10 max-w-3xl space-y-4">
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-orange-500/10 text-orange-400 border border-orange-500/20">
                  <span className="w-1.5 h-1.5 rounded-full bg-orange-500 animate-pulse" />
                  System Model v1.2 Active
                </span>
                <h1 className="text-4xl sm:text-5xl font-black tracking-tight text-white leading-tight">
                  IT Support Ticket <br className="hidden sm:inline" />
                  <span className="bg-gradient-to-r from-orange-400 via-amber-400 to-orange-500 bg-clip-text text-transparent">
                    Classification Dashboard
                  </span>
                </h1>
                <p className="text-base sm:text-lg text-zinc-400 font-normal leading-relaxed">
                  Leveraging cutting-edge machine learning and natural language processing to automatically categorize
                  incoming support issues, identify duplicated requests, and dramatically accelerate resolving times.
                </p>
              </div>
            </div>

            {/* Stat Cards Row */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
              {/* Stat 1: Total Tickets */}
              <div className="group relative overflow-hidden rounded-2xl border border-white/[0.06] bg-white/[0.01] hover:bg-white/[0.02] p-6 transition-all duration-300 hover:border-orange-500/20 hover:shadow-[0_8px_30px_rgb(0,0,0,0.4)]">
                <div className="absolute top-0 right-0 w-24 h-24 bg-gradient-to-bl from-orange-500/5 to-transparent rounded-bl-full transition-all duration-500 group-hover:from-orange-500/10" />
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Total Dataset</span>
                    <span className="p-2 rounded-lg bg-white/[0.02] border border-white/[0.04] text-orange-400 group-hover:scale-110 transition-all duration-300">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                      </svg>
                    </span>
                  </div>
                  <div>
                    <h3 className="text-3xl font-black text-white tracking-tight">2,229</h3>
                    <p className="text-xs text-zinc-400 mt-1 font-medium">Total Tickets Handled</p>
                  </div>
                </div>
              </div>

              {/* Stat 2: Duplicates */}
              <div className="group relative overflow-hidden rounded-2xl border border-white/[0.06] bg-white/[0.01] hover:bg-white/[0.02] p-6 transition-all duration-300 hover:border-orange-500/20 hover:shadow-[0_8px_30px_rgb(0,0,0,0.4)]">
                <div className="absolute top-0 right-0 w-24 h-24 bg-gradient-to-bl from-orange-500/5 to-transparent rounded-bl-full transition-all duration-500 group-hover:from-orange-500/10" />
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Redundancy</span>
                    <span className="p-2 rounded-lg bg-white/[0.02] border border-white/[0.04] text-orange-400 group-hover:scale-110 transition-all duration-300">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                      </svg>
                    </span>
                  </div>
                  <div>
                    <h3 className="text-3xl font-black text-white tracking-tight">610</h3>
                    <p className="text-xs text-zinc-400 mt-1 font-medium">Duplicates Detected</p>
                  </div>
                </div>
              </div>

              {/* Stat 3: Clean Samples */}
              <div className="group relative overflow-hidden rounded-2xl border border-white/[0.06] bg-white/[0.01] hover:bg-white/[0.02] p-6 transition-all duration-300 hover:border-orange-500/20 hover:shadow-[0_8px_30px_rgb(0,0,0,0.4)]">
                <div className="absolute top-0 right-0 w-24 h-24 bg-gradient-to-bl from-orange-500/5 to-transparent rounded-bl-full transition-all duration-500 group-hover:from-orange-500/10" />
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Filtered Data</span>
                    <span className="p-2 rounded-lg bg-white/[0.02] border border-white/[0.04] text-orange-400 group-hover:scale-110 transition-all duration-300">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.957 11.957 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                      </svg>
                    </span>
                  </div>
                  <div>
                    <h3 className="text-3xl font-black text-white tracking-tight">962</h3>
                    <p className="text-xs text-zinc-400 mt-1 font-medium">Clean Unique Samples</p>
                  </div>
                </div>
              </div>

              {/* Stat 4: Best F1 Score */}
              <div className="group relative overflow-hidden rounded-2xl border border-white/[0.06] bg-white/[0.01] hover:bg-white/[0.02] p-6 transition-all duration-300 hover:border-orange-500/20 hover:shadow-[0_8px_30px_rgb(0,0,0,0.4)]">
                <div className="absolute top-0 right-0 w-24 h-24 bg-gradient-to-bl from-orange-500/5 to-transparent rounded-bl-full transition-all duration-500 group-hover:from-orange-500/10" />
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Accuracy</span>
                    <span className="p-2 rounded-lg bg-orange-500/10 border border-orange-500/20 text-orange-400 group-hover:scale-110 transition-all duration-300">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                      </svg>
                    </span>
                  </div>
                  <div>
                    <h3 className="text-3xl font-black text-white tracking-tight bg-gradient-to-r from-orange-400 to-amber-400 bg-clip-text text-transparent">
                      0.74
                    </h3>
                    <p className="text-xs text-zinc-400 mt-1 font-medium">Best F1 Accuracy</p>
                  </div>
                </div>
              </div>
            </div>

            {/* ==============================================================
               1. ARCHITECTURE FLOWCHART SECTION
               ============================================================== */}
            <section className="space-y-6">
              <div className="border-l-2 border-orange-500 pl-4">
                <h3 className="text-xl font-bold text-white tracking-tight">1. Pipeline Architecture Flowchart</h3>
                <p className="text-xs text-zinc-400 mt-1">
                  A visual pipeline mapping support ticket ingestion, multi-path feature extraction, and model classification.
                </p>
              </div>

              <div className="relative rounded-2xl border border-white/[0.06] bg-white/[0.01] p-6 lg:p-8">
                {/* Horizontal flow on larger screens, vertical stacking on mobile */}
                <div className="flex flex-col md:flex-row items-center justify-between gap-6 md:gap-4 relative">
                  
                  {/* Node A: Raw Ticket */}
                  <div className="w-full md:w-1/5 flex flex-col items-center justify-center p-4 bg-white/[0.02] border border-white/[0.06] rounded-xl text-center group hover:border-orange-500/20 transition-all duration-300">
                    <div className="w-8 h-8 rounded-lg bg-orange-500/10 flex items-center justify-center text-orange-400 mb-2">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <span className="text-xs font-bold text-white">Raw Ticket</span>
                    <span className="text-[10px] text-zinc-500 mt-1">Ingested support requests</span>
                  </div>

                  {/* Connect arrow 1 */}
                  <div className="flex items-center justify-center text-orange-500/40 md:rotate-0 rotate-90">
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                    </svg>
                  </div>

                  {/* Node B: Clean & Deduplicate */}
                  <div className="w-full md:w-1/5 flex flex-col items-center justify-center p-4 bg-white/[0.02] border border-white/[0.06] rounded-xl text-center group hover:border-orange-500/20 transition-all duration-300">
                    <div className="w-8 h-8 rounded-lg bg-orange-500/10 flex items-center justify-center text-orange-400 mb-2">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                      </svg>
                    </div>
                    <span className="text-xs font-bold text-white">Clean & Deduplicate</span>
                    <span className="text-[10px] text-zinc-500 mt-1">Noise repair & duplicate removal</span>
                  </div>

                  {/* Connect arrow 2 */}
                  <div className="flex items-center justify-center text-orange-500/40 md:rotate-0 rotate-90">
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                    </svg>
                  </div>

                  {/* Node C: The 4 Paths */}
                  <div className="w-full md:w-2/5 flex flex-col gap-2.5">
                    {/* Path 1: TF-IDF */}
                    <div className="flex items-center justify-between p-2.5 bg-black/40 border border-white/[0.04] rounded-lg text-[10px]">
                      <span className="text-zinc-300 font-medium">TF-IDF Features</span>
                      <span className="text-zinc-500">➔</span>
                      <span className="text-zinc-400">Logistic Regression</span>
                    </div>

                    {/* Path 2: Sentence Embeddings (Winner) */}
                    <div className="relative flex items-center justify-between p-2.5 bg-orange-500/5 border border-orange-500/50 rounded-lg text-[10px] shadow-[0_0_12px_rgba(249,115,22,0.15)]">
                      <div className="absolute top-0 right-2 -translate-y-1/2 bg-orange-500 text-neutral-950 font-black text-[8px] px-1.5 py-0.5 rounded-full uppercase tracking-wide">
                        Winner
                      </div>
                      <span className="text-orange-300 font-bold">Sentence Embeddings</span>
                      <span className="text-orange-400 font-bold">➔</span>
                      <span className="text-orange-200 font-semibold">Logistic Regression</span>
                    </div>

                    {/* Path 3: GPT-4o mini */}
                    <div className="flex items-center justify-between p-2.5 bg-black/40 border border-white/[0.04] rounded-lg text-[10px]">
                      <span className="text-zinc-300 font-medium">GPT-4o mini</span>
                      <span className="text-zinc-500">➔</span>
                      <span className="text-zinc-400">Prompt Classification</span>
                    </div>

                    {/* Path 4: DistilBERT */}
                    <div className="flex items-center justify-between p-2.5 bg-black/40 border border-white/[0.04] rounded-lg text-[10px]">
                      <span className="text-zinc-300 font-medium">DistilBERT + LoRA</span>
                      <span className="text-zinc-500">➔</span>
                      <span className="text-zinc-400">Fine-tuned Classifier</span>
                    </div>
                  </div>

                  {/* Connect arrow 3 */}
                  <div className="flex items-center justify-center text-orange-500/40 md:rotate-0 rotate-90">
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                    </svg>
                  </div>

                  {/* Node D: Predicted Class */}
                  <div className="w-full md:w-1/5 flex flex-col items-center justify-center p-4 bg-orange-500/10 border border-orange-500/30 rounded-xl text-center group hover:border-orange-500/50 hover:shadow-[0_0_15px_rgba(249,115,22,0.1)] transition-all duration-300">
                    <div className="w-8 h-8 rounded-lg bg-orange-500/20 flex items-center justify-center text-orange-400 mb-2">
                      <svg className="w-4 h-4 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                      </svg>
                    </div>
                    <span className="text-xs font-bold text-white">Predicted Class</span>
                    <span className="text-[10px] text-zinc-400 mt-1">Final ticket categorization</span>
                  </div>

                </div>
              </div>
            </section>

            {/* ==============================================================
               2. DATA SECTION — 2 CHARTS SIDE BY SIDE
               ============================================================== */}
            <section className="space-y-6">
              <div className="border-l-2 border-orange-500 pl-4">
                <h3 className="text-xl font-bold text-white tracking-tight">2. Dataset & Class Distribution</h3>
                <p className="text-xs text-zinc-400 mt-1">
                  Comprehensive analysis of ticket categories and language frequencies across the entire cleaned dataset.
                </p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Left Chart: Class Distribution (Horizontal Bar Chart) */}
                <div className="rounded-2xl border border-white/[0.06] bg-white/[0.01] p-6 space-y-4">
                  <h4 className="text-sm font-bold text-zinc-300">Class Label Distribution</h4>
                  <div className="h-64 flex items-center justify-center">
                    {mounted ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          layout="vertical"
                          data={classDistribution}
                          margin={{ top: 10, right: 10, left: 25, bottom: 5 }}
                        >
                          <defs>
                            <linearGradient id="horizontalOrangeGrad" x1="0" y1="0" x2="1" y2="0">
                              <stop offset="0%" stopColor="#f97316" stopOpacity={0.2} />
                              <stop offset="100%" stopColor="#f59e0b" stopOpacity={0.8} />
                            </linearGradient>
                          </defs>
                          <XAxis type="number" stroke="#52525b" fontSize={10} tickLine={false} />
                          <YAxis
                            dataKey="name"
                            type="category"
                            stroke="#71717a"
                            fontSize={9}
                            tickLine={false}
                            width={90}
                          />
                          <Tooltip content={<CustomTooltip />} />
                          <Bar dataKey="value" fill="url(#horizontalOrangeGrad)" radius={[0, 4, 4, 0]}>
                            {classDistribution.map((entry, index) => (
                              <Cell key={`cell-${index}`} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="text-xs text-zinc-500 font-mono animate-pulse">Loading Chart Data...</div>
                    )}
                  </div>
                </div>

                {/* Right Chart: Language Distribution (Donut Chart) */}
                <div className="rounded-2xl border border-white/[0.06] bg-white/[0.01] p-6 space-y-4">
                  <h4 className="text-sm font-bold text-zinc-300">Ticket Language Distribution</h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 items-center">
                    <div className="h-48 flex items-center justify-center">
                      {mounted ? (
                        <ResponsiveContainer width="100%" height="100%">
                          <PieChart>
                            <Pie
                              data={languageDistribution}
                              cx="50%"
                              cy="50%"
                              innerRadius={50}
                              outerRadius={70}
                              paddingAngle={4}
                              dataKey="value"
                            >
                              {languageDistribution.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={LANG_COLORS[index % LANG_COLORS.length]} />
                              ))}
                            </Pie>
                            <Tooltip content={<CustomTooltip />} />
                          </PieChart>
                        </ResponsiveContainer>
                      ) : (
                        <div className="text-xs text-zinc-500 font-mono animate-pulse">Loading Chart Data...</div>
                      )}
                    </div>
                    {/* Language Custom Legends */}
                    <div className="space-y-2">
                      {languageDistribution.map((item, index) => (
                        <div key={`lang-${index}`} className="flex items-center justify-between text-xs">
                          <div className="flex items-center gap-2">
                            <span
                              className="w-2.5 h-2.5 rounded-full inline-block"
                              style={{ backgroundColor: LANG_COLORS[index % LANG_COLORS.length] }}
                            />
                            <span className="text-zinc-400 font-medium">{item.name}</span>
                          </div>
                          <span className="font-mono text-zinc-200">
                            {item.value} <span className="text-[10px] text-zinc-500">({item.percentage})</span>
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </section>

            {/* ==============================================================
               3. PREPROCESSING — 4 CARDS IN A 2x2 GRID
               ============================================================== */}
            <section className="space-y-6">
              <div className="border-l-2 border-orange-500 pl-4">
                <h3 className="text-xl font-bold text-white tracking-tight">3. Data Preprocessing & Validation</h3>
                <p className="text-xs text-zinc-400 mt-1">
                  Rigorous data sanitization protocols implemented to group duplicates, scrub label noise, and isolate testing baselines.
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                {/* Preprocessing Card 1: Deduplication */}
                <div className="group relative overflow-hidden rounded-2xl border border-white/[0.06] bg-white/[0.01] p-5 space-y-3 transition-all duration-300 hover:border-orange-500/20">
                  <div className="absolute top-0 right-0 w-16 h-16 bg-gradient-to-bl from-orange-500/5 to-transparent rounded-bl-full" />
                  <div className="flex items-center gap-3">
                    <span className="flex items-center justify-center w-6 h-6 rounded-md bg-orange-500/10 text-orange-400 text-xs font-black font-mono">
                      01
                    </span>
                    <h4 className="text-sm font-bold text-white tracking-tight">Deduplication</h4>
                  </div>
                  <p className="text-xs text-zinc-400 leading-relaxed font-normal">
                    The training set dataset size was optimized and shrank cleanly from 1572 down to 962 samples by stripping redundant requests.
                  </p>
                </div>

                {/* Preprocessing Card 2: EOL class removal */}
                <div className="group relative overflow-hidden rounded-2xl border border-white/[0.06] bg-white/[0.01] p-5 space-y-3 transition-all duration-300 hover:border-orange-500/20">
                  <div className="absolute top-0 right-0 w-16 h-16 bg-gradient-to-bl from-orange-500/5 to-transparent rounded-bl-full" />
                  <div className="flex items-center gap-3">
                    <span className="flex items-center justify-center w-6 h-6 rounded-md bg-orange-500/10 text-orange-400 text-xs font-black font-mono">
                      02
                    </span>
                    <h4 className="text-sm font-bold text-white tracking-tight">EOL Class Removal</h4>
                  </div>
                  <p className="text-xs text-zinc-400 leading-relaxed font-normal">
                    Scrubbed the End-of-Life category because investigation revealed 44 out of 45 submitted samples were identical automated templates.
                  </p>
                </div>

                {/* Preprocessing Card 3: Label cleaning */}
                <div className="group relative overflow-hidden rounded-2xl border border-white/[0.06] bg-white/[0.01] p-5 space-y-3 transition-all duration-300 hover:border-orange-500/20">
                  <div className="absolute top-0 right-0 w-16 h-16 bg-gradient-to-bl from-orange-500/5 to-transparent rounded-bl-full" />
                  <div className="flex items-center gap-3">
                    <span className="flex items-center justify-center w-6 h-6 rounded-md bg-orange-500/10 text-orange-400 text-xs font-black font-mono">
                      03
                    </span>
                    <h4 className="text-sm font-bold text-white tracking-tight">Label Cleaning</h4>
                  </div>
                  <p className="text-xs text-zinc-400 leading-relaxed font-normal">
                    Resolved structural label noise where instances of identical ticket text were submitted under two entirely separate classifications.
                  </p>
                </div>

                {/* Preprocessing Card 4: Train/test preservation */}
                <div className="group relative overflow-hidden rounded-2xl border border-white/[0.06] bg-white/[0.01] p-5 space-y-3 transition-all duration-300 hover:border-orange-500/20">
                  <div className="absolute top-0 right-0 w-16 h-16 bg-gradient-to-bl from-orange-500/5 to-transparent rounded-bl-full" />
                  <div className="flex items-center gap-3">
                    <span className="flex items-center justify-center w-6 h-6 rounded-md bg-orange-500/10 text-orange-400 text-xs font-black font-mono">
                      04
                    </span>
                    <h4 className="text-sm font-bold text-white tracking-tight">Train/Test Preservation</h4>
                  </div>
                  <p className="text-xs text-zinc-400 leading-relaxed font-normal">
                    Guaranteed model audit validity by holding the evaluation test dataset completely untouched throughout the preprocessing pipeline.
                  </p>
                </div>
              </div>
            </section>

            {/* ==============================================================
               4. RESULTS CHART
               ============================================================== */}
            <section className="space-y-6">
              <div className="border-l-2 border-orange-500 pl-4">
                <h3 className="text-xl font-bold text-white tracking-tight">4. Comparative Model Performance (Macro F1)</h3>
                <p className="text-xs text-zinc-400 mt-1">
                  Evaluation metrics across all eight model architectures, highlighting our optimal sentence embeddings approach.
                </p>
              </div>

              <div className="rounded-2xl border border-white/[0.06] bg-white/[0.01] p-6 space-y-4">
                <div className="h-72 flex items-center justify-center">
                  {mounted ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={modelsResults} margin={{ top: 20, right: 10, left: 10, bottom: 25 }}>
                        <XAxis
                          dataKey="name"
                          stroke="#71717a"
                          fontSize={9}
                          tickLine={false}
                          angle={-15}
                          textAnchor="end"
                          height={50}
                        />
                        <YAxis
                          stroke="#71717a"
                          fontSize={10}
                          tickLine={false}
                          domain={[0, 1]}
                          ticks={[0, 0.2, 0.4, 0.6, 0.8, 1.0]}
                        />
                        <Tooltip content={<CustomTooltip />} />
                        <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                          {modelsResults.map((entry, index) => {
                            let color = "#c2410c"; // Muted orange fallback color
                            let opacity = 0.5;
                            
                            if (entry.isWinner) {
                              color = "#22c55e"; // Winner is Green
                              opacity = 1.0;
                            } else if (entry.isFailure) {
                              color = "#ef4444"; // Failure is Red
                              opacity = 1.0;
                            } else if (entry.name === "GPT few-shot" || entry.name === "Original Keras dirty") {
                              opacity = 0.8;
                            }
                            
                            return (
                              <Cell
                                key={`cell-${index}`}
                                fill={color}
                                opacity={opacity}
                              />
                            );
                          })}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="text-xs text-zinc-500 font-mono animate-pulse">Loading Chart Data...</div>
                  )}
                </div>
                
                {/* Color Guide Legends */}
                <div className="flex flex-wrap items-center justify-center gap-6 text-[10px] text-zinc-500 font-mono font-medium pt-2 border-t border-white/[0.04]">
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded bg-[#22c55e] inline-block" />
                    <span className="text-zinc-300">Winner Model (0.74 F1)</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded bg-[#ef4444] inline-block" />
                    <span className="text-zinc-300">Failure Run (0.11 F1)</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded bg-[#c2410c] inline-block opacity-60" />
                    <span className="text-zinc-400">Baseline / Alternate Models</span>
                  </div>
                </div>
              </div>
            </section>

            {/* ==============================================================
               5. RESULTS TABLE
               ============================================================== */}
            <section className="space-y-6">
              <div className="border-l-2 border-orange-500 pl-4">
                <h3 className="text-xl font-bold text-white tracking-tight">5. Performance Metrics Reference Table</h3>
                <p className="text-xs text-zinc-400 mt-1">
                  Detailed breakdown of F1-scores, accuracy rates, real-world ticket testing, custom evaluations, and core insights.
                </p>
              </div>

              <div className="rounded-2xl border border-white/[0.06] bg-white/[0.01] overflow-hidden shadow-2xl">
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse text-xs">
                    <thead>
                      <tr className="bg-white/[0.02] border-b border-white/[0.06] text-zinc-400 font-mono">
                        <th className="py-3.5 px-4 font-semibold uppercase tracking-wider text-[10px]">Model</th>
                        <th className="py-3.5 px-4 font-semibold uppercase tracking-wider text-[10px] text-center">Macro F1</th>
                        <th className="py-3.5 px-4 font-semibold uppercase tracking-wider text-[10px] text-center">Accuracy</th>
                        <th className="py-3.5 px-4 font-semibold uppercase tracking-wider text-[10px] text-center">Real Tickets</th>
                        <th className="py-3.5 px-4 font-semibold uppercase tracking-wider text-[10px] text-center">Custom 50</th>
                        <th className="py-3.5 px-4 font-semibold uppercase tracking-wider text-[10px] pl-6">Core Takeaway</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/[0.04]">
                      {/* Row 1: Original Keras */}
                      <tr className="hover:bg-white/[0.01] transition-all">
                        <td className="py-3 px-4 font-bold text-zinc-200">Original Keras (dirty)</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-400">0.72</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-400">0.79</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-500">—</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-500">—</td>
                        <td className="py-3 px-4 pl-6">
                          <span className="inline-flex px-2 py-0.5 rounded text-[10px] font-bold bg-red-500/10 text-red-400 border border-red-500/20">
                            Inflated by dirty data
                          </span>
                        </td>
                      </tr>

                      {/* Row 2: TF-IDF + LR */}
                      <tr className="hover:bg-white/[0.01] transition-all">
                        <td className="py-3 px-4 font-bold text-zinc-200">TF-IDF + LR</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-400">0.69</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-400">0.75</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-500">—</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-500">—</td>
                        <td className="py-3 px-4 pl-6">
                          <span className="inline-flex px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/10 text-amber-400 border border-amber-500/20">
                            Honest baseline
                          </span>
                        </td>
                      </tr>

                      {/* Row 3: Embeddings + LR (WINNER) */}
                      <tr className="bg-orange-500/[0.02] hover:bg-orange-500/[0.04] transition-all">
                        <td className="py-3 px-4 font-black text-orange-400 flex items-center gap-1.5">
                          <span className="w-1.5 h-1.5 rounded-full bg-orange-500" />
                          Embeddings + LR
                        </td>
                        <td className="py-3 px-4 text-center font-bold font-mono text-orange-300">0.74</td>
                        <td className="py-3 px-4 text-center font-bold font-mono text-orange-400">0.78</td>
                        <td className="py-3 px-4 text-center font-bold font-mono text-orange-300">0.83</td>
                        <td className="py-3 px-4 text-center font-bold font-mono text-orange-400">0.59</td>
                        <td className="py-3 px-4 pl-6">
                          <span className="inline-flex px-2 py-0.5 rounded text-[10px] font-bold bg-green-500/15 text-green-400 border border-green-500/30">
                            Best overall
                          </span>
                        </td>
                      </tr>

                      {/* Row 4: GPT zero-shot */}
                      <tr className="hover:bg-white/[0.01] transition-all">
                        <td className="py-3 px-4 font-bold text-zinc-200">GPT zero-shot</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-400">0.64</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-400">0.67</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-400">0.71</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-400">0.80</td>
                        <td className="py-3 px-4 pl-6">
                          <span className="inline-flex px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/10 text-amber-400 border border-amber-500/20">
                            Flexible, not robust
                          </span>
                        </td>
                      </tr>

                      {/* Row 5: GPT few-shot */}
                      <tr className="hover:bg-white/[0.01] transition-all">
                        <td className="py-3 px-4 font-bold text-zinc-200">GPT few-shot</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-400">0.71</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-400">0.72</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-400">0.68</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-400">0.86</td>
                        <td className="py-3 px-4 pl-6">
                          <span className="inline-flex px-2 py-0.5 rounded text-[10px] font-bold bg-green-500/10 text-green-400 border border-green-500/20">
                            Strong on clean prompts
                          </span>
                        </td>
                      </tr>

                      {/* Row 6: LoRA A */}
                      <tr className="hover:bg-white/[0.01] transition-all">
                        <td className="py-3 px-4 font-bold text-zinc-200">LoRA A</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-400">0.70</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-400">0.78</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-400">0.69</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-400">0.48</td>
                        <td className="py-3 px-4 pl-6">
                          <span className="inline-flex px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/10 text-amber-400 border border-amber-500/20">
                            Needs more data
                          </span>
                        </td>
                      </tr>

                      {/* Row 7: LoRA B */}
                      <tr className="hover:bg-white/[0.01] transition-all">
                        <td className="py-3 px-4 font-bold text-zinc-200">LoRA B</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-400">0.11</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-400">0.30</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-500">—</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-500">—</td>
                        <td className="py-3 px-4 pl-6">
                          <span className="inline-flex px-2 py-0.5 rounded text-[10px] font-bold bg-red-500/10 text-red-400 border border-red-500/20">
                            Failure run
                          </span>
                        </td>
                      </tr>

                      {/* Row 8: Embeddings + synthetic */}
                      <tr className="hover:bg-white/[0.01] transition-all">
                        <td className="py-3 px-4 font-bold text-zinc-200">Embeddings + synthetic</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-400">0.72</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-400">0.78</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-400">0.73</td>
                        <td className="py-3 px-4 text-center font-mono text-zinc-400">0.84</td>
                        <td className="py-3 px-4 pl-6">
                          <span className="inline-flex px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/10 text-amber-400 border border-amber-500/20">
                            Better external, weaker real
                          </span>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </section>

            {/* ==============================================================
               6. CONFUSION MATRIX GALLERY
               ============================================================== */}
            <section className="space-y-6">
              <div className="border-l-2 border-orange-500 pl-4">
                <h3 className="text-xl font-bold text-white tracking-tight">6. Model Confusion Matrix Gallery</h3>
                <p className="text-xs text-zinc-400 mt-1">
                  Class-by-class confusion matrix selector to audit misclassifications, class recall rates, and model errors.
                </p>
              </div>

              <div className="space-y-6">
                {/* Horizontal Switcher Tab Buttons */}
                <div className="flex flex-wrap items-center gap-2 p-1.5 bg-white/[0.01] border border-white/[0.05] rounded-xl">
                  {matrices.map((matrix) => (
                    <button
                      key={matrix.id}
                      onClick={() => setActiveMatrix(matrix.id)}
                      className={`px-4 py-2 rounded-lg text-xs font-semibold tracking-wide transition-all duration-300 cursor-pointer ${
                        activeMatrix === matrix.id
                          ? "bg-orange-500 text-neutral-950 shadow-md font-bold"
                          : "text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.03]"
                      }`}
                    >
                      {matrix.label}
                    </button>
                  ))}
                </div>

                {/* Display Area for Active Confusion Matrix */}
                <div className="mx-auto max-w-[700px] w-full rounded-2xl border border-white/[0.06] bg-[#0c0c12]/60 p-4 flex flex-col items-center justify-center shadow-2xl">
                  <img
                    src={`http://localhost:5000/static/images/${activeMatrixObj.filename}`}
                    alt={`Confusion Matrix for ${activeMatrixObj.label}`}
                    className="w-full h-auto rounded-xl object-contain"
                  />
                </div>
              </div>
            </section>

            {/* ==============================================================
               7. LORA TRAINING CURVES — 2 IMAGES SIDE BY SIDE
               ============================================================== */}
            <section className="space-y-6">
              <div className="border-l-2 border-orange-500 pl-4">
                <h3 className="text-xl font-bold text-white tracking-tight">7. LoRA Fine-Tuning Loss Curves</h3>
                <p className="text-xs text-zinc-400 mt-1">
                  Side-by-side comparison of training loss convergence (Model A) versus gradient divergence and failure (Model B).
                </p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Model A Training Curve */}
                <div className="space-y-3">
                  {loraAError ? (
                    <FallbackCurve model="A" />
                  ) : (
                    <div className="rounded-2xl border border-white/[0.06] bg-white/[0.01] p-5 flex flex-col items-center justify-center min-h-[220px]">
                      <img
                        src="http://localhost:5000/static/images/lora_model_a_training_curve.png"
                        alt="LoRA Model A Training Curve"
                        onError={() => setLoraAError(true)}
                        className="mx-auto rounded-xl max-h-[280px] w-auto border border-white/[0.04] object-contain shadow-xl"
                      />
                      <span className="text-[10px] text-zinc-500 mt-2 font-mono">lora_model_a_training_curve.png</span>
                    </div>
                  )}
                </div>

                {/* Model B Training Curve */}
                <div className="space-y-3">
                  {loraBError ? (
                    <FallbackCurve model="B" />
                  ) : (
                    <div className="rounded-2xl border border-white/[0.06] bg-white/[0.01] p-5 flex flex-col items-center justify-center min-h-[220px]">
                      <img
                        src="http://localhost:5000/static/images/lora_model_b_training_curve.png"
                        alt="LoRA Model B Training Curve"
                        onError={() => setLoraBError(true)}
                        className="mx-auto rounded-xl max-h-[280px] w-auto border border-white/[0.04] object-contain shadow-xl"
                      />
                      <span className="text-[10px] text-zinc-500 mt-2 font-mono">lora_model_b_training_curve.png</span>
                    </div>
                  )}
                </div>
              </div>
            </section>

            {/* ==============================================================
               8. KEY FINDINGS — 2x2 GRID OF CARDS
               ============================================================== */}
            <section className="space-y-6">
              <div className="border-l-2 border-orange-500 pl-4">
                <h3 className="text-xl font-bold text-white tracking-tight">8. Executive Insights & Lessons Learned</h3>
                <p className="text-xs text-zinc-400 mt-1">
                  Four foundational learnings established through auditing neural representations, data quality, and LLM constraints.
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                {/* Finding 1: Dirty data */}
                <div className="relative overflow-hidden rounded-2xl border border-white/[0.06] bg-white/[0.01] p-6 hover:border-orange-500/20 transition-all duration-300 space-y-3">
                  <div className="flex items-center gap-3">
                    <span className="text-xl">🔍</span>
                    <h4 className="text-sm font-bold text-zinc-200 tracking-tight">Dirty Data Inflated Metrics</h4>
                  </div>
                  <p className="text-xs text-zinc-400 leading-relaxed font-normal">
                    Automated templates artificially inflated early evaluation performance. For example, EOL obtained an flawless 1.00 F1 score solely due to 44 identical templates.
                  </p>
                </div>

                {/* Finding 2: Embeddings */}
                <div className="relative overflow-hidden rounded-2xl border border-white/[0.06] bg-white/[0.01] p-6 hover:border-orange-500/20 transition-all duration-300 space-y-3">
                  <div className="flex items-center gap-3">
                    <span className="text-xl">🧠</span>
                    <h4 className="text-sm font-bold text-zinc-200 tracking-tight">Embeddings Beat TF-IDF</h4>
                  </div>
                  <p className="text-xs text-zinc-400 leading-relaxed font-normal">
                    Continuous semantic representations outperformed standard bag-of-words pipelines, boosting F1 classification scores by +0.05 overall, and +0.09 on real tickets.
                  </p>
                </div>

                {/* Finding 3: GPT */}
                <div className="relative overflow-hidden rounded-2xl border border-white/[0.06] bg-white/[0.01] p-6 hover:border-orange-500/20 transition-all duration-300 space-y-3">
                  <div className="flex items-center gap-3">
                    <span className="text-xl">⚡</span>
                    <h4 className="text-sm font-bold text-zinc-200 tracking-tight">GPT Wins on Clean Prompts Only</h4>
                  </div>
                  <p className="text-xs text-zinc-400 leading-relaxed font-normal">
                    Large language models demonstrated extreme prompt dependency, scoring 0.86 F1 on carefully curated custom test sets, but dropping to 0.68 F1 on raw real tickets.
                  </p>
                </div>

                {/* Finding 4: Synthetic data */}
                <div className="relative overflow-hidden rounded-2xl border border-white/[0.06] bg-white/[0.01] p-6 hover:border-orange-500/20 transition-all duration-300 space-y-3">
                  <div className="flex items-center gap-3">
                    <span className="text-xl">🔬</span>
                    <h4 className="text-sm font-bold text-zinc-200 tracking-tight">Synthetic Data Shifted Optimization</h4>
                  </div>
                  <p className="text-xs text-zinc-400 leading-relaxed font-normal">
                    Synthetically augmented samples skewed the boundary optimization of the embedding vectors, causing real-ticket evaluation scores to decay from 0.83 down to 0.73 F1.
                  </p>
                </div>
              </div>
            </section>

          </div>
        ) : (
          /* ==============================================================
             DEMO VIEW
             ============================================================== */
          <div style={{ maxWidth: '900px', margin: '0 auto', width: '100%', padding: '0 24px' }} className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500 py-6">
            <style dangerouslySetInnerHTML={{ __html: `
              textarea.no-scrollbar::-webkit-scrollbar {
                display: none !important;
              }
              textarea.no-scrollbar {
                -ms-overflow-style: none !important;
                scrollbar-width: none !important;
              }
            ` }} />
            {/* Header */}
            <div className="text-center space-y-2">
              <h2 className="text-3xl font-black text-white tracking-tight">Classify a Ticket</h2>
              <p className="text-xs text-zinc-300 font-mono tracking-wide">
                Enter an IT support ticket to classify
              </p>
            </div>

            {/* Input Form Card */}
            <div style={{ marginLeft: 'auto', marginRight: 'auto', maxWidth: '900px' }} className="space-y-3">
              <div className="flex items-center justify-between px-1">
                <span className="text-[10px] font-bold text-zinc-300 uppercase tracking-widest">
                  Ticket Content
                </span>
              </div>
              <div className="relative rounded-3xl border border-white/[0.15] bg-[#0c0c12]/80 p-3 shadow-2xl focus-within:border-[#c96a2a]/40 focus-within:ring-1 focus-within:ring-[#c96a2a]/10 transition-all duration-300 overflow-hidden">
                <div className="flex items-end gap-3 px-2 py-1">
                  <textarea
                     ref={textareaRef}
                     rows={1}
                     value={ticketText}
                     onChange={(e) => setTicketText(e.target.value)}
                     placeholder="Paste an IT support ticket..."
                     className="flex-1 bg-transparent text-zinc-100 placeholder-zinc-500 focus:outline-none text-sm leading-relaxed max-h-[200px] overflow-y-auto py-1.5 no-scrollbar"
                     style={{ height: '24px', resize: 'none', WebkitAppearance: 'none' }}
                  />
                  
                  {ticketText.trim() !== "" && (
                    <button
                      onClick={handleClassify}
                      disabled={isLoading}
                      className="flex items-center justify-center w-8 h-8 rounded-xl hover:opacity-90 active:scale-[0.95] text-white transition-all duration-300 shadow-md cursor-pointer shrink-0 disabled:opacity-40"
                      style={{ backgroundColor: '#c96a2a' }}
                      title="Classify Ticket"
                    >
                      {isLoading ? (
                        <svg className="animate-spin h-4 w-4 text-neutral-950" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                        </svg>
                      ) : (
                        <svg className="w-4 h-4 stroke-[3]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 10.5 12 3m0 0 7.5 7.5M12 3v18" />
                        </svg>
                      )}
                    </button>
                  )}
                </div>
              </div>

              {predictionResults && (
                <div className="flex justify-start px-2">
                  <span
                    onClick={handleClear}
                    className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors cursor-pointer select-none font-medium underline"
                  >
                    Clear inputs & results
                  </span>
                </div>
              )}

              {errorMessage && (
                <p className="text-xs font-semibold text-red-500 mt-2 font-mono text-center">
                  {errorMessage}
                </p>
              )}
            </div>

            {/* Results Area */}
            {predictionResults && (
              <div style={{ marginLeft: 'auto', marginRight: 'auto', maxWidth: '900px' }} className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <div className="flex items-center justify-between border-b border-white/[0.08] pb-3">
                  <h3 className="text-xs font-bold text-zinc-200 uppercase tracking-widest">
                    Prediction Results
                  </h3>
                  <span className="text-[10px] text-zinc-300 font-bold uppercase tracking-wider font-mono">
                    Active Models: 6/6
                  </span>
                </div>

                {/* 1. Best Model - Full Width Highlighted Card */}
                {(() => {
                  const bestModel = modelsConfig.find((m) => m.key === "embeddings");
                  const result = bestModel ? predictionResults[bestModel.key] : null;
                  if (!result) return null;
                  return (
                    <div className="rounded-2xl border border-[#c96a2a]/60 bg-[#c96a2a]/[0.02] p-6 space-y-4 hover:border-[#c96a2a]/80 transition-all duration-300 relative shadow-[0_8px_30px_rgba(201,106,42,0.05)]">
                      <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-[#c96a2a]/40 to-transparent" />
                      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                        <div className="space-y-1">
                          <span className="text-[10px] uppercase font-extrabold tracking-widest text-[#c96a2a] block">
                            ⭐ Best Model
                          </span>
                          <h4 className="text-sm font-bold text-zinc-100">
                            {bestModel.label}
                          </h4>
                          <h5 className="text-2xl font-black tracking-tight text-[#c96a2a]">
                            {result.label}
                          </h5>
                        </div>
                        <div className="sm:text-right space-y-2 min-w-[140px]">
                          <div className="flex items-center justify-between text-xs font-mono">
                            <span className="text-zinc-300 font-bold uppercase tracking-wider text-[10px]">Confidence</span>
                            <span className="text-zinc-100 font-black">
                              {(result.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="w-full h-1.5 bg-white/[0.04] rounded-full overflow-hidden">
                            <div
                              className="h-full bg-[#c96a2a] transition-all duration-500"
                              style={{ width: `${result.confidence * 100}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })()}

                {/* 2. Remaining 5 Models Grid */}
                <div style={{ justifyContent: 'center' }} className="flex flex-wrap gap-6">
                  {modelsConfig
                    .filter((m) => m.key !== "embeddings")
                    .map((model) => {
                      const result = predictionResults[model.key];
                      if (!result) return null;
                      return (
                        <div
                          key={model.key}
                          className="group relative overflow-hidden rounded-2xl border border-white/[0.15] bg-white/[0.01] p-5 space-y-4 hover:border-[#c96a2a]/40 transition-all duration-300 w-full sm:w-[calc(50%-12px)] md:w-[calc(33.333%-16px)] min-w-[260px] max-w-[282px]"
                        >
                          <div className="space-y-1">
                            <span className="text-[10px] uppercase font-extrabold tracking-wider text-zinc-300 block">
                              {model.label}
                            </span>
                            <h5 className="text-base font-black tracking-tight text-[#c96a2a]">
                              {result.label}
                            </h5>
                          </div>
                          
                          <div className="space-y-1.5">
                            <div className="flex items-center justify-between text-xs font-mono">
                              <span className="text-zinc-300 font-bold uppercase tracking-wider text-[10px]">Confidence</span>
                              <span className="text-zinc-100 font-bold">
                                {(result.confidence * 100).toFixed(0)}%
                              </span>
                            </div>
                            <div className="w-full h-1 bg-white/[0.04] rounded-full overflow-hidden">
                              <div
                                className="h-full bg-[#c96a2a] transition-all duration-500"
                                style={{ width: `${result.confidence * 100}%` }}
                              />
                            </div>
                          </div>
                        </div>
                      );
                    })}
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
