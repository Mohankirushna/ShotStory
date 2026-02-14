import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Brain, ChevronDown, ChevronRight, Cpu, Eye, Layers, Search,
  CheckCircle2, AlertCircle, FileText, ArrowRight, Shield,
} from "lucide-react";

const PHASE_ICONS = {
  "Visual Perception": Eye,
  "Object Detection": Search,
  "Scene Understanding (VQA)": Brain,
  "Attention Analysis": Layers,
  "Narrative Construction": FileText,
  "Attribution & Explanation": Shield,
};

const CONFIDENCE_COLORS = {
  high: "text-green-400 bg-green-500/15",
  medium: "text-yellow-400 bg-yellow-500/15",
  low: "text-red-400 bg-red-500/15",
};

export default function XAIReasoningPanel({ xaiReasoning }) {
  const [activeTab, setActiveTab] = useState("chain");
  const [expandedStep, setExpandedStep] = useState(null);

  if (!xaiReasoning) return null;

  const { reasoning_chain, decision_log, sentence_map, transparency } = xaiReasoning;

  const tabs = [
    { key: "chain", label: "Reasoning Chain", icon: Brain },
    { key: "decisions", label: "Decision Log", icon: Cpu },
    { key: "mapping", label: "Feature→Story Map", icon: ArrowRight },
    { key: "transparency", label: "Transparency", icon: Shield },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
      className="glass-card overflow-hidden"
    >
      {/* Header */}
      <div className="border-b border-white/10 bg-gradient-to-r from-purple-500/10 to-brand-500/10 px-5 py-4">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-purple-500/20">
            <Brain className="h-4.5 w-4.5 text-purple-400" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white tracking-wide">
              XAI Reasoning &amp; Explainability
            </h3>
            <p className="text-[11px] text-gray-500">
              Step-by-step transparency into model decisions
            </p>
          </div>
        </div>
      </div>

      {/* Tab bar */}
      <div className="flex border-b border-white/5 bg-white/[0.02] px-2">
        {tabs.map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className={`flex items-center gap-1.5 px-3 py-2.5 text-[11px] font-medium transition-colors border-b-2 ${
              activeTab === key
                ? "border-purple-400 text-purple-300"
                : "border-transparent text-gray-500 hover:text-gray-300"
            }`}
          >
            <Icon className="h-3 w-3" />
            {label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="p-4 max-h-[600px] overflow-y-auto">
        <AnimatePresence mode="wait">
          {activeTab === "chain" && (
            <ReasoningChain
              key="chain"
              chain={reasoning_chain}
              expandedStep={expandedStep}
              setExpandedStep={setExpandedStep}
            />
          )}
          {activeTab === "decisions" && (
            <DecisionLog key="decisions" log={decision_log} />
          )}
          {activeTab === "mapping" && (
            <SentenceMap key="mapping" mappings={sentence_map} />
          )}
          {activeTab === "transparency" && (
            <TransparencySummary key="transparency" data={transparency} />
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}

/* ============================================================
   Sub-components
   ============================================================ */

function ReasoningChain({ chain, expandedStep, setExpandedStep }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="space-y-2"
    >
      {chain.map((step) => {
        const Icon = PHASE_ICONS[step.phase] || Brain;
        const open = expandedStep === step.step;
        return (
          <div
            key={step.step}
            className="rounded-xl border border-white/5 bg-white/[0.02] overflow-hidden"
          >
            <button
              onClick={() => setExpandedStep(open ? null : step.step)}
              className="flex w-full items-center gap-3 px-4 py-3 text-left"
            >
              {/* Step number */}
              <span className="flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full bg-purple-500/20 text-[10px] font-bold text-purple-300">
                {step.step}
              </span>
              <Icon className="h-4 w-4 flex-shrink-0 text-purple-400" />
              <div className="min-w-0 flex-1">
                <p className="text-xs font-semibold text-gray-200 truncate">
                  {step.phase}
                </p>
                <p className="text-[10px] text-gray-500 truncate">
                  {step.model}
                </p>
              </div>
              <span
                className={`rounded-md px-2 py-0.5 text-[10px] font-medium ${
                  CONFIDENCE_COLORS[step.confidence] || CONFIDENCE_COLORS.medium
                }`}
              >
                {step.confidence}
              </span>
              {open ? (
                <ChevronDown className="h-3.5 w-3.5 text-gray-500" />
              ) : (
                <ChevronRight className="h-3.5 w-3.5 text-gray-500" />
              )}
            </button>

            <AnimatePresence>
              {open && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="border-t border-white/5"
                >
                  <div className="space-y-2.5 px-4 py-3">
                    <div>
                      <p className="text-[10px] font-medium uppercase text-gray-500 mb-1">
                        What happened
                      </p>
                      <p className="text-xs text-gray-300 leading-relaxed">
                        {step.action}
                      </p>
                    </div>
                    <div>
                      <p className="text-[10px] font-medium uppercase text-gray-500 mb-1">
                        Result
                      </p>
                      <p className="text-xs text-gray-200 leading-relaxed bg-white/[0.03] rounded-lg p-2.5">
                        {step.result}
                      </p>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        );
      })}
    </motion.div>
  );
}

function DecisionLog({ log }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="space-y-3"
    >
      {log.map((entry, i) => (
        <div
          key={i}
          className="rounded-xl border border-white/5 bg-white/[0.02] p-3.5"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-white">
              {entry.decision}
            </span>
            <span className="rounded-md bg-blue-500/15 px-2 py-0.5 text-[10px] font-medium text-blue-300">
              {entry.model}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-2 mb-2">
            <div>
              <p className="text-[10px] font-medium uppercase text-gray-500">Input</p>
              <p className="text-[11px] text-gray-400">{entry.input}</p>
            </div>
            <div>
              <p className="text-[10px] font-medium uppercase text-gray-500">Output</p>
              <p className="text-[11px] text-gray-300">{entry.output}</p>
            </div>
          </div>
          <div className="flex items-start gap-1.5 rounded-lg bg-purple-500/5 p-2">
            <AlertCircle className="mt-0.5 h-3 w-3 flex-shrink-0 text-purple-400" />
            <p className="text-[11px] text-purple-300">{entry.why}</p>
          </div>
        </div>
      ))}
    </motion.div>
  );
}

function SentenceMap({ mappings }) {
  if (!mappings || mappings.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="py-8 text-center text-sm text-gray-500"
      >
        No direct sentence-level mappings found.
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="space-y-3"
    >
      {mappings.map((m, i) => (
        <div
          key={i}
          className="rounded-xl border border-white/5 bg-white/[0.02] p-3.5"
        >
          <div className="flex items-center gap-2 mb-2">
            <span className="rounded-md bg-brand-500/15 px-2 py-0.5 text-[10px] font-medium text-brand-300 capitalize">
              {m.feature_type}
            </span>
            <ArrowRight className="h-3 w-3 text-gray-600" />
            <span className="text-xs font-medium text-white">
              &ldquo;{m.feature_value}&rdquo;
            </span>
          </div>
          <div className="space-y-1.5">
            {m.matched_sentences.map((s, j) => (
              <div
                key={j}
                className="flex items-start gap-2 rounded-lg bg-white/[0.03] p-2"
              >
                <CheckCircle2 className="mt-0.5 h-3 w-3 flex-shrink-0 text-green-400" />
                <p className="text-[11px] text-gray-300 leading-relaxed">
                  {s.sentence}
                </p>
              </div>
            ))}
          </div>
        </div>
      ))}
    </motion.div>
  );
}

function TransparencySummary({ data }) {
  if (!data) return null;

  const verdictColors = {
    "Fully Transparent": "text-green-400 bg-green-500/15 border-green-500/30",
    "Mostly Transparent": "text-yellow-400 bg-yellow-500/15 border-yellow-500/30",
    "Partially Transparent": "text-orange-400 bg-orange-500/15 border-orange-500/30",
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="space-y-4"
    >
      {/* Verdict */}
      <div
        className={`rounded-xl border p-4 text-center ${
          verdictColors[data.explainability_verdict] || verdictColors["Partially Transparent"]
        }`}
      >
        <Shield className="mx-auto mb-2 h-8 w-8" />
        <p className="text-lg font-bold">{data.explainability_verdict}</p>
        <p className="mt-1 text-xs opacity-80">
          {data.feature_coverage_pct}% of detected features appear in the story
        </p>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-3">
        <StatCard
          label="Visual Features Detected"
          value={data.total_visual_features}
        />
        <StatCard
          label="Features Used in Story"
          value={data.features_used_in_story}
        />
        <StatCard
          label="Feature Coverage"
          value={`${data.feature_coverage_pct}%`}
        />
        <StatCard
          label="Avg Attribution Score"
          value={`${data.avg_attribution_score}%`}
        />
      </div>

      {/* Models used */}
      <div className="rounded-xl border border-white/5 bg-white/[0.02] p-3.5">
        <p className="mb-2 text-[10px] font-medium uppercase text-gray-500">
          Models Used in Pipeline
        </p>
        <div className="space-y-1.5">
          {data.models_used.map((m, i) => (
            <div key={i} className="flex items-center gap-2">
              <Cpu className="h-3 w-3 text-purple-400" />
              <p className="text-[11px] text-gray-300">{m}</p>
            </div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}

function StatCard({ label, value }) {
  return (
    <div className="rounded-xl border border-white/5 bg-white/[0.02] p-3 text-center">
      <p className="text-xl font-bold text-white">{value}</p>
      <p className="mt-0.5 text-[10px] text-gray-500">{label}</p>
    </div>
  );
}
