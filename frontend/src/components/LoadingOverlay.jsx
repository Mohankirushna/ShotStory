import { motion } from "framer-motion";
import { Loader2, ScanEye, Brain, BookOpen, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";

const STEPS = [
  { icon: ScanEye, label: "Analyzing visual features..." },
  { icon: Brain, label: "Detecting objects & scene attributes..." },
  { icon: BookOpen, label: "Generating story from visual context..." },
  { icon: Sparkles, label: "Building visual explanations..." },
];

export default function LoadingOverlay({ preview }) {
  const [step, setStep] = useState(0);

  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setStep((s) => (s < STEPS.length - 1 ? s + 1 : s));
    }, 8000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const ticker = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(ticker);
  }, []);

  const mins = Math.floor(elapsed / 60);
  const secs = elapsed % 60;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="mx-auto max-w-lg text-center"
    >
      {/* Thumbnail */}
      {preview && (
        <div className="relative mx-auto mb-8 h-56 w-80 overflow-hidden rounded-2xl border border-white/10">
          <img
            src={preview}
            alt="Uploaded"
            className="h-full w-full object-cover opacity-40"
          />
          {/* Pulsing overlay */}
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="absolute h-20 w-20 rounded-full bg-brand-500/30 animate-ring" />
            <Loader2 className="relative h-10 w-10 animate-spin text-brand-400" />
          </div>
        </div>
      )}

      {/* Steps */}
      <div className="space-y-4">
        {STEPS.map(({ icon: Icon, label }, i) => {
          const active = i === step;
          const done = i < step;
          return (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: done || active ? 1 : 0.35, x: 0 }}
              transition={{ delay: i * 0.15 }}
              className={`flex items-center gap-3 rounded-xl px-5 py-3 transition-colors ${
                active
                  ? "border border-brand-500/30 bg-brand-500/10 text-brand-300"
                  : done
                  ? "text-green-400"
                  : "text-gray-600"
              }`}
            >
              <Icon className={`h-5 w-5 ${active ? "animate-pulse" : ""}`} />
              <span className="text-sm font-medium">{label}</span>
              {done && <span className="ml-auto text-xs">&#10003;</span>}
              {active && (
                <Loader2 className="ml-auto h-4 w-4 animate-spin text-brand-400" />
              )}
            </motion.div>
          );
        })}
      </div>

      <div className="mt-6 space-y-1 text-xs text-gray-600">
        <p className="font-mono text-gray-400">
          {mins}:{secs.toString().padStart(2, "0")} elapsed
        </p>
        <p>
          First run downloads models (~2 GB) and may take{" "}
          <strong className="text-gray-400">5–10 minutes</strong>.
          Subsequent runs will be much faster.
        </p>
      </div>
    </motion.div>
  );
}
