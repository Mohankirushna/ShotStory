import { useState } from "react";
import { motion } from "framer-motion";
import { Layers, Image as ImageIcon, ScanEye, Box } from "lucide-react";

const VIEWS = [
  { key: "original", label: "Original", Icon: ImageIcon },
  { key: "attention", label: "Attention Map", Icon: ScanEye },
  { key: "objects", label: "Object Detection", Icon: Box },
];

export default function AttributionMap({ original, visualizations }) {
  const [view, setView] = useState("original");

  const srcs = {
    original: `data:image/png;base64,${original}`,
    attention: `data:image/png;base64,${visualizations.attention_overlay}`,
    objects: `data:image/png;base64,${visualizations.object_overlay}`,
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card overflow-hidden"
    >
      {/* Toolbar */}
      <div className="flex items-center gap-1 border-b border-white/10 px-3 py-2">
        <Layers className="mr-2 h-4 w-4 text-gray-500" />
        {VIEWS.map(({ key, label, Icon }) => (
          <button
            key={key}
            onClick={() => setView(key)}
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
              view === key
                ? "bg-brand-500/20 text-brand-300"
                : "text-gray-500 hover:bg-white/5 hover:text-gray-300"
            }`}
          >
            <Icon className="h-3.5 w-3.5" />
            {label}
          </button>
        ))}
      </div>

      {/* Image */}
      <div className="relative">
        <img
          src={srcs[view]}
          alt={view}
          className="h-auto w-full object-contain"
        />
        {view === "attention" && (
          <div className="absolute bottom-3 left-3 rounded-lg bg-black/70 px-3 py-1.5 text-[11px] text-gray-300 backdrop-blur">
            Warm regions = higher model attention
          </div>
        )}
      </div>
    </motion.div>
  );
}
