import { BookOpen } from "lucide-react";
import { motion } from "framer-motion";

export default function StoryDisplay({ story }) {
  if (!story) return null;

  const paragraphs = story.text
    .split("\n")
    .map((p) => p.trim())
    .filter(Boolean);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
      className="glass-card p-6 sm:p-8"
    >
      {/* Header */}
      <div className="mb-6 flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-500/20">
          <BookOpen className="h-5 w-5 text-brand-400" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-white">Generated Story</h3>
          <p className="text-xs text-gray-500">
            Method: <span className="text-brand-400">{story.method}</span>
          </p>
        </div>
      </div>

      {/* Title */}
      <h4 className="mb-4 font-serif text-2xl font-semibold italic text-brand-300">
        &ldquo;{story.title}&rdquo;
      </h4>

      {/* Body */}
      <div className="story-text space-y-3 text-[15px] leading-relaxed text-gray-300">
        {paragraphs.map((p, i) => (
          <p key={i}>{p}</p>
        ))}
      </div>
    </motion.div>
  );
}
