import { motion } from "framer-motion";
import { ArrowLeft } from "lucide-react";
import AttributionMap from "./AttributionMap";
import StoryDisplay from "./StoryDisplay";
import FeaturePanel from "./FeaturePanel";
import XAIReasoningPanel from "./XAIReasoningPanel";

export default function ResultsView({ results, onReset }) {
  const {
    original_image,
    story,
    caption,
    scene_attributes,
    objects,
    visualizations,
    attributions,
    xai_reasoning,
  } = results;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
    >
      {/* Top bar */}
      <div className="mb-6 flex items-center justify-between">
        <button
          onClick={onReset}
          className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-4 py-2 text-sm font-medium text-gray-300 transition-colors hover:bg-white/10"
        >
          <ArrowLeft className="h-4 w-4" />
          New Image
        </button>
        <p className="text-xs text-gray-600">
          Analysis complete &middot;{" "}
          {objects.length} object{objects.length !== 1 ? "s" : ""} detected
        </p>
      </div>

      {/* Three-column layout */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Left – image + overlays + story */}
        <div className="space-y-6">
          <AttributionMap
            original={original_image}
            visualizations={visualizations}
          />
          <StoryDisplay story={story} />
        </div>

        {/* Center – features & attributions */}
        <div>
          <FeaturePanel
            caption={caption}
            sceneAttributes={scene_attributes}
            objects={objects}
            attributions={attributions}
          />
        </div>

        {/* Right – XAI reasoning */}
        <div>
          <XAIReasoningPanel xaiReasoning={xai_reasoning} />
        </div>
      </div>
    </motion.div>
  );
}
