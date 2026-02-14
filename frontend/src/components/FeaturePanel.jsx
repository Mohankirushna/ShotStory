import { motion } from "framer-motion";
import {
  Eye, MapPin, Clock, Cloud, Palette, Heart, Zap, User, MessageSquare,
} from "lucide-react";

const ATTR_ICONS = {
  mood: Heart,
  setting: MapPin,
  time_of_day: Clock,
  weather: Cloud,
  colors: Palette,
  emotions: Heart,
  activity: Zap,
  main_subject: User,
};

export default function FeaturePanel({ caption, sceneAttributes, objects, attributions }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      className="space-y-5"
    >
      {/* Caption */}
      <div className="glass-card p-5">
        <div className="mb-3 flex items-center gap-2">
          <MessageSquare className="h-4 w-4 text-brand-400" />
          <h4 className="text-sm font-semibold text-white">Image Caption</h4>
        </div>
        <p className="text-sm leading-relaxed text-gray-300 italic">&ldquo;{caption}&rdquo;</p>
      </div>

      {/* Scene Attributes */}
      <div className="glass-card p-5">
        <div className="mb-3 flex items-center gap-2">
          <Eye className="h-4 w-4 text-brand-400" />
          <h4 className="text-sm font-semibold text-white">Scene Attributes</h4>
        </div>
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(sceneAttributes).map(([key, val]) => {
            const Icon = ATTR_ICONS[key] || Eye;
            return (
              <div
                key={key}
                className="flex items-start gap-2 rounded-lg bg-white/[0.03] p-2.5"
              >
                <Icon className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-brand-500" />
                <div className="min-w-0">
                  <p className="text-[10px] font-medium uppercase tracking-wider text-gray-500">
                    {key.replace(/_/g, " ")}
                  </p>
                  <p className="truncate text-xs text-gray-300">{val}</p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Detected Objects */}
      {objects.length > 0 && (
        <div className="glass-card p-5">
          <div className="mb-3 flex items-center gap-2">
            <Zap className="h-4 w-4 text-brand-400" />
            <h4 className="text-sm font-semibold text-white">
              Detected Objects{" "}
              <span className="text-gray-500">({objects.length})</span>
            </h4>
          </div>
          <div className="space-y-2">
            {objects.map((obj, i) => (
              <div
                key={i}
                className="flex items-center justify-between rounded-lg bg-white/[0.03] px-3 py-2"
              >
                <span className="text-sm text-gray-300 capitalize">
                  {obj.label}
                </span>
                <div className="flex items-center gap-2">
                  <div className="h-1.5 w-20 overflow-hidden rounded-full bg-gray-800">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-brand-500 to-brand-400"
                      style={{ width: `${obj.confidence * 100}%` }}
                    />
                  </div>
                  <span className="w-10 text-right text-[11px] text-gray-500">
                    {(obj.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Attributions */}
      {attributions.length > 0 && (
        <div className="glass-card p-5">
          <div className="mb-3 flex items-center gap-2">
            <Zap className="h-4 w-4 text-brand-400" />
            <h4 className="text-sm font-semibold text-white">
              Feature → Story Attributions
            </h4>
          </div>
          <div className="space-y-3">
            {attributions.slice(0, 8).map((attr, i) => (
              <div
                key={i}
                className="rounded-lg border border-white/5 bg-white/[0.02] p-3"
              >
                <div className="mb-1 flex items-center justify-between">
                  <span className="rounded-md bg-brand-500/15 px-2 py-0.5 text-xs font-medium text-brand-300 capitalize">
                    {attr.feature_type}
                  </span>
                  <span className="text-[11px] text-gray-500">
                    importance: {(attr.importance_score * 100).toFixed(0)}%
                  </span>
                </div>
                <p className="mb-1 text-sm font-medium text-gray-200">
                  {attr.visual_feature}
                </p>
                <p className="text-xs text-gray-500">{attr.story_influence}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  );
}
