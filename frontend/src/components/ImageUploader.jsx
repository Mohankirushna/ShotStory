import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, ImageIcon, AlertCircle } from "lucide-react";
import { motion } from "framer-motion";

export default function ImageUploader({ onUpload, error }) {
  const onDrop = useCallback(
    (accepted) => {
      if (accepted.length) onUpload(accepted[0]);
    },
    [onUpload]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".png", ".jpg", ".jpeg", ".webp"] },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024,
  });

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="mx-auto max-w-2xl"
    >
      {/* Heading */}
      <div className="mb-10 text-center">
        <h2 className="mb-3 text-4xl font-extrabold tracking-tight text-white">
          Upload an <span className="gradient-text">Image</span>
        </h2>
        <p className="text-gray-400">
          Drop a photo and ShotStory will generate a narrative with transparent
          visual explanations showing <em>why</em> each story element was chosen.
        </p>
      </div>

      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={`group cursor-pointer rounded-2xl border-2 border-dashed p-14 text-center transition-all duration-300 ${
          isDragActive
            ? "scale-[1.02] border-brand-400 bg-brand-400/10"
            : "border-gray-700 hover:border-brand-500/50 hover:bg-white/[0.03]"
        }`}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center gap-4">
          {isDragActive ? (
            <Upload className="h-16 w-16 animate-bounce text-brand-400" />
          ) : (
            <ImageIcon className="h-16 w-16 text-gray-600 transition-colors group-hover:text-gray-400" />
          )}
          <div>
            <p className="text-lg font-medium text-gray-300">
              {isDragActive
                ? "Drop your image here"
                : "Drag & drop an image here"}
            </p>
            <p className="mt-1 text-sm text-gray-500">
              or click to browse &middot; PNG, JPG, WebP up to 10 MB
            </p>
          </div>
        </div>
      </div>

      {/* Error */}
      {error && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-5 flex items-start gap-3 rounded-xl border border-red-800/60 bg-red-900/20 p-4"
        >
          <AlertCircle className="mt-0.5 h-5 w-5 flex-shrink-0 text-red-400" />
          <p className="text-sm text-red-300">{error}</p>
        </motion.div>
      )}
    </motion.div>
  );
}
