import { Camera, Sparkles } from "lucide-react";

export default function Header() {
  return (
    <header className="border-b border-white/5 bg-gray-950/80 backdrop-blur-lg sticky top-0 z-50">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4 sm:px-6 lg:px-8">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-brand-400 to-brand-600 shadow-lg shadow-brand-500/25">
            <Camera className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white">
              Shot<span className="gradient-text">Story</span>
            </h1>
            <p className="hidden text-xs text-gray-500 sm:block">
              Explainable Image-to-Story Generation
            </p>
          </div>
        </div>

        {/* Badge */}
        <div className="flex items-center gap-1.5 rounded-full border border-brand-500/30 bg-brand-500/10 px-3 py-1 text-xs font-medium text-brand-400">
          <Sparkles className="h-3 w-3" />
          Visual Feature Attribution
        </div>
      </div>
    </header>
  );
}
