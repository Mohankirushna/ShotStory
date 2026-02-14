import { useState } from "react";
import Header from "./components/Header";
import ImageUploader from "./components/ImageUploader";
import ResultsView from "./components/ResultsView";
import LoadingOverlay from "./components/LoadingOverlay";
import { analyzeImage } from "./api/client";

export default function App() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadedPreview, setUploadedPreview] = useState(null);

  const handleUpload = async (file) => {
    setLoading(true);
    setError(null);
    setResults(null);
    setUploadedPreview(URL.createObjectURL(file));

    try {
      const data = await analyzeImage(file);
      if (data.success) {
        setResults(data);
      } else {
        setError("Analysis returned an unexpected response.");
      }
    } catch (err) {
      const msg =
        err.response?.data?.detail ||
        err.message ||
        "Something went wrong. Is the backend running?";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResults(null);
    setError(null);
    setUploadedPreview(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950">
      <Header />

      <main className="mx-auto max-w-7xl px-4 pb-20 pt-6 sm:px-6 lg:px-8">
        {!results && !loading && (
          <ImageUploader onUpload={handleUpload} error={error} />
        )}

        {loading && <LoadingOverlay preview={uploadedPreview} />}

        {results && (
          <ResultsView results={results} onReset={handleReset} />
        )}
      </main>
    </div>
  );
}
