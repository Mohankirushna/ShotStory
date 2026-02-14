import axios from "axios";

const api = axios.create({ baseURL: "/api", timeout: 600_000 }); // 10-min timeout for first-run model downloads

/**
 * Upload an image and receive the full analysis + story + attributions.
 */
export async function analyzeImage(file) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post("/analyze", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function healthCheck() {
  const { data } = await api.get("/health");
  return data;
}

export async function preloadModels() {
  const { data } = await api.post("/preload");
  return data;
}
