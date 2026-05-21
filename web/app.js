const form = document.querySelector("#generateForm");
const output = document.querySelector("#output");
const outputLabel = document.querySelector("#outputLabel");
const charCount = document.querySelector("#charCount");
const generateButton = document.querySelector("#generateButton");
const copyButton = document.querySelector("#copyButton");
const modelStatus = document.querySelector("#modelStatus");
const lengthInput = document.querySelector("#length");
const lengthValue = document.querySelector("#lengthValue");
const temperatureInput = document.querySelector("#temperature");
const temperatureValue = document.querySelector("#temperatureValue");

function syncRangeLabels() {
  lengthValue.textContent = lengthInput.value;
  temperatureValue.textContent = Number(temperatureInput.value).toFixed(1);
}

async function loadModelStatus() {
  try {
    const response = await fetch("/api/models");
    const data = await response.json();
    const available = [];
    if (data.ngram_omer) available.push("Ömer Seyfettin");
    if (data.ngram) available.push("N-gram");
    if (data.transformer) available.push("Transformer");
    modelStatus.textContent = available.length ? `${available.join(" + ")} hazır` : "Model bulunamadı";
  } catch {
    modelStatus.textContent = "Sunucu bekleniyor";
  }
}

function setOutput(text, isError = false) {
  output.classList.toggle("error", isError);
  output.textContent = text;
  charCount.textContent = `${text.length} karakter`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData(form);
  const payload = {
    prompt: data.get("prompt"),
    model: data.get("model"),
    length: Number(data.get("length")),
    temperature: Number(data.get("temperature")),
    top_k: Number(data.get("topK")),
    seed: Number(data.get("seed")),
  };

  generateButton.disabled = true;
  outputLabel.textContent = "Üretiliyor";
  setOutput("");

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Üretim başarısız");
    outputLabel.textContent = `${payload.model} sonucu`;
    setOutput(result.text);
  } catch (error) {
    outputLabel.textContent = "Hata";
    setOutput(error.message, true);
  } finally {
    generateButton.disabled = false;
  }
});

copyButton.addEventListener("click", async () => {
  const text = output.textContent.trim();
  if (!text) return;
  await navigator.clipboard.writeText(text);
  outputLabel.textContent = "Kopyalandı";
});

lengthInput.addEventListener("input", syncRangeLabels);
temperatureInput.addEventListener("input", syncRangeLabels);

syncRangeLabels();
loadModelStatus();
