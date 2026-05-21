from __future__ import annotations

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from turkish_story_lm import NGramLanguageModel


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
NGRAM_MODELS = {
    "ngram": ROOT / "runs" / "ngram_tr.json",
    "ngram_omer": ROOT / "runs" / "ngram_omer_seyfettin.json",
}
NGRAM_CACHE: dict[str, NGramLanguageModel] = {}


class StoryLMHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def log_message(self, format: str, *args) -> None:
        print(f"[web] {self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/models":
            self.write_json(
                {
                    "ngram": NGRAM_MODELS["ngram"].exists(),
                    "ngram_omer": NGRAM_MODELS["ngram_omer"].exists(),
                    "transformer": (ROOT / "runs" / "transformer_tr.pt").exists(),
                }
            )
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/generate":
            self.send_error(404)
            return

        try:
            payload = self.read_json()
            text = generate_story(payload)
            self.write_json({"text": text})
        except Exception as exc:
            self.write_json({"error": str(exc)}, status=400)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw or "{}")

    def write_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def generate_story(payload: dict) -> str:
    model_name = payload.get("model", "ngram")
    prompt = str(payload.get("prompt", "Bir sabah")).strip() or "Bir sabah"
    length = int(payload.get("length", 500))
    temperature = float(payload.get("temperature", 0.8))
    top_k = int(payload.get("top_k", 8))
    seed = int(payload.get("seed", 7))

    if length < 1 or length > 4000:
        raise ValueError("length must be between 1 and 4000")
    if not 0.1 <= temperature <= 2.0:
        raise ValueError("temperature must be between 0.1 and 2.0")
    if top_k < 1 or top_k > 64:
        raise ValueError("top_k must be between 1 and 64")

    if model_name in NGRAM_MODELS:
        model_path = NGRAM_MODELS[model_name]
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path.relative_to(ROOT)} not found; train the model first")
        model = NGRAM_CACHE.get(model_name)
        if model is None:
            model = NGramLanguageModel.load(model_path)
            NGRAM_CACHE[model_name] = model
        return model.generate(
            prompt=prompt,
            max_new_chars=length,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
        )

    if model_name == "transformer":
        return generate_transformer(prompt, length, temperature, top_k, seed)

    raise ValueError("model must be ngram, ngram_omer, or transformer")


def generate_transformer(
    prompt: str,
    length: int,
    temperature: float,
    top_k: int,
    seed: int,
) -> str:
    from turkish_story_lm.tokenizer import CharTokenizer
    from turkish_story_lm.transformer import TinyTransformerLM, TransformerConfig, require_torch

    torch, _, _ = require_torch()
    model_path = ROOT / "runs" / "transformer_tr.pt"
    if not model_path.exists():
        raise FileNotFoundError("runs/transformer_tr.pt not found; train the Transformer first")

    torch.manual_seed(seed)
    checkpoint = torch.load(model_path, map_location="cpu")
    tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"])
    model = TinyTransformerLM(TransformerConfig.from_dict(checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state"])
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    generated = model.generate(
        idx,
        max_new_tokens=length,
        temperature=temperature,
        top_k=top_k,
    )[0].tolist()
    return tokenizer.decode(generated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local Turkish story LM web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), StoryLMHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"serving {url}")
    print("press Ctrl+C to stop")
    server.serve_forever()


if __name__ == "__main__":
    main()
