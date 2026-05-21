from __future__ import annotations

import json
import urllib.request


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    base_url = "http://127.0.0.1:7860"
    models = get_json(f"{base_url}/api/models")
    print(f"models={models}")
    result = post_json(
        f"{base_url}/api/generate",
        {
            "model": "ngram",
            "prompt": "Bir sabah",
            "length": 120,
            "temperature": 0.7,
            "top_k": 8,
            "seed": 7,
        },
    )
    text = result["text"]
    print(f"generated_chars={len(text)}")
    print(text[:160])


if __name__ == "__main__":
    main()
