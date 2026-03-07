"""
rag.py
Core RAG pipeline for Sutradhar API:
  1. Embed query using Pinecone inference
  2. Retrieve relevant passages from Pinecone
  3. Generate answer using Sarvam-M
  4. Translate answer using Sarvam Translate
"""

import os
import json
import requests
from pinecone import Pinecone
from sarvamai import SarvamAI
from dotenv import load_dotenv

load_dotenv()

# ── Clients ───────────────────────────────────────────────────────────────────
SARVAM_API_KEY  = os.getenv("SARVAM_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX  = os.getenv("PINECONE_INDEX", "sutradhar")

sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
pc            = Pinecone(api_key=PINECONE_API_KEY)

# ── Load config ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "storytellers.json")) as f:
    STORYTELLERS = {s["id"]: s for s in json.load(f)["storytellers"]}

with open(os.path.join(BASE_DIR, "scriptures.json")) as f:
    SCRIPTURES = {s["id"]: s for s in json.load(f)["scriptures"]}

# ── Supported languages ───────────────────────────────────────────────────────
LANGUAGE_CODES = {
    "English":   "en-IN",
    "Hindi":     "hi-IN",
    "Tamil":     "ta-IN",
    "Telugu":    "te-IN",
    "Kannada":   "kn-IN",
    "Malayalam": "ml-IN",
    "Bengali":   "bn-IN",
    "Marathi":   "mr-IN",
    "Gujarati":  "gu-IN",
    "Punjabi":   "pa-IN",
    "Odia":      "or-IN",
}

TTS_VOICES = {
    "English":   "shubh",
    "Hindi":     "anushka",
    "Tamil":     "abhilasha",
    "Telugu":    "anushka",
    "Kannada":   "anushka",
    "Malayalam": "anushka",
    "Bengali":   "anushka",
    "Marathi":   "anushka",
    "Gujarati":  "anushka",
    "Punjabi":   "anushka",
    "Odia":      "anushka",
}


# ── Pinecone index ────────────────────────────────────────────────────────────
def get_index():
    return pc.Index(PINECONE_INDEX)


# ── Embed text via Pinecone inference ─────────────────────────────────────────
def embed_text(text: str) -> list[float]:
    result = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[text],
        parameters={"input_type": "query"}
    )
    return result[0].values


# ── Retrieve passages from Pinecone ──────────────────────────────────────────
def retrieve_passages(query: str, scripture: str, n_results: int = 4) -> list[dict]:
    index     = get_index()
    namespace = SCRIPTURES[scripture]["pinecone_namespace"]
    vector    = embed_text(query)

    results = index.query(
        vector=vector,
        top_k=n_results,
        namespace=namespace,
        include_metadata=True
    )

    passages = []
    for match in results.matches:
        meta = match.metadata or {}
        passages.append({
            "text":       meta.get("text", ""),
            "source":     meta.get("source", ""),
            "kanda":      meta.get("kanda", ""),
            "topic":      meta.get("topic", ""),
            "score":      round(match.score, 3)
        })
    return passages


# ── Generate answer using Sarvam-M ────────────────────────────────────────────
def generate_answer(query: str, passages: list[dict], storyteller_id: str) -> str:
    storyteller = STORYTELLERS[storyteller_id]

    context_parts = []
    for i, p in enumerate(passages, 1):
        label = f"[Passage {i}"
        if p.get("kanda"):
            label += f" — {p['kanda']}"
        if p.get("source"):
            label += f", Source: {p['source']}"
        label += "]"
        context_parts.append(f"{label}\n{p['text']}")
    context = "\n\n".join(context_parts)

    user_message = (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer as {storyteller['name']}:"
    )

    url     = "https://api.sarvam.ai/v1/chat/completions"
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type":         "application/json"
    }
    payload = {
        "model": "sarvam-m",
        "messages": [
            {"role": "system", "content": storyteller["system_prompt"]},
            {"role": "user",   "content": user_message}
        ],
        "max_tokens": 512
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


# ── Sarvam Translate ──────────────────────────────────────────────────────────
def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    if source_lang == target_lang:
        return text

    url     = "https://api.sarvam.ai/translate"
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type":         "application/json"
    }
    payload = {
        "input":                text,
        "source_language_code": source_lang,
        "target_language_code": target_lang,
        "model":                "mayura:v1",
        "mode":                 "formal"
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json().get("translated_text", text)


# ── Speech to Text (Saarika v2.5) ─────────────────────────────────────────────
def speech_to_text(audio_bytes: bytes, language: str = "English") -> dict:
    lang_code = LANGUAGE_CODES.get(language, "en-IN")

    try:
        url     = "https://api.sarvam.ai/speech-to-text"
        headers = {"api-subscription-key": SARVAM_API_KEY}
        files   = {"file": ("audio.wav", audio_bytes, "audio/wav")}
        data    = {"model": "saarika:v2.5", "language_code": lang_code}

        response = requests.post(url, headers=headers, files=files, data=data)
        response.raise_for_status()
        result = response.json()
        return {
            "transcript":    result.get("transcript", ""),
            "language_code": result.get("language_code", lang_code),
            "error":         None
        }
    except Exception as e:
        return {"transcript": "", "language_code": lang_code, "error": str(e)}


# ── Text to Speech (Bulbul v3) ────────────────────────────────────────────────
def text_to_speech(text: str, language: str = "English") -> dict:
    import base64
    lang_code = LANGUAGE_CODES.get(language, "en-IN")
    speaker   = TTS_VOICES.get(language, "shubh")

    if len(text) > 2500:
        text = text[:2490] + "..."

    try:
        response     = sarvam_client.text_to_speech.convert(
            text=text,
            target_language_code=lang_code,
            model="bulbul:v3",
            speaker=speaker
        )
        audio_bytes  = base64.b64decode(response.audios[0])
        return {"audio_bytes": audio_bytes, "error": None}
    except Exception as e:
        return {"audio_bytes": None, "error": str(e)}


# ── Main ask pipeline ─────────────────────────────────────────────────────────
def ask(
    query:       str,
    scripture:   str = "ramayana",
    storyteller: str = "valmiki",
    language:    str = "English"
) -> dict:
    lang_code = LANGUAGE_CODES.get(language, "en-IN")

    try:
        # Step 1 — Translate query to English for retrieval
        query_en = query
        if lang_code != "en-IN":
            query_en = translate_text(query, source_lang=lang_code, target_lang="en-IN")

        # Step 2 — Retrieve relevant passages (filter low-relevance)
        RELEVANCE_THRESHOLD = 0.45
        all_passages = retrieve_passages(query_en, scripture=scripture)
        passages = [p for p in all_passages if p["score"] >= RELEVANCE_THRESHOLD]

        # Step 3 — Generate answer in English
        answer_en = generate_answer(query_en, passages, storyteller_id=storyteller)

        # Step 4 — Translate answer back to user's language
        answer = answer_en
        if lang_code != "en-IN":
            answer = translate_text(answer_en, source_lang="en-IN", target_lang=lang_code)

        return {
            "answer":      answer,
            "answer_en":   answer_en,
            "passages":    passages,
            "scripture":   scripture,
            "storyteller": storyteller,
            "language":    language,
            "error":       None
        }

    except Exception as e:
        return {
            "answer":    None,
            "passages":  [],
            "scripture": scripture,
            "language":  language,
            "error":     str(e)
        }