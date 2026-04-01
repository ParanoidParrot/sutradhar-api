# Sutradhar API
### Multilingual AI storyteller backend for Indian epics and scriptures

---

## Stack
| Component | Technology |
|-----------|------------|
| API Framework | FastAPI |
| LLM | Sarvam-M (24B) |
| Translation | Sarvam Translate (mayura:v1) |
| Speech to Text | Sarvam Saarika v2.5 |
| Text to Speech | Sarvam Bulbul v3 |
| Vector Database | Pinecone |
| Embeddings | multilingual-e5-large (Pinecone inference) |

---

## Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/sutradhar-api
cd sutradhar-api

# 2. Create venv with Python 3.11
python3.11 -m venv venv311
source venv311/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add API keys
cp .env.example .env
# Fill in SARVAM_API_KEY and PINECONE_API_KEY

# 5. Create Pinecone index (run once)
python3 setup_pinecone.py

# 6. Seed knowledge base (run once)
python3 ingest.py --seed

# 7. Run the API
uvicorn main:app --reload
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/scriptures` | List available scriptures |
| GET | `/storytellers` | List all storytellers |
| GET | `/storytellers/{scripture}` | Storytellers for a scripture |
| GET | `/languages` | Supported languages |
| POST | `/ask` | Ask a text question |
| POST | `/ask/voice` | Ask via audio file |
| POST | `/tts` | Text to speech |

---

## Adding New Documents

```bash
# Ingest a PDF
python3 ingest.py --file ramayana_vol1.pdf --scripture ramayana --source "Valmiki Ramayana Vol 1" --kanda "Bala Kanda"

# Ingest a URL
python3 ingest.py --url https://sacred-texts.com/hin/rama/rama.htm --scripture ramayana --source "Sacred Texts"

# Ingest a Word doc
python3 ingest.py --file notes.docx --scripture ramayana --source "My Notes"
```

---

## Adding New Scriptures
1. Add namespace to `scriptures.json`
2. Add storyteller to `storytellers.json`
3. Run `ingest.py` with the new scripture namespace