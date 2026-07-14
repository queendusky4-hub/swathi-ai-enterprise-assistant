# Swathi AI — Multilingual Tamil Assistant

A portfolio-ready conversational AI application supporting **Tamil, Tanglish, and English**. It combines deterministic intent routing, an optional fine-tuned BERT classifier, SQLite conversation history, and an optional internet-hosted OpenAI-compatible LLM.

## Architecture

```text
Streamlit UI → language detection → rule router → BERT intent classifier → hosted LLM fallback
                                      ↓
                              multilingual replies
                                      ↓
                                SQLite history
```

## Features

- Tamil, Tanglish and English language detection
- Fast rule-based responses for known intents
- Optional local fine-tuned BERT intent classifier
- Optional hosted LLM for open-ended questions
- Persistent SQLite conversation history
- FastAPI endpoints with automatic OpenAPI documentation
- Docker and Docker Compose
- pytest, Ruff and GitHub Actions CI
- Safe fallback when the model or internet provider is unavailable

## Local setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
streamlit run app.py
```


## Run the API

```bash
uvicorn swathi_ai.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for Swagger UI.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"vanakkam","online":false,"show_all_formats":false}'
```

Available endpoints:

- `GET /health`
- `GET /model/status`
- `POST /chat`

## Add the trained model

Place the complete Hugging Face model directory in `model/`. It must include the model weights (`model.safetensors` or `pytorch_model.bin`) together with `config.json` and tokenizer files. Training checkpoints such as `scheduler.pt`, `rng_state.pth`, and `training_args.bin` are not required for inference.

## Enable online AI

Copy `.env.example` to `.env` and configure any service exposing an OpenAI-compatible `/chat/completions` API:

```env
LLM_BASE_URL=https://your-provider.example/v1
LLM_API_KEY=your-secret
LLM_MODEL=your-model-name
```

Never commit `.env` or API keys.

## Tests and quality

```bash
pytest -q
ruff check src tests app.py
docker build -t swathi-ai .
# or run both UI and API
docker compose up --build
```

## Deployment

For Streamlit Community Cloud, connect this GitHub repository, use `app.py` as the entry point, and add environment values through the platform's secrets/settings. The BERT model can be committed only when its size is suitable; otherwise download it from a model registry during deployment.

## Project status

The uploaded files did not include model weights, so the repository runs immediately using rule-based routing and optional hosted LLM. Add the original trained weights to activate BERT inference.
