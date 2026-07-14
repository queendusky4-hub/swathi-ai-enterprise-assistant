# Replace these files now

Copy the following files into your current project, preserving the same paths:

- `app.py`
- `src/swathi_ai/config.py`
- `src/swathi_ai/database.py`
- `src/swathi_ai/api.py`

Also ensure `python-dotenv` is installed:

```powershell
python -m pip install python-dotenv
python -m pip install -e .
```

Restart FastAPI:

```powershell
python -m uvicorn swathi_ai.api:app --reload
```

Restart Streamlit in a second terminal:

```powershell
python -m streamlit run app.py
```

Check:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/model/status`
- `http://127.0.0.1:8000/docs`
- `http://localhost:8501`
