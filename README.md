# AdvoMate (Legal Advisor)

A Django-based legal advisor application with an embedded rule-based / neural chatbot for basic legal Q&A.

## Quick start 

1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run migrations and start Django server:

```bash
python Ai_based_Advo_Chat\legal_advisor\manage.py migrate
python Ai_based_Advo_Chat\legal_advisor\manage.py runserver
```

## Generated artifacts & important files 🔧

These files are generated or large/binary artifacts and should NOT be committed to git (they are already ignored by `.gitignore`):

- `chatbot_model.h5` — trained Keras/TensorFlow model
- `words.pkl`, `classes.pkl` — tokenization/vocabulary pickles
- `db.sqlite3` — local SQLite database (development only)
- `static/media/` (uploaded user files)

> Note: If you need to recreate the chatbot model locally, follow the steps below.

## Regenerating the chatbot model (training) 🔁

1. Ensure `intents.json` is present at `Ai_based_Advo_Chat/legal_advisor/intents.json`.
2. From the repository root run:

```bash
python Ai_based_Advo_Chat\legal_advisor\train_chatbot.py
```

This script uses NLTK and Keras/TensorFlow and will download the required NLTK resources if missing.

If you prefer, you can run the training script from inside the `Ai_based_Advo_Chat/legal_advisor/` directory.

## Notes & Best Practices ✅

- Do not commit generated model files or uploaded media; keep them out of source control.
- If you accidentally committed large files, consider removing them from history (BFG or git filter-repo).
- For reproducible environments, pin package versions in `requirements.txt` or use `pip freeze > requirements.txt` from a clean venv.

