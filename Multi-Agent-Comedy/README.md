# Multi‑Agent Conversation & Stand‑up Comedy (AutoGen demo)

This notebook explores a **multi‑agent conversation** setup that collaborates to draft short stand‑up comedy bits.

## What's inside
- `L1_Multi-Agent_Conversation_and_Stand-up_Comedy.ipynb` — the main notebook
- `utils.py` — helper to load your `OPENAI_API_KEY` from environment or `.env`
- `requirements.txt` — minimal dependencies
- `.gitignore`

## Setup
```bash
# (optional) create a venv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
# set your key (Linux/macOS)
export OPENAI_API_KEY=sk-...
# or create a .env file with: OPENAI_API_KEY=sk-...
```

## Run
Open the notebook and run all cells:
```bash
jupyter notebook L1_Multi-Agent_Conversation_and_Stand-up_Comedy.ipynb
```

## Notes
- The notebook uses the OpenAI API via AutoGen; usage may incur costs.
- If you run into `ModuleNotFoundError: autogen`, ensure `pip install autogen` completed successfully.
