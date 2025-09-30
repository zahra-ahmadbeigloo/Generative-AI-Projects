import os

def get_openai_api_key() -> str:
    # Return the OpenAI API key from environment or a local .env file.
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key

    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip()
    raise RuntimeError("OPENAI_API_KEY is not set. Export it or add to a .env file.")
