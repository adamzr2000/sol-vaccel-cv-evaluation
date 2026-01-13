import os
from pathlib import Path
from secrets import token_urlsafe

from dotenv import load_dotenv

ENV_FILE = Path(__file__).parent / ".env"

DEFAULT_SECRETS = {
    "SESSION_SECRET_KEY": token_urlsafe(32),
}


def generate_env_file(path=ENV_FILE, defaults=DEFAULT_SECRETS):
    if not path.exists():
        with open(path, "w") as f:
            for key, value in defaults.items():
                f.write(f"{key}={value}\n")


def load_secrets():
    generate_env_file()
    load_dotenv(dotenv_path=ENV_FILE)
    return {
        "SESSION_SECRET_KEY": os.getenv("SESSION_SECRET_KEY"),
    }