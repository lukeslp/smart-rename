
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console

# Constants
APP_NAME = "smart-rename"
CONFIG_DIR = Path.home() / ".config" / APP_NAME
CACHE_DIR = Path.home() / ".cache" / APP_NAME
LOG_DIR = Path.home() / ".local" / "share" / APP_NAME

# Ensure directories exist
for directory in [CONFIG_DIR, CACHE_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = CONFIG_DIR / "config.json"

console = Console()

# Supported AI providers
PROVIDERS = {
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "text_model": "gpt-4o-mini",
        "vision_model": "gpt-4o"
    },
    "ollama": {
        "env_key": None,  # No API key needed
        "base_url": "http://localhost:11434/v1",
        "text_model": "llama3.2",
        "vision_model": "llava"
    }
}


def load_config() -> Dict[str, Any]:
    """Load configuration from file or environment."""
    config = {}

    # Load from file
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError:
            pass

    # Environment variables override file config
    if os.getenv("OPENAI_API_KEY"):
        config["openai_api_key"] = os.getenv("OPENAI_API_KEY")

    if os.getenv("AI_PROVIDER"):
        config["ai_provider"] = os.getenv("AI_PROVIDER")

    if os.getenv("UNPAYWALL_EMAIL"):
        config["unpaywall_email"] = os.getenv("UNPAYWALL_EMAIL")

    return config


def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def get_ai_provider() -> str:
    """Get the configured AI provider (openai or ollama)."""
    config = load_config()
    provider = config.get("ai_provider", "openai").lower()

    # Validate provider
    if provider not in PROVIDERS:
        return "openai"
    return provider


def get_ai_api_key() -> Optional[str]:
    """Get the API key for the configured provider."""
    config = load_config()
    provider = get_ai_provider()

    if provider == "ollama":
        return "ollama"  # No key needed, return placeholder

    # Check config first, then environment
    key_name = f"{provider}_api_key"
    if key_name in config:
        return config[key_name]

    env_key = PROVIDERS[provider]["env_key"]
    return os.getenv(env_key) if env_key else None


def get_ai_config(vision: bool = False) -> Dict[str, Any]:
    """Get full AI configuration for the current provider.

    Args:
        vision: If True, return vision-capable model config
    """
    provider = get_ai_provider()
    api_key = get_ai_api_key()

    model_key = "vision_model" if vision else "text_model"

    return {
        "provider": provider,
        "api_key": api_key,
        "base_url": PROVIDERS[provider]["base_url"],
        "model": PROVIDERS[provider][model_key]
    }


def get_unpaywall_email() -> str:
    """Get Unpaywall email, default to None if not set."""
    config = load_config()
    return config.get("unpaywall_email", "anonymous@example.com")
