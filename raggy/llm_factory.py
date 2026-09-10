"""Provider-agnostic LLM construction.

Generation can run locally via Ollama (default) or remotely via a cloud
provider. API keys are read from the provider's standard environment
variable at construction time (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``,
``GEMINI_API_KEY``); they are never stored in config.yaml.
"""

import logging
import shutil
import subprocess

from langchain_core.language_models import BaseChatModel

from .progress import ProgressCallback

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("ollama", "openai", "anthropic", "google")

# Anthropic requires an explicit max_tokens; the others are fine with a sane
# default. Kept as a constant so remotes behave consistently out of the box.
# A value is only mandatory for anthropic, but setting it uniformly avoids
# surprise across providers.
_DEFAULT_MAX_TOKENS = 1024


def _stream_pull(model: str, progress: ProgressCallback) -> None:
    """Pull ``model`` through Ollama's API, reporting progress as it downloads.

    The ``ollama pull`` CLI draws its own progress bars straight to the
    terminal, which a front end can neither label nor place. Pulling over the
    API instead yields the same work as ``status``/``completed``/``total``
    events, so the caller decides how it looks.
    """
    from ollama import pull

    for event in pull(model, stream=True):
        if event.total:
            progress(f"pulling {model} ...", event.completed or 0, event.total)
        elif event.status:
            progress(f"pulling {model} ... ({event.status})")


def _normalized(model: str) -> str:
    """The name Ollama stores: a bare name means the ``:latest`` tag."""
    return model if ":" in model else f"{model}:latest"


def _installed_models_via_api() -> set[str]:
    """Model names already available, asked of the running server.

    The HTTP API is the primary source on purpose. The desktop app on Windows
    installs the server without putting the ``ollama`` CLI on PATH, so a
    CLI-only check would refuse to answer questions on a machine where every
    model is present and the server is up.
    """
    from ollama import Client

    response = Client().list()
    models = getattr(response, "models", None) or response.get("models") or []
    names: set[str] = set()
    for entry in models:
        name = getattr(entry, "model", None) or getattr(entry, "name", None)
        if name is None and isinstance(entry, dict):
            name = entry.get("model") or entry.get("name")
        if name:
            names.add(_normalized(str(name)))
    return names


def _installed_models_via_cli(exe: str) -> set[str]:
    """Model names already available, read from ``ollama list``.

    ``ollama list`` reports downloaded models without touching the network. Its
    output is tab-delimited with the model name in the first column.
    """
    listed = subprocess.run([exe, "list"], capture_output=True, text=True, check=False)
    names: set[str] = set()
    for line in listed.stdout.splitlines():
        fields = line.split()
        if fields:
            names.add(_normalized(fields[0]))
    return names


def ensure_ollama_model(model: str, progress: ProgressCallback | None = None) -> bool:
    """Pull ``model`` into the local Ollama instance if it isn't already present.

    Returns True if a pull was performed (model was missing), False if the model
    was already present. Raises ``RuntimeError`` if no model list can be read
    (Ollama not running / not installed) or the pull itself fails.

    Presence is checked against the running server first and the CLI second, so
    a machine where Ollama runs as an app with no CLI on PATH still works.

    Without ``progress`` the pull runs as ``ollama pull <model>`` and Ollama
    renders its own output; pass a callback to receive the download progress
    instead and keep the terminal under the caller's control.
    """
    requested = _normalized(model)
    via_cli = False
    try:
        if requested in _installed_models_via_api():
            return False
    except Exception as e:  # noqa: BLE001 - any transport/library failure falls back
        logger.debug("Could not list installed models over the API: %s", e)
        via_cli = True

    if via_cli:
        exe = shutil.which("ollama")
        if exe is None:
            raise RuntimeError(
                "Could not reach the model service and the `ollama` CLI is not on "
                "PATH. Start Ollama (`ollama serve`, or the desktop app) and try "
                "again."
            )
        try:
            if requested in _installed_models_via_cli(exe):
                return False
        except OSError as cli_error:
            raise RuntimeError(f"Failed to run ollama CLI: {cli_error}") from cli_error

    logger.info("Pulling Ollama model '%s' (first run may take a while)...", model)
    try:
        # The CLI is only an option when it exists: pulling over the API is
        # equivalent and works on a machine that installed the desktop app.
        exe = shutil.which("ollama")
        if progress is None and exe is not None:
            subprocess.run([exe, "pull", model], check=True)
        else:
            _stream_pull(model, progress)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to pull Ollama model '{model}': {e.stderr.strip() or e}"
        ) from e
    except Exception as e:  # API transport/protocol failures
        raise RuntimeError(f"Failed to pull Ollama model '{model}': {e}") from e
    return True


def get_llm(
    provider: str,
    model_name: str,
    temperature: float,
) -> BaseChatModel:
    """Initialize and return a chat language model for ``provider``.

    All supported providers implement LangChain's ``BaseChatModel`` interface,
    so callers use ``.invoke()`` / ``.stream()`` identically regardless of
    which one is selected; only construction differs.
    """
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model_name, temperature=temperature)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=None,  # falls back to OPENAI_API_KEY env var
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            api_key=None,  # falls back to ANTHROPIC_API_KEY env var
            max_tokens=_DEFAULT_MAX_TOKENS,
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            api_key=None,  # falls back to GEMINI_API_KEY env var
        )

    raise ValueError(
        f"Unknown llm_provider {provider!r}; expected one of {SUPPORTED_PROVIDERS}"
    )
