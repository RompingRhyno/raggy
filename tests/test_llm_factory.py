import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from raggy import llm_factory


def _fake_provider_module(name: str, attrs: dict):
    mod = ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def test_get_llm_ollama(monkeypatch):
    class FakeChatOllama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "langchain_ollama",
        _fake_provider_module("langchain_ollama", {"ChatOllama": FakeChatOllama}),
    )

    llm = llm_factory.get_llm("ollama", "llama3.2", 0.0)

    assert isinstance(llm, FakeChatOllama)
    assert llm.kwargs == {"model": "llama3.2", "temperature": 0.0}


def test_get_llm_openai(monkeypatch):
    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        _fake_provider_module("langchain_openai", {"ChatOpenAI": FakeChatOpenAI}),
    )

    llm = llm_factory.get_llm("openai", "gpt-4o", 0.7)

    assert isinstance(llm, FakeChatOpenAI)
    assert llm.kwargs["model"] == "gpt-4o"
    assert llm.kwargs["temperature"] == 0.7
    assert llm.kwargs["api_key"] is None  # resolves from env at runtime


def test_get_llm_anthropic_sets_max_tokens(monkeypatch):
    class FakeChatAnthropic:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "langchain_anthropic",
        _fake_provider_module(
            "langchain_anthropic", {"ChatAnthropic": FakeChatAnthropic}
        ),
    )

    llm = llm_factory.get_llm("anthropic", "claude-sonnet", 0.0)

    assert isinstance(llm, FakeChatAnthropic)
    assert llm.kwargs["model"] == "claude-sonnet"
    assert llm.kwargs["api_key"] is None
    assert llm.kwargs["max_tokens"] == llm_factory._DEFAULT_MAX_TOKENS


def test_get_llm_google(monkeypatch):
    class FakeChatGoogle:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "langchain_google_genai",
        _fake_provider_module(
            "langchain_google_genai", {"ChatGoogleGenerativeAI": FakeChatGoogle}
        ),
    )

    llm = llm_factory.get_llm("google", "gemini-1.5-pro", 0.3)

    assert isinstance(llm, FakeChatGoogle)
    assert llm.kwargs["model"] == "gemini-1.5-pro"
    assert llm.kwargs["temperature"] == 0.3
    assert llm.kwargs["api_key"] is None


def test_get_llm_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown llm_provider"):
        llm_factory.get_llm("grok", "grok-1", 0.0)


def without_the_api(monkeypatch):
    """Force the CLI fallback: pretend the Ollama server cannot be reached.

    Presence is normally checked over the HTTP API first (the Windows desktop
    app installs a server with no CLI on PATH), so these CLI-focused tests have
    to take the API out of the picture to exercise the branch they are about.
    """

    def unavailable():
        raise ConnectionError("connection refused")

    monkeypatch.setattr(llm_factory, "_installed_models_via_api", unavailable)


def test_ensure_ollama_model_already_present(monkeypatch):
    without_the_api(monkeypatch)
    monkeypatch.setattr(llm_factory.shutil, "which", lambda _: "/usr/local/bin/ollama")

    def fake_run(cmd, **kwargs):
        assert kwargs.pop("capture_output") is True
        assert kwargs.pop("text") is True
        assert kwargs.pop("check", False) is False

        class Result:
            stdout = (
                "NAME\tID\tSIZE\tMODIFIED\n"
                "llama3.2:latest\ta80c4f17acd5\t2.0 GB\t2 days ago\n"
                "nomic-embed-text:latest\t0a109f422b47\t274 MB\t2 days ago\n"
            )
            stderr = ""

        return Result()

    monkeypatch.setattr(llm_factory.subprocess, "run", fake_run)

    assert llm_factory.ensure_ollama_model("llama3.2") is False


def test_ensure_ollama_model_pulls_missing(monkeypatch):
    without_the_api(monkeypatch)
    monkeypatch.setattr(llm_factory.shutil, "which", lambda _: "/usr/local/bin/ollama")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((list(cmd), dict(kwargs)))

        class Result:
            stdout = "NAME\tID\tSIZE\tMODIFIED\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(llm_factory.subprocess, "run", fake_run)

    assert llm_factory.ensure_ollama_model("llama3.2") is True
    assert calls[-1][0] == ["/usr/local/bin/ollama", "pull", "llama3.2"]
    assert calls[-1][1]["check"] is True


def test_ensure_ollama_model_no_cli_raises(monkeypatch):
    without_the_api(monkeypatch)
    monkeypatch.setattr(llm_factory.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="CLI is not on"):
        llm_factory.ensure_ollama_model("llama3.2")


def test_ensure_ollama_model_uses_the_api_when_it_is_reachable(monkeypatch):
    """The Windows desktop app runs a server with no CLI on PATH."""
    monkeypatch.setattr(
        llm_factory, "_installed_models_via_api", lambda: {"llama3.2:latest"}
    )
    monkeypatch.setattr(llm_factory.shutil, "which", lambda _: None)

    assert llm_factory.ensure_ollama_model("llama3.2") is False


def test_ensure_ollama_model_pulls_over_the_api_without_a_cli(monkeypatch):
    """A model missing from a CLI-less install is still fetched, over the API."""
    monkeypatch.setattr(llm_factory, "_installed_models_via_api", lambda: set())
    monkeypatch.setattr(llm_factory.shutil, "which", lambda _: None)
    pulled = []

    def fake_pull(model, stream=False):
        pulled.append((model, stream))
        return []

    monkeypatch.setattr("ollama.pull", fake_pull)

    assert llm_factory.ensure_ollama_model("llama3.2") is True
    assert pulled == [("llama3.2", True)]


def test_ensure_ollama_model_already_present_with_explicit_latest_tag(monkeypatch):
    without_the_api(monkeypatch)
    monkeypatch.setattr(llm_factory.shutil, "which", lambda _: "/usr/local/bin/ollama")

    def fake_run(cmd, **kwargs):
        class Result:
            stdout = (
                "NAME\tID\tSIZE\tMODIFIED\n"
                "llama3.2:latest\ta80c4f17acd5\t2.0 GB\t2 days ago\n"
            )
            stderr = ""

        return Result()

    monkeypatch.setattr(llm_factory.subprocess, "run", fake_run)
    assert llm_factory.ensure_ollama_model("llama3.2:latest") is False


def test_ensure_ollama_model_missing_tag_triggers_pull(monkeypatch):
    without_the_api(monkeypatch)
    monkeypatch.setattr(llm_factory.shutil, "which", lambda _: "/usr/local/bin/ollama")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((list(cmd), dict(kwargs)))

        class Result:
            stdout = (
                "NAME\tID\tSIZE\tMODIFIED\n"
                "llama3.2:latest\ta80c4f17acd5\t2.0 GB\t2 days ago\n"
            )
            stderr = ""

        return Result()

    monkeypatch.setattr(llm_factory.subprocess, "run", fake_run)

    assert llm_factory.ensure_ollama_model("llama3.2:q4_K_M") is True
    assert calls[-1][0] == ["/usr/local/bin/ollama", "pull", "llama3.2:q4_K_M"]


def test_ensure_ollama_model_pull_failure_raises_runtime_error(monkeypatch):
    without_the_api(monkeypatch)
    monkeypatch.setattr(llm_factory.shutil, "which", lambda _: "/usr/local/bin/ollama")

    def fake_run(cmd, **kwargs):
        if cmd[1] == "list":

            class ListResult:
                stdout = "NAME\tID\tSIZE\tMODIFIED\n"
                stderr = ""

            return ListResult()

        exc = subprocess.CalledProcessError(1, cmd)
        exc.stderr = "error: pull model manifest: file does not exist"
        raise exc

    monkeypatch.setattr(llm_factory.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="Failed to pull Ollama model 'ghost'"):
        llm_factory.ensure_ollama_model("ghost")


def _fake_pull_module(events, calls):
    """A stand-in for the ``ollama`` package exposing a streaming ``pull``."""

    def pull(model, stream=False):
        calls.append((model, stream))
        return iter(events)

    return _fake_provider_module("ollama", {"pull": pull})


def test_ensure_ollama_model_streams_pull_progress(monkeypatch):
    without_the_api(monkeypatch)
    monkeypatch.setattr(llm_factory.shutil, "which", lambda _: "/usr/local/bin/ollama")

    ran = []

    def fake_run(cmd, **kwargs):
        ran.append(list(cmd))

        class Result:
            stdout = "NAME\tID\tSIZE\tMODIFIED\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(llm_factory.subprocess, "run", fake_run)

    events = [
        SimpleNamespace(status="pulling manifest", completed=None, total=None),
        SimpleNamespace(status="pulling 970aa74c", completed=100, total=274),
        SimpleNamespace(status="pulling 970aa74c", completed=274, total=274),
        SimpleNamespace(status="success", completed=None, total=None),
    ]
    calls = []
    monkeypatch.setitem(sys.modules, "ollama", _fake_pull_module(events, calls))

    reported = []
    assert (
        llm_factory.ensure_ollama_model(
            "nomic-embed-text",
            progress=lambda message, completed=None, total=None: reported.append(
                (message, completed, total)
            ),
        )
        is True
    )

    assert calls == [("nomic-embed-text", True)]
    assert reported == [
        ("pulling nomic-embed-text ... (pulling manifest)", None, None),
        ("pulling nomic-embed-text ...", 100, 274),
        ("pulling nomic-embed-text ...", 274, 274),
        ("pulling nomic-embed-text ... (success)", None, None),
    ]
    # The pull went over the API; the CLI was only used to list what is present.
    assert ran == [["/usr/local/bin/ollama", "list"]]


def test_ensure_ollama_model_streamed_pull_failure_raises_runtime_error(monkeypatch):
    without_the_api(monkeypatch)
    monkeypatch.setattr(llm_factory.shutil, "which", lambda _: "/usr/local/bin/ollama")

    def fake_run(cmd, **kwargs):
        class Result:
            stdout = "NAME\tID\tSIZE\tMODIFIED\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(llm_factory.subprocess, "run", fake_run)

    def exploding_pull(model, stream=False):
        raise ConnectionError("[Errno 61] Connection refused")

    monkeypatch.setitem(
        sys.modules, "ollama", _fake_provider_module("ollama", {"pull": exploding_pull})
    )

    with pytest.raises(RuntimeError, match="Failed to pull Ollama model 'ghost'"):
        llm_factory.ensure_ollama_model("ghost", progress=lambda *a, **kw: None)
