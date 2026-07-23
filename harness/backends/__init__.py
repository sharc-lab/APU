"""Backend implementations for cloud/local model endpoints."""

from harness.backends.base import Backend, ModelCallResult
from harness.backends.cloud_openai import CloudOpenAIBackend
from harness.backends.local_ollama import LocalOllamaBackend

__all__ = ["Backend", "ModelCallResult", "CloudOpenAIBackend", "LocalOllamaBackend"]
