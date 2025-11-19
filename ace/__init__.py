"""Agentic Context Engineering (ACE) reproduction framework."""

from typing import Optional
from .playbook import Bullet, Playbook
from .delta import DeltaOperation, DeltaBatch
from .llm import LLMClient, DummyLLMClient, TransformersLLMClient
from .roles import (
    Generator,
    ReplayGenerator,
    Reflector,
    Curator,
    GeneratorOutput,
    ReflectorOutput,
    CuratorOutput,
)
from .adaptation import (
    OfflineAdapter,
    OnlineAdapter,
    Sample,
    TaskEnvironment,
    SimpleEnvironment,
    EnvironmentResult,
    AdapterStepResult,
)

# Import SimpleAgent (high-level convenience wrapper)
try:
    from .agent import SimpleAgent as _SimpleAgent

    SimpleAgent: Optional[type] = _SimpleAgent
except ImportError:
    SimpleAgent: Optional[type] = None  # type: ignore

# Import optional feature detection
from .features import has_opik, has_litellm

# Import observability components if available
if has_opik():
    try:
        from .observability import OpikIntegration as _OpikIntegration

        OpikIntegration: Optional[type] = _OpikIntegration
        OBSERVABILITY_AVAILABLE = True
    except ImportError:
        OpikIntegration: Optional[type] = None  # type: ignore
        OBSERVABILITY_AVAILABLE = False
else:
    OpikIntegration: Optional[type] = None  # type: ignore
    OBSERVABILITY_AVAILABLE = False

# Import production LLM clients if available
if has_litellm():
    try:
        from .llm_providers import LiteLLMClient as _LiteLLMClient

        LiteLLMClient: Optional[type] = _LiteLLMClient
        LITELLM_AVAILABLE = True
    except ImportError:
        LiteLLMClient: Optional[type] = None  # type: ignore
        LITELLM_AVAILABLE = False
else:
    LiteLLMClient: Optional[type] = None  # type: ignore
    LITELLM_AVAILABLE = False

# Import integrations (browser-use, etc.) if available
try:
    from .integrations import (
        ACEAgent as _ACEAgent,
        wrap_playbook_context as _wrap_playbook_context,
        BROWSER_USE_AVAILABLE as _BROWSER_USE_AVAILABLE,
    )

    ACEAgent: Optional[type] = _ACEAgent
    wrap_playbook_context: Optional[type] = _wrap_playbook_context  # type: ignore
    BROWSER_USE_AVAILABLE = _BROWSER_USE_AVAILABLE
except ImportError:
    ACEAgent: Optional[type] = None  # type: ignore
    wrap_playbook_context: Optional[type] = None  # type: ignore
    BROWSER_USE_AVAILABLE = False

__all__ = [
    # Core components
    "Bullet",
    "Playbook",
    "DeltaOperation",
    "DeltaBatch",
    "LLMClient",
    "DummyLLMClient",
    "TransformersLLMClient",
    "LiteLLMClient",
    "Generator",
    "ReplayGenerator",
    "Reflector",
    "Curator",
    "GeneratorOutput",
    "ReflectorOutput",
    "CuratorOutput",
    "OfflineAdapter",
    "OnlineAdapter",
    "Sample",
    "TaskEnvironment",
    "SimpleEnvironment",
    "EnvironmentResult",
    "AdapterStepResult",
    # Out-of-box agents
    "SimpleAgent",
    "ACEAgent",  # Browser-use integration
    # Utilities
    "wrap_playbook_context",
    # Feature flags
    "OpikIntegration",
    "LITELLM_AVAILABLE",
    "OBSERVABILITY_AVAILABLE",
    "BROWSER_USE_AVAILABLE",
]
