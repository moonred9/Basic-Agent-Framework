class AgentFrameworkError(Exception):
    """Base exception for the framework."""


class ToolNotFoundError(AgentFrameworkError):
    """Raised when a requested tool is not registered."""


class ToolExecutionError(AgentFrameworkError):
    """Raised when a tool fails to execute."""


class ParseError(AgentFrameworkError):
    """Raised when LLM output cannot be parsed into a supported action."""


class MaxStepsExceededError(AgentFrameworkError):
    """Raised when the agent reaches max_steps without an answer."""
