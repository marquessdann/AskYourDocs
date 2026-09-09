class AskYourDocsError(Exception):
    """Base class for domain errors that the API translates into clean JSON responses."""


class NoRelevantContextError(AskYourDocsError):
    """Raised when no ingested chunk is similar enough to the question to answer safely."""

    def __init__(self, best_score: float, threshold: float):
        self.best_score = best_score
        self.threshold = threshold
        super().__init__(
            f"No chunk reached the minimum relevance score (best={best_score:.3f}, "
            f"required>={threshold:.3f})."
        )


class EmptyKnowledgeBaseError(AskYourDocsError):
    """Raised when a question is asked before any document has been ingested."""


class LLMProviderError(AskYourDocsError):
    """Raised when the configured LLM or embedding provider fails or is misconfigured."""


class UnsupportedFileError(AskYourDocsError):
    """Raised when the uploaded file is not a supported/parsable PDF."""


class RateLimitExceededError(AskYourDocsError):
    """Raised when a client exceeds the configured request rate."""

    def __init__(self, retry_after_seconds: float):
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"Rate limit exceeded. Retry after {retry_after_seconds:.0f}s.")
