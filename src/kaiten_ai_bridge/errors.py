"""Domain-specific failures without user content in their string representation."""

from __future__ import annotations


class BridgeError(Exception):
    """Base error carrying a stable, log-safe code."""

    code = "bridge_error"
    retryable = False

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.code)


class ConfigurationError(BridgeError):
    code = "configuration_error"


class InvalidWebhookError(BridgeError):
    code = "invalid_webhook"


class SourceNotReadyError(BridgeError):
    code = "source_not_ready"
    retryable = True


class KaitenUnavailableError(BridgeError):
    code = "kaiten_unavailable"
    retryable = True


class KaitenProtocolError(BridgeError):
    code = "kaiten_protocol_error"
    retryable = True


class ReceiverUnavailableError(BridgeError):
    code = "receiver_unavailable"
    retryable = True


class ReceiverRejectedError(BridgeError):
    code = "receiver_rejected"
    retryable = True


class DataRejectedError(BridgeError):
    """A user-correctable request problem; do not retry the same event version."""

    code = "data_rejected"

    def __init__(self, code: str, user_message: str) -> None:
        self.code = code
        self.user_message = user_message
        super().__init__(code)


class UnsupportedResultFilesError(BridgeError):
    code = "result_files_not_configured"
