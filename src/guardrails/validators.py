"""Guardrails AI validators for input/output validation."""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    value: str
    errors: list[str] = field(default_factory=list)


class InputValidator:
    """Validate user input before processing."""

    INJECTION_PATTERNS = [
        r"ignore\s+(previous|all|any)\s+instructions",
        r"forget\s+(everything|all|previous)",
        r"disregard\s+(all|previous|any)",
        r"new\s+instructions?\s*:",
        r"system\s*:\s*you\s+are",
        r"pretend\s+you\s+are",
    ]

    def __init__(
        self,
        max_length: int = 10000,
        check_injection: bool = True,
    ):
        """Initialize input validator.

        Args:
            max_length: Maximum allowed input length.
            check_injection: Whether to check for injection attempts.
        """
        self.max_length = max_length
        self.check_injection = check_injection
        self._injection_regex = re.compile(
            "|".join(self.INJECTION_PATTERNS), re.IGNORECASE
        )

    def validate(self, text: str) -> ValidationResult:
        """Validate input text.

        Args:
            text: The input text to validate.

        Returns:
            ValidationResult with validity and any errors.
        """
        errors = []

        # Check for empty input
        if not text or not text.strip():
            errors.append("Input is empty or contains only whitespace")
            return ValidationResult(is_valid=False, value=text, errors=errors)

        # Check length
        if len(text) > self.max_length:
            errors.append(
                f"Input length ({len(text)}) exceeds maximum ({self.max_length})"
            )

        # Check for injection attempts
        if self.check_injection and self._injection_regex.search(text):
            errors.append("Input contains potential injection patterns")

        return ValidationResult(
            is_valid=len(errors) == 0,
            value=text,
            errors=errors,
        )


class OutputValidator:
    """Validate model output before returning to user."""

    PII_PATTERNS = [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # Phone number
        r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",  # SSN
        r"\b\d{16}\b",  # Credit card (basic)
    ]

    TOXIC_PATTERNS = [
        r"\b(idiot|stupid|dumb|moron)\b",
        r"\bi\s+hate\s+you\b",
        r"\byou\s+are\s+(an?\s+)?(idiot|stupid|worthless)\b",
    ]

    def __init__(
        self,
        check_pii: bool = True,
        check_toxicity: bool = True,
        max_length: int = 50000,
    ):
        """Initialize output validator.

        Args:
            check_pii: Whether to check for PII.
            check_toxicity: Whether to check for toxic content.
            max_length: Maximum allowed output length.
        """
        self.check_pii = check_pii
        self.check_toxicity = check_toxicity
        self.max_length = max_length

        self._pii_regex = re.compile("|".join(self.PII_PATTERNS), re.IGNORECASE)
        self._toxic_regex = re.compile("|".join(self.TOXIC_PATTERNS), re.IGNORECASE)

    def validate(self, text: str) -> ValidationResult:
        """Validate output text.

        Args:
            text: The output text to validate.

        Returns:
            ValidationResult with validity and any errors.
        """
        errors = []

        # Check for empty output
        if not text or not text.strip():
            errors.append("Output is empty")
            return ValidationResult(is_valid=False, value=text, errors=errors)

        # Check length
        if len(text) > self.max_length:
            errors.append(
                f"Output length ({len(text)}) exceeds maximum ({self.max_length})"
            )

        # Check for PII
        if self.check_pii and self._pii_regex.search(text):
            errors.append("Output contains potential PII (email, phone, SSN, etc.)")

        # Check for toxicity
        if self.check_toxicity and self._toxic_regex.search(text):
            errors.append("Output contains potentially toxic content")

        return ValidationResult(
            is_valid=len(errors) == 0,
            value=text,
            errors=errors,
        )


def validate_input(
    text: str,
    max_length: int = 10000,
    check_injection: bool = True,
) -> ValidationResult:
    """Convenience function to validate input.

    Args:
        text: The input text to validate.
        max_length: Maximum allowed length.
        check_injection: Whether to check for injection.

    Returns:
        ValidationResult.
    """
    validator = InputValidator(
        max_length=max_length,
        check_injection=check_injection,
    )
    return validator.validate(text)


def validate_output(
    text: str,
    check_pii: bool = True,
    check_toxicity: bool = True,
    max_length: int = 50000,
) -> ValidationResult:
    """Convenience function to validate output.

    Args:
        text: The output text to validate.
        check_pii: Whether to check for PII.
        check_toxicity: Whether to check for toxicity.
        max_length: Maximum allowed length.

    Returns:
        ValidationResult.
    """
    validator = OutputValidator(
        check_pii=check_pii,
        check_toxicity=check_toxicity,
        max_length=max_length,
    )
    return validator.validate(text)
