"""Local administration outcomes and operation-bound authorization, without GUI/storage imports."""
from dataclasses import dataclass, field
from enum import Enum
import re
from .attendance import DeviceSettings


class AuthOutcome(Enum):
    AUTHENTICATED = 'authenticated'
    INCORRECT = 'incorrect'
    LOCKED = 'locked'
    SETUP_REQUIRED = 'setup_required'
    INVALID = 'invalid'
    FAILED = 'failed'


@dataclass(frozen=True)
class AuthResult:
    outcome: AuthOutcome
    retry_after: int = 0


def validate_pin(pin, confirmation=None, *, new=False):
    if not isinstance(pin, str) or re.fullmatch(r'[0-9]{6}', pin) is None:
        raise ValueError('PIN must contain exactly six ASCII digits')
    if confirmation is not None and pin != confirmation:
        raise ValueError('PIN confirmation does not match')
    if new and (len(set(pin)) == 1 or pin in ('123456', '654321', '012345', '543210', '121212')):
        raise ValueError('Choose a nontrivial PIN')
    return pin


@dataclass(frozen=True)
class OperationAuthorization:
    """Service-issued capability; validity is checked and consumed at the write boundary."""
    _consume: object = field(repr=False, compare=False)

    def require(self, scope):
        self._consume(scope)


class MaintenanceOperation(Enum):
    HISTORY = 'delete_history'
    ADMIN_DATA = 'reset_admin_data'
    PREFERENCES = 'reset_preferences'


@dataclass(frozen=True)
class MaintenancePreview:
    operation: MaintenanceOperation
    sessions: int
    records: int
    participants: int
    fingerprint: str
    preferences: DeviceSettings | None = None
