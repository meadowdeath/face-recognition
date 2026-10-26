"""Read one aggregate JSON report; atomically publish configuration only."""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path

from face_recognition.domain.models.open_set_config import ConfigurationFreezeError, FrozenOpenSetConfiguration
from face_recognition.infrastructure.persistence.baseline_evaluation_report import write_aggregate_report_json


@dataclass(frozen=True)
class AggregateDevelopmentReport:
    payload: dict = field(repr=False)
    sha256: str


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError


def read_aggregate_development_report(path: Path) -> AggregateDevelopmentReport:
    """Hash the same byte snapshot that is decoded; no other files are opened."""
    try:
        if Path(path).suffix.lower() != ".json":
            raise ValueError
        content = Path(path).read_bytes()
        data = json.loads(content.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        if not isinstance(data, dict):
            raise ValueError
        return AggregateDevelopmentReport(data, sha256(content).hexdigest())
    except Exception:
        raise ConfigurationFreezeError("Unable to read a valid aggregate development JSON report") from None


def frozen_configuration_to_json(configuration: FrozenOpenSetConfiguration) -> str:
    if configuration != FrozenOpenSetConfiguration(configuration.provenance, configuration.development_evidence):
        raise ConfigurationFreezeError("Only the exact frozen development configuration may be published")
    data = asdict(configuration)
    data["architecture"]["svm_parameters"] = dict(data["architecture"]["svm_parameters"])
    data["architecture"]["scaler_parameters"] = dict(data["architecture"]["scaler_parameters"])
    data["metric_definitions"]["definitions"] = dict(data["metric_definitions"]["definitions"])
    try:
        return json.dumps(data, allow_nan=False, sort_keys=True, indent=2) + "\n"
    except (ValueError, TypeError):
        raise ConfigurationFreezeError("Invalid frozen configuration metadata") from None


def write_frozen_configuration(configuration: FrozenOpenSetConfiguration, output: Path, development_report: Path) -> None:
    write_aggregate_report_json(frozen_configuration_to_json(configuration), output, (development_report,))
