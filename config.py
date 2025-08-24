from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml
import json
from pathlib import Path


@dataclass
class CapsuleConfig:
    mass: float  # kg
    diameter: float  # m
    initial_position: float = 0.0  # m
    initial_velocity: float = 0.0  # m/s


@dataclass
class TubeConfig:
    length: float  # m
    inner_diameter: float  # m


@dataclass
class CoilConfig:
    center: float  # m
    length: float  # m
    force: float  # N
    name: Optional[str] = None


@dataclass
class SimulationConfig:
    dt: float = 1e-3  # s
    max_time: float = 10.0  # s
    stop_at_exit: bool = True


@dataclass
class ExportConfig:
    format: str = "parquet"  # json, json.gz, parquet, hdf5
    base_path: str = "outputs/simulation"
    json_compress: bool = True
    parquet_compression: str = "snappy"


@dataclass
class SimulationSpec:
    """Complete specification for a simulation run."""
    capsule: CapsuleConfig
    tube: TubeConfig
    coils: List[CoilConfig]
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SimulationSpec:
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str | Path) -> SimulationSpec:
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SimulationSpec:
        """Create from dictionary (from API or config file)."""
        return cls(
            capsule=CapsuleConfig(**data['capsule']),
            tube=TubeConfig(**data['tube']),
            coils=[CoilConfig(**coil) for coil in data['coils']],
            simulation=SimulationConfig(**data.get('simulation', {})),
            export=ExportConfig(**data.get('export', {})),
            metadata=data.get('metadata', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'capsule': {
                'mass': self.capsule.mass,
                'diameter': self.capsule.diameter,
                'initial_position': self.capsule.initial_position,
                'initial_velocity': self.capsule.initial_velocity,
            },
            'tube': {
                'length': self.tube.length,
                'inner_diameter': self.tube.inner_diameter,
            },
            'coils': [
                {
                    'center': coil.center,
                    'length': coil.length,
                    'force': coil.force,
                    'name': coil.name,
                }
                for coil in self.coils
            ],
            'simulation': {
                'dt': self.simulation.dt,
                'max_time': self.simulation.max_time,
                'stop_at_exit': self.simulation.stop_at_exit,
            },
            'export': {
                'format': self.export.format,
                'base_path': self.export.base_path,
                'json_compress': self.export.json_compress,
                'parquet_compression': self.export.parquet_compression,
            },
            'metadata': self.metadata
        }

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    def to_json(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Predefined configurations for common scenarios
DEFAULT_ASSIGNMENT_CONFIG = SimulationSpec(
    capsule=CapsuleConfig(mass=1.0, diameter=0.083),
    tube=TubeConfig(length=0.5, inner_diameter=0.09),
    coils=[
        CoilConfig(center=0.075, length=0.3, force=10.0, name="C1"),
        CoilConfig(center=0.15, length=0.3, force=10.0, name="C2"),
        CoilConfig(center=0.225, length=0.3, force=10.0, name="C3"),
        CoilConfig(center=0.3, length=0.3, force=10.0, name="C4"),
        CoilConfig(center=0.375, length=0.3, force=10.0, name="C5"),
        CoilConfig(center=0.45, length=0.3, force=10.0, name="C6"),
    ],
    simulation=SimulationConfig(dt=1e-3, max_time=2.0, stop_at_exit=True),
    export=ExportConfig(format="parquet", base_path="outputs/default_scenario"),
    metadata={"scenario": "default_assignment", "description": "Assignment default parameters"}
)


def validate_simulation_spec(spec: SimulationSpec) -> List[str]:
    """Validate simulation specification and return list of errors."""
    errors = []

    # Validate capsule
    if spec.capsule.mass <= 0:
        errors.append("Capsule mass must be positive")
    if spec.capsule.diameter <= 0:
        errors.append("Capsule diameter must be positive")

    # Validate tube
    if spec.tube.length <= 0:
        errors.append("Tube length must be positive")
    if spec.tube.inner_diameter <= 0:
        errors.append("Tube inner diameter must be positive")
    if spec.capsule.diameter >= spec.tube.inner_diameter:
        errors.append("Capsule diameter must be smaller than tube inner diameter")

    # Validate coils
    if not spec.coils:
        errors.append("At least one coil must be specified")

    for i, coil in enumerate(spec.coils):
        if coil.length <= 0:
            errors.append(f"Coil {i + 1} length must be positive")
        if coil.force < 0:
            errors.append(f"Coil {i + 1} force must be non-negative")
        if coil.center < 0 or coil.center > spec.tube.length:
            errors.append(f"Coil {i + 1} center must be within tube bounds")

    # Validate simulation parameters
    if spec.simulation.dt <= 0:
        errors.append("Time step (dt) must be positive")
    if spec.simulation.dt > 0.01:
        errors.append("Time step (dt) should be <= 0.01 for numerical stability")
    if spec.simulation.max_time <= 0:
        errors.append("Max simulation time must be positive")

    # Validate export settings
    valid_formats = {"json", "json.gz", "parquet", "hdf5"}
    if spec.export.format not in valid_formats:
        errors.append(f"Export format must be one of: {valid_formats}")

    return errors