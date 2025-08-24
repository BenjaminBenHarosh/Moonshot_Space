from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt


# --- Domain objects ----------------------------------------------------------

@dataclass
class Capsule:
    mass: float  # kg
    diameter: float  # m (kept for completeness)
    x0: float = 0.0  # m
    v0: float = 0.0  # m/s


@dataclass
class Tube:
    length: float  # m
    inner_diameter: float  # m


@dataclass
class Coil:
    center: float  # m (position of coil center along the tube)
    length: float  # m (axial span of active force)
    force: float  # N (constant magnitude when engaged)
    name: Optional[str] = None

    @property
    def start(self) -> float:
        return self.center - 0.5 * self.length

    @property
    def end(self) -> float:
        return self.center + 0.5 * self.length


# --- Coil placement helpers -------------------------------------------------

def evenly_spaced_coils_inside(tube: Tube, n: int, coil_length: float, force: float) -> List[Coil]:
    """
    Place n coils evenly so that each coil is fully inside the tube.
    Centers are positioned in [coil_length/2, tube.length - coil_length/2].
    This maximizes usable acceleration force within the tube.
    """
    if coil_length > tube.length:
        raise ValueError("coil_length must be <= tube length")
    if n <= 0:
        return []

    if n == 1:
        # Single coil should be at tube center
        centers = [tube.length / 2.0]
    else:
        half = coil_length / 2.0
        centers = np.linspace(half, tube.length - half, n)

    return [Coil(center=float(c), length=coil_length, force=force, name=f"C{i + 1}")
            for i, c in enumerate(centers)]


def spans_from_entrance(tube: Tube, n: int, coil_length: float, force: float) -> List[Coil]:
    """
    Place n coils so that the FIRST coil starts at x=0 (immediate acceleration).
    Subsequent coils are spaced evenly by start positions.
    Overlaps are allowed if coil_length is large vs spacing.
    """
    if coil_length <= 0:
        raise ValueError("coil_length must be > 0")
    if coil_length > tube.length:
        raise ValueError("coil_length must be <= tube length")
    if n <= 0:
        return []

    # Space the starts evenly from 0 to (tube.length - coil_length)
    if n == 1:
        starts = [0.0]
    else:
        max_start = max(0.0, tube.length - coil_length)
        starts = np.linspace(0.0, max_start, n)

    coils = []
    for i, s in enumerate(starts):
        center = float(s) + 0.5 * coil_length
        # Clamp center to ensure coil stays within tube bounds
        center = min(max(center, 0.5 * coil_length), tube.length - 0.5 * coil_length)
        coils.append(Coil(center=center, length=coil_length, force=force, name=f"C{i + 1}"))

    return coils


@dataclass
class SimulationResult:
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    events: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    def to_dataframe(self):
        import pandas as pd

        # Get energy and power data from metadata if available
        kinetic_energy = self.metadata.get("kinetic_energy_j", [0.0] * len(self.time))
        force_applied = self.metadata.get("force_applied_n", [0.0] * len(self.time))
        power_consumption = self.metadata.get("power_consumption_w", [0.0] * len(self.time))

        return pd.DataFrame(
            {
                "time": self.time,
                "position": self.position,
                "velocity": self.velocity,
                "acceleration": self.acceleration,
                "kinetic_energy": kinetic_energy,
                "force_applied": force_applied,
                "power_consumption": power_consumption,
            }
        )


# --- Core simulator ----------------------------------------------------------

class Simulator:
    """
    1D capsule accelerated by electromagnetic coils using span-based force model.

    The capsule experiences constant force while passing through each coil's span
    [coil.start, coil.end]. Multiple coil forces add linearly when spans overlap.
    Outside all coils, the capsule moves at constant velocity.

    Integration: semi-implicit Euler (velocity updated before position)
    """

    def __init__(
            self,
            capsule: Capsule,
            tube: Tube,
            coils: List[Coil],
            dt: float = 1e-3,
            max_time: float = 10.0,
            stop_at_exit: bool = True,
    ):
        self.capsule = capsule
        self.tube = tube
        self.coils = coils
        self.dt = dt
        self.max_time = max_time
        self.stop_at_exit = stop_at_exit

    # In Simulator class

    def _is_coil_engaged(self, x: float, coil) -> bool:
        """Return True if position x is within the coil's active span."""
        eps = 1e-12
        left = coil.center - coil.length / 2.0
        right = coil.center + coil.length / 2.0
        return (left - eps) <= x <= (right + eps)

    def _total_force(self, x: float) -> float:
        """Sum constant forces from all coils whose spans contain x."""
        F = 0.0
        for coil in self.coils:
            if self._is_coil_engaged(x, coil):
                F += coil.force  # positive = forward thrust
        return F

    def _calculate_theoretical_work_done(self) -> float:
        """
        Calculate theoretical work done by all coils.

        This assumes the capsule travels through the entire length of each coil,
        which may overestimate work if coils extend beyond the tube or if the
        capsule doesn't traverse the full coil length.
        """
        total_work = 0.0
        for coil in self.coils:
            # Work = Force × distance
            # For overlapping coils, this gives an upper bound estimate
            effective_length = min(coil.length, self.tube.length)
            total_work += coil.force * effective_length
        return total_work

    def run(self) -> SimulationResult:
        """Run the simulation and return results (pos & acc aligned at the same instant)."""
        m = self.capsule.mass
        dt = self.dt
        tube_length = self.tube.length

        # Initial state
        t = 0.0
        x = self.capsule.x0
        v = self.capsule.v0

        # Output arrays
        times, positions, velocities, accelerations = [], [], [], []
        kinetic_energies, forces_applied, power_consumption = [], [], []
        events: List[Dict[str, Any]] = []

        # Event tracking
        previously_engaged = [False] * len(self.coils)
        t_exit = None
        has_exited = False

        # Helper to compute total force at a position (with tiny tolerance when used by _is_coil_engaged)
        def total_force_at(pos: float) -> float:
            return self._total_force(pos) if pos < tube_length else 0.0

        while t <= self.max_time:
            # Early stop after exit if requested
            if self.stop_at_exit and has_exited:
                break

            # Exit event detection at CURRENT x (before stepping)
            if not has_exited and x >= tube_length:
                t_exit = t
                has_exited = True
                events.append({
                    "event": "tube_exit",
                    "time": t,
                    "position": x,
                    "velocity": v,
                })

            # ---- integration step (semi-implicit Euler) using acceleration at CURRENT x ----
            F_now = total_force_at(x)
            a_now = F_now / m
            v = v + a_now * dt
            x = x + v * dt
            t = t + dt
            # -----------------------------------------------------------------------------

            # Recompute force/acceleration at the NEW position so recordings match position
            F_rec = total_force_at(x)
            a_rec = F_rec / m

            # Coil engagement events based on NEW position (which we are recording)
            if x < tube_length:
                for i, coil in enumerate(self.coils):
                    engaged = self._is_coil_engaged(x, coil)
                    if engaged and not previously_engaged[i]:
                        events.append({
                            "event": "coil_engaged",
                            "coil": coil.name or f"C{i + 1}",
                            "time": t,
                            "position": x,
                            "force": coil.force,
                        })
                    elif not engaged and previously_engaged[i]:
                        events.append({
                            "event": "coil_disengaged",
                            "coil": coil.name or f"C{i + 1}",
                            "time": t,
                            "position": x,
                        })
                    previously_engaged[i] = engaged

            # ---- record (position, acceleration, etc. are from the SAME instant/position) ----
            times.append(t)
            positions.append(x)
            velocities.append(v)
            accelerations.append(a_rec)

            ke = 0.5 * m * v * v
            power = F_rec * v
            kinetic_energies.append(ke)
            forces_applied.append(F_rec)
            power_consumption.append(power)
            # -------------------------------------------------------------------------------

        # Energy metrics
        initial_ke = 0.5 * m * self.capsule.v0 ** 2
        final_ke = kinetic_energies[-1] if kinetic_energies else initial_ke
        energy_gained = final_ke - initial_ke

        theoretical_work = self._calculate_theoretical_work_done()
        if theoretical_work > 1e-12:
            efficiency = (energy_gained / theoretical_work) * 100.0
        else:
            efficiency = 0.0 if abs(energy_gained) < 1e-12 else 100.0

        # Peaks
        max_acceleration = max(accelerations) if accelerations else 0.0
        max_force = max(forces_applied) if forces_applied else 0.0
        max_power = max(power_consumption) if power_consumption else 0.0

        # Average power during acceleration (until exit if known)
        if t_exit is not None and power_consumption:
            exit_idx = min(len(times) - 1, int(t_exit / dt))
            avg_power_accel = (sum(power_consumption[:exit_idx + 1]) / (exit_idx + 1)) if exit_idx >= 0 else 0.0
        else:
            avg_power_accel = (sum(power_consumption) / len(power_consumption)) if power_consumption else 0.0

        return SimulationResult(
            time=np.asarray(times, dtype=float),
            position=np.asarray(positions, dtype=float),
            velocity=np.asarray(velocities, dtype=float),
            acceleration=np.asarray(accelerations, dtype=float),
            events=events,
            metadata={
                "dt_s": dt,
                "tube_length_m": tube_length,
                "capsule_mass_kg": self.capsule.mass,
                "initial_velocity_mps": self.capsule.v0,
                "initial_position_m": self.capsule.x0,
                "coil_centers_m": [c.center for c in self.coils],
                "coil_length_m": self.coils[0].length if self.coils else None,
                "coil_force_n": self.coils[0].force if self.coils else None,
                "num_coils": len(self.coils),
                "max_time_s": self.max_time,
                "exit_time_s": t_exit,
                "final_velocity_mps": velocities[-1] if velocities else 0.0,
                "final_position_m": positions[-1] if positions else 0.0,
                # Energy metrics
                "initial_kinetic_energy_j": initial_ke,
                "final_kinetic_energy_j": final_ke,
                "energy_gained_j": energy_gained,
                "theoretical_work_j": theoretical_work,
                "energy_efficiency_percent": efficiency,
                # Performance metrics
                "max_acceleration_ms2": max_acceleration,
                "max_force_n": max_force,
                "max_power_w": max_power,
                "avg_power_during_accel_w": avg_power_accel,
                # Traces
                "kinetic_energy_j": kinetic_energies,
                "force_applied_n": forces_applied,
                "power_consumption_w": power_consumption,
            },
        )

    def plot_coils(self):
        """Visualize coil positions along the tube."""
        plt.figure(figsize=(10, 2))

        # Draw tube
        plt.hlines(1, 0, self.tube.length, colors='black', linewidth=4, label='Tube')

        # Draw coils
        for i, coil in enumerate(self.coils):
            plt.hlines(1, coil.start, coil.end, colors='red', linewidth=12, alpha=0.7)
            plt.text(coil.center, 1.02, coil.name or f"C{i + 1}",
                     ha="center", va="bottom", fontweight='bold')

        plt.title("Coil Positions Inside Tube")
        plt.xlabel("Position (m)")
        plt.ylabel("")
        plt.yticks([])
        plt.xlim(-0.02, self.tube.length + 0.02)
        plt.ylim(0.9, 1.15)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_kinematics(result: SimulationResult, t_max: Optional[float] = None):
        """Plot position, velocity, and acceleration vs time."""
        t = result.time
        # Use actual simulation end time, not the configured max_time
        t_limit = t_max if t_max is not None else (t[-1] if len(t) > 0 else 1.0)

        plots = [
            ("position", "Position (m)", "Position vs Time"),
            ("velocity", "Velocity (m/s)", "Velocity vs Time"),
            ("acceleration", "Acceleration (m/s²)", "Acceleration vs Time"),
        ]

        for attr_name, ylabel, title in plots:
            y = getattr(result, attr_name)
            plt.figure(figsize=(10, 6))
            plt.plot(t, y, linewidth=2)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel("Time (s)")
            plt.ylabel(ylabel)
            plt.xlim(0.0, t_limit)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()