from simulator import Capsule, Tube, Simulator, evenly_spaced_coils_inside
from exporter import export_timeseries

def demo_scenario():
    """Run the default scenario from the assignment with continuous motion."""
    print("=== Running Default Assignment Scenario ===")

    # Default scenario parameters
    tube = Tube(length=0.5, inner_diameter=0.09)
    capsule = Capsule(mass=1.0, diameter=0.083, x0=0.0, v0=0.0)

    # 6 coils, each 0.3m long, providing 10N force
    coils = evenly_spaced_coils_inside(tube, n=6, coil_length=0.3, force=10.0)

    # Create and run simulation - continue after tube exit for physics validation
    sim = Simulator(
        capsule=capsule,
        tube=tube,
        coils=coils,
        dt=1e-3,  # 1ms time step
        max_time=2.0,  # Run for 2 seconds to show ballistic phase
        stop_at_exit=False  # Continue after tube exit to validate physics
    )

    result = sim.run()

    # Print summary
    exit_time = result.metadata.get('exit_time_s')
    print(f"Simulation completed:")
    print(f"  Tube exit time: {exit_time:.4f} s" if exit_time else "  Did not exit tube")
    print(f"  Final simulation time: {result.time[-1]:.4f} s")
    print(f"  Final position: {result.metadata.get('final_position_m', 0):.4f} m")
    print(f"  Final velocity: {result.metadata.get('final_velocity_mps', 0):.2f} m/s")
    print(f"  Distance after exit: {result.metadata.get('final_position_m', 0) - tube.length:.4f} m")
    print(f"  Number of events: {len(result.events)}")

    # Energy analysis
    print(f"\nEnergy Analysis:")
    print(f"  Initial kinetic energy: {result.metadata.get('initial_kinetic_energy_j', 0):.3f} J")
    print(f"  Final kinetic energy: {result.metadata.get('final_kinetic_energy_j', 0):.3f} J")
    print(f"  Energy gained: {result.metadata.get('energy_gained_j', 0):.3f} J")
    print(f"  Theoretical work by coils: {result.metadata.get('theoretical_work_j', 0):.3f} J")
    print(f"  Energy efficiency: {result.metadata.get('energy_efficiency_percent', 0):.1f}%")

    # Performance metrics
    print(f"\nPerformance Metrics:")
    print(f"  Maximum acceleration: {result.metadata.get('max_acceleration_ms2', 0):.2f} m/s²")
    print(f"  Maximum force applied: {result.metadata.get('max_force_n', 0):.1f} N")
    print(f"  Maximum power: {result.metadata.get('max_power_w', 0):.1f} W")
    print(f"  Average power during acceleration: {result.metadata.get('avg_power_during_accel_w', 0):.1f} W")

    # Validate constant velocity after tube exit
    if exit_time:
        exit_idx = None
        for i, t in enumerate(result.time):
            if t >= exit_time:
                exit_idx = i
                break

        if exit_idx and exit_idx < len(result.velocity) - 10:
            post_exit_velocities = result.velocity[exit_idx:exit_idx + 10]
            vel_variation = max(post_exit_velocities) - min(post_exit_velocities)
            print(f"  Velocity variation after exit: {vel_variation:.6f} m/s (physics check)")

    # Visualize results
    print("\nGenerating plots...")
    sim.plot_coils()
    Simulator.plot_kinematics(result)

    # Export results
    print("Exporting results...")
    output_path = export_timeseries(result, base_path="outputs/default_scenario", fmt="parquet")
    print(f"Results exported to: {output_path}")

    return result

def main():
    """Run all demonstration scenarios."""
    print("Capsule Acceleration Simulator - Span Force Model")
    print("=" * 50)

    # Run different scenarios
    default_result = demo_scenario()

    print("\n" + "=" * 50)
    print("All scenarios completed!")
    print("Check the 'outputs/' directory for exported data files.")


if __name__ == "__main__":
    main()