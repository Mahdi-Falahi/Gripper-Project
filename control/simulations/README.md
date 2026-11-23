# Gripper control simulations

[Project overview](../../README.md)

These Simulink models explore how an electric gripper moves toward an object, switches to force control after contact, holds the object, and releases it. Each model generates its own intent signal and simulates the drive, motor, gearbox, jaws, object contact, and sensor feedback.

## Model variants

| Model | What it illustrates |
| --- | --- |
| [`EMG_Gripper_System.slx`](EMG_Gripper_System.slx) | Baseline hybrid controller with PI force regulation and a motor-current-based estimate of grip force. |
| [`EMG_Gripper_System_enhanced.slx`](EMG_Gripper_System_enhanced.slx) | Adds simulated left and right strain gauges, a jaw-speed reference for approach, and contact and grip-hold refinements. The current-based force estimate is logged for comparison. |
| [`EMG_Gripper_System_LADRC.slx`](EMG_Gripper_System_LADRC.slx) | Builds on the strain-gauge model and uses second-order linear active disturbance rejection control (LADRC) in force hold. Its extended state observer estimates force, force rate, and the lumped disturbance. |

## Run a simulation

The models were saved with MATLAB and Simulink R2025a. From this folder in MATLAB, open the model you want and select **Run** in Simulink. For example, run and plot the LADRC force response from the MATLAB command window.

```matlab
model = 'EMG_Gripper_System_LADRC';
open_system(model);
out = sim(model);
plot(out.force_ref_log.Time, out.force_ref_log.Data, '--', ...
     out.force_log.Time, out.force_log.Data);
xlabel('Time (s)');
ylabel('Force (N)');
legend('Force reference', 'Force feedback');
```

Start with **System Response Scope** or compare `force_ref_log` and `force_log`. Check `mode_log`, `position_log`, and `current_log` to relate force changes to the controller state, jaw motion, and actuator current. The enhanced and LADRC models also expose `true_contact_force_log` for the simulated contact force. The LADRC model adds `eso_disturbance_log` for observer diagnostics.

## Included plots

| Plot | Comparison |
| --- | --- |
| [`conversion to strain gauge.png`](conversion%20to%20strain%20gauge.png) | Simulated strain-gauge and motor-current force estimates against the model's contact force. |
| [`enhancements on gripping.png`](enhancements%20on%20gripping.png) | Baseline and enhanced approach/contact behavior, including peak force, contact loss, and hold error. |
| [`conversion to adrc.png`](conversion%20to%20adrc.png) | Conventional and LADRC force responses during contact and a grip-force ramp, with summary metrics. |

The plots show simulation behavior. Bench tests will establish how the sensing and controllers perform on the physical gripper.
