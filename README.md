# EMG-driven gripper for proportional grasping

Ongoing undergraduate research in electrical engineering at Sharif University of Technology. The project develops a myoelectric interface for a single-degree-of-freedom electric gripper.

## Problem

A user needs to control both how fast the jaws move and how firmly they grip an object. Surface electromyography (sEMG) measures forearm muscle activity, but noise at rest and overlapping muscle activation make it difficult to infer when the user intends to move or how much effort they intend to apply. Once the jaws touch an object, the resulting force also depends on the gripper and the object, including drivetrain friction and contact stiffness.

## Approach

1. **Decode intent from sEMG.** We process signals from a forearm finger flexor and extensor to detect when movement starts, infer its direction, and estimate proportional commands. The repository's offline experiments use prepared Ninapro DB2 recordings to study onset detection and velocity estimation. The planned decoder also estimates the desired grip force for use after contact.
2. **Execute the command with local feedback.** The controller is designed to use position feedback for jaw speed before contact and strain-gauge feedback for force regulation after contact. A supervisory controller manages approach, contact, hold, and release. We are studying conventional PI/PID force control and linear active disturbance rejection control (LADRC).

The planned interface carries the operating mode and separate speed and force references to the controller. The controller returns contact and status information so the decoder can select the appropriate reference as the grasp progresses.

## Repository guide

| Where | What to look for |
| --- | --- |
| [Project overview (PDF)](project%20overview%20%28gets%20updated%20over%20time%29.pdf) | Two-page summary of the research, preliminary simulation, and validation plan. |
| [`EMG/simulations/`](EMG/simulations/README.md) | MATLAB scripts for onset detection, kinematic labeling, and feature extraction, a Python analysis notebook, and an STM32 sampling sketch. The guide explains the prepared data inputs and which scripts produce each feature table. |
| [`control/simulations/`](control/simulations/README.md) | Three Simulink model variants and plots from the sensing and controller comparisons. The guide explains how to run a model and inspect its outputs. |

## Current status

The repository contains offline EMG experiments and control simulations. The Simulink models use generated intent signals, so readers can explore the controller without EMG recordings. Bench calibration, matched controller trials on the gripper, and integration with the live EMG decoder remain planned work.
