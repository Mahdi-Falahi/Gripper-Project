# EMG processing and analysis

[Project overview](../../README.md)

These experiments use recorded surface electromyography (sEMG) and hand kinematics to study movement onset, direction, and proportional velocity. The scripts focus on hand opening and fist closing in the prepared Ninapro DB2 Exercise 1 recordings. They use signals associated with the extensor digitorum communis (EDC) and flexor digitorum superficialis (FDS).

## Data and setup

Use MATLAB with Signal Processing Toolbox for the filtering and autoregressive features. For notebook analyses, use Jupyter with Python 3, NumPy, pandas, SciPy, and scikit-learn. Some experiments also import statsmodels, Matplotlib, CatBoost, XGBoost, or imbalanced-learn.

The scripts expect locally prepared subject files. For example, the default folder convention for subject 5 is `datasets/DB2/DB2_s5/S5_E1_A1_pruned.mat`, relative to the repository root. The MAT file must contain the project's `pruned_table` with EMG, joint angle, stimulus and repetition columns. The kinematic onset script also reads subject and exercise columns. For the velocity workflow, provide the two EMG columns as `emg_edc` and `emg_fds`. The prepared recordings and derived data tables are external to this repository.

## Run the velocity analysis

1. Set MATLAB's current folder to the repository root and add the scripts to the path with `addpath(fullfile(pwd, 'EMG', 'simulations'))`.
2. Set `pruned_file` in `exercise1_tkin_marker.m` and `subject_id` in `exercise1_angle_profile_examiner.m` and `velocity_metric_extractor.m` to the same subject. Run the scripts in that order. They produce the kinematic onset log, the trapezoid milestone log, and the dynamic velocity feature table in the subject folder.
3. Open `metric_analysis.ipynb` and run the velocity experiment you want with the notebook kernel's working directory set to the repository root. Check that cell's subject list and inputs. Cohort experiments need the corresponding exports for each subject, and several cells also read the trapezoid milestone log.

## File guide

### Kinematic labels and feature exports

| File | Purpose |
| --- | --- |
| [`exercise1_tkin_marker.m`](exercise1_tkin_marker.m) | Finds kinematic movement onsets from joint angle and velocity. Writes `*_pruned_tkin_log.csv`. |
| [`exercise1_angle_profile_examiner.m`](exercise1_angle_profile_examiner.m) | Uses the onset log to label takeoff, steady hold, and release milestones. Writes `*_kinematic_trapezoid_log.csv`. |
| [`velocity_metric_extractor.m`](velocity_metric_extractor.m) | Exports windowed EMG features, joint velocity targets, and hold flags to `*_dynamic_velocity_features.csv`. |
| [`metric_calculator.m`](metric_calculator.m) | Extracts EMG feature windows around rest and kinematic onsets. Writes `*_pruned_emg_features.csv` and a matching MAT file. |
| [`metric_calculator_expanded.m`](metric_calculator_expanded.m) | Sweeps window lengths and offsets relative to kinematic onset. Writes `*_pruned_emg_sliding_features.csv` and a matching MAT file. |
| [`metric_calculator_active_phase.m`](metric_calculator_active_phase.m) and [`metric_calculator_active_phase_modified.m`](metric_calculator_active_phase_modified.m) | Batch-extract phase-specific EMG and kinematic metrics using the trapezoid logs. The modified script revises the speed and hold measurements. Both write `*_feature_metrics.csv` to the same location, so choose the variant for the experiment. |

### Detector experiments and acquisition

| File | Purpose |
| --- | --- |
| [`exercise1_processing.m`](exercise1_processing.m) | Replays recorded signals for onset detection and direction estimation using calibration from the first repetition. |
| [`exercise1_onset_detector.m`](exercise1_onset_detector.m) and [`exercise1_onset_detector_modified.m`](exercise1_onset_detector_modified.m) | Replay a continuous recording and log detections. The modified version adds circular buffer rollback and excludes kinematic plateaus from the onset search. |
| [`onset_detector_advanced.m`](onset_detector_advanced.m) | Separates rapid onset detection from burst confirmation and direction estimation. Writes the same event log filenames as the two preceding detectors. |
| [`first_stage_of_system.m`](first_stage_of_system.m) and [`first_stage_of_system_modified.m`](first_stage_of_system_modified.m) | Explore adaptive onset detection and speed estimates. The modified version groups nearby onset candidates before updating its adaptive threshold. |
| [`metric_analysis.ipynb`](metric_analysis.ipynb) | Experiments for feature screening, calibration, onset analysis, and velocity decoding. Each experiment reads its corresponding MATLAB exports. |
| [`sampler_prog.ino`](sampler_prog.ino) | Standalone STM32 Arduino sketch that samples one analog input at a nominal 2 kHz and streams 12-bit ADC readings over serial. |
