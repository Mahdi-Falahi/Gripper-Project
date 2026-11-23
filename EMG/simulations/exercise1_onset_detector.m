% ==============================================================================
% REAL-TIME ONSET DETECTION & DIRECTIONAL STEERING (NinaPro DB2)
% Continuous Timeline Streaming with Visual-Cue Shading & Pruned Data Ingestion
% ==============================================================================
clear; clc; close all;

% --- 1. Direct Paths & Ingestion ---
subject_id = 5;
mat_path  = sprintf('datasets/DB2/DB2_s%d/S%d_E1_A1_pruned.mat', subject_id, subject_id);
tkin_path = sprintf('datasets/DB2/DB2_s%d/S%d_E1_A1_pruned_tkin_log.csv', subject_id, subject_id);

if ~exist(mat_path, 'file'),  mat_path  = sprintf('S%d_E1_A1_pruned.mat', subject_id); end
if ~exist(tkin_path, 'file'), tkin_path = sprintf('S%d_E1_A1_pruned_tkin_log.csv', subject_id); end

fprintf('[INGESTION] Loading Pruned MAT : %s\n', mat_path);
data = load(mat_path, 'pruned_table');
pt   = data.pruned_table;

fprintf('[INGESTION] Loading tkin CSV   : %s\n', tkin_path);
tkin_table = readtable(tkin_path);

% Target movements: 5 = Hand Open, 6 = Fist Clench
target_movements = [5, 6];
tkin_table = tkin_table(ismember(tkin_table.movement, target_movements), :);

if ismember('tkin_angle', tkin_table.Properties.VariableNames)
    tkin_col = 'tkin_angle';
elseif ismember('t_ang', tkin_table.Properties.VariableNames)
    tkin_col = 't_ang';
else
    error('Could not find "tkin_angle" or "t_ang" in %s', tkin_path);
end

% --- 2. Extract Signals Directly from pruned_table ---
fs_emg    = 2000;
theta     = double(pt.joint_angle);  % Pre-filtered 1-DoF excursion (deg)
emg_edc   = double(pt.emg_edc);      % Pre-extracted Ch 9 (EDC - Extensor)
emg_fds   = double(pt.emg_fds);      % Pre-extracted Ch 10 (FDS - Flexor)
stim_data = double(pt.stimulus);     % Visual cue / stimulus state
rep_data  = double(pt.repetition);   % Repetition counter (1-6)

N_pts     = height(pt);
t         = (0:N_pts-1)' / fs_emg;
theta_vel = [0; diff(theta) * fs_emg]; % Instantaneous angular velocity (deg/s)

% --- 3. Extract Exact Visual Cue Blocks for Timeline Shading ---
% Contiguous block parsing directly on stim_data
shade_blocks = struct('mov', {}, 'rep', {}, 'tStart', {}, 'tEnd', {}, 'color', {});
color_palette = containers.Map({5, 6}, {[0.75, 0.88, 1.00], [1.00, 0.82, 0.82]}); % Blue for Open, Red for Fist

for m = target_movements
    idx = find(stim_data == m);
    if isempty(idx), continue; end
    
    breaks = find(diff(idx) > 1);
    s_idx  = idx([1; breaks + 1]);
    e_idx  = idx([breaks; end]);
    
    for k = 1:numel(s_idx)
        s_k = s_idx(k);
        e_k = e_idx(k);
        r_k = mode(rep_data(s_k:e_k));
        shade_blocks(end+1) = struct(...
            'mov', m, ...
            'rep', r_k, ...
            'tStart', t(s_k), ...
            'tEnd', t(e_k), ...
            'color', color_palette(m));
    end
end
% Sort shade blocks chronologically
[~, sort_order] = sort([shade_blocks.tStart]);
shade_blocks = shade_blocks(sort_order);

% --- 4. Baseline Calibration: Longest Inter-Trial Velocity Rest Search ---
disp('[CALIBRATION] Locating longest unprompted quiet rest window via velocity...');

vel_near_zero     = 3.0;                 % deg/s mechanical stillness threshold
min_rest_dur_pts  = round(1.0 * fs_emg); % Minimum 1.0 s requirement
settle_buffer_pts = round(0.5 * fs_emg); % 500 ms post-release buffer

is_rest_sample = (stim_data == 0);

% Mask 500 ms post-movement release to avoid viscoelastic tendon recoil
active_releases = find(diff([0; stim_data > 0]) < 0);
for ar = 1:numel(active_releases)
    idx_settle_start = active_releases(ar);
    idx_settle_end   = min(numel(is_rest_sample), idx_settle_start + settle_buffer_pts);
    is_rest_sample(idx_settle_start:idx_settle_end) = false;
end

is_quiet = is_rest_sample & (abs(theta_vel) <= vel_near_zero);

quiet_diff    = diff([0; is_quiet; 0]);
block_starts  = find(quiet_diff == 1);
block_ends    = find(quiet_diff == -1) - 1;
block_lengths = block_ends - block_starts + 1;

if isempty(block_lengths) || max(block_lengths) < min_rest_dur_pts
    disp('[WARN] No contiguous 1.0 s block <= 3.0 deg/s. Taking minimum velocity interval.');
    rest_blocks_diff = diff([0; is_rest_sample; 0]);
    r_starts = find(rest_blocks_diff == 1);
    r_ends   = find(rest_blocks_diff == -1) - 1;
    valid_r  = find((r_ends - r_starts + 1) >= min_rest_dur_pts);
    
    best_mean_vel  = Inf;
    best_idx_start = 1;
    for vr = 1:numel(valid_r)
        seg_idx = r_starts(valid_r(vr)) : r_ends(valid_r(vr));
        mov_v = movmean(abs(theta_vel(seg_idx)), min_rest_dur_pts, 'Endpoints', 'discard');
        [min_v, rel_idx] = min(mov_v);
        if min_v < best_mean_vel
            best_mean_vel  = min_v;
            best_idx_start = seg_idx(rel_idx);
        end
    end
    idx_cal_start = best_idx_start;
    idx_cal_end   = idx_cal_start + min_rest_dur_pts - 1;
else
    [~, best_block] = max(block_lengths);
    idx_cal_start   = block_starts(best_block);
    cal_len         = min(block_lengths(best_block), round(2.5 * fs_emg)); % Cap at 2.5 s
    idx_cal_end     = idx_cal_start + cal_len - 1;
end

fprintf('[CALIBRATION] Rest segment: %.2f s to %.2f s (Duration: %.2f s | Mean |Vel|: %.2f deg/s)\n', ...
        t(idx_cal_start), t(idx_cal_end), t(idx_cal_end) - t(idx_cal_start), ...
        mean(abs(theta_vel(idx_cal_start:idx_cal_end))));

% Sift resting features for 50 ms Mahalanobis Tripwire (3D Subspace)
win_dm_pts   = round(0.050 * fs_emg); % 50 ms window
step_cal_pts = round(0.010 * fs_emg); % 10 ms hop
rest_feats   = [];

for i = idx_cal_start : step_cal_pts : (idx_cal_end - win_dm_pts + 1)
    w_e = emg_edc(i : i + win_dm_pts - 1);
    w_f = emg_fds(i : i + win_dm_pts - 1);
    rest_feats = [rest_feats; extract_features_dm(w_e, w_f)]; %#ok<AGROW>
end

mu_rest      = mean(rest_feats, 1);
cov_rest     = cov(rest_feats) + 1e-4 * eye(size(rest_feats, 2)); % Shrinkage regularization
inv_cov_rest = inv(cov_rest);

% --- 5. 1-Shot Directional Steering Calibration (Repetition 1) ---
disp('[CALIBRATION] Computing 4-feature Variant 2 LDA via Rep 1 shrinkage...');

row_m5_r1 = tkin_table(tkin_table.movement == 5 & tkin_table.repetition == 1, :);
row_m6_r1 = tkin_table(tkin_table.movement == 6 & tkin_table.repetition == 1, :);

tkin_m5 = row_m5_r1.(tkin_col)(1);
tkin_m6 = row_m6_r1.(tkin_col)(1);

win_40_pts  = round(0.040 * fs_emg); % 40 ms directional steering snapshot
idx_snap_m5 = round(tkin_m5 * fs_emg);
idx_snap_m6 = round(tkin_m6 * fs_emg);

x5_calib = extract_features_v2(emg_edc(idx_snap_m5 : idx_snap_m5 + win_40_pts - 1), ...
                               emg_fds(idx_snap_m5 : idx_snap_m5 + win_40_pts - 1));
x6_calib = extract_features_v2(emg_edc(idx_snap_m6 : idx_snap_m6 + win_40_pts - 1), ...
                               emg_fds(idx_snap_m6 : idx_snap_m6 + win_40_pts - 1));

% Population Prior: [antagonist_diff_ratio, edc_Mobility, fds_LOG, edc_Complexity]
w_pop = [-1.50, -0.65, 0.45, 0.35];
b_pop = -0.10;

% Regularized Intercept Shrinkage (alpha = 0.70)
alpha_shrink = 0.70;
pol = sign(w_pop * (x6_calib - x5_calib)');
if pol == 0, pol = 1.0; end

w_eff    = w_pop * pol;
b_calib  = -w_eff * (0.5 * (x5_calib + x6_calib))';
b_hybrid = (1.0 - alpha_shrink) * (b_pop * pol) + alpha_shrink * b_calib;

% --- 6. Continuous Timeline Streaming Engine ---
fprintf('[STREAMING] Sweeping detector across continuous timeline in 10 ms hops...\n');

step_pts = round(0.010 * fs_emg); % 10 ms hop (100 Hz update rate)
raw_indices = 1 : step_pts : (N_pts - win_dm_pts);

% --------------------------------------------------------------------------
% CONSTRUCT EXCLUSION MASK FOR SEARCH SPACE
% Excludes sustained plateau and release: [tStart + 1.0s, tEnd + 1.0s]
% --------------------------------------------------------------------------
is_excluded_sample = false(N_pts, 1);
for k = 1:numel(shade_blocks)
    idx_ex_start = round((shade_blocks(k).tStart + 2.0) * fs_emg);
    idx_ex_end   = min(N_pts, round((shade_blocks(k).tEnd + 1.0) * fs_emg));
    if idx_ex_start <= idx_ex_end
        is_excluded_sample(idx_ex_start:idx_ex_end) = true;
    end
end

% Prune sample_indices so evaluation windows never sample from excluded zones
win_end_indices = raw_indices + win_dm_pts - 1;
sample_indices  = raw_indices(~is_excluded_sample(win_end_indices));

% State Machine Initialization
STATE_REST       = 0;
STATE_REFRACTORY = 1;
current_state    = STATE_REST;

consec_hits        = 0;
cooldown_samples   = 0;
min_refractory_pts = round(0.0 * fs_emg); 
detections         = [];                 

for s = 1:numel(sample_indices)
    cur_idx = sample_indices(s);
    t_now   = t(cur_idx + win_dm_pts - 1);
    
    % Reset state if crossing an excluded gap back into baseline rest
    if s > 1 && (cur_idx - sample_indices(s - 1)) > step_pts
        current_state    = STATE_REST;
        consec_hits      = 0;
        cooldown_samples = 0;
    end
    
    % Sliding 50 ms state detection window
    w_e = emg_edc(cur_idx : cur_idx + win_dm_pts - 1);
    w_f = emg_fds(cur_idx : cur_idx + win_dm_pts - 1);
    
    phi_dm = extract_features_dm(w_e, w_f);
    diff_u = phi_dm - mu_rest;
    dm_val = sqrt(diff_u * inv_cov_rest * diff_u');
    
    if current_state == STATE_REST
        if dm_val > 3.0
            consec_hits = consec_hits + 1;
            if consec_hits >= 2
                t_onset = t_now;
                
                idx_dir = min(N_pts - win_40_pts + 1, cur_idx + win_dm_pts);
                w_e_dir = emg_edc(idx_dir : idx_dir + win_40_pts - 1);
                w_f_dir = emg_fds(idx_dir : idx_dir + win_40_pts - 1);
                
                phi_v2 = extract_features_v2(w_e_dir, w_f_dir);
                score  = w_eff * phi_v2' + b_hybrid;
                pred_dir = (score > 0)*6 + (score <= 0)*5;
                
                detections = [detections; t_onset, pred_dir, theta(cur_idx + win_dm_pts - 1)]; %#ok<AGROW>
                
                current_state    = STATE_REFRACTORY;
                cooldown_samples = min_refractory_pts;
                consec_hits      = 0;
            end
        else
            consec_hits = 0;
        end
        
    elseif current_state == STATE_REFRACTORY
        if cooldown_samples > 0
            cooldown_samples = cooldown_samples - step_pts;
        elseif dm_val < 2.0
            current_state = STATE_REST;
            consec_hits   = 0;
        end
    end
end

fprintf('[COMPLETION] Detected Onsets: %d | Ground-Truth Prompts: %d\n', ...
        size(detections, 1), height(tkin_table));

% ==============================================================================
% 6B. CHRONOLOGICAL TIMESERIES EVENT STREAM
% Logs every discrete event (CUE_START, CUE_END, TKIN, MYO_ONSET) in temporal order
% ==============================================================================
disp('[LOGGING] Compiling chronological timeseries event stream...');

event_records = {}; % Structure: {timestamp_s, event_type, theta_deg, active_prompt, repetition, classified_dir}

% 1. Ingest Every Visual Cue Transition (Start & End)
for k = 1:numel(shade_blocks)
    sb = shade_blocks(k);
    
    % Cue Start Event
    ang_c_start = interp1(t, theta, sb.tStart);
    event_records(end+1, :) = {sb.tStart, 'CUE_START', ang_c_start, sb.mov, sb.rep, NaN}; %#ok<AGROW>
    
    % Cue End Event
    ang_c_end = interp1(t, theta, sb.tEnd);
    event_records(end+1, :) = {sb.tEnd, 'CUE_END', ang_c_end, sb.mov, sb.rep, NaN}; %#ok<AGROW>
end

% 2. Ingest Every Physical Kinematic Takeoff (tkin_angle)
for k = 1:height(tkin_table)
    t_k = tkin_table.(tkin_col)(k);
    if ~isnan(t_k) && t_k >= t(1) && t_k <= t(end)
        ang_k = interp1(t, theta, t_k);
        event_records(end+1, :) = {t_k, 'TKIN', ang_k, tkin_table.movement(k), tkin_table.repetition(k), NaN}; %#ok<AGROW>
    end
end

% 3. Ingest EVERY Detected Myoelectric Onset (Zero Drops / Unrestricted)
for d = 1:size(detections, 1)
    t_det   = detections(d, 1);
    pred_d  = detections(d, 2);
    ang_det = detections(d, 3);
    
    % Sample prompt state and repetition active at this exact millisecond
    idx_sample  = min(numel(stim_data), max(1, round(t_det * fs_emg)));
    active_stim = stim_data(idx_sample);
    active_rep  = rep_data(idx_sample);
    
    event_records(end+1, :) = {t_det, 'MYO_ONSET', ang_det, active_stim, active_rep, pred_d}; %#ok<AGROW>
end

% 4. Assemble and Sort Chronological Timeseries Table
event_timeseries_log = cell2table(event_records, 'VariableNames', ...
    {'timestamp_s', 'event_type', 'theta_deg', 'active_prompt', 'repetition', 'classified_dir'});

% Sort all events strictly by timestamp
event_timeseries_log = sortrows(event_timeseries_log, 'timestamp_s');

% 5. Compute Real-Time Latency Relative to Nearest Physical Takeoff (tkin)
tkin_rows = strcmp(event_timeseries_log.event_type, 'TKIN');
tkin_times = event_timeseries_log.timestamp_s(tkin_rows);
dt_tkin_ms = nan(height(event_timeseries_log), 1);

for row_i = 1:height(event_timeseries_log)
    if strcmp(event_timeseries_log.event_type{row_i}, 'MYO_ONSET') && ~isempty(tkin_times)
        % Signed latency: negative = sEMG lead, positive = sEMG lag
        [min_abs_dt, nearest_idx] = min(abs(event_timeseries_log.timestamp_s(row_i) - tkin_times));
        dt_tkin_ms(row_i) = (event_timeseries_log.timestamp_s(row_i) - tkin_times(nearest_idx)) * 1000.0;
    end
end
event_timeseries_log.latency_to_nearest_tkin_ms = dt_tkin_ms;

% 6. Save Chronological Stream to CSV and MAT
csv_save_path = sprintf('datasets/DB2/DB2_s%d/S%d_E1_A1_timeseries_event_log.csv', subject_id, subject_id);
mat_save_path = sprintf('datasets/DB2/DB2_s%d/S%d_E1_A1_timeseries_event_log.mat', subject_id, subject_id);

writetable(event_timeseries_log, csv_save_path);
save(mat_save_path, 'event_timeseries_log');

fprintf('[LOGGING] Saved Chronological Event Log:\n  --> %s\n  --> %s\n\n', csv_save_path, mat_save_path);

% Print preview of the event stream
disp('========================================================================================================');
disp('                                  CHRONOLOGICAL EVENT TIMESERIES LOG                                    ');
disp('========================================================================================================');
disp(event_timeseries_log(1:min(25, height(event_timeseries_log)), :));
disp('========================================================================================================');

% --- 7. Timeline Diagnostic Visualization ---
fig = figure('Color', 'w', 'Position', [50, 100, 1350, 620], ...
             'Name', sprintf('Continuous Timeline Benchmark (S%d)', subject_id));
hold on; grid on; box on;

y_limits = [min(theta) - 10, max(theta) + 15];

% A. Draw Visual Cue Background Shadows Directly from stim_data Transitions
for k = 1:numel(shade_blocks)
    sb = shade_blocks(k);
    x_patch = [sb.tStart, sb.tEnd, sb.tEnd, sb.tStart];
    y_patch = [y_limits(1), y_limits(1), y_limits(2), y_limits(2)];
    
    p_s = patch(x_patch, y_patch, sb.color, 'EdgeColor', 'none', 'FaceAlpha', 0.45, 'HandleVisibility', 'off');
    uistack(p_s, 'bottom'); % Ensure shading stays behind signal lines
    
    if sb.mov == 5
        txt_str = sprintf('Cue: Open (R%d)', sb.rep);
        txt_col = [0.10, 0.40, 0.70];
    else
        txt_str = sprintf('Cue: Fist (R%d)', sb.rep);
        txt_col = [0.85, 0.25, 0.20];
    end
    text(sb.tStart + 0.15, y_limits(2) - 3.5, txt_str, 'Color', txt_col, ...
         'FontSize', 8.5, 'FontWeight', 'bold');
end

% B. Continuous Finger PIP Joint Excursion Curve
p_theta = plot(t, theta, 'Color', [0.15, 0.15, 0.15], 'LineWidth', 1.6, 'DisplayName', 'Finger PIP Angle (\theta)');

% C. Ground-Truth Kinematic Onsets (Imported from CSV)
p_tkin = plot(nan, nan, 'bd', 'MarkerSize', 8.5, 'MarkerFaceColor', [0.4, 0.7, 1.0], ...
              'LineWidth', 1.5, 'DisplayName', 'Ground Truth t_{kin, angle} (from CSV)');

for k = 1:height(tkin_table)
    t_k = tkin_table.(tkin_col)(k);
    if ~isnan(t_k) && t_k >= t(1) && t_k <= t(end)
        xline(t_k, 'b--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
        y_val = interp1(t, theta, t_k);
        plot(t_k, y_val, 'bd', 'MarkerSize', 8.5, 'MarkerFaceColor', [0.4, 0.7, 1.0], ...
             'LineWidth', 1.5, 'HandleVisibility', 'off');
    end
end

% D. Streaming Onset Detections (Classified Direction Color-Coding)
p_open = plot(nan, nan, 'o', 'MarkerSize', 10, 'LineWidth', 2.0, ...
              'MarkerEdgeColor', [0.0, 0.45, 0.1], 'MarkerFaceColor', [0.2, 0.85, 0.3], ...
              'DisplayName', 'Detected Onset: Hand Open (M5)');
p_fist = plot(nan, nan, 's', 'MarkerSize', 10, 'LineWidth', 2.0, ...
              'MarkerEdgeColor', [0.55, 0.0, 0.35], 'MarkerFaceColor', [0.95, 0.2, 0.55], ...
              'DisplayName', 'Detected Onset: Fist / Clench (M6)');

for d = 1:size(detections, 1)
    t_det = detections(d, 1);
    pred  = detections(d, 2);
    y_det = detections(d, 3);
    
    if pred == 5
        plot(t_det, y_det, 'o', 'MarkerSize', 11, 'LineWidth', 2.0, ...
             'MarkerEdgeColor', [0.0, 0.45, 0.1], 'MarkerFaceColor', [0.2, 0.85, 0.3], ...
             'HandleVisibility', 'off');
        xline(t_det, ':', 'Color', [0.0, 0.6, 0.2], 'LineWidth', 1.4, 'HandleVisibility', 'off');
    else
        plot(t_det, y_det, 's', 'MarkerSize', 11, 'LineWidth', 2.0, ...
             'MarkerEdgeColor', [0.55, 0.0, 0.35], 'MarkerFaceColor', [0.95, 0.2, 0.55], ...
             'HandleVisibility', 'off');
        xline(t_det, ':', 'Color', [0.75, 0.1, 0.45], 'LineWidth', 1.4, 'HandleVisibility', 'off');
    end
end

% Legend placeholders for cue shadows
h_m5 = patch(nan, nan, color_palette(5), 'EdgeColor', 'none', 'FaceAlpha', 0.45, 'DisplayName', 'Cue: Hand Open (M5)');
h_m6 = patch(nan, nan, color_palette(6), 'EdgeColor', 'none', 'FaceAlpha', 0.45, 'DisplayName', 'Cue: Fist Clench (M6)');

xlim([t(1), t(end)]);
ylim(y_limits);
xlabel('Continuous Timeline (seconds)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('PIP Excursion Angle (degrees)', 'FontSize', 11, 'FontWeight', 'bold');
title(sprintf('Continuous Onset Detection & Steering: S%d (Pruned Ingestion | Visual Cue Shading)', subject_id), ...
      'FontSize', 12, 'FontWeight', 'bold');
legend([p_theta, h_m5, h_m6, p_tkin, p_open, p_fist], 'Location', 'southoutside', ...
       'Orientation', 'horizontal', 'FontSize', 9.5);

% ==============================================================================
% HELPER FEATURE FUNCTIONS
% ==============================================================================
function feats = extract_features_dm(edc, fds)
    eps_val = 1e-6;
    rms_e = sqrt(mean(edc.^2));
    rms_f = sqrt(mean(fds.^2));
    ratio = (rms_e - rms_f) / (rms_e + rms_f + eps_val);
    dx_e  = diff(edc);
    mob_e = sqrt(var(dx_e) / (var(edc) + 1e-12));
    log_f = exp(mean(log(abs(fds) + eps_val)));
    feats = [ratio, mob_e, log_f];
end

function feats = extract_features_v2(edc, fds)
    eps_val = 1e-6;
    rms_e = sqrt(mean(edc.^2));
    rms_f = sqrt(mean(fds.^2));
    ratio = (rms_e - rms_f) / (rms_e + rms_f + eps_val);
    dx_e  = diff(edc);
    ddx_e = diff(dx_e);
    mob_e = sqrt(var(dx_e) / (var(edc) + 1e-12));
    mob_dxe = sqrt(var(ddx_e) / (var(dx_e) + 1e-12));
    comp_e  = mob_dxe / (mob_e + 1e-12);
    log_f   = exp(mean(log(abs(fds) + eps_val)));
    feats   = [ratio, mob_e, log_f, comp_e];
end