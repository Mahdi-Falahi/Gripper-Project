% ==============================================================================
% REAL-TIME ONSET DETECTION & DIRECTIONAL STEERING (NinaPro DB2)
% Ingests pre-processed 'pruned_table' and precomputed 'pruned_tkin_log.csv'
% ==============================================================================
clear; clc; close all;

% --- 1. Direct Paths & Ingestion ---
subject_id = 1;
mat_path  = sprintf('datasets/DB2/DB2_s%d/S%d_E1_A1_pruned.mat', subject_id, subject_id);
tkin_path = sprintf('datasets/DB2/DB2_s%d/S%d_E1_A1_pruned_tkin_log.csv', subject_id, subject_id);

if ~exist(mat_path, 'file'),  mat_path  = sprintf('S%d_E1_A1_pruned.mat', subject_id); end
if ~exist(tkin_path, 'file'), tkin_path = sprintf('S%d_E1_A1_pruned_tkin_log.csv', subject_id); end

fprintf('[INGESTION] Loading Pruned MAT : %s\n', mat_path);
data = load(mat_path, 'pruned_table');
pt   = data.pruned_table;

fprintf('[INGESTION] Loading tkin CSV   : %s\n', tkin_path);
tkin_table = readtable(tkin_path);

% Filter CSV for target 1-DoF gestures (M5 = Open, M6 = Fist)
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
theta     = double(pt.joint_angle);  % Already 12 Hz low-pass filtered 1-DoF excursion
emg_edc   = double(pt.emg_edc);      % Pre-extracted Ch 9 (EDC - Extensor)
emg_fds   = double(pt.emg_fds);      % Pre-extracted Ch 10 (FDS - Flexor)
stim_data = double(pt.stimulus);     % Clinical stimulus prompt
rep_data  = double(pt.repetition);   % Repetition counter (1-6)

t         = (0:height(pt)-1)' / fs_emg;
theta_vel = [0; diff(theta) * fs_emg]; % Clean derivative (kinematics already filtered)

% --- 3. Longest Inter-Trial Velocity Rest Search (Mahalanobis Baseline) ---
disp('[CALIBRATION] Locating longest unprompted quiet rest window via velocity...');

vel_near_zero     = 3.0;                 % deg/s mechanical stillness floor
min_rest_dur_pts  = round(1.0 * fs_emg); % Minimum 1.0 s requirement
settle_buffer_pts = round(0.5 * fs_emg); % 500 ms post-release buffer

% Isolate unprompted baseline regions
is_rest_sample = (stim_data == 0);

% Mask out 500 ms post-movement release to avoid viscoelastic tendon recoil
active_releases = find(diff([0; stim_data > 0]) < 0);
for ar = 1:numel(active_releases)
    idx_settle_start = active_releases(ar);
    idx_settle_end   = min(numel(is_rest_sample), idx_settle_start + settle_buffer_pts);
    is_rest_sample(idx_settle_start:idx_settle_end) = false;
end

% Find contiguous quiet regions
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

% --- 4. 1-Shot Directional Steering Calibration (Repetition 1) ---
disp('[CALIBRATION] Computing 4-feature Variant 2 LDA via Rep 1 shrinkage...');

row_m5_r1 = tkin_table(tkin_table.movement == 5 & tkin_table.repetition == 1, :);
row_m6_r1 = tkin_table(tkin_table.movement == 6 & tkin_table.repetition == 1, :);

tkin_m5 = row_m5_r1.(tkin_col)(1);
tkin_m6 = row_m6_r1.(tkin_col)(1);

win_40_pts  = round(0.040 * fs_emg); % 40 ms directional steering window
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

% --- 5. Continuous Timeline Streaming Engine ---
fprintf('[STREAMING] Sweeping detector across continuous timeline in 10 ms hops...\n');

idx_stream_start = max(1, round((tkin_table.t_prompt(1) - 1.0) * fs_emg));
idx_stream_end   = min(numel(theta), round((tkin_table.t_prompt(end) + 3.0) * fs_emg));

step_pts = round(0.010 * fs_emg); % 10 ms hop (100 Hz update rate)
sample_indices = idx_stream_start : step_pts : (idx_stream_end - win_dm_pts);

STATE_REST       = 0;
STATE_REFRACTORY = 1;
current_state    = STATE_REST;

consec_hits        = 0;
cooldown_samples   = 0;
min_refractory_pts = round(1.2 * fs_emg); % 1.2 s refractory lockout
detections         = [];                 % [t_onset, pred_dir (5 or 6), theta_at_onset]

for s = 1:numel(sample_indices)
    cur_idx = sample_indices(s);
    t_now   = t(cur_idx + win_dm_pts - 1);
    
    % Sliding 50 ms state detection window
    w_e = emg_edc(cur_idx : cur_idx + win_dm_pts - 1);
    w_f = emg_fds(cur_idx : cur_idx + win_dm_pts - 1);
    
    phi_dm = extract_features_dm(w_e, w_f);
    diff_u = phi_dm - mu_rest;
    dm_val = sqrt(diff_u * inv_cov_rest * diff_u');
    
    if current_state == STATE_REST
        % Tripwire Threshold: DM > 3.0 sustained for 2 consecutive steps (20 ms)
        if dm_val > 3.0
            consec_hits = consec_hits + 1;
            if consec_hits >= 2
                t_onset = t_now;
                
                % Snapshot 40 ms post-detection window for directional steering
                idx_dir = min(numel(theta) - win_40_pts + 1, cur_idx + win_dm_pts);
                w_e_dir = emg_edc(idx_dir : idx_dir + win_40_pts - 1);
                w_f_dir = emg_fds(idx_dir : idx_dir + win_40_pts - 1);
                
                phi_v2 = extract_features_v2(w_e_dir, w_f_dir);
                score  = w_eff * phi_v2' + b_hybrid;
                pred_dir = (score > 0)*6 + (score <= 0)*5; % 5 = Hand Open, 6 = Fist
                
                detections = [detections; t_onset, pred_dir, theta(cur_idx + win_dm_pts - 1)]; %#ok<AGROW>
                
                % Engage refractory state to block re-triggering during sustained contraction
                current_state    = STATE_REFRACTORY;
                cooldown_samples = min_refractory_pts;
                consec_hits      = 0;
            end
        else
            consec_hits = 0;
        end
        
    elseif current_state == STATE_REFRACTORY
        % Decrement timer; re-arm only once both electrical and mechanical states settle
        if cooldown_samples > 0
            cooldown_samples = cooldown_samples - step_pts;
        elseif dm_val < 2.0 && abs(theta_vel(cur_idx)) < 8.0
            current_state = STATE_REST;
            consec_hits   = 0;
        end
    end
end

fprintf('[COMPLETION] Detected Onsets: %d | Ground-Truth Prompts: %d\n', ...
        size(detections, 1), height(tkin_table));

% --- 6. Timeline Diagnostic Visualization ---
figure('Color', 'w', 'Position', [50, 100, 1350, 620], ...
       'Name', sprintf('Continuous Timeline Benchmark (S%d)', subject_id));
hold on; grid on; box on;

t_start_eval = t(idx_stream_start);
t_end_eval   = t(idx_stream_end);
y_limits = [min(theta(idx_stream_start:idx_stream_end)) - 10, ...
            max(theta(idx_stream_start:idx_stream_end)) + 15];

% A. Visual Cue Active Shadows
for k = 1:height(tkin_table)
    ts = tkin_table.t_prompt(k);
    p_end = find(t >= ts & stim_data == 0, 1, 'first');
    if isempty(p_end), te = ts + 2.0; else, te = t(p_end); end
    
    m_val = tkin_table.movement(k);
    r_val = tkin_table.repetition(k);
    
    if m_val == 5
        patch([ts, te, te, ts], [y_limits(1), y_limits(1), y_limits(2), y_limits(2)], ...
              [0.85, 0.95, 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
        text(ts + 0.1, y_limits(2) - 3.5, sprintf('Open (R%d)', r_val), ...
             'Color', [0.1, 0.5, 0.1], 'FontSize', 8.5, 'FontWeight', 'bold');
    else
        patch([ts, te, te, ts], [y_limits(1), y_limits(1), y_limits(2), y_limits(2)], ...
              [0.95, 0.85, 0.90], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
        text(ts + 0.1, y_limits(2) - 3.5, sprintf('Fist (R%d)', r_val), ...
             'Color', [0.6, 0.1, 0.4], 'FontSize', 8.5, 'FontWeight', 'bold');
    end
end
p_cue = patch(nan, nan, [0.9, 0.9, 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.6, 'DisplayName', 'Visual Cue Active');

% B. Continuous Finger PIP Joint Excursion
p_theta = plot(t, theta, 'Color', [0.2, 0.2, 0.2], 'LineWidth', 1.8, 'DisplayName', 'Finger PIP Angle (\theta)');

% C. Ground-Truth Kinematic Onsets (Imported from CSV)
p_tkin = plot(nan, nan, 'bd', 'MarkerSize', 8.5, 'MarkerFaceColor', [0.4, 0.7, 1.0], ...
              'LineWidth', 1.5, 'DisplayName', 'Ground Truth t_{kin, angle} (from CSV)');

for k = 1:height(tkin_table)
    t_k = tkin_table.(tkin_col)(k);
    if ~isnan(t_k) && t_k >= t_start_eval && t_k <= t_end_eval
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

xlim([t_start_eval, t_end_eval]);
ylim(y_limits);
xlabel('Continuous Timeline (seconds)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('PIP Excursion Angle (degrees)', 'FontSize', 11, 'FontWeight', 'bold');
title(sprintf('Continuous Onset Detection & Steering: S%d (Ingesting pruned\\_table)', subject_id), ...
      'FontSize', 12, 'FontWeight', 'bold');
legend([p_theta, p_cue, p_tkin, p_open, p_fist], 'Location', 'southoutside', ...
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