% ==============================================================================
% REAL-TIME DYNAMIC ONSET DETECTION (NinaPro DB2)
% Ingestion, Circular State Buffer Rollback, Kinematic-Gated Plateau Exclusion
% ==============================================================================
clear; clc; close all;

% --- 1. Direct Paths & Ingestion ---
subject_id = 4;
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

[~, sort_order] = sort([shade_blocks.tStart]);
shade_blocks = shade_blocks(sort_order);

% --- 4. Baseline Calibration: Seed Search on Unprompted Rest ---
disp('[CALIBRATION] Locating initial quiet rest baseline window...');
vel_near_zero     = 3.0;                 % deg/s stillness threshold
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
    disp('[WARN] No contiguous 1.0 s block <= 3.0 deg/s. Selecting minimum velocity interval.');
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

t_cal_start = t(idx_cal_start);
t_cal_end   = t(idx_cal_end);

fprintf('[CALIBRATION] Rest seed interval: %.2f s to %.2f s (Duration: %.2f s)\n', ...
        t_cal_start, t_cal_end, t_cal_end - t_cal_start);

% Extract initial 3D feature subspace for 50 ms Mahalanobis Tripwire
win_dm_pts   = round(0.050 * fs_emg); % 50 ms window
step_cal_pts = round(0.010 * fs_emg); % 10 ms hop
rest_feats   = [];

for i = idx_cal_start : step_cal_pts : (idx_cal_end - win_dm_pts + 1)
    w_e = emg_edc(i : i + win_dm_pts - 1);
    w_f = emg_fds(i : i + win_dm_pts - 1);
    rest_feats = [rest_feats; extract_features_dm(w_e, w_f)]; %#ok<AGROW>
end

mu_rest      = mean(rest_feats, 1);
cov_rest     = cov(rest_feats) + 1e-4 * eye(size(rest_feats, 2));
inv_cov_rest = inv(cov_rest);

% --- 5. Continuous Streaming Engine with Circular State Buffer Rollback ---
fprintf('[STREAMING] Running pure-sEMG detector with circular state rollback...\n');
step_pts = round(0.010 * fs_emg); % 10 ms hop (100 Hz update rate)
raw_indices = 1 : step_pts : (N_pts - win_dm_pts);

% -------------------------------------------------------------------------
% SEARCH SPACE EXCLUSION MASK: [t_angle + 250 ms, tEnd + 1.0s]
% Masks sustained plateau and release to prevent flight-notch double-triggers
% -------------------------------------------------------------------------
is_excluded_sample = false(N_pts, 1);
for k = 1:numel(shade_blocks)
    sb = shade_blocks(k);
    
    % Match the corresponding kinematic takeoff for this movement & repetition
    match_row = tkin_table(tkin_table.movement == sb.mov & tkin_table.repetition == sb.rep, :);
    if ~isempty(match_row) && ~isnan(match_row.(tkin_col)(1))
        t_ang_k    = match_row.(tkin_col)(1);
        t_ex_start = t_ang_k + 0.250; % Cutoff search 250 ms post-kinematic takeoff
    else
        t_ex_start = sb.tStart + 1.0; % Fallback if kinematic takeoff is absent
    end
    t_ex_end = sb.tEnd + 2.5;         % Release buffer
    
    idx_ex_start = max(1, min(N_pts, round(t_ex_start * fs_emg)));
    idx_ex_end   = max(1, min(N_pts, round(t_ex_end * fs_emg)));
    
    if idx_ex_start <= idx_ex_end
        is_excluded_sample(idx_ex_start:idx_ex_end) = true;
    end
end

win_end_indices = raw_indices + win_dm_pts - 1;
sample_indices  = raw_indices(~is_excluded_sample(win_end_indices));

% State Machine Parameters
STATE_REST         = 0;
STATE_REFRACTORY   = 1;

current_state      = STATE_REST;
consec_hits        = 0;
quiet_hops         = 0;
rearm_hops_target  = 10;                    % 10 hops = 100 ms quiet rest required to re-arm
min_refractory_pts = round(0.0 * fs_emg);   % Kept at 0 ms per specification
cooldown_samples   = 0;
alpha_floor        = 0.015;                 % Baseline adaptation rate

% Circular State Buffer (20 slots x 10 ms = 200 ms lookback window)
history_len        = 20;
mu_buf             = repmat(mu_rest, history_len, 1);
cov_buf            = repmat(cov_rest, [1, 1, history_len]);
buf_ptr            = 1;
restore_idx        = 1;

t_candidate        = NaN;
theta_candidate    = NaN;
detections         = [];                    % Stores [t_onset, theta_onset]

% Track sample intervals where FIFO actively accepts updates
fifo_fill_mask = false(N_pts, 1);
% (Optional) Include the initial seed calibration window:
fifo_fill_mask(idx_cal_start:idx_cal_end) = true;

for s = 1:numel(sample_indices)
    cur_idx = sample_indices(s);
    t_now   = t(cur_idx + win_dm_pts - 1);
    
    % Reset state if skipping across an excluded plateau boundary back to rest
    if s > 1 && (cur_idx - sample_indices(s - 1)) > step_pts
        current_state    = STATE_REST;
        consec_hits      = 0;
        cooldown_samples = 0;
        quiet_hops       = 0;
        mu_buf           = repmat(mu_rest, history_len, 1);
        cov_buf          = repmat(cov_rest, [1, 1, history_len]);
        buf_ptr          = 1;
    end
    
    % Sliding 50 ms state detection window
    w_e = emg_edc(cur_idx : cur_idx + win_dm_pts - 1);
    w_f = emg_fds(cur_idx : cur_idx + win_dm_pts - 1);
    
    phi_dm = extract_features_dm(w_e, w_f);
    diff_u = phi_dm - mu_rest;
    dm_val = sqrt(diff_u * inv_cov_rest * diff_u');
    
    if current_state == STATE_REST
        if dm_val <= 3.0
            % Full-Manifold Rest Adaptation (Prevents Covariance Collapse)
            mu_rest      = (1 - alpha_floor) * mu_rest + alpha_floor * phi_dm;
            dev          = phi_dm - mu_rest;
            cov_rest     = (1 - alpha_floor) * cov_rest + alpha_floor * (dev' * dev);
            inv_cov_rest = inv(cov_rest + 1e-4 * eye(3));
            
            % Push Clean State to Circular Buffer
            buf_ptr = mod(buf_ptr, history_len) + 1;
            mu_buf(buf_ptr, :)     = mu_rest;
            cov_buf(:, :, buf_ptr) = cov_rest;

            % MARK SAMPLES WHERE FIFO IS FILLED
            idx_hop_end = min(N_pts, cur_idx + win_dm_pts - 1);
            fifo_fill_mask(cur_idx:idx_hop_end) = true;
            
            consec_hits = 0;
            
        elseif dm_val > 3.0
            % Potential Departure Detected
            if consec_hits == 0
                % STORE START OF TRANSIENT (Backdating Anchor)
                t_candidate     = t_now;
                theta_candidate = theta(cur_idx + win_dm_pts - 1);
                
                % Snapshot rollback target: points to oldest slot in FIFO (~200 ms ago)
                restore_idx     = mod(buf_ptr, history_len) + 1;
            end
            consec_hits = consec_hits + 1;
            
            if consec_hits >= 2 % 20 ms confirmation reached
                % COMMIT BACKDATED ONSET TIMESTAMP
                detections = [detections; t_candidate, theta_candidate]; %#ok<AGROW>
                
                % ROLLBACK: Revert baseline to pre-ramp state (200 ms prior)
                mu_rest      = mu_buf(restore_idx, :);
                cov_rest     = cov_buf(:, :, restore_idx);
                inv_cov_rest = inv(cov_rest + 1e-4 * eye(3));
                
                % Reset history buffer to clean state
                mu_buf   = repmat(mu_rest, history_len, 1);
                cov_buf  = repmat(cov_rest, [1, 1, history_len]);
                buf_ptr  = 1;
                
                current_state    = STATE_REFRACTORY;
                cooldown_samples = min_refractory_pts; % 0 ms
                consec_hits      = 0;
                quiet_hops       = 0;
            end
        end
        
    elseif current_state == STATE_REFRACTORY
        % 1. Mandatory lockout countdown
        if cooldown_samples > 0
            cooldown_samples = cooldown_samples - step_pts;
            quiet_hops = 0;
            
        % 2. After lockout, verify 100 ms sustained electrical quiet
        elseif dm_val < 2.0
            quiet_hops = quiet_hops + 1;
            if quiet_hops >= rearm_hops_target
                current_state = STATE_REST;
                quiet_hops    = 0;
            end
        else
            quiet_hops = 0;
        end
    end
end

fprintf('[COMPLETION] Detected Onsets: %d | Ground-Truth Prompts: %d\n', ...
        size(detections, 1), height(tkin_table));

% --- 6. Chronological Timeseries Event Stream Logging ---
disp('[LOGGING] Compiling chronological timeseries event stream...');
event_records = {};

% 1. Visual Cue Events
for k = 1:numel(shade_blocks)
    sb = shade_blocks(k);
    event_records(end+1, :) = {sb.tStart, 'CUE_START', interp1(t, theta, sb.tStart), sb.mov, sb.rep, NaN}; %#ok<AGROW>
    event_records(end+1, :) = {sb.tEnd,   'CUE_END',   interp1(t, theta, sb.tEnd),   sb.mov, sb.rep, NaN}; %#ok<AGROW>
end

% 2. Physical Kinematic Takeoffs (tkin_angle)
for k = 1:height(tkin_table)
    t_k = tkin_table.(tkin_col)(k);
    if ~isnan(t_k) && t_k >= t(1) && t_k <= t(end)
        event_records(end+1, :) = {t_k, 'TKIN', interp1(t, theta, t_k), ...
                                   tkin_table.movement(k), tkin_table.repetition(k), NaN}; %#ok<AGROW>
    end
end

% 3. Detected Myoelectric Onsets
for d = 1:size(detections, 1)
    t_det   = detections(d, 1);
    ang_det = detections(d, 2);
    
    idx_sample  = min(numel(stim_data), max(1, round(t_det * fs_emg)));
    active_stim = stim_data(idx_sample);
    active_rep  = rep_data(idx_sample);
    
    event_records(end+1, :) = {t_det, 'MYO_ONSET', ang_det, active_stim, active_rep, NaN}; %#ok<AGROW>
end

event_timeseries_log = cell2table(event_records, 'VariableNames', ...
    {'timestamp_s', 'event_type', 'theta_deg', 'active_prompt', 'repetition', 'classified_dir'});
event_timeseries_log = sortrows(event_timeseries_log, 'timestamp_s');

% Signed latency to nearest tkin
tkin_times = event_timeseries_log.timestamp_s(strcmp(event_timeseries_log.event_type, 'TKIN'));
dt_tkin_ms = nan(height(event_timeseries_log), 1);
for row_i = 1:height(event_timeseries_log)
    if strcmp(event_timeseries_log.event_type{row_i}, 'MYO_ONSET') && ~isempty(tkin_times)
        [~, nearest_idx] = min(abs(event_timeseries_log.timestamp_s(row_i) - tkin_times));
        dt_tkin_ms(row_i) = (event_timeseries_log.timestamp_s(row_i) - tkin_times(nearest_idx)) * 1000.0;
    end
end
event_timeseries_log.latency_to_nearest_tkin_ms = dt_tkin_ms;

csv_save_path = sprintf('datasets/DB2/DB2_s%d/S%d_E1_A1_timeseries_event_log.csv', subject_id, subject_id);
mat_save_path = sprintf('datasets/DB2/DB2_s%d/S%d_E1_A1_timeseries_event_log.mat', subject_id, subject_id);
writetable(event_timeseries_log, csv_save_path);
save(mat_save_path, 'event_timeseries_log');
fprintf('[LOGGING] Saved Chronological Event Log: %s\n', csv_save_path);

% --- 7. Diagnostic Timeline Visualization (3-Subplot Stack) ---
fig = figure('Color', 'w', 'Position', [50, 40, 1350, 920], ...
             'Name', sprintf('Continuous Timeline Benchmark (S%d)', subject_id));

% Symmetric limits for direct amplitude comparison between channels
max_emg_volt = max([max(abs(emg_edc)), max(abs(emg_fds))]) * 1.1;
if max_emg_volt == 0, max_emg_volt = 1.0; end
y_lim_emg = [-max_emg_volt, max_emg_volt];

% --------------------------------------------------------------------------
% SUBPLOT 1: Raw EDC (Ch 9 - Extensor Digitorum Communis)
% --------------------------------------------------------------------------
ax1 = subplot(5, 1, 1); hold on; grid on; box on;

% Rest calibration shadow
patch([t_cal_start, t_cal_end, t_cal_end, t_cal_start], ...
      [y_lim_emg(1), y_lim_emg(1), y_lim_emg(2), y_lim_emg(2)], ...
      [0.55, 0.55, 0.55], 'EdgeColor', 'none', 'FaceAlpha', 0.40, 'HandleVisibility', 'off');

% Visual cue shadows
for k = 1:numel(shade_blocks)
    sb = shade_blocks(k);
    patch([sb.tStart, sb.tEnd, sb.tEnd, sb.tStart], ...
          [y_lim_emg(1), y_lim_emg(1), y_lim_emg(2), y_lim_emg(2)], ...
          sb.color, 'EdgeColor', 'none', 'FaceAlpha', 0.35, 'HandleVisibility', 'off');
end

% Raw EDC Signal
plot(t, emg_edc, 'Color', [0.15, 0.45, 0.85], 'LineWidth', 1.0, 'DisplayName', 'Raw EDC (Extensor)');

% Onset vertical lines
for d = 1:size(detections, 1)
    xline(detections(d, 1), 'r:', 'LineWidth', 1.3, 'HandleVisibility', 'off');
end

ylim(y_lim_emg);
ylabel('EDC Amplitude (V)', 'FontSize', 10, 'FontWeight', 'bold');
title(sprintf('Pure-sEMG Onset Detection: S%d (Pruned Ingestion | Dynamic Rest Tracking)', subject_id), ...
      'FontSize', 11, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 9);

% --------------------------------------------------------------------------
% SUBPLOT 2: Raw FDS (Ch 10 - Flexor Digitorum Superficialis)
% --------------------------------------------------------------------------
ax2 = subplot(5, 1, 2); hold on; grid on; box on;

% Rest calibration shadow
patch([t_cal_start, t_cal_end, t_cal_end, t_cal_start], ...
      [y_lim_emg(1), y_lim_emg(1), y_lim_emg(2), y_lim_emg(2)], ...
      [0.55, 0.55, 0.55], 'EdgeColor', 'none', 'FaceAlpha', 0.40, 'HandleVisibility', 'off');

% Visual cue shadows
for k = 1:numel(shade_blocks)
    sb = shade_blocks(k);
    patch([sb.tStart, sb.tEnd, sb.tEnd, sb.tStart], ...
          [y_lim_emg(1), y_lim_emg(1), y_lim_emg(2), y_lim_emg(2)], ...
          sb.color, 'EdgeColor', 'none', 'FaceAlpha', 0.35, 'HandleVisibility', 'off');
end

% Raw FDS Signal
plot(t, emg_fds, 'Color', [0.85, 0.30, 0.20], 'LineWidth', 1.0, 'DisplayName', 'Raw FDS (Flexor)');

% Onset vertical lines
for d = 1:size(detections, 1)
    xline(detections(d, 1), 'r:', 'LineWidth', 1.3, 'HandleVisibility', 'off');
end

ylim(y_lim_emg);
ylabel('FDS Amplitude (V)', 'FontSize', 10, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 9);

% --------------------------------------------------------------------------
% SUBPLOT 3: Original Excursion & Kinematics (Untouched)
% --------------------------------------------------------------------------
ax3 = subplot(5, 1, [3, 5]);
hold on; grid on; box on;
y_limits = [min(theta) - 10, max(theta) + 15];

% A. Rest Calibration Shadow
p_cal = patch([t_cal_start, t_cal_end, t_cal_end, t_cal_start], ...
              [y_limits(1), y_limits(1), y_limits(2), y_limits(2)], ...
              [0.55, 0.55, 0.55], 'EdgeColor', 'none', 'FaceAlpha', 0.40, ...
              'DisplayName', 'Rest Calibration Window');
uistack(p_cal, 'bottom');

% B. Visual Cue Background Shadows
for k = 1:numel(shade_blocks)
    sb = shade_blocks(k);
    p_s = patch([sb.tStart, sb.tEnd, sb.tEnd, sb.tStart], ...
                [y_limits(1), y_limits(1), y_limits(2), y_limits(2)], ...
                sb.color, 'EdgeColor', 'none', 'FaceAlpha', 0.45, 'HandleVisibility', 'off');
    uistack(p_s, 'bottom');
    
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

% C. Finger Joint Angle Excursion
p_theta = plot(t, theta, 'Color', [0.15, 0.15, 0.15], 'LineWidth', 1.6, 'DisplayName', 'Finger PIP Angle (\theta)');

% D. Ground-Truth Kinematic Takeoffs
p_tkin = plot(nan, nan, 'bd', 'MarkerSize', 8.5, 'MarkerFaceColor', [0.4, 0.7, 1.0], ...
              'LineWidth', 1.5, 'DisplayName', 'Ground Truth t_{kin} (Takeoff)');
for k = 1:height(tkin_table)
    t_k = tkin_table.(tkin_col)(k);
    if ~isnan(t_k) && t_k >= t(1) && t_k <= t(end)
        xline(t_k, 'b--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
        plot(t_k, interp1(t, theta, t_k), 'bd', 'MarkerSize', 8.5, ...
             'MarkerFaceColor', [0.4, 0.7, 1.0], 'LineWidth', 1.5, 'HandleVisibility', 'off');
    end
end

% E. Detected Myoelectric Onsets (Backdated)
p_onset = plot(nan, nan, 'rv', 'MarkerSize', 9, 'LineWidth', 1.8, ...
               'MarkerFaceColor', [1.0, 0.2, 0.2], 'DisplayName', 'Detected t_{onset} (Backdated)');
for d = 1:size(detections, 1)
    t_det = detections(d, 1);
    y_det = detections(d, 2);
    
    plot(t_det, y_det, 'rv', 'MarkerSize', 9, 'LineWidth', 1.8, ...
         'MarkerFaceColor', [1.0, 0.2, 0.2], 'HandleVisibility', 'off');
    xline(t_det, 'r:', 'LineWidth', 1.4, 'HandleVisibility', 'off');
end

% Shading Legend Placeholders
h_m5 = patch(nan, nan, color_palette(5), 'EdgeColor', 'none', 'FaceAlpha', 0.45, 'DisplayName', 'Cue: Hand Open (M5)');
h_m6 = patch(nan, nan, color_palette(6), 'EdgeColor', 'none', 'FaceAlpha', 0.45, 'DisplayName', 'Cue: Fist Clench (M6)');

xlim([t(1), t(end)]);
ylim(y_limits);
xlabel('Continuous Timeline (seconds)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('PIP Excursion Angle (degrees)', 'FontSize', 11, 'FontWeight', 'bold');
legend([p_theta, p_cal, h_m5, h_m6, p_tkin, p_onset], 'Location', 'southoutside', ...
       'Orientation', 'horizontal', 'FontSize', 9.5);

% Link all three subplots for synchronized horizontal panning and zooming
linkaxes([ax1, ax2, ax3], 'x');

% ==============================================================================
% 8. DEDICATED WINDOW: FIFO ACTIVE FILLING ZONES & KINEMATICS
% ==============================================================================
fig2 = figure('Color', 'w', 'Position', [100, 150, 1350, 520], ...
              'Name', sprintf('FIFO Ingestion Windows vs Kinematics (S%d)', subject_id));
hold on; grid on; box on;
y_limits_f2 = [min(theta) - 10, max(theta) + 15];

% A. Shadow All Continuous FIFO Filling Regions (Soft Amber/Gold)
d_fifo   = diff([0; fifo_fill_mask; 0]);
f_starts = find(d_fifo == 1);
f_ends   = find(d_fifo == -1) - 1;

p_fifo_legend = patch(nan, nan, [1.0, 0.90, 0.65], 'EdgeColor', 'none', ...
                      'FaceAlpha', 0.55, 'DisplayName', 'FIFO Actively Ingesting');

for k = 1:numel(f_starts)
    t_s = t(f_starts(k));
    t_e = t(f_ends(k));
    p_f = patch([t_s, t_e, t_e, t_s], ...
                [y_limits_f2(1), y_limits_f2(1), y_limits_f2(2), y_limits_f2(2)], ...
                [1.0, 0.90, 0.65], 'EdgeColor', 'none', 'FaceAlpha', 0.55, 'HandleVisibility', 'off');
    uistack(p_f, 'bottom');
end

% B. Visual Cue Background Shadows
for k = 1:numel(shade_blocks)
    sb = shade_blocks(k);
    p_s = patch([sb.tStart, sb.tEnd, sb.tEnd, sb.tStart], ...
                [y_limits_f2(1), y_limits_f2(1), y_limits_f2(2), y_limits_f2(2)], ...
                sb.color, 'EdgeColor', 'none', 'FaceAlpha', 0.35, 'HandleVisibility', 'off');
    uistack(p_s, 'bottom');
    
    if sb.mov == 5
        txt_str = sprintf('Cue: Open (R%d)', sb.rep);
        txt_col = [0.10, 0.40, 0.70];
    else
        txt_str = sprintf('Cue: Fist (R%d)', sb.rep);
        txt_col = [0.85, 0.25, 0.20];
    end
    text(sb.tStart + 0.15, y_limits_f2(2) - 3.5, txt_str, 'Color', txt_col, ...
         'FontSize', 8.5, 'FontWeight', 'bold');
end

% C. Continuous Finger PIP Joint Angle Excursion
p_theta_f2 = plot(t, theta, 'Color', [0.15, 0.15, 0.15], 'LineWidth', 1.6, ...
                  'DisplayName', 'Finger PIP Angle (\theta)');

% D. Ground-Truth Kinematic Takeoffs (tkin)
p_tkin_f2 = plot(nan, nan, 'bd', 'MarkerSize', 8.5, 'MarkerFaceColor', [0.4, 0.7, 1.0], ...
                 'LineWidth', 1.5, 'DisplayName', 'Ground Truth t_{kin}');
for k = 1:height(tkin_table)
    t_k = tkin_table.(tkin_col)(k);
    if ~isnan(t_k) && t_k >= t(1) && t_k <= t(end)
        xline(t_k, 'b--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
        plot(t_k, interp1(t, theta, t_k), 'bd', 'MarkerSize', 8.5, ...
             'MarkerFaceColor', [0.4, 0.7, 1.0], 'LineWidth', 1.5, 'HandleVisibility', 'off');
    end
end

% Visual Cue Placeholders for Legend
h_m5_f2 = patch(nan, nan, color_palette(5), 'EdgeColor', 'none', 'FaceAlpha', 0.35, 'DisplayName', 'Cue: Hand Open (M5)');
h_m6_f2 = patch(nan, nan, color_palette(6), 'EdgeColor', 'none', 'FaceAlpha', 0.35, 'DisplayName', 'Cue: Fist Clench (M6)');

xlim([t(1), t(end)]);
ylim(y_limits_f2);
xlabel('Continuous Timeline (seconds)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('PIP Excursion Angle (degrees)', 'FontSize', 11, 'FontWeight', 'bold');
title(sprintf('FIFO Active Window Analysis vs Kinematics (S%d)', subject_id), ...
      'FontSize', 12, 'FontWeight', 'bold');
legend([p_theta_f2, p_fifo_legend, h_m5_f2, h_m6_f2, p_tkin_f2], ...
       'Location', 'southoutside', 'Orientation', 'horizontal', 'FontSize', 9.5);

% ==============================================================================
% HELPER FUNCTION
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