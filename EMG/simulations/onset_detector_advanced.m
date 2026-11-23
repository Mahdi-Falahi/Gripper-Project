% ==============================================================================
% REAL-TIME DYNAMIC ONSET DETECTION & DIRECTIONAL STEERING (NinaPro DB2)
% Architecture: Asynchronous Dual-Engine (30 ms Window | 10 ms Hop | 100 Hz Loop)
% Engine 1: 4D Bilateral Onset Tripwire [EDC_MAV, EDC_Mob, FDS_WAMP, FDS_WL]
% Engine 2: 40 ms Motor-Locked Burst Examination & Directional Decoding (NDI)
% Adaptation: Ledoit-Wolf Baseline Manifold with Circular State Rollback (200 ms)
% Calibration: Dual-Phase Partitioned (Phase 1: M5 Trials 1-6 | Phase 2: M6 Trials 7-12)
% ==============================================================================
clear; clc; close all;

% --- 1. Direct Paths & Ingestion ---
subject_id = 9;
mat_path  = sprintf('datasets/DB2/DB2_s%d/S%d_E1_A1_pruned.mat', subject_id, subject_id);
tkin_path = sprintf('datasets/DB2/DB2_s%d/S%d_E1_A1_pruned_tkin_log.csv', subject_id, subject_id);
if ~exist(mat_path, 'file'),  mat_path  = sprintf('S%d_E1_A1_pruned.mat', subject_id); end
if ~exist(tkin_path, 'file'), tkin_path = sprintf('S%d_E1_A1_pruned_tkin_log.csv', subject_id); end

fprintf('[INGESTION] Loading Pruned MAT : %s\n', mat_path);
data = load(mat_path, 'pruned_table');
pt   = data.pruned_table;

fprintf('[INGESTION] Loading tkin CSV   : %s\n', tkin_path);
tkin_table = readtable(tkin_path);

% Target movements: 5 = Hand Open (EDC agonist), 6 = Fist Clench (FDS agonist)
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
emg_edc   = double(pt.emg_edc);      % Ch 9 (EDC - Extensor Digitorum Communis)
emg_fds   = double(pt.emg_fds);      % Ch 10 (FDS - Flexor Digitorum Superficialis)
stim_data = double(pt.stimulus);     % Visual cue / stimulus state
rep_data  = double(pt.repetition);   % Repetition counter (1-6)
N_pts     = height(pt);
t         = (0:N_pts-1)' / fs_emg;
theta_vel = [0; diff(theta) * fs_emg]; % Instantaneous angular velocity (deg/s)

% --- 3. Extract Visual Cue Blocks for Timeline Shading ---
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

blocks_m5 = shade_blocks([shade_blocks.mov] == 5);
blocks_m6 = shade_blocks([shade_blocks.mov] == 6);

% --- 4. Baseline Calibration: Dual-Phase Partitioned Rest Search ---
disp('[CALIBRATION] Locating Phase 1 and Phase 2 rest baseline windows...');
vel_near_zero     = 3.0;                 % deg/s stillness threshold
min_rest_dur_pts  = round(1.0 * fs_emg); % Minimum 1.0 s requirement
settle_buffer_pts = round(0.5 * fs_emg); % 500 ms post-release buffer
is_rest_sample    = (stim_data == 0);

% Mask 500 ms post-movement release to avoid viscoelastic tendon recoil
active_releases = find(diff([0; stim_data > 0]) < 0);
for ar = 1:numel(active_releases)
    idx_settle_start = active_releases(ar);
    idx_settle_end   = min(numel(is_rest_sample), idx_settle_start + settle_buffer_pts);
    is_rest_sample(idx_settle_start:idx_settle_end) = false;
end

% ==================== SEED 1: PRE-MOVEMENT 5 (TRIALS 1 TO 6) ====================
t_m5_first_cue    = blocks_m5(1).tStart;
idx_m5_first_cue  = max(1, round(t_m5_first_cue * fs_emg));

is_pre_cue_rest1  = is_rest_sample;
is_pre_cue_rest1(idx_m5_first_cue:end) = false;
is_pre_cue_quiet1 = is_pre_cue_rest1 & (abs(theta_vel) <= vel_near_zero);

quiet_diff1    = diff([0; is_pre_cue_quiet1; 0]);
block_starts1  = find(quiet_diff1 == 1);
block_ends1    = find(quiet_diff1 == -1) - 1;
block_lengths1 = block_ends1 - block_starts1 + 1;
valid_pre1     = find(block_lengths1 >= min_rest_dur_pts);

if ~isempty(valid_pre1)
    [~, best_sub1] = max(block_lengths1(valid_pre1));
    b1             = valid_pre1(best_sub1);
    idx_cal1_start = block_starts1(b1);
    cal1_len       = min(block_lengths1(b1), round(2.5 * fs_emg));
    idx_cal1_end   = idx_cal1_start + cal1_len - 1;
else
    disp('[WARN] Pre-M5 stillness < 1.0 s. Selecting minimum-jitter window...');
    search_lim1 = max(min_rest_dur_pts + 1, idx_m5_first_cue);
    mov_v1      = movmean(abs(theta_vel(1:search_lim1)), min_rest_dur_pts, 'Endpoints', 'discard');
    [~, b_idx1] = min(mov_v1);
    idx_cal1_start = b_idx1;
    idx_cal1_end   = idx_cal1_start + min_rest_dur_pts - 1;
end
t_cal1_start = t(idx_cal1_start);
t_cal1_end   = t(idx_cal1_end);
fprintf('[CALIBRATION 1] Phase 1 (M5) Rest Seed: %.2f s to %.2f s (Duration: %.2f s)\n', ...
        t_cal1_start, t_cal1_end, t_cal1_end - t_cal1_start);

% ==================== SEED 2: PRE-MOVEMENT 6 (TRIALS 7 TO 12) ====================
idx_m5_release   = min(N_pts, round(blocks_m5(end).tEnd * fs_emg) + settle_buffer_pts);
idx_m6_first_cue = max(1, round(blocks_m6(1).tStart * fs_emg));

is_pause_rest2   = false(size(is_rest_sample));
if idx_m5_release < idx_m6_first_cue
    is_pause_rest2(idx_m5_release:idx_m6_first_cue) = is_rest_sample(idx_m5_release:idx_m6_first_cue);
end
is_pause_quiet2 = is_pause_rest2 & (abs(theta_vel) <= vel_near_zero);

quiet_diff2    = diff([0; is_pause_quiet2; 0]);
block_starts2  = find(quiet_diff2 == 1);
block_ends2    = find(quiet_diff2 == -1) - 1;
block_lengths2 = block_ends2 - block_starts2 + 1;
valid_pause2   = find(block_lengths2 >= min_rest_dur_pts);

if ~isempty(valid_pause2)
    [~, best_sub2] = max(block_lengths2(valid_pause2));
    b2             = valid_pause2(best_sub2);
    idx_cal2_start = block_starts2(b2);
    cal2_len       = min(block_lengths2(b2), round(2.5 * fs_emg));
    idx_cal2_end   = idx_cal2_start + cal2_len - 1;
else
    disp('[WARN] Inter-movement pause stillness < 1.0 s. Selecting minimum-jitter window...');
    search_seg2 = abs(theta_vel(idx_m5_release:idx_m6_first_cue));
    if numel(search_seg2) >= min_rest_dur_pts
        mov_v2 = movmean(search_seg2, min_rest_dur_pts, 'Endpoints', 'discard');
        [~, b_idx2] = min(mov_v2);
        idx_cal2_start = idx_m5_release + b_idx2 - 1;
        idx_cal2_end   = idx_cal2_start + min_rest_dur_pts - 1;
    else
        idx_cal2_start = idx_m5_release;
        idx_cal2_end   = idx_m6_first_cue - 1;
    end
end
t_cal2_start = t(idx_cal2_start);
t_cal2_end   = t(idx_cal2_end);
fprintf('[CALIBRATION 2] Phase 2 (M6) Rest Seed: %.2f s to %.2f s (Duration: %.2f s)\n', ...
        t_cal2_start, t_cal2_end, t_cal2_end - t_cal2_start);

% Common Windowing Parameters
win_dm_pts      = round(0.030 * fs_emg); % 30 ms window
step_pts        = round(0.010 * fs_emg); % 10 ms hop
min_sigma_floor = [2.13e-07, 0.0352, 0.1386, 6.50e-06];

% Extract and regularize Manifold 1 (Phase 1: Movement 5)
rest_feats_1 = [];
for i = idx_cal1_start : step_pts : (idx_cal1_end - win_dm_pts + 1)
    w_e = emg_edc(i : i + win_dm_pts - 1);
    w_f = emg_fds(i : i + win_dm_pts - 1);
    rest_feats_1 = [rest_feats_1; extract_onset_features_4d(w_e, w_f)]; %#ok<AGROW>
end
mu_scale_1    = mean(rest_feats_1, 1);
sigma_scale_1 = max(std(rest_feats_1, 0, 1), min_sigma_floor);
Z_rest_1      = (rest_feats_1 - mu_scale_1) ./ sigma_scale_1;
mu_rest_1     = mean(Z_rest_1, 1);
[cov_rest_1, inv_cov_rest_1, rho_seed_1] = ledoit_wolf_standardized(Z_rest_1);

% Extract and regularize Manifold 2 (Phase 2: Movement 6)
rest_feats_2 = [];
for i = idx_cal2_start : step_pts : (idx_cal2_end - win_dm_pts + 1)
    w_e = emg_edc(i : i + win_dm_pts - 1);
    w_f = emg_fds(i : i + win_dm_pts - 1);
    rest_feats_2 = [rest_feats_2; extract_onset_features_4d(w_e, w_f)]; %#ok<AGROW>
end
mu_scale_2    = mean(rest_feats_2, 1);
sigma_scale_2 = max(std(rest_feats_2, 0, 1), min_sigma_floor);
Z_rest_2      = (rest_feats_2 - mu_scale_2) ./ sigma_scale_2;
mu_rest_2     = mean(Z_rest_2, 1);
[cov_rest_2, inv_cov_rest_2, rho_seed_2] = ledoit_wolf_standardized(Z_rest_2);

tau_tripwire   = 4.0313; % Chi-Square departure tripwire (alpha = 0.0027, df = 4)

% --- 5. Continuous Streaming Engine with Circular State Rollback ---
fprintf('[STREAMING] Initializing Dual-Engine Controller with Partitioned Search...\n');

% Exclusion mask for plateaus and release periods
is_excluded_sample = false(N_pts, 1);
is_excluded_sample(1:idx_cal1_end) = true;               % Mask calibration 1
is_excluded_sample(idx_cal2_start:idx_cal2_end) = true; % Mask calibration 2

for k = 1:numel(shade_blocks)
    sb = shade_blocks(k);
    match_row = tkin_table(tkin_table.movement == sb.mov & tkin_table.repetition == sb.rep, :);
    if ~isempty(match_row) && ~isnan(match_row.(tkin_col)(1))
        t_ang_k    = match_row.(tkin_col)(1);
        t_ex_start = t_ang_k + 0.250; % Cutoff 250 ms post-kinematic takeoff
    else
        t_ex_start = sb.tStart + 1.0;
    end
    t_ex_end = sb.tEnd + 2.0;         % Post-release buffer
    
    idx_ex_start = max(1, min(N_pts, round(t_ex_start * fs_emg)));
    idx_ex_end   = max(1, min(N_pts, round(t_ex_end * fs_emg)));
    if idx_ex_start <= idx_ex_end
        is_excluded_sample(idx_ex_start:idx_ex_end) = true;
    end
end

% Partitioned evaluation indices: M5 (Trials 1-6) followed by M6 (Trials 7-12)
raw_indices_m5 = (idx_cal1_end + 1) : step_pts : (idx_cal2_start - win_dm_pts);
win_ends_m5    = raw_indices_m5 + win_dm_pts - 1;
sample_idx_m5  = raw_indices_m5(~is_excluded_sample(win_ends_m5));

raw_indices_m6 = (idx_cal2_end + 1) : step_pts : (N_pts - win_dm_pts);
win_ends_m6    = raw_indices_m6 + win_dm_pts - 1;
sample_idx_m6  = raw_indices_m6(~is_excluded_sample(win_ends_m6));

sample_indices = [sample_idx_m5, sample_idx_m6];
N_phase1_hops  = numel(sample_idx_m5);

% State Machine Parameters
STATE_REST       = 0;
STATE_EXAMINE    = 1;
STATE_REFRACTORY = 2;

current_state      = STATE_REST;
consec_hits        = 0;
quiet_hops         = 0;
rearm_hops_target  = 5;
alpha_floor        = 0.002;
tau_gate_adapt     = tau_tripwire;

% Initialize active baseline manifold to Phase 1 (Movement 5)
mu_scale     = mu_scale_1;
sigma_scale  = sigma_scale_1;
mu_rest      = mu_rest_1;
cov_rest     = cov_rest_1;
inv_cov_rest = inv_cov_rest_1;
rho_seed     = rho_seed_1;

% Circular State Buffer (20 slots x 10 ms = 200 ms lookback window)
history_len        = 20;
mu_buf             = repmat(mu_rest, history_len, 1);
cov_buf            = repmat(cov_rest, [1, 1, history_len]);
buf_ptr            = 1;
restore_idx        = 1;

t_candidate        = NaN;
theta_candidate    = NaN;
idx_candidate      = 1;

exam_hops_total    = 4;
exam_hops_left     = 0;
exam_dm_accum      = [];

detections          = [];
rejected_detections = [];
dm_stream           = nan(numel(sample_indices), 1);
t_stream            = nan(numel(sample_indices), 1);

fifo_fill_mask = false(N_pts, 1);
fifo_fill_mask(idx_cal1_start:idx_cal1_end) = true;
fifo_fill_mask(idx_cal2_start:idx_cal2_end) = true;

% ==================== SINGLE UNIFIED STREAMING LOOP ====================
for s = 1:numel(sample_indices)
    cur_idx = sample_indices(s);
    t_now   = t(cur_idx + win_dm_pts - 1);
    
    % Switch to Phase 2 Manifold at Movement 6 boundary
    if s == N_phase1_hops + 1
        mu_scale     = mu_scale_2;
        sigma_scale  = sigma_scale_2;
        mu_rest      = mu_rest_2;
        cov_rest     = cov_rest_2;
        inv_cov_rest = inv_cov_rest_2;
        rho_seed     = rho_seed_2;
        
        current_state = STATE_REST;
        consec_hits   = 0;
        quiet_hops    = 0;
        mu_buf        = repmat(mu_rest, history_len, 1);
        cov_buf       = repmat(cov_rest, [1, 1, history_len]);
        buf_ptr       = 1;
    % Reset state machine across excluded plateau boundaries back to rest
    elseif s > 1 && (cur_idx - sample_indices(s - 1)) > step_pts
        current_state = STATE_REST;
        consec_hits   = 0;
        quiet_hops    = 0;
        mu_buf        = repmat(mu_rest, history_len, 1);
        cov_buf       = repmat(cov_rest, [1, 1, history_len]);
        buf_ptr       = 1;
    end
    
    % Sliding 30 ms state detection window
    w_e = emg_edc(cur_idx : cur_idx + win_dm_pts - 1);
    w_f = emg_fds(cur_idx : cur_idx + win_dm_pts - 1);
    
    phi_raw = extract_onset_features_4d(w_e, w_f);
    
    % Standardize incoming features and clamp extreme numerical spikes
    z_feat = (phi_raw - mu_scale) ./ sigma_scale;
    z_feat = max(min(z_feat, 20.0), -20.0); % Clamp to +/- 20 sigma
    
    % Compute Mahalanobis Distance on active standardized manifold
    diff_u = z_feat - mu_rest;
    dm_val = sqrt(diff_u * inv_cov_rest * diff_u');
    
    dm_stream(s) = dm_val;
    t_stream(s)  = t_now;
    
    switch current_state
        case STATE_REST
            if dm_val <= tau_gate_adapt
                % Continuous shrinkage-regularized adaptation
                mu_rest  = (1 - alpha_floor) * mu_rest + alpha_floor * z_feat;
                dev      = z_feat - mu_rest;
                
                s_sample = (1.0 - rho_seed) * (dev' * dev) + rho_seed * eye(4);
                cov_rest = (1 - alpha_floor) * cov_rest + alpha_floor * s_sample;
                inv_cov_rest = inv(cov_rest + 1e-4 * eye(4));
                
                buf_ptr = mod(buf_ptr, history_len) + 1;
                mu_buf(buf_ptr, :)     = mu_rest;
                cov_buf(:, :, buf_ptr) = cov_rest;
                
                idx_hop_end = min(N_pts, cur_idx + win_dm_pts - 1);
                fifo_fill_mask(cur_idx:idx_hop_end) = true;
                consec_hits = 0;
                
            elseif dm_val >= tau_tripwire
                if consec_hits == 0
                    t_candidate     = t_now;
                    theta_candidate = theta(cur_idx + win_dm_pts - 1);
                    idx_candidate   = cur_idx;
                    restore_idx     = mod(buf_ptr, history_len) + 1;
                end
                consec_hits = consec_hits + 1;
                
                if consec_hits >= 3
                    current_state   = STATE_EXAMINE;
                    exam_hops_left  = exam_hops_total;
                    exam_dm_accum   = dm_val;
                    consec_hits     = 0;
                end
            else
                consec_hits = 0;
            end
            
        case STATE_EXAMINE
            exam_dm_accum  = [exam_dm_accum, dm_val]; %#ok<AGROW>
            exam_hops_left = exam_hops_left - 1;
            
            if exam_hops_left <= 0
                mean_exam_dm = mean(exam_dm_accum);
                
                if mean_exam_dm >= 2.50
                    % Valid onset confirmed
                    w_burst_e = emg_edc(idx_candidate : min(N_pts, idx_candidate + round(0.040 * fs_emg)));
                    w_burst_f = emg_fds(idx_candidate : min(N_pts, idx_candidate + round(0.040 * fs_emg)));
                    
                    rms_e = sqrt(mean(w_burst_e.^2));
                    rms_f = sqrt(mean(w_burst_f.^2));
                    ndi   = (rms_e - rms_f) / (rms_e + rms_f + 1e-6);
                    
                    if ndi >= 0.0
                        classified_dir = 5; % Hand Open
                    else
                        classified_dir = 6; % Fist Clench
                    end
                    
                    detections = [detections; t_candidate, theta_candidate, classified_dir]; %#ok<AGROW>
                    
                    mu_rest      = mu_buf(restore_idx, :);
                    cov_rest     = cov_buf(:, :, restore_idx);
                    inv_cov_rest = inv(cov_rest + 1e-4 * eye(4));
                    
                    mu_buf   = repmat(mu_rest, history_len, 1);
                    cov_buf  = repmat(cov_rest, [1, 1, history_len]);
                    buf_ptr  = 1;
                    
                    current_state = STATE_REFRACTORY;
                    quiet_hops    = 0;
                else
                    % Reject twitch
                    rejected_detections = [rejected_detections; t_candidate, theta_candidate]; %#ok<AGROW>
                    
                    mu_rest      = mu_buf(restore_idx, :);
                    cov_rest     = cov_buf(:, :, restore_idx);
                    inv_cov_rest = inv(cov_rest + 1e-4 * eye(4));
                    
                    current_state = STATE_REST;
                    consec_hits   = 0;
                end
            end
            
        case STATE_REFRACTORY
            if dm_val < 3.0
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

fprintf('[COMPLETION] Detected Onsets: %d | Rejected Candidates: %d | Ground-Truth Prompts: %d\n', ...
        size(detections, 1), size(rejected_detections, 1), height(tkin_table));

% --- 6. Chronological Timeseries Event Stream Logging ---
disp('[LOGGING] Compiling chronological timeseries event stream...');
event_records = {};
for k = 1:numel(shade_blocks)
    sb = shade_blocks(k);
    event_records(end+1, :) = {sb.tStart, 'CUE_START', interp1(t, theta, sb.tStart), sb.mov, sb.rep, NaN}; %#ok<AGROW>
    event_records(end+1, :) = {sb.tEnd,   'CUE_END',   interp1(t, theta, sb.tEnd),   sb.mov, sb.rep, NaN}; %#ok<AGROW>
end
for k = 1:height(tkin_table)
    t_k = tkin_table.(tkin_col)(k);
    if ~isnan(t_k) && t_k >= t(1) && t_k <= t(end)
        event_records(end+1, :) = {t_k, 'TKIN', interp1(t, theta, t_k), ...
                                   tkin_table.movement(k), tkin_table.repetition(k), NaN}; %#ok<AGROW>
    end
end
for d = 1:size(detections, 1)
    t_det   = detections(d, 1);
    ang_det = detections(d, 2);
    dir_det = detections(d, 3);
    
    idx_sample  = min(numel(stim_data), max(1, round(t_det * fs_emg)));
    active_stim = stim_data(idx_sample);
    active_rep  = rep_data(idx_sample);
    
    event_records(end+1, :) = {t_det, 'MYO_ONSET', ang_det, active_stim, active_rep, dir_det}; %#ok<AGROW>
end
event_timeseries_log = cell2table(event_records, 'VariableNames', ...
    {'timestamp_s', 'event_type', 'theta_deg', 'active_prompt', 'repetition', 'classified_dir'});
event_timeseries_log = sortrows(event_timeseries_log, 'timestamp_s');
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

% --- 7. Diagnostic Timeline Visualization: DM (Smaller) & Kinematics (Bigger) ---
fig = figure('Color', 'w', 'Position', [50, 40, 1350, 840], ...
             'Name', sprintf('Dual-Engine DM & Kinematic Timeline (S%d)', subject_id));

% --------------------------------------------------------------------------
% SUBPLOT 1 (TOP, SMALLER): Mahalanobis Distance Trajectory (D_M) - 1/3 Height
% --------------------------------------------------------------------------
ax1 = subplot(3, 1, 1); hold on; grid on; box on;
p_cal_dm1 = patch([t_cal1_start, t_cal1_end, t_cal1_end, t_cal1_start], ...
                  [0, 0, 50, 50], [0.55, 0.55, 0.55], ...
                  'EdgeColor', 'none', 'FaceAlpha', 0.40, 'DisplayName', 'Rest Cal 1 (M5)');
uistack(p_cal_dm1, 'bottom');

p_cal_dm2 = patch([t_cal2_start, t_cal2_end, t_cal2_end, t_cal2_start], ...
                  [0, 0, 50, 50], [0.45, 0.45, 0.45], ...
                  'EdgeColor', 'none', 'FaceAlpha', 0.40, 'DisplayName', 'Rest Cal 2 (M6)');
uistack(p_cal_dm2, 'bottom');

for k = 1:numel(shade_blocks)
    sb = shade_blocks(k);
    p_s = patch([sb.tStart, sb.tEnd, sb.tEnd, sb.tStart], ...
                [0, 0, 50, 50], sb.color, 'EdgeColor', 'none', ...
                'FaceAlpha', 0.35, 'HandleVisibility', 'off');
    uistack(p_s, 'bottom');
end
p_dm = plot(t_stream, dm_stream, 'Color', [0.15, 0.15, 0.20], 'LineWidth', 1.2, ...
            'DisplayName', 'Mahalanobis Distance (D_M)');
yline(tau_tripwire, 'r--', 'LineWidth', 1.5, ...
      'DisplayName', sprintf('Tripwire & Adaptation Boundary (\\tau = %.2f)', tau_tripwire));
yline(2.50, 'Color', [0.85, 0.45, 0.10], 'LineStyle', ':', 'LineWidth', 1.4, ...
      'DisplayName', 'Engine 2 Hurdle (2.50)');
for d = 1:size(detections, 1)
    xline(detections(d, 1), 'b-', 'LineWidth', 1.3, 'HandleVisibility', 'off');
end
p_rej_dm = plot(nan, nan, 'x', 'MarkerSize', 8.5, 'LineWidth', 2.0, ...
                'Color', [0.45, 0.45, 0.45], 'DisplayName', 'Rejected Onset (Twitch)');
for r = 1:size(rejected_detections, 1)
    t_rej = rejected_detections(r, 1);
    dm_at_rej = interp1(t_stream(~isnan(dm_stream)), dm_stream(~isnan(dm_stream)), t_rej, 'nearest');
    plot(t_rej, dm_at_rej, 'x', 'MarkerSize', 8.5, 'LineWidth', 2.0, ...
         'Color', [0.45, 0.45, 0.45], 'HandleVisibility', 'off');
    xline(t_rej, ':', 'Color', [0.55, 0.55, 0.55], 'LineWidth', 1.2, 'HandleVisibility', 'off');
end
ylabel('D_M', 'FontSize', 10, 'FontWeight', 'bold');
title(sprintf('Dual-Engine Diagnostics: S%d (Top: D_M Subspace | Bottom: Joint Kinematics)', subject_id), ...
      'FontSize', 11, 'FontWeight', 'bold');
xlim([t(1), t(end)]);
ylim([0, max([max(dm_stream), tau_tripwire * 2, 12])]);
legend([p_dm, p_cal_dm1, p_cal_dm2, p_rej_dm], 'Location', 'northeast', 'FontSize', 8.5);

% --------------------------------------------------------------------------
% SUBPLOT 2 (BOTTOM, BIGGER): Finger Joint Kinematics - 2/3 Height
% --------------------------------------------------------------------------
ax2 = subplot(3, 1, [2, 3]); hold on; grid on; box on;
y_limits = [min(theta) - 10, max(theta) + 15];
p_cal1 = patch([t_cal1_start, t_cal1_end, t_cal1_end, t_cal1_start], ...
               [y_limits(1), y_limits(1), y_limits(2), y_limits(2)], ...
               [0.55, 0.55, 0.55], 'EdgeColor', 'none', 'FaceAlpha', 0.40, ...
               'DisplayName', 'Rest Cal 1 (M5)');
uistack(p_cal1, 'bottom');

p_cal2 = patch([t_cal2_start, t_cal2_end, t_cal2_end, t_cal2_start], ...
               [y_limits(1), y_limits(1), y_limits(2), y_limits(2)], ...
               [0.45, 0.45, 0.45], 'EdgeColor', 'none', 'FaceAlpha', 0.40, ...
               'DisplayName', 'Rest Cal 2 (M6)');
uistack(p_cal2, 'bottom');

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
         'FontSize', 9.0, 'FontWeight', 'bold');
end
p_theta = plot(t, theta, 'Color', [0.15, 0.15, 0.15], 'LineWidth', 1.8, ...
               'DisplayName', 'Finger PIP Angle (\theta)');
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
p_onset_m5 = plot(nan, nan, '^', 'MarkerSize', 9.5, 'LineWidth', 1.6, ...
                  'Color', [0.10, 0.45, 0.85], 'MarkerFaceColor', [0.35, 0.75, 1.0], ...
                  'DisplayName', 'Confirmed Onset (Hand Open)');
p_onset_m6 = plot(nan, nan, 'v', 'MarkerSize', 9.5, 'LineWidth', 1.6, ...
                  'Color', [0.85, 0.15, 0.15], 'MarkerFaceColor', [1.0, 0.35, 0.35], ...
                  'DisplayName', 'Confirmed Onset (Fist Clench)');
for d = 1:size(detections, 1)
    t_det   = detections(d, 1);
    ang_det = detections(d, 2);
    dir_det = detections(d, 3);
    
    if dir_det == 5
        plot(t_det, ang_det, '^', 'MarkerSize', 9.5, 'LineWidth', 1.6, ...
             'Color', [0.10, 0.45, 0.85], 'MarkerFaceColor', [0.35, 0.75, 1.0], 'HandleVisibility', 'off');
        xline(t_det, ':', 'Color', [0.10, 0.45, 0.85], 'LineWidth', 1.4, 'HandleVisibility', 'off');
    else
        plot(t_det, ang_det, 'v', 'MarkerSize', 9.5, 'LineWidth', 1.6, ...
             'Color', [0.85, 0.15, 0.15], 'MarkerFaceColor', [1.0, 0.35, 0.35], 'HandleVisibility', 'off');
        xline(t_det, ':', 'Color', [0.85, 0.15, 0.15], 'LineWidth', 1.4, 'HandleVisibility', 'off');
    end
end
p_onset_rej = plot(nan, nan, 'x', 'MarkerSize', 10.0, 'LineWidth', 2.0, ...
                   'Color', [0.45, 0.45, 0.45], 'DisplayName', 'Rejected Candidate Onset');
for r = 1:size(rejected_detections, 1)
    t_rej   = rejected_detections(r, 1);
    ang_rej = rejected_detections(r, 2);
    
    plot(t_rej, ang_rej, 'x', 'MarkerSize', 10.0, 'LineWidth', 2.0, ...
         'Color', [0.45, 0.45, 0.45], 'HandleVisibility', 'off');
    xline(t_rej, ':', 'Color', [0.55, 0.55, 0.55], 'LineWidth', 1.2, 'HandleVisibility', 'off');
end
h_m5 = patch(nan, nan, color_palette(5), 'EdgeColor', 'none', 'FaceAlpha', 0.45, 'DisplayName', 'Cue: Open (M5)');
h_m6 = patch(nan, nan, color_palette(6), 'EdgeColor', 'none', 'FaceAlpha', 0.45, 'DisplayName', 'Cue: Fist (M6)');
xlim([t(1), t(end)]);
ylim(y_limits);
xlabel('Continuous Timeline (seconds)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('PIP Excursion Angle (degrees)', 'FontSize', 11, 'FontWeight', 'bold');
legend([p_theta, p_cal1, p_cal2, h_m5, h_m6, p_tkin, p_onset_m5, p_onset_m6, p_onset_rej], ...
       'Location', 'southoutside', 'Orientation', 'horizontal', 'FontSize', 9.5);
linkaxes([ax1, ax2], 'x');

% ==============================================================================
% 8. DEDICATED WINDOW: FIFO ACTIVE FILLING ZONES & KINEMATICS
% ==============================================================================
fig2 = figure('Color', 'w', 'Position', [100, 150, 1350, 520], ...
              'Name', sprintf('FIFO Ingestion Windows vs Kinematics (S%d)', subject_id));
hold on; grid on; box on;
y_limits_f2 = [min(theta) - 10, max(theta) + 15];
d_fifo   = diff([0; fifo_fill_mask; 0]);
f_starts = find(d_fifo == 1);
f_ends   = find(d_fifo == -1) - 1;
p_fifo_legend = patch(nan, nan, [1.0, 0.90, 0.65], 'EdgeColor', 'none', ...
                      'FaceAlpha', 0.55, 'DisplayName', 'FIFO Actively Ingesting (Adaptation)');
for k = 1:numel(f_starts)
    t_s = t(f_starts(k));
    t_e = t(f_ends(k));
    p_f = patch([t_s, t_e, t_e, t_s], ...
                [y_limits_f2(1), y_limits_f2(1), y_limits_f2(2), y_limits_f2(2)], ...
                [1.0, 0.90, 0.65], 'EdgeColor', 'none', 'FaceAlpha', 0.55, 'HandleVisibility', 'off');
    uistack(p_f, 'bottom');
end
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
p_theta_f2 = plot(t, theta, 'Color', [0.15, 0.15, 0.15], 'LineWidth', 1.6, ...
                  'DisplayName', 'Finger PIP Angle (\theta)');
p_tkin_f2  = plot(nan, nan, 'bd', 'MarkerSize', 8.5, 'MarkerFaceColor', [0.4, 0.7, 1.0], ...
                  'LineWidth', 1.5, 'DisplayName', 'Ground Truth t_{kin}');
for k = 1:height(tkin_table)
    t_k = tkin_table.(tkin_col)(k);
    if ~isnan(t_k) && t_k >= t(1) && t_k <= t(end)
        xline(t_k, 'b--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
        plot(t_k, interp1(t, theta, t_k), 'bd', 'MarkerSize', 8.5, ...
             'MarkerFaceColor', [0.4, 0.7, 1.0], 'LineWidth', 1.5, 'HandleVisibility', 'off');
    end
end
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
% HELPER FUNCTIONS
% ==============================================================================
function feats = extract_onset_features_4d(edc, fds)
    mav_e = mean(abs(edc));
    dx_e  = diff(edc);
    mob_e = sqrt(var(dx_e) / (var(edc) + 1e-12));
    
    wamp_thresh = 1e-5; % 10 uV threshold
    wamp_f = sum(abs(diff(fds)) >= wamp_thresh);
    wl_f   = sum(abs(diff(fds)));
    
    feats = [mav_e, mob_e, double(wamp_f), wl_f];
end

function [sigma_shrunk, inv_sigma, rho] = ledoit_wolf_standardized(Z)
    % Operates on dimensionless, z-scored features (mean 0, var 1)
    [n, p] = size(Z);
    mu     = mean(Z, 1);
    Zc     = Z - mu;
    S      = (Zc' * Zc) / n;
    
    % In standardized space, shrinkage target is the identity matrix I
    target = eye(p);
    d2     = norm(S - target, 'fro')^2;
    
    b_bar2 = 0;
    for i = 1:n
        y_i    = Zc(i, :)' * Zc(i, :);
        b_bar2 = b_bar2 + norm(y_i - S, 'fro')^2;
    end
    b_bar2 = b_bar2 / (n^2);
    
    % Guarantee at least 15% shrinkage intensity to prevent singular collapse
    rho          = min(1.0, max(0.15, b_bar2 / max(d2, 1e-12)));
    sigma_shrunk = (1.0 - rho) * S + rho * target;
    inv_sigma    = inv(sigma_shrunk + 1e-4 * eye(p));
end