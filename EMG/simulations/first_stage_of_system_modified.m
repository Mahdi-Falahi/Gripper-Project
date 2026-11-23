% ==============================================================================
% REAL-TIME ADAPTIVE DUAL-ENGINE ONSET DETECTOR & TRAJECTORY EXAMINER
% Architecture: Candidate Cluster Queue with 150 ms Deferred Auto-Tuning
% Ingests: S{id}_E1_A1_pruned.mat & S{id}_E1_A1_kinematic_trapezoid_log.csv
% Features:
%   1. Stage 0 Rest Noise Floor Calibration (Neutral Gray Shading)
%   2. Stage 1 Dynamic Range & Velocity Anchor (Rep 1 Calibration Gray Shading)
%   3. Candidate Cluster Grace Engine:
%        - Allows Engine 1 to fire repeated candidates without self-stoppage
%        - Defers Engine 1 auto-tuning/punishment across a 150 ms window
%        - If ANY candidate confirms -> Rescues earliest candidate timestamp (GREEN)
%        - If cluster expires without confirmation -> Final Rejection (RED) + Auto-Tune
%   4. Search space exclusion mask: [t_kin + 0.250 s, tEnd + 1.0 s]
% ==============================================================================
clear; clc; close all;

% --- 1. Paths & Ingestion ---
subject_id = 1; % Changeable to any active DB2 subject

base_dir = fullfile('datasets', 'DB2', sprintf('DB2_s%d', subject_id));
if ~exist(base_dir, 'dir')
    base_dir = '.';
end

mat_path = fullfile(base_dir, sprintf('S%d_E1_A1_pruned.mat', subject_id));
csv_path = fullfile(base_dir, sprintf('S%d_E1_A1_kinematic_trapezoid_log.csv', subject_id));

if ~exist(csv_path, 'file')
    csv_path = fullfile(base_dir, sprintf('S%d_E1_A1_pruned_tkin_log.csv', subject_id));
end

fprintf('[INGESTION] Loading MAT: %s\n', mat_path);
data = load(mat_path, 'pruned_table');
pt   = data.pruned_table;

fprintf('[INGESTION] Loading Kinematics: %s\n', csv_path);
k_table = readtable(csv_path);

% Filter target 1-DoF movements (5 = Hand Open, 6 = Fist Clench)
target_movements = [5, 6];
k_table = k_table(ismember(k_table.movement, target_movements), :);

if ismember('t1_takeoff_s', k_table.Properties.VariableNames)
    tkin_col = 't1_takeoff_s';
elseif ismember('tkin_angle', k_table.Properties.VariableNames)
    tkin_col = 'tkin_angle';
elseif ismember('t_ang', k_table.Properties.VariableNames)
    tkin_col = 't_ang';
else
    error('Could not find kinematic takeoff column in %s', csv_path);
end

% --- 2. Signal Extraction ---
fs_emg    = 2000;
theta     = double(pt.joint_angle);  % PIP joint angle (deg)
emg_edc   = double(pt.emg_edc);      % Ch 9 (EDC - Extensor)
emg_fds   = double(pt.emg_fds);      % Ch 10 (FDS - Flexor)
stim_data = double(pt.stimulus);     % Visual stimulus prompt
rep_data  = double(pt.repetition);   % Repetition counter (1-6)
N_pts     = height(pt);
t         = (0:N_pts-1)' / fs_emg;
theta_vel = [0; diff(theta) * fs_emg]; % Kinematic velocity (deg/s)

% --- 3. Extract Prompt Blocks for Timeline Shading ---
shade_blocks = struct('mov', {}, 'rep', {}, 'tStart', {}, 'tEnd', {}, 'color', {}, 'is_calib', {});
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
        
        is_cal = (r_k == 1);
        if is_cal
            blk_col = [0.70, 0.70, 0.70]; % Calibration Gray (Rep 1)
        elseif m == 5
            blk_col = [0.75, 0.88, 1.00]; % Test Open (Blue)
        else
            blk_col = [1.00, 0.82, 0.82]; % Test Fist (Red/Salmon)
        end
        
        shade_blocks(end+1) = struct(...
            'mov', m, 'rep', r_k, ...
            'tStart', t(s_k), 'tEnd', t(e_k), ...
            'color', blk_col, 'is_calib', is_cal);
    end
end
[~, sort_order] = sort([shade_blocks.tStart]);
shade_blocks = shade_blocks(sort_order);

% --- 4. STAGE 0: Rest Baseline Calibration (Seed Search) ---
disp('[CALIBRATION] Stage 0: Locating quiet rest seed window...');
vel_near_zero     = 3.0;                 % deg/s stillness threshold
min_rest_dur_pts  = round(1.0 * fs_emg); % Minimum 1.0 s
settle_buffer_pts = round(0.5 * fs_emg); % 500 ms buffer post-release
is_rest_sample    = (stim_data == 0);

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
    cal_len         = min(block_lengths(best_block), round(2.5 * fs_emg));
    idx_cal_end     = idx_cal_start + cal_len - 1;
end

t_cal_start = t(idx_cal_start);
t_cal_end   = t(idx_cal_end);
fprintf('[CALIBRATION] Rest Seed Interval: %.2f s to %.2f s\n', t_cal_start, t_cal_end);

% Extract Engine 1 3D Subspace for Mahalanobis Tripwire
win_dm_pts   = round(0.050 * fs_emg); % 50 ms window
step_cal_pts = round(0.010 * fs_emg); % 10 ms hop
rest_feats   = [];
wl_edc_rest  = [];
wl_fds_rest  = [];

for i = idx_cal_start : step_cal_pts : (idx_cal_end - win_dm_pts + 1)
    w_e = emg_edc(i : i + win_dm_pts - 1);
    w_f = emg_fds(i : i + win_dm_pts - 1);
    rest_feats = [rest_feats; extract_features_dm(w_e, w_f)]; %#ok<AGROW>
    wl_edc_rest(end+1) = sum(abs(diff(w_e))); %#ok<AGROW>
    wl_fds_rest(end+1) = sum(abs(diff(w_f))); %#ok<AGROW>
end

mu_rest      = mean(rest_feats, 1);
cov_rest     = cov(rest_feats) + 1e-4 * eye(size(rest_feats, 2));
inv_cov_rest = inv(cov_rest);

% Engine 2 Noise Floor References
wl_floor_edc = mean(wl_edc_rest);
wl_std_edc   = std(wl_edc_rest);
wl_floor_fds = mean(wl_fds_rest);
wl_std_fds   = std(wl_fds_rest);

% --- 5. STAGE 1: Calibration Trials Dynamic Anchor (Repetition 1) ---
disp('[CALIBRATION] Stage 1: Ingesting Repetition 1 dynamic calibration anchors...');
row_m5_r1 = k_table(k_table.movement == 5 & k_table.repetition == 1, :);
row_m6_r1 = k_table(k_table.movement == 6 & k_table.repetition == 1, :);

tkin_m5_r1 = row_m5_r1.(tkin_col)(1);
tkin_m6_r1 = row_m6_r1.(tkin_col)(1);

idx_m5 = max(1, round(tkin_m5_r1 * fs_emg));
idx_m6 = max(1, round(tkin_m6_r1 * fs_emg));
win_cal_wl = round(0.050 * fs_emg);

w_edc_m5_r1 = emg_edc(idx_m5 - round(0.020*fs_emg) : idx_m5 + win_cal_wl);
w_fds_m6_r1 = emg_fds(idx_m6 - round(0.020*fs_emg) : idx_m6 + win_cal_wl);

wl_ref_edc = max(wl_std_edc * 5, sum(abs(diff(w_edc_m5_r1))));
wl_ref_fds = max(wl_std_fds * 5, sum(abs(diff(w_fds_m6_r1))));

if ismember('rise_dur_ms', row_m5_r1.Properties.VariableNames) && ismember('theta2_deg', row_m5_r1.Properties.VariableNames)
    v_ref_m5 = abs(row_m5_r1.theta2_deg(1) - row_m5_r1.theta1_deg(1)) / max(0.1, row_m5_r1.rise_dur_ms(1)/1000);
    v_ref_m6 = abs(row_m6_r1.theta2_deg(1) - row_m6_r1.theta1_deg(1)) / max(0.1, row_m6_r1.rise_dur_ms(1)/1000);
else
    v_ref_m5 = 120.0;
    v_ref_m6 = 120.0;
end

% --- 6. Streaming Setup & Kinematic Exclusion Gating ---
disp('[STREAMING] Initializing candidate cluster streaming pipeline...');
step_pts = round(0.010 * fs_emg); % 10 ms hop (100 Hz)
raw_indices = 1 : step_pts : (N_pts - win_dm_pts);

% Search space exclusion mask: [t_kin + 0.250 s, tEnd + 1.0 s]
is_excluded_sample = false(N_pts, 1);
for k = 1:numel(shade_blocks)
    sb = shade_blocks(k);
    match_row = k_table(k_table.movement == sb.mov & k_table.repetition == sb.rep, :);
    if ~isempty(match_row) && ~isnan(match_row.(tkin_col)(1))
        t_ang_k    = match_row.(tkin_col)(1);
        t_ex_start = t_ang_k + 0.250;
    else
        t_ex_start = sb.tStart + 1.0;
    end
    t_ex_end = sb.tEnd + 1.0; % 1.0 s post visual cue
    
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
rearm_hops_target  = 10;                  % 100 ms quiet rest required to re-arm
cooldown_samples   = 0;
alpha_floor        = 0.015;               % Rest adaptation rate

% Engine 1 Adaptive Tripwire Parameters
dm_thresh_base     = 3.0;
dm_thresh_current  = dm_thresh_base;
dm_decay_rate      = 0.2;

% Circular State Buffer (20 slots x 10 ms = 200 ms FIFO lookback)
history_len        = 20;
mu_buf             = repmat(mu_rest, history_len, 1);
cov_buf            = repmat(cov_rest, [1, 1, history_len]);
buf_ptr            = 1;

% --- Candidate Cluster Queue Data Structures ---
cluster_active      = false;
cluster_t_start     = NaN;
cluster_restore_idx = 1;
cluster_grace_s     = 0.200;              % 200 ms cluster grace lifespan
cand_spacing_pts    = round(0.020 * fs_emg); % Minimum 20 ms spacing between candidate records
last_cand_idx       = -inf;
candidates          = struct('t_cand', {}, 'cur_idx', {}, 'theta_cand', {}, 'evaluated', {}, 'phi_dm', {});

confirmed_onsets   = []; % [t_onset, theta_onset, est_speed, direction]
rejected_onsets    = []; % [t_cand, theta_cand, dm_val, direction]
dm_trace           = zeros(numel(sample_indices), 3);

% --- 7. Continuous Streaming Engine ---
for s = 1:numel(sample_indices)
    cur_idx = sample_indices(s);
    t_now   = t(cur_idx + win_dm_pts - 1);
    
    % Adaptive threshold relaxation
    if dm_thresh_current > dm_thresh_base
        dm_thresh_current = max(dm_thresh_base, dm_thresh_current - dm_decay_rate);
    end
    
    % Buffer reset across excluded boundaries
    if s > 1 && (cur_idx - sample_indices(s - 1)) > step_pts
        current_state       = STATE_REST;
        consec_hits         = 0;
        cooldown_samples    = 0;
        quiet_hops          = 0;
        dm_thresh_current   = dm_thresh_base;
        cluster_active      = false;
        candidates          = struct('t_cand', {}, 'cur_idx', {}, 'theta_cand', {}, 'evaluated', {}, 'phi_dm', {});
        last_cand_idx       = -inf;
        mu_buf              = repmat(mu_rest, history_len, 1);
        cov_buf             = repmat(cov_rest, [1, 1, history_len]);
        buf_ptr             = 1;
    end
    
    w_e = emg_edc(cur_idx : cur_idx + win_dm_pts - 1);
    w_f = emg_fds(cur_idx : cur_idx + win_dm_pts - 1);
    
    phi_dm = extract_features_dm(w_e, w_f);
    diff_u = phi_dm - mu_rest;
    dm_val = sqrt(diff_u * inv_cov_rest * diff_u');
    
    dm_trace(s, :) = [t_now, dm_val, dm_thresh_current];
    
    % ----------------------------------------------------------------------
    % STATE: REST (Engine 1 Monitoring & Candidate Cluster Generation)
    % ----------------------------------------------------------------------
    if current_state == STATE_REST
        
        % 1. Baseline Adaptation (Runs strictly while NO cluster is active)
        if dm_val <= dm_thresh_current
            if ~cluster_active
                mu_rest      = (1 - alpha_floor) * mu_rest + alpha_floor * phi_dm;
                dev          = phi_dm - mu_rest;
                cov_rest     = (1 - alpha_floor) * cov_rest + alpha_floor * (dev' * dev);
                inv_cov_rest = inv(cov_rest + 1e-4 * eye(3));
                
                buf_ptr = mod(buf_ptr, history_len) + 1;
                mu_buf(buf_ptr, :)     = mu_rest;
                cov_buf(:, :, buf_ptr) = cov_rest;
            end
            consec_hits = 0;
            
        elseif dm_val > dm_thresh_current
            consec_hits = consec_hits + 1;
            
            % Tripwire Confirmation (20 ms sustained)
            if consec_hits >= 2
                if ~cluster_active
                    % INITIATE NEW CLUSTER
                    cluster_active      = true;
                    cluster_t_start     = t_now;
                    cluster_restore_idx = mod(buf_ptr, history_len) + 1; % 200 ms prior state
                    candidates          = struct('t_cand', {}, 'cur_idx', {}, 'theta_cand', {}, 'evaluated', {}, 'phi_dm', {});
                    last_cand_idx       = -inf;
                end
                
                % Add candidate if minimum spacing is satisfied
                if (cur_idx - last_cand_idx) >= cand_spacing_pts
                    new_cand.t_cand     = t_now;
                    new_cand.cur_idx    = cur_idx;
                    new_cand.theta_cand = theta(cur_idx + win_dm_pts - 1);
                    new_cand.evaluated  = false;
                    new_cand.phi_dm     = phi_dm;
                    candidates(end+1)   = new_cand; %#ok<AGROW>
                    last_cand_idx       = cur_idx;
                end
            end
        end
        
        % 2. Candidate Evaluation within Active Cluster
        if cluster_active
            cluster_confirmed = false;
            
            for c = 1:numel(candidates)
                if ~candidates(c).evaluated
                    % Extract 40 ms causal window around candidate
                    idx_exam_start = max(1, candidates(c).cur_idx - round(0.010 * fs_emg));
                    idx_exam_end   = min(N_pts, candidates(c).cur_idx + win_dm_pts - 1);
                    w_e_exam       = emg_edc(idx_exam_start:idx_exam_end);
                    w_f_exam       = emg_fds(idx_exam_start:idx_exam_end);
                    
                    rms_e = sqrt(mean(w_e_exam.^2));
                    rms_f = sqrt(mean(w_f_exam.^2));
                    ndi   = (rms_e - rms_f) / (rms_e + rms_f + 1e-6);
                    
                    if ndi >= 0
                        pred_dir = 5; % Extension / Open (EDC Agonist)
                        w_ago    = w_e_exam;
                        wl_floor = wl_floor_edc;
                        wl_std   = wl_std_edc;
                        wl_ref   = wl_ref_edc;
                        v_ref    = v_ref_m5;
                    else
                        pred_dir = 6; % Flexion / Fist (FDS Agonist)
                        w_ago    = w_f_exam;
                        wl_floor = wl_floor_fds;
                        wl_std   = wl_std_fds;
                        wl_ref   = wl_ref_fds;
                        v_ref    = v_ref_m6;
                    end
                    
                    wl_ago  = sum(abs(diff(w_ago)));
                    half_pt = round(length(w_ago) / 2);
                    rms_h1  = sqrt(mean(w_ago(1:half_pt).^2));
                    rms_h2  = sqrt(mean(w_ago(half_pt+1:end).^2));
                    epr     = rms_h2 / (rms_h1 + 1e-6);
                    
                    pass_complexity = (wl_ago >= (wl_floor + 2.5 * wl_std));
                    pass_persist    = (epr >= 0.60);
                    pass_polar      = (abs(ndi) >= 0.12);
                    
                    if pass_complexity && pass_persist && pass_polar
                        % --------------------------------------------------
                        % BRANCH A: ONSET CONFIRMED (GREEN)
                        % --------------------------------------------------
                        wl_norm   = max(0.1, min(2.0, (wl_ago - wl_floor) / (wl_ref - wl_floor + 1e-6)));
                        est_speed = v_ref * wl_norm;
                        if pred_dir == 6, est_speed = -est_speed; end
                        
                        % Commit Onset at the EARLIEST Candidate (Captures full EMD lead!)
                        t_confirmed   = candidates(1).t_cand;
                        ang_confirmed = candidates(1).theta_cand;
                        confirmed_onsets = [confirmed_onsets; t_confirmed, ang_confirmed, est_speed, pred_dir]; %#ok<AGROW>
                        
                        % Roll back baseline to uncorrupted pre-cluster state (~200 ms prior)
                        mu_rest      = mu_buf(cluster_restore_idx, :);
                        cov_rest     = cov_buf(:, :, cluster_restore_idx);
                        inv_cov_rest = inv(cov_rest + 1e-4 * eye(3));
                        mu_buf       = repmat(mu_rest, history_len, 1);
                        cov_buf      = repmat(cov_rest, [1, 1, history_len]);
                        buf_ptr      = 1;
                        
                        % Engage flight refractory; NO baseline poisoning / threshold inflation
                        current_state    = STATE_REFRACTORY;
                        cooldown_samples = 0;
                        consec_hits      = 0;
                        quiet_hops       = 0;
                        
                        cluster_active    = false;
                        candidates        = struct('t_cand', {}, 'cur_idx', {}, 'theta_cand', {}, 'evaluated', {}, 'phi_dm', {});
                        cluster_confirmed = true;
                        break;
                    else
                        % Provisional failure: do NOT tune Engine 1 yet! Keep monitoring.
                        candidates(c).evaluated = true;
                    end
                end
            end
            
            % 3. Check Cluster Lifespan Expiration (150 ms Grace Window)
            if cluster_active && ~cluster_confirmed
                if (t_now - cluster_t_start) >= cluster_grace_s
                    % ------------------------------------------------------
                    % BRANCH B: CLUSTER EXPIRED WITHOUT CONFIRMATION (RED)
                    % ------------------------------------------------------
                    % Confirmed genuine artifact: mark the first candidate red
                    rejected_onsets = [rejected_onsets; candidates(1).t_cand, candidates(1).theta_cand, dm_val, pred_dir]; %#ok<AGROW>
                    
                    % NOW and ONLY NOW apply Engine 1 Auto-Tuning safely:
                    dm_thresh_current = 4.2; % Inflate threshold temporarily
                    dev_art = candidates(1).phi_dm - mu_rest;
                    cov_rest = cov_rest + 0.15 * (dev_art' * dev_art); % Desensitize disturbance axis
                    inv_cov_rest = inv(cov_rest + 1e-4 * eye(3));
                    
                    % Brief 60 ms micro-lockout to settle artifact tail
                    cooldown_samples = round(0.060 * fs_emg);
                    consec_hits      = 0;
                    
                    cluster_active = false;
                    candidates     = struct('t_cand', {}, 'cur_idx', {}, 'theta_cand', {}, 'evaluated', {}, 'phi_dm', {});
                end
            end
        end
        
    % ----------------------------------------------------------------------
    % STATE: REFRACTORY (Flight & Recovery Settle)
    % ----------------------------------------------------------------------
    elseif current_state == STATE_REFRACTORY
        if cooldown_samples > 0
            cooldown_samples = cooldown_samples - step_pts;
            quiet_hops = 0;
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

fprintf('[RESULT] Confirmed: %d | Rejected: %d | Prompts: %d\n', ...
        size(confirmed_onsets, 1), size(rejected_onsets, 1), height(k_table));

% ==============================================================================
% 1. TKEO COMPUTATION (Place after signal extraction)
% ==============================================================================
% Convert Volts to Millivolts (mV) and remove DC offset
edc_mv = (emg_edc - mean(emg_edc(idx_cal_start:idx_cal_end))) * 1000.0;
fds_mv = (emg_fds - mean(emg_fds(idx_cal_start:idx_cal_end))) * 1000.0;

% Discrete TKEO in mV^2
tkeo_raw_edc = [0; edc_mv(2:end-1).^2 - edc_mv(1:end-2) .* edc_mv(3:end); 0];
tkeo_raw_fds = [0; fds_mv(2:end-1).^2 - fds_mv(1:end-2) .* fds_mv(3:end); 0];

% Rectify numerical residuals and smooth (25 ms span)
tkeo_span = round(0.025 * fs_emg);
tkeo_edc  = movmean(max(0, tkeo_raw_edc), tkeo_span);
tkeo_fds  = movmean(max(0, tkeo_raw_fds), tkeo_span);

% ==============================================================================
% 2. 3-SUBPLOT DIAGNOSTIC VISUALIZATION (Drop-in replacement for Section 8)
% ==============================================================================
fig = figure('Color', 'w', 'Position', [40, 30, 1420, 930], ...
             'Name', sprintf('Dual-Engine Architecture with TKEO Tracking: Subject %d', subject_id));

% ------------------------------------------------------------------------------
% SUBPLOT 1: Kinematics & Onset Markings (Shrunk to top row)
% ------------------------------------------------------------------------------
ax1 = subplot(3, 1, 1); hold on; grid on; box on;
y_limits_th = [min(theta) - 10, max(theta) + 20];

% Stage 0 Rest Baseline Shadow
p_cal_rest = patch([t_cal_start, t_cal_end, t_cal_end, t_cal_start], ...
                   [y_limits_th(1), y_limits_th(1), y_limits_th(2), y_limits_th(2)], ...
                   [0.82, 0.82, 0.82], 'EdgeColor', 'none', 'FaceAlpha', 0.65, ...
                   'DisplayName', 'Stage 0: Rest Baseline Window');

% Stage 1 & Prompt Shading
for k = 1:numel(shade_blocks)
    sb = shade_blocks(k);
    patch([sb.tStart, sb.tEnd, sb.tEnd, sb.tStart], ...
          [y_limits_th(1), y_limits_th(1), y_limits_th(2), y_limits_th(2)], ...
          sb.color, 'EdgeColor', 'none', 'FaceAlpha', 0.45, 'HandleVisibility', 'off');
      
    if sb.is_calib
        txt_str = sprintf('CALIB: M%d (R%d)', sb.mov, sb.rep);
        txt_col = [0.25, 0.25, 0.25];
    elseif sb.mov == 5
        txt_str = sprintf('Open (R%d)', sb.rep);
        txt_col = [0.10, 0.40, 0.70];
    else
        txt_str = sprintf('Fist (R%d)', sb.rep);
        txt_col = [0.85, 0.20, 0.20];
    end
    text(sb.tStart + 0.12, y_limits_th(2) - 5.0, txt_str, 'Color', txt_col, ...
         'FontSize', 8.0, 'FontWeight', 'bold');
end

p_cal_trial = patch(nan, nan, [0.70, 0.70, 0.70], 'EdgeColor', 'none', 'FaceAlpha', 0.60, ...
                    'DisplayName', 'Stage 1: Calibration Trials (Rep 1)');

p_theta = plot(t, theta, 'Color', [0.15, 0.15, 0.15], 'LineWidth', 1.5, 'DisplayName', 'Finger PIP Angle (\theta)');

p_tkin = plot(nan, nan, 'bd', 'MarkerSize', 8.0, 'MarkerFaceColor', [0.4, 0.7, 1.0], ...
              'LineWidth', 1.4, 'DisplayName', 'Ground Truth t_{kin}');
for k = 1:height(k_table)
    t_k = k_table.(tkin_col)(k);
    if ~isnan(t_k) && t_k >= t(1) && t_k <= t(end)
        xline(t_k, 'b--', 'LineWidth', 1.1, 'HandleVisibility', 'off');
        plot(t_k, interp1(t, theta, t_k), 'bd', 'MarkerSize', 8.0, ...
             'MarkerFaceColor', [0.4, 0.7, 1.0], 'LineWidth', 1.4, 'HandleVisibility', 'off');
    end
end

p_green = plot(nan, nan, 'o', 'MarkerSize', 8.5, 'LineWidth', 1.8, ...
               'MarkerEdgeColor', [0.0, 0.50, 0.15], 'MarkerFaceColor', [0.2, 0.85, 0.3], ...
               'DisplayName', 'Confirmed Genuine Onset');
for d = 1:size(confirmed_onsets, 1)
    t_c   = confirmed_onsets(d, 1);
    ang_c = confirmed_onsets(d, 2);
    spd_c = confirmed_onsets(d, 3);
    
    plot(t_c, ang_c, 'o', 'MarkerSize', 8.5, 'LineWidth', 1.8, ...
         'MarkerEdgeColor', [0.0, 0.50, 0.15], 'MarkerFaceColor', [0.2, 0.85, 0.3], ...
         'HandleVisibility', 'off');
    xline(t_c, 'g-', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    text(t_c + 0.05, ang_c + 8.0, sprintf('%+.0f°/s', spd_c), ...
         'Color', [0.0, 0.50, 0.15], 'FontSize', 7.5, 'FontWeight', 'bold');
end

p_red = plot(nan, nan, 'x', 'MarkerSize', 9.5, 'LineWidth', 2.0, ...
             'MarkerEdgeColor', [0.85, 0.10, 0.10], 'DisplayName', 'Rejected False Burst (Artifact)');
for r = 1:size(rejected_onsets, 1)
    t_r   = rejected_onsets(r, 1);
    ang_r = rejected_onsets(r, 2);
    plot(t_r, ang_r, 'x', 'MarkerSize', 9.5, 'LineWidth', 2.0, ...
         'MarkerEdgeColor', [0.85, 0.10, 0.10], 'HandleVisibility', 'off');
    xline(t_r, 'r:', 'LineWidth', 1.1, 'HandleVisibility', 'off');
end

ylim(y_limits_th);
xlim([t(1), t(end)]);
ylabel('Angle (deg)', 'FontSize', 10, 'FontWeight', 'bold');
title(sprintf('Real-Time Dual-Engine Onset Validation & Trajectory Tracking: Subject %d', subject_id), ...
      'FontSize', 11, 'FontWeight', 'bold');
legend([p_theta, p_cal_rest, p_cal_trial, p_tkin, p_green, p_red], 'Location', 'northeast', ...
       'Orientation', 'horizontal', 'FontSize', 8.0);

% ------------------------------------------------------------------------------
% SUBPLOT 2: Teager-Kaiser Energy Operator (TKEO) of EDC & FDS
% ------------------------------------------------------------------------------
ax2 = subplot(3, 1, 2); hold on; grid on; box on;
max_tkeo = max([max(tkeo_edc), max(tkeo_fds)]) * 1.15;
if max_tkeo <= 0, max_tkeo = 1.0; end
y_limits_tkeo = [0, max_tkeo];

% Calibration & Cue Shadows
patch([t_cal_start, t_cal_end, t_cal_end, t_cal_start], ...
      [0, 0, y_limits_tkeo(2), y_limits_tkeo(2)], ...
      [0.82, 0.82, 0.82], 'EdgeColor', 'none', 'FaceAlpha', 0.65, 'HandleVisibility', 'off');

for k = 1:numel(shade_blocks)
    sb = shade_blocks(k);
    patch([sb.tStart, sb.tEnd, sb.tEnd, sb.tStart], ...
          [0, 0, y_limits_tkeo(2), y_limits_tkeo(2)], ...
          sb.color, 'EdgeColor', 'none', 'FaceAlpha', 0.40, 'HandleVisibility', 'off');
end

p_tkeo_edc = plot(t, tkeo_edc, 'Color', [0.10, 0.45, 0.85], 'LineWidth', 1.3, 'DisplayName', 'TKEO EDC (Extensor)');
p_tkeo_fds = plot(t, tkeo_fds, 'Color', [0.85, 0.25, 0.20], 'LineWidth', 1.3, 'DisplayName', 'TKEO FDS (Flexor)');

for d = 1:size(confirmed_onsets, 1)
    xline(confirmed_onsets(d, 1), 'g-', 'LineWidth', 1.1, 'HandleVisibility', 'off');
end
for r = 1:size(rejected_onsets, 1)
    xline(rejected_onsets(r, 1), 'r:', 'LineWidth', 1.1, 'HandleVisibility', 'off');
end

ylim(y_limits_tkeo);
xlim([t(1), t(end)]);
ylabel('TKEO Energy (V^2)', 'FontSize', 10, 'FontWeight', 'bold');
legend([p_tkeo_edc, p_tkeo_fds], 'Location', 'northeast', 'FontSize', 8.5);

% ------------------------------------------------------------------------------
% SUBPLOT 3: Engine 1 Mahalanobis Distance & Adaptive Tripwire Threshold
% ------------------------------------------------------------------------------
ax3 = subplot(3, 1, 3); hold on; grid on; box on;

patch([t_cal_start, t_cal_end, t_cal_end, t_cal_start], [0, 0, 10, 10], ...
      [0.82, 0.82, 0.82], 'EdgeColor', 'none', 'FaceAlpha', 0.65, 'HandleVisibility', 'off');

for k = 1:numel(shade_blocks)
    sb = shade_blocks(k);
    patch([sb.tStart, sb.tEnd, sb.tEnd, sb.tStart], [0, 0, 10, 10], ...
          sb.color, 'EdgeColor', 'none', 'FaceAlpha', 0.40, 'HandleVisibility', 'off');
end

p_dm    = plot(dm_trace(:, 1), dm_trace(:, 2), 'Color', [0.35, 0.35, 0.35], 'LineWidth', 1.0, ...
               'DisplayName', 'D_M(t)');
p_adapt = plot(dm_trace(:, 1), dm_trace(:, 3), 'Color', [0.85, 0.35, 0.10], 'LineWidth', 1.4, ...
               'DisplayName', 'Adaptive Tripwire Threshold');

for d = 1:size(confirmed_onsets, 1)
    xline(confirmed_onsets(d, 1), 'g-', 'LineWidth', 1.1, 'HandleVisibility', 'off');
end
for r = 1:size(rejected_onsets, 1)
    xline(rejected_onsets(r, 1), 'r:', 'LineWidth', 1.1, 'HandleVisibility', 'off');
end

ylim([0, 10]);
xlim([t(1), t(end)]);
xlabel('Timeline (s)', 'FontSize', 10, 'FontWeight', 'bold');
ylabel('D_M Distance', 'FontSize', 10, 'FontWeight', 'bold');
legend([p_dm, p_adapt], 'Location', 'northeast', 'FontSize', 8.5);

linkaxes([ax1, ax2, ax3], 'x');

% Helper feature extraction for Mahalanobis baseline
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