% ==============================================================================
% DYNAMIC FEATURE & VELOCITY EXPORTER WITH TRAPEZOID HOLD FLAGGING
% Ingests: pruned_table (.mat) + kinematic_trapezoid_log (.csv)
% Features: 30 Multi-domain sEMG metrics (40 ms window | 10 ms hop | 100 Hz)
% Target  : Continuous joint velocity (theta_vel_deg_s)
% Mask    : is_hold_plateau (centered between P_start and R_start +/- 50 ms)
% ==============================================================================
clear; clc; close all;

% --- 1. Ingestion Paths ---
subject_id = 5;
sub_dir    = sprintf('datasets/DB2/DB2_s%d', subject_id);
mat_path   = fullfile(sub_dir, sprintf('S%d_E1_A1_pruned.mat', subject_id));
trap_path  = fullfile(sub_dir, sprintf('S%d_E1_A1_kinematic_trapezoid_log.csv', subject_id));

if ~exist(mat_path, 'file'),  mat_path  = sprintf('S%d_E1_A1_pruned.mat', subject_id); end
if ~exist(trap_path, 'file'), trap_path = sprintf('S%d_E1_A1_kinematic_trapezoid_log.csv', subject_id); end

fprintf('[INGESTION] Loading MAT : %s\n', mat_path);
data = load(mat_path, 'pruned_table');
pt   = data.pruned_table;

fprintf('[INGESTION] Loading Trapezoid Milestones CSV : %s\n', trap_path);
trap_table = readtable(trap_path);

% Identify milestone column names
if ismember('t2_plat_start_s', trap_table.Properties.VariableNames)
    col_pstart = 't2_plat_start_s';
    col_rstart = 't3_plat_end_s';
elseif ismember('t2', trap_table.Properties.VariableNames)
    col_pstart = 't2';
    col_rstart = 't3';
else
    error('Could not identify P_start (t2) and R_start (t3) in %s', trap_path);
end

% --- 2. Signal Extraction & Filtered Velocity ---
fs_emg    = 2000;
theta     = double(pt.joint_angle);  % PIP joint excursion (deg)
emg_edc   = double(pt.emg_edc);      % Ch 9 (EDC)
emg_fds   = double(pt.emg_fds);      % Ch 10 (FDS)
stim_data = double(pt.stimulus);     % Visual cue (0, 5, 6)
rep_data  = double(pt.repetition);   % Repetition counter (1-6)
N_pts     = height(pt);
t         = (0:N_pts-1)' / fs_emg;

% === HARDWARE FIX: SWAP CHANNELS FOR SUBJECT 5 ===
if subject_id == 5
    fprintf('[CORRECTION] Subject 5 detected: Inverting EDC (Ch 9) and FDS (Ch 10)...\n');
    temp_chan = emg_edc;
    emg_edc   = emg_fds;
    emg_fds   = temp_chan;
end
% ==================================================

% Kinematic velocity with 5 Hz zero-phase lowpass filter (removes glove jitter)
raw_vel    = [0; diff(theta) * fs_emg];
[b_v, a_v] = butter(2, 5 / (fs_emg / 2), 'low');
theta_vel  = filtfilt(b_v, a_v, raw_vel);

% --- 3. Extract Visual Cue Blocks for Timeline Reference ---
target_movements = [5, 6];
shade_blocks = struct('mov', {}, 'rep', {}, 'tStart', {}, 'tEnd', {});
color_palette = containers.Map({5, 6}, {[0.75, 0.88, 1.00], [1.00, 0.82, 0.82]});

for m = target_movements
    idx = find(stim_data == m);
    if isempty(idx), continue; end
    
    breaks = find(diff(idx) > 1);
    s_idx  = idx([1; breaks + 1]);
    e_idx  = idx([breaks; end]);
    
    for k = 1:numel(s_idx)
        r_k = mode(rep_data(s_idx(k):e_idx(k)));
        shade_blocks(end+1) = struct(...
            'mov', m, 'rep', r_k, ...
            'tStart', t(s_idx(k)), 'tEnd', t(e_idx(k)));
    end
end
[~, sort_order] = sort([shade_blocks.tStart]);
shade_blocks = shade_blocks(sort_order);

% --- 4. Sliding Window Multi-Metric Feature Extraction ---
win_pts       = round(0.040 * fs_emg); % 40 ms window (80 samples)
hop_pts       = round(0.010 * fs_emg); % 10 ms step (20 samples = 100 Hz cadence)
frame_indices = (win_pts : hop_pts : N_pts)';
mid_idx = frame_indices - round(win_pts / 2);
N_frames      = numel(frame_indices);
t_frames      = t(mid_idx);
theta_frames  = theta(mid_idx);
vel_frames    = theta_vel(mid_idx);
stim_frames   = stim_data(mid_idx);
rep_frames    = rep_data(mid_idx);

fprintf('[PROCESSING] Extracting 30 rich metrics across %d frames (100 Hz cadence)...\n', N_frames);

feats_edc = zeros(N_frames, 13);
feats_fds = zeros(N_frames, 13);

for f = 1:N_frames
    end_i   = frame_indices(f);
    start_i = end_i - win_pts + 1;
    
    w_e = emg_edc(start_i:end_i);
    w_f = emg_fds(start_i:end_i);
    
    feats_edc(f, :) = compute_single_channel_features(w_e);
    feats_fds(f, :) = compute_single_channel_features(w_f);
end

% Bilateral interaction features
mav_e = feats_edc(:, 1);
mav_f = feats_fds(:, 1);
rms_e = feats_edc(:, 2);
rms_f = feats_fds(:, 2);
wl_e  = feats_edc(:, 4);
wl_f  = feats_fds(:, 4);

ndi_mav = (mav_e - mav_f) ./ (mav_e + mav_f + 1e-6);
ndi_rms = (rms_e - rms_f) ./ (rms_e + rms_f + 1e-6);
ndi_wl  = (wl_e - wl_f)   ./ (wl_e + wl_f + 1e-6);
cci     = (2.0 * min(mav_e, mav_f)) ./ (mav_e + mav_f + 1e-6);

bilateral_feats = [ndi_mav, ndi_rms, ndi_wl, cci];

% --- 5. Precise Isometric Hold Flagging (Trapezoid P_start to R_start +/- 50 ms) ---
v_zero_band     = 5.0;  % deg/s velocity ceiling for stationary hold
buffer_sec      = 0.050; % 50 ms inspection margin
is_hold_plateau = false(N_frames, 1);

for row = 1:height(trap_table)
    t_pstart = trap_table.(col_pstart)(row);
    t_rstart = trap_table.(col_rstart)(row);
    
    if isnan(t_pstart) || isnan(t_rstart) || (t_rstart <= t_pstart)
        continue;
    end
    
    % Expand boundary by 50 ms buffer
    t_search_start = max(0, t_pstart - buffer_sec);
    t_search_end   = t_rstart + buffer_sec;
    
    % Locate frames falling within this candidate interval
    cand_frame_mask = (t_frames >= t_search_start) & (t_frames <= t_search_end);
    cand_indices    = find(cand_frame_mask);
    
    if isempty(cand_indices), continue; end
    
    % Flag frames where velocity is settled near zero
    for idx_f = cand_indices'
        if abs(vel_frames(idx_f)) <= v_zero_band
            is_hold_plateau(idx_f) = true;
        end
    end
end

% Morphological bridge: close sub-50 ms velocity blips during static hold
hold_min_gap_pts = round(0.500 / (hop_pts / fs_emg)); % 5 frames = 50 ms
is_hold_plateau  = ~bwareaopen(~is_hold_plateau, hold_min_gap_pts);

% --- 6. Export to CSV (Retaining Exact File Name) ---
feature_names_base = {'MAV', 'RMS', 'VAR', 'WL', 'ZC', 'SSC', 'WAMP', ...
                      'Hjorth_Mob', 'Hjorth_Comp', 'AR1', 'AR2', 'AR3', 'AR4'};
edc_col_names = cellfun(@(s) ['EDC_', s], feature_names_base, 'UniformOutput', false);
fds_col_names = cellfun(@(s) ['FDS_', s], feature_names_base, 'UniformOutput', false);
bi_col_names  = {'NDI_MAV', 'NDI_RMS', 'NDI_WL', 'CoContraction_Index'};

all_feature_cols = [edc_col_names, fds_col_names, bi_col_names];

data_matrix = [t_frames, theta_frames, vel_frames, stim_frames, rep_frames, ...
               double(is_hold_plateau), feats_edc, feats_fds, bilateral_feats];

var_names = [{'timestamp_s', 'theta_deg', 'theta_vel_deg_s', 'stimulus', 'repetition', ...
              'is_hold_plateau'}, all_feature_cols];

export_table = array2table(data_matrix, 'VariableNames', var_names);

csv_filename  = sprintf('S%d_E1_A1_dynamic_velocity_features.csv', subject_id);
csv_save_path = fullfile(sub_dir, csv_filename);
if ~exist(sub_dir, 'dir') && ~isempty(sub_dir), mkdir(sub_dir); end

writetable(export_table, csv_save_path);
fprintf('[SUCCESS] Exported dataset with %d rows and %d columns to:\n  --> %s\n', ...
        height(export_table), width(export_table), csv_save_path);

% --- 7. Diagnostic Timeline Visualization: Angle & Velocity with Shading ---
fig = figure('Color', 'w', 'Position', [50, 50, 1380, 750], ...
             'Name', sprintf('Kinematic Trapezoid Hold Segmentation (S%d)', subject_id));

hold_color = [1.00, 0.85, 0.35]; % Amber gold for Flagged Hold

% Extract hold intervals for patch plotting
hold_diff   = diff([0; is_hold_plateau; 0]);
hold_starts = find(hold_diff == 1);
hold_ends   = find(hold_diff == -1) - 1;

% --------------------------------------------------------------------------
% SUBPLOT 1: Finger PIP Excursion Angle & Hold Flag
% --------------------------------------------------------------------------
ax1 = subplot(2, 1, 1); hold on; grid on; box on;
y_limits_ang = [min(theta) - 10, max(theta) + 15];

% 1. Visual Cue Prompt Shading
for k = 1:numel(shade_blocks)
    sb = shade_blocks(k);
    patch([sb.tStart, sb.tEnd, sb.tEnd, sb.tStart], ...
          [y_limits_ang(1), y_limits_ang(1), y_limits_ang(2), y_limits_ang(2)], ...
          color_palette(sb.mov), 'EdgeColor', 'none', 'FaceAlpha', 0.35, 'HandleVisibility', 'off');
    
    if sb.mov == 5
        txt_str = sprintf('Cue: Open (R%d)', sb.rep);
        txt_col = [0.10, 0.40, 0.70];
    else
        txt_str = sprintf('Cue: Fist (R%d)', sb.rep);
        txt_col = [0.85, 0.25, 0.20];
    end
    text(sb.tStart + 0.15, y_limits_ang(2) - 3.5, txt_str, 'Color', txt_col, ...
         'FontSize', 8.5, 'FontWeight', 'bold');
end

% 2. Shaded Hold Overlay on Angle
p_hold_leg = patch(nan, nan, hold_color, 'EdgeColor', 'none', 'FaceAlpha', 0.65, ...
                   'DisplayName', 'Flagged Isometric Hold (P_{start} to R_{start})');
for h = 1:numel(hold_starts)
    t_h_s = t_frames(hold_starts(h));
    t_h_e = t_frames(hold_ends(h));
    p_h = patch([t_h_s, t_h_e, t_h_e, t_h_s], ...
                [y_limits_ang(1), y_limits_ang(1), y_limits_ang(2), y_limits_ang(2)], ...
                hold_color, 'EdgeColor', 'none', 'FaceAlpha', 0.65, 'HandleVisibility', 'off');
    uistack(p_h, 'bottom');
end

% 3. Trapezoid Milestone Markers
for row = 1:height(trap_table)
    t2 = trap_table.(col_pstart)(row);
    t3 = trap_table.(col_rstart)(row);
    if ~isnan(t2)
        plot(t2, interp1(t, theta, t2), 'd', 'MarkerSize', 7.0, ...
             'MarkerFaceColor', [0.2, 0.5, 0.9], 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
    end
    if ~isnan(t3)
        plot(t3, interp1(t, theta, t3), '^', 'MarkerSize', 7.0, ...
             'MarkerFaceColor', [0.9, 0.3, 0.2], 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
    end
end

p_ang  = plot(t, theta, 'k-', 'LineWidth', 1.6, 'DisplayName', 'Finger PIP Angle (\theta)');
h_p2   = plot(nan, nan, 'd', 'MarkerFaceColor', [0.2, 0.5, 0.9], 'MarkerEdgeColor', 'k', 'DisplayName', 't_2 (P_{start})');
h_p3   = plot(nan, nan, '^', 'MarkerFaceColor', [0.9, 0.3, 0.2], 'MarkerEdgeColor', 'k', 'DisplayName', 't_3 (R_{start})');
h_cue5 = patch(nan, nan, color_palette(5), 'EdgeColor', 'none', 'FaceAlpha', 0.35, 'DisplayName', 'Cue: Hand Open (M5)');
h_cue6 = patch(nan, nan, color_palette(6), 'EdgeColor', 'none', 'FaceAlpha', 0.35, 'DisplayName', 'Cue: Fist Clench (M6)');

ylabel('Angle (deg)', 'FontWeight', 'bold');
title(sprintf('Kinematic Trapezoid Hold Segmentation: Subject %d (40 ms Window | 100 Hz)', subject_id), ...
      'FontSize', 11, 'FontWeight', 'bold');
xlim([t(1), t(end)]); ylim(y_limits_ang);
legend([p_ang, h_p2, h_p3, p_hold_leg, h_cue5, h_cue6], ...
       'Location', 'southoutside', 'Orientation', 'horizontal', 'FontSize', 8.5);

% --------------------------------------------------------------------------
% SUBPLOT 2: Joint Angular Velocity & Hold Flag
% --------------------------------------------------------------------------
ax2 = subplot(2, 1, 2); hold on; grid on; box on;
y_limits_vel = [min(theta_vel) - 20, max(theta_vel) + 20];

% 1. Visual Cue Prompt Shading
for k = 1:numel(shade_blocks)
    sb = shade_blocks(k);
    patch([sb.tStart, sb.tEnd, sb.tEnd, sb.tStart], ...
          [y_limits_vel(1), y_limits_vel(1), y_limits_vel(2), y_limits_vel(2)], ...
          color_palette(sb.mov), 'EdgeColor', 'none', 'FaceAlpha', 0.25, 'HandleVisibility', 'off');
end

% 2. Shaded Hold Overlay on Velocity
for h = 1:numel(hold_starts)
    t_h_s = t_frames(hold_starts(h));
    t_h_e = t_frames(hold_ends(h));
    p_h = patch([t_h_s, t_h_e, t_h_e, t_h_s], ...
                [y_limits_vel(1), y_limits_vel(1), y_limits_vel(2), y_limits_vel(2)], ...
                hold_color, 'EdgeColor', 'none', 'FaceAlpha', 0.65, 'HandleVisibility', 'off');
    uistack(p_h, 'bottom');
end

p_vel = plot(t, theta_vel, 'Color', [0.10, 0.25, 0.70], 'LineWidth', 1.5, ...
             'DisplayName', 'Joint Velocity (\theta_{vel})');
yline(v_zero_band, 'r--', 'LineWidth', 1.1, ...
      'DisplayName', sprintf('\\pm%.1f deg/s (Near-Zero Band)', v_zero_band));
yline(-v_zero_band, 'r--', 'LineWidth', 1.1, 'HandleVisibility', 'off');
yline(0, 'k:', 'LineWidth', 1.0, 'HandleVisibility', 'off');

xlabel('Continuous Timeline (seconds)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Velocity (deg/s)', 'FontWeight', 'bold');
xlim([t(1), t(end)]); ylim(y_limits_vel);
legend([p_vel], 'Location', 'northeast', 'FontSize', 8.5);

linkaxes([ax1, ax2], 'x');

% ==============================================================================
% HELPER FUNCTIONS
% ==============================================================================
function feats = compute_single_channel_features(w)
    mav_val = mean(abs(w));
    rms_val = sqrt(mean(w.^2));
    var_val = var(w);
    
    dw     = diff(w);
    wl_val = sum(abs(dw));
    
    deadband = 1.5e-5;
    zc_count = 0;
    for i = 1:numel(w)-1
        if (w(i) * w(i+1) < 0) && abs(w(i) - w(i+1)) >= deadband
            zc_count = zc_count + 1;
        end
    end
    
    ssc_count = 0;
    for i = 2:numel(w)-1
        if ((w(i) - w(i-1)) * (w(i) - w(i+1)) > 0) && ...
           (abs(w(i) - w(i-1)) >= deadband || abs(w(i) - w(i+1)) >= deadband)
            ssc_count = ssc_count + 1;
        end
    end
    
    wamp_count = sum(abs(dw) >= 1.0e-5);
    
    var_w   = var(w) + 1e-12;
    var_dw  = var(dw) + 1e-12;
    ddw     = diff(dw);
    var_ddw = var(ddw) + 1e-12;
    
    mob_val  = sqrt(var_dw / var_w);
    comp_val = sqrt(var_ddw / var_dw) / (mob_val + 1e-6);
    
    w_centered = w - mean(w);
    r = xcorr(w_centered, 4, 'biased');
    R_mat = toeplitz(r(5:8));
    rhs   = r(6:9);
    if rcond(R_mat) > 1e-8
        ar_coeffs = -(R_mat \ rhs)';
    else
        ar_coeffs = zeros(1, 4);
    end
    
    feats = [mav_val, rms_val, var_val, wl_val, double(zc_count), double(ssc_count), ...
             double(wamp_count), mob_val, comp_val, ar_coeffs];
end