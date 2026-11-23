% ==============================================================================
% KINEMATIC TRAPEZOID MILESTONE EXTRACTOR (4-Point Segmentation)
% Ingests: pruned_table (.mat) + tkin_log (.csv)
% Extracts: t1 (Takeoff), t2 (Plateau Start), t_max, t3 (Release Start), t4 (Rest Settle)
% ==============================================================================
clear; clc; close all;

% --- 1. Ingestion Paths ---
subject_id = 6;
mat_path  = sprintf('datasets/DB2/DB2_s%d/S%d_E1_A1_pruned.mat', subject_id, subject_id);
tkin_path = sprintf('datasets/DB2/DB2_s%d/S%d_E1_A1_pruned_tkin_log.csv', subject_id, subject_id);

if ~exist(mat_path, 'file'),  mat_path  = sprintf('S%d_E1_A1_pruned.mat', subject_id); end
if ~exist(tkin_path, 'file'), tkin_path = sprintf('S%d_E1_A1_pruned_tkin_log.csv', subject_id); end

fprintf('[INGESTION] Loading MAT : %s\n', mat_path);
data = load(mat_path, 'pruned_table');
pt   = data.pruned_table;

fprintf('[INGESTION] Loading CSV : %s\n', tkin_path);
tkin_table = readtable(tkin_path);

% Filter target movements (5 = Open, 6 = Fist)
target_movements = [5, 6];
tkin_table = tkin_table(ismember(tkin_table.movement, target_movements), :);

if ismember('tkin_angle', tkin_table.Properties.VariableNames)
    tkin_col = 'tkin_angle';
elseif ismember('t_ang', tkin_table.Properties.VariableNames)
    tkin_col = 't_ang';
else
    error('Could not find "tkin_angle" or "t_ang" column in %s', tkin_path);
end

% --- 2. Signal Extraction ---
fs_emg    = 2000;
theta     = double(pt.joint_angle);
stim_data = double(pt.stimulus);
rep_data  = double(pt.repetition);
N_pts     = height(pt);
t         = (0:N_pts-1)' / fs_emg;

% 50 ms moving average to suppress minor digit tremor/sensor ripple
smooth_span  = round(0.050 * fs_emg);
theta_smooth = movmean(theta, smooth_span);

% --- 3. Extract Trapezoidal Kinematic Milestones per Trial ---
fprintf('[EXTRACTION] Segmenting trapezoidal phases across %d trials...\n', height(tkin_table));
records = cell(height(tkin_table), 16);
for k = 1:height(tkin_table)
    m  = tkin_table.movement(k);
    r  = tkin_table.repetition(k);
    t1 = tkin_table.(tkin_col)(k);
    
    if isnan(t1) || t1 < t(1) || t1 > t(end)
        fprintf('  [WARN] Skipping M%d R%d: invalid t_angle.\n', m, r);
        continue;
    end
    
    % Locate visual stimulus prompt bounds for this movement & repetition
    stim_indices = find(stim_data == m & rep_data == r);
    if isempty(stim_indices)
        t_cue_end = t1 + 4.0; % Safe fallback
    else
        t_cue_end = t(stim_indices(end));
    end
    
    % Search window: from t1 (takeoff) to 1.5 s post-stimulus release
    idx_start  = max(1, round(t1 * fs_emg));
    idx_search = min(N_pts, round((t_cue_end + 1.5) * fs_emg));
    
    t_win    = t(idx_start : idx_search);
    th_win   = theta_smooth(idx_start : idx_search);
    th_raw_w = theta(idx_start : idx_search);
    
    % 1. Takeoff Baseline Anchor (Point 1)
    theta_1 = th_raw_w(1);
    
    % --- A. INITIAL-TRAJECTORY DIRECTION LOCK ---
    % Inspect only the first 350 ms post-takeoff to lock voluntary direction
    init_dur_pts = min(numel(th_win), round(0.350 * fs_emg));
    init_delta   = th_win(1:init_dur_pts) - theta_1;
    [~, idx_init_max] = max(abs(init_delta));
    
    dir_sign = sign(init_delta(idx_init_max));
    if dir_sign == 0, dir_sign = 1; end % Fallback default
    
    % Direction-normalized excursion (voluntary contraction is strictly positive)
    y_exc = dir_sign * (th_win - theta_1);
    
    % --- B. CAUSAL ACTIVE-WINDOW PEAK BOUNDING ---
    % Peak search restricted to prompt window (+ 300 ms motor follow-through)
    idx_active_limit = min(numel(t_win), round((t_cue_end - t1 + 0.30) * fs_emg));
    if idx_active_limit < init_dur_pts
        idx_active_limit = numel(t_win);
    end
    
    [y_max, idx_peak] = max(y_exc(1:idx_active_limit));
    t_max     = t_win(idx_peak);
    theta_max = th_raw_w(idx_peak);
    
    % Guard against non-movements (< 3 deg excursion)
    if y_max < 3.0
        fprintf('  [WARN] M%d R%d excursion too small (%.2f deg). Flagging.\n', m, r, y_max);
        continue;
    end
    
    % Clamp negative excursions (rebound/hyperextension undershoots) to zero
    y_exc_clamped = max(0, y_exc);
    
    % --- C. TRAPEZOID PHASE SEGMENTATION (Terminal Descent Anchoring) ---
    th_plat_thresh = 0.85 * y_max; % 85% defines sustained plateau for ARRIVAL
    th_10_rest     = 0.10 * y_max; % 10% defines baseline return floor
    
    % 2. Plateau Arrival (Point 2): First entry into plateau
    rel_idx_2 = find(y_exc_clamped(1:idx_peak) >= th_plat_thresh, 1, 'first');
    if isempty(rel_idx_2), rel_idx_2 = idx_peak; end
    t2      = t_win(rel_idx_2);
    theta_2 = th_raw_w(rel_idx_2);
    
    % Terminal Release Descent: Last point holding >= 50% excursion
    % Bypasses transient dips, tremor, and hold-phase rebounds
    idx_term_mid = find(y_exc_clamped >= 0.50 * y_max, 1, 'last');
    if isempty(idx_term_mid) || idx_term_mid < rel_idx_2
        idx_term_mid = idx_peak;
    end
    
    % 3. Plateau Release (Point 3): Identify the "corner" of the terminal descent
    % By finding the maximum deviation from a secant line drawn between the peak 
    % and the 50% terminal drop, we robustly identify the start of the descent 
    % while ignoring post-peak bleeds, settling, or small pre-release bumps.
    if idx_term_mid > idx_peak
        y_seg = y_exc_clamped(idx_peak : idx_term_mid);
        
        % Create a straight linear drop from peak down to 50% mark
        y_secant = linspace(y_seg(1), y_seg(end), numel(y_seg));
        y_secant = reshape(y_secant, size(y_seg)); % Ensure dimension match
        
        % The corner (release start) is the point furthest ABOVE this linear drop
        [~, max_diff_idx] = max(y_seg - y_secant);
        rel_idx_3 = idx_peak + max_diff_idx - 1;
        
        % Sanity check: release cannot happen before plateau arrival
        if rel_idx_3 < rel_idx_2
            rel_idx_3 = rel_idx_2;
        end
    else
        rel_idx_3 = idx_term_mid;
    end
    t3      = t_win(rel_idx_3);
    theta_3 = th_raw_w(rel_idx_3);
    
    % 4. Baseline Return Settle (Point 4): Trace FORWARD from terminal descent
    rel_offset_4 = find(y_exc_clamped(idx_term_mid:end) <= th_10_rest, 1, 'first');
    if ~isempty(rel_offset_4)
        rel_idx_4 = idx_term_mid + rel_offset_4 - 1;
    else
        % Fallback: point of minimum local excursion post-descent
        [~, min_local_idx] = min(y_exc_clamped(idx_term_mid:end));
        rel_idx_4 = idx_term_mid + min_local_idx - 1;
    end
    t4      = t_win(rel_idx_4);
    theta_4 = th_raw_w(rel_idx_4);
    
    % Direction classification label
    if dir_sign > 0
        dir_label = 'EXT'; % Extension (Hand Open)
    else
        dir_label = 'FLX'; % Flexion (Fist Clench)
    end
    
    % Phase Durations
    dur_rise_ms = (t2 - t1) * 1000.0;
    dur_hold_ms = (t3 - t2) * 1000.0;
    dur_fall_ms = (t4 - t3) * 1000.0;
    
    records(k, :) = {m, r, dir_label, ...
                     t1, theta_1, ...
                     t2, theta_2, ...
                     t_max, theta_max, ...
                     t3, theta_3, ...
                     t4, theta_4, ...
                     dur_rise_ms, dur_hold_ms, dur_fall_ms};
end

% Assemble and Filter Output Table
valid_rows = ~cellfun(@isempty, records(:, 1));
kinematic_trapezoid_table = cell2table(records(valid_rows, :), 'VariableNames', { ...
    'movement', 'repetition', 'direction', ...
    't1_takeoff_s', 'theta1_deg', ...
    't2_plat_start_s', 'theta2_deg', ...
    't_max_s', 'theta_max_deg', ...
    't3_plat_end_s', 'theta3_deg', ...
    't4_rest_settle_s', 'theta4_deg', ...
    'rise_dur_ms', 'hold_dur_ms', 'fall_dur_ms'});

% --- 4. Export Table to CSV ---
out_csv = sprintf('datasets/DB2/DB2_s%d/S%d_E1_A1_kinematic_trapezoid_log.csv', subject_id, subject_id);
out_dir = fileparts(out_csv);
if ~isempty(out_dir) && ~exist(out_dir, 'dir'), mkdir(out_dir); end
writetable(kinematic_trapezoid_table, out_csv);
fprintf('[SAVED] Kinematic Trapezoid Milestones: %s\n', out_csv);

% Display Table
disp(kinematic_trapezoid_table(:, 1:11));

% --- 5. Diagnostic Visualization ---
fig = figure('Color', 'w', 'Position', [50, 100, 1350, 600], ...
             'Name', sprintf('Kinematic Trapezoid Segmentation (S%d)', subject_id));
hold on; grid on; box on;
y_limits = [min(theta) - 10, max(theta) + 15];

% Plot Joint Angle
plot(t, theta, 'Color', [0.25, 0.25, 0.25], 'LineWidth', 1.5, 'DisplayName', 'PIP Angle (\theta)');

% Color definitions
c_rise = [0.85, 0.92, 1.00]; % Light blue (Rise)
c_hold = [1.00, 0.95, 0.70]; % Amber (Plateau Hold)
c_fall = [1.00, 0.88, 0.88]; % Light red (Return)

for row = 1:height(kinematic_trapezoid_table)
    tb = kinematic_trapezoid_table(row, :);
    
    % Shading Phase 1: Rise [t1, t2]
    patch([tb.t1_takeoff_s, tb.t2_plat_start_s, tb.t2_plat_start_s, tb.t1_takeoff_s], ...
          [y_limits(1), y_limits(1), y_limits(2), y_limits(2)], ...
          c_rise, 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
      
    % Shading Phase 2: Plateau Hold [t2, t3]
    patch([tb.t2_plat_start_s, tb.t3_plat_end_s, tb.t3_plat_end_s, tb.t2_plat_start_s], ...
          [y_limits(1), y_limits(1), y_limits(2), y_limits(2)], ...
          c_hold, 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
      
    % Shading Phase 3: Return [t3, t4]
    patch([tb.t3_plat_end_s, tb.t4_rest_settle_s, tb.t4_rest_settle_s, tb.t3_plat_end_s], ...
          [y_limits(1), y_limits(1), y_limits(2), y_limits(2)], ...
          c_fall, 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Point 1: Takeoff (Green Circle)
    plot(tb.t1_takeoff_s, tb.theta1_deg, 'o', 'MarkerSize', 7.5, ...
         'MarkerFaceColor', [0.1, 0.8, 0.2], 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
     
    % Point 2: Plateau Arrival (Blue Diamond)
    plot(tb.t2_plat_start_s, tb.theta2_deg, 'd', 'MarkerSize', 7.5, ...
         'MarkerFaceColor', [0.2, 0.5, 0.9], 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
     
    % Point Max: Peak Deviation (Magenta Square)
    plot(tb.t_max_s, tb.theta_max_deg, 's', 'MarkerSize', 6.5, ...
         'MarkerFaceColor', [0.8, 0.2, 0.8], 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
     
    % Point 3: Release Start (Red Up-Triangle)
    plot(tb.t3_plat_end_s, tb.theta3_deg, '^', 'MarkerSize', 7.5, ...
         'MarkerFaceColor', [0.9, 0.3, 0.2], 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
     
    % Point 4: Return Settled (Black Down-Triangle)
    plot(tb.t4_rest_settle_s, tb.theta4_deg, 'v', 'MarkerSize', 7.5, ...
         'MarkerFaceColor', [0.4, 0.4, 0.4], 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
end

% Dummy plot handles for legend
h_p1   = plot(nan, nan, 'o', 'MarkerFaceColor', [0.1, 0.8, 0.2], 'MarkerEdgeColor', 'k', 'DisplayName', 't_1 Takeoff');
h_p2   = plot(nan, nan, 'd', 'MarkerFaceColor', [0.2, 0.5, 0.9], 'MarkerEdgeColor', 'k', 'DisplayName', 't_2 Plateau Start (90%)');
h_pmax = plot(nan, nan, 's', 'MarkerFaceColor', [0.8, 0.2, 0.8], 'MarkerEdgeColor', 'k', 'DisplayName', 't_{max} Peak Excursion');
h_p3   = plot(nan, nan, '^', 'MarkerFaceColor', [0.9, 0.3, 0.2], 'MarkerEdgeColor', 'k', 'DisplayName', 't_3 Release Start (90%)');
h_p4   = plot(nan, nan, 'v', 'MarkerFaceColor', [0.4, 0.4, 0.4], 'MarkerEdgeColor', 'k', 'DisplayName', 't_4 Baseline Settled (10%)');

h_sh_rise = patch(nan, nan, c_rise, 'EdgeColor', 'none', 'DisplayName', 'Rise Phase');
h_sh_hold = patch(nan, nan, c_hold, 'EdgeColor', 'none', 'DisplayName', 'Plateau Hold Phase');
h_sh_fall = patch(nan, nan, c_fall, 'EdgeColor', 'none', 'DisplayName', 'Return Phase');

xlim([t(1), t(end)]);
ylim(y_limits);
xlabel('Continuous Timeline (seconds)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Joint Angle \theta (degrees)', 'FontSize', 11, 'FontWeight', 'bold');
title(sprintf('Trapezoidal Kinematic Segmentation (S%d) | 4 Key Transition Points', subject_id), ...
      'FontSize', 12, 'FontWeight', 'bold');
legend([h_p1, h_p2, h_pmax, h_p3, h_p4, h_sh_rise, h_sh_hold, h_sh_fall], ...
       'Location', 'southoutside', 'Orientation', 'horizontal', 'FontSize', 9);