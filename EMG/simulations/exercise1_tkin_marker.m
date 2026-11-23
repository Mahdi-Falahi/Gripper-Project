%% --- Detection of Kinematic Onset (t_kin) with 300 ms Near-Zero Rest Baseline ---
clear; clc; close all;

% --- Configuration ---
pruned_file = 'datasets/DB2/DB2_s12/S12_E1_A1_pruned.mat'; % Adjust path as needed
fs_emg      = 2000;                                     % Sampling frequency (Hz)

% Dynamic Rest Baseline Window Parameters
base_dur_ms        = 300;                               % Required continuous rest duration (ms)
base_dur_pts       = round((base_dur_ms / 1000) * fs_emg); % 600 samples
vel_near_zero      = 3.0;                               % Velocity near-zero threshold (deg/s)

% Onset Detection Parameters
min_sustain_ms_ang = 40;                                % Sustain filter for angle (ms)
min_sustain_pts_ang = round((min_sustain_ms_ang / 1000) * fs_emg);
min_sustain_ms_vel = 40;                                % Sustain filter for velocity (ms)
min_sustain_pts_vel = round((min_sustain_ms_vel / 1000) * fs_emg);

% --- 1. Load Data ---
load(pruned_file, 'pruned_table');
[file_dir, file_name, ~] = fileparts(pruned_file);

t     = (0:height(pruned_table)-1)' / fs_emg;
theta = pruned_table.joint_angle;
stim  = double(pruned_table.stimulus);
rep   = double(pruned_table.repetition);
subj  = pruned_table.subject(1);
ex    = pruned_table.exercise(1);

% Compute kinematic velocity (deg/s)
theta_vel = gradient(theta, 1 / fs_emg);

% --- 2. Identify Contiguous Movement Repetitions ---
active_idx = find(stim > 0 & rep > 0);
breaks     = find(diff(active_idx) > 1);
rep_starts = active_idx([1; breaks + 1]);
rep_ends   = active_idx([breaks; end]);
N_reps     = numel(rep_starts);

% Preallocate results struct and baseline storage
results = struct('repetition', {}, 'movement', {}, 't_prompt', {}, ...
                 't_base_start', {}, 't_base_end', {}, ...
                 'tkin_angle', {}, 'tkin_vel', {}, 'diff_vel_vs_angle_ms', {});
base_windows = zeros(N_reps, 2); % [t_start, t_end] for gray plotting

% --- 3. Extract 300 ms Near-Zero Baselines & Detect t_kin ---
for k = 1:N_reps
    idx_p = rep_starts(k); % Prompt arrival (t_v)
    m_val = stim(idx_p);
    r_val = rep(idx_p);
    
    % Define the inter-movement rest search boundary (before current prompt)
    if k == 1
        idx_rest_start = max(1, idx_p - round(3.5 * fs_emg));
    else
        idx_rest_start = max(1, rep_ends(k - 1) + 1);
    end
    idx_rest_end = max(idx_rest_start + base_dur_pts, idx_p - 1);
    
    % Search for 300 ms continuous window where |theta_vel| <= vel_near_zero
    rest_seg_vel = abs(theta_vel(idx_rest_start:idx_rest_end));
    max_in_win   = movmax(rest_seg_vel, [0, base_dur_pts - 1], 'Endpoints', 'discard');
    
    cand_idx = find(max_in_win <= vel_near_zero);
    if ~isempty(cand_idx)
        best_rel = cand_idx(end); % Closest to the visual cue
    else
        [~, best_rel] = min(max_in_win); % Fallback: lowest peak velocity window
    end
    
    idx_b_start = idx_rest_start + best_rel - 1;
    idx_b_end   = idx_b_start + base_dur_pts - 1;
    
    % Store baseline timing for plot shadowing
    base_windows(k, :) = [t(idx_b_start), t(idx_b_end)];
    
    % Calculate resting statistics over this 300 ms interval
    base_theta     = median(theta(idx_b_start:idx_b_end));
    base_theta_std = std(theta(idx_b_start:idx_b_end));
    base_vel_std   = std(theta_vel(idx_b_start:idx_b_end));
    
    % --- SEARCH STARTS IMMEDIATELY AFTER GRAY BASELINE ---
    idx_search_start = idx_b_end + 1;
    idx_search_end   = min(numel(theta), idx_p + round(2.5 * fs_emg));
    search_range     = idx_search_start:idx_search_end;
    
    % --- Method A: Angle Peak Backward Search ---
    post_cue_range = idx_p:idx_search_end;
    
    % 1. Find max and min angle post visual cue
    [max_ang, rel_max] = max(theta(post_cue_range));
    [min_ang, rel_min] = min(theta(post_cue_range));
    
    % Identify whether the primary movement excursion is positive or negative
    if abs(max_ang - base_theta) >= abs(min_ang - base_theta)
        idx_peak = idx_p + rel_max - 1;
        peak_dev = abs(max_ang - base_theta);
    else
        idx_peak = idx_p + rel_min - 1;
        peak_dev = abs(min_ang - base_theta);
    end

    ang_thresh = max([3 * base_theta_std, 0.03 * peak_dev, 0.8]);
    
    % 2. Step backwards from peak until reaching baseline threshold
    if peak_dev > ang_thresh
        back_range = idx_peak:-1:idx_search_start;
        idx_onset  = find(abs(theta(back_range) - base_theta) <= ang_thresh, 1, 'first');
        if ~isempty(idx_onset)
            t_ang = t(back_range(idx_onset));
        else
            t_ang = t(idx_search_start); % Reached search boundary
        end
    else
        t_ang = NaN; % Excursion did not break resting noise floor
    end
    
    % --- Method B: Clamped Velocity Threshold ---
    vel_thresh = min(max(2 * base_vel_std, 10.0), 15.0);
    vel_diff   = abs(theta_vel(search_range));
    vel_bool   = vel_diff > vel_thresh;
    vel_detect = movsum(vel_bool, [0, min_sustain_pts_vel - 1]) >= min_sustain_pts_vel;
    first_vel  = find(vel_detect, 1, 'first');
    if ~isempty(first_vel)
        t_vel = t(search_range(first_vel));
    else
        t_vel = NaN;
    end
    
    % Store results
    results(k).repetition           = r_val;
    results(k).movement             = m_val;
    results(k).t_prompt             = t(idx_p);
    results(k).t_base_start         = t(idx_b_start);
    results(k).t_base_end           = t(idx_b_end);
    results(k).tkin_angle           = t_ang;
    results(k).tkin_vel             = t_vel;
    results(k).diff_vel_vs_angle_ms = (t_vel - t_ang) * 1000;
end

% --- 4. Write Timing Log to CSV ---
log_table = struct2table(results);
log_file  = fullfile(file_dir, [file_name, '_tkin_log.csv']);
writetable(log_table, log_file);
fprintf('Kinematic Onset Log saved to: %s\n', log_file);
disp(log_table);

% --- 5. Generate Annotated Plot with Movement & Gray Rest Shading ---
fig = figure('Color', 'w', 'Position', [100, 100, 1250, 580], ...
             'Name', sprintf('S%d E%d Kinematic Onset with Dynamic 300ms Rest Baseline', subj, ex));

% --- Left Y-Axis: Joint Flexion Angle ---
yyaxis left
ax = gca;
ax.YColor = [0.15, 0.15, 0.15];
hold on;

color_palette = containers.Map({5, 6}, {[0.75, 0.88, 1.00], [1.00, 0.82, 0.82]});
yl_ang = [min(theta) - 10, max(theta) + 15];

% A. Draw active movement shading (Blue for M5, Red for M6)
for k = 1:N_reps
    m = stim(rep_starts(k));
    c = [0.9, 0.9, 0.9];
    if isKey(color_palette, m), c = color_palette(m); end
    p_mov = patch([t(rep_starts(k)), t(rep_ends(k)), t(rep_ends(k)), t(rep_starts(k))], ...
                  [yl_ang(1), yl_ang(1), yl_ang(2), yl_ang(2)], c, ...
                  'EdgeColor', 'none', 'FaceAlpha', 0.40, 'HandleVisibility', 'off');
    uistack(p_mov, 'bottom');
end

% B. Draw 300 ms Near-Zero Rest Baseline Gray Shading
for k = 1:N_reps
    tb1 = base_windows(k, 1);
    tb2 = base_windows(k, 2);
    p_gray = patch([tb1, tb2, tb2, tb1], [yl_ang(1), yl_ang(1), yl_ang(2), yl_ang(2)], ...
                   [0.65, 0.65, 0.65], 'EdgeColor', 'none', 'FaceAlpha', 0.55, ...
                   'HandleVisibility', 'off');
    uistack(p_gray, 'bottom');
end

% Plot Angle Trajectory
h_ang = plot(t, theta, 'Color', [0.15, 0.15, 0.15], 'LineWidth', 1.4, ...
             'DisplayName', 'Angle \theta(t)');

% Plot Detected Onset Markers
valid_ang = ~isnan([results.tkin_angle]);
valid_vel = ~isnan([results.tkin_vel]);

t_angs  = [results(valid_ang).tkin_angle];
th_angs = interp1(t, theta, t_angs);
t_vels  = [results(valid_vel).tkin_vel];
th_vels = interp1(t, theta, t_vels);

h_ang_pts = scatter(t_angs, th_angs, 48, [0.85, 0.33, 0.10], 'o', 'filled', ...
                    'DisplayName', 't_{kin} (Angle Threshold)');
h_vel_pts = scatter(t_vels, th_vels, 48, [0.00, 0.55, 0.20], '^', 'filled', ...
                    'DisplayName', 't_{kin} (Velocity Threshold)');

for i = 1:numel(t_angs)
    xline(t_angs(i), ':', 'Color', [0.85, 0.33, 0.10, 0.7], 'LineWidth', 1.0, 'HandleVisibility', 'off');
end
for i = 1:numel(t_vels)
    xline(t_vels(i), '--', 'Color', [0.00, 0.55, 0.20, 0.7], 'LineWidth', 1.0, 'HandleVisibility', 'off');
end

ylabel('Joint Flexion \theta (deg)', 'FontWeight', 'bold');
ylim(yl_ang);

% --- Right Y-Axis: Joint Velocity Overlay ---
yyaxis right
ax.YColor = [0.10, 0.45, 0.85];
hold on;

h_vel = plot(t, theta_vel, 'Color', [0.10, 0.45, 0.85, 0.60], 'LineWidth', 0.9, ...
             'LineStyle', '-', 'DisplayName', 'Velocity \theta''(t)');
yline(0, ':', 'Color', [0.10, 0.45, 0.85, 0.35], 'HandleVisibility', 'off');

ylabel('Joint Velocity \theta'' (deg/s)', 'FontWeight', 'bold');
max_vel_mag = max(abs(theta_vel));
ylim([-max_vel_mag * 1.35, max_vel_mag * 1.35]);

% --- Labels & Legend ---
xlabel('Time (seconds)', 'FontWeight', 'bold');
title(sprintf('Subject %d | Exercise %d: Dynamic 300ms Rest Baseline & Kinematic Onset', subj, ex), 'FontSize', 11);
grid on; box on;
xlim([t(1), t(end)]);

% Dummy patches for the legend
h_gray = patch(NaN, NaN, [0.65, 0.65, 0.65], 'EdgeColor', 'none', 'FaceAlpha', 0.55, ...
               'DisplayName', '300 ms Rest Baseline');
h_m5   = patch(NaN, NaN, color_palette(5), 'EdgeColor', 'none', 'FaceAlpha', 0.40, ...
               'DisplayName', 'M5 (Hand Open)');
h_m6   = patch(NaN, NaN, color_palette(6), 'EdgeColor', 'none', 'FaceAlpha', 0.40, ...
               'DisplayName', 'M6 (Fist Clench)');

legend([h_ang, h_vel, h_vel_pts, h_ang_pts, h_gray, h_m5, h_m6], ...
       'Location', 'northeast', 'Orientation', 'horizontal');

% --- 6. Save Plot ---
plot_file = fullfile(file_dir, [file_name, '_tkin_dynamic_baseline.png']);
drawnow;
exportgraphics(fig, plot_file, 'Resolution', 300);
fprintf('Annotated Plot saved to     : %s\n', plot_file);