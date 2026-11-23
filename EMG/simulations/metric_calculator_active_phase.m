% ==============================================================================
% BATCH EMG-KINEMATIC FEATURE METRIC EXTRACTOR (NinaPro DB2)
% Ingests: S{id}_E1_A1_pruned.mat (emg_edc, emg_fds, joint_angle)
%          S{id}_E1_A1_kinematic_trapezoid_log.csv
% Features: Causal EMD shift, Onset & Offset EMD, RED, iEMG, WL, MNF,
%           Hjorth Mobility/Complexity, NDI, CCI, Decay Rate, Antagonist Brake
% Exports: datasets/DB2/DB2_s{id}/S{id}_E1_A1_feature_metrics.csv
% ==============================================================================
clear; clc;

fs_emg      = 2000;
rms_span    = round(0.050 * fs_emg); % 50 ms smoothing window
exclude_sub = [11];                  % Explicitly exclude Subject 11

% 4th-order Butterworth bandpass (20 - 450 Hz)
[b_filt, a_filt] = butter(4, [20, 450] / (fs_emg / 2), 'bandpass');

% Discover dataset base path
base_dir = fullfile('datasets', 'DB2');
if ~exist(base_dir, 'dir') && exist(fullfile('..', 'datasets', 'DB2'), 'dir')
    base_dir = fullfile('..', 'datasets', 'DB2');
end

sub_dirs = dir(fullfile(base_dir, 'DB2_s*'));
sub_dirs = sub_dirs([sub_dirs.isdir]);
fprintf('[PIPELINE] Found %d subject directories.\n', numel(sub_dirs));

for s = 1:numel(sub_dirs)
    folder_name = sub_dirs(s).name;
    sid = str2double(strrep(folder_name, 'DB2_s', ''));
    
    if ismember(sid, exclude_sub)
        fprintf('--> Skipping S%d (Cohort Excluded).\n', sid);
        continue;
    end
    
    s_path   = fullfile(sub_dirs(s).folder, folder_name);
    mat_file = fullfile(s_path, sprintf('S%d_E1_A1_pruned.mat', sid));
    csv_file = fullfile(s_path, sprintf('S%d_E1_A1_kinematic_trapezoid_log.csv', sid));
    
    if ~exist(mat_file, 'file') || ~exist(csv_file, 'file')
        fprintf('  [WARN] S%d missing MAT or CSV log in %s. Skipping.\n', sid, s_path);
        continue;
    end
    
    fprintf('--> Processing S%d...\n', sid);
    
    data    = load(mat_file, 'pruned_table');
    pt      = data.pruned_table;
    k_table = readtable(csv_file);
    
    % Native column extraction
    theta   = double(pt.joint_angle);
    raw_edc = double(pt.emg_edc);
    raw_fds = double(pt.emg_fds);
    N_pts   = length(theta);
    
    % Bandpass and compute continuous linear envelopes
    filt_edc = filtfilt(b_filt, a_filt, raw_edc);
    filt_fds = filtfilt(b_filt, a_filt, raw_fds);
    
    env_edc = sqrt(movmean(filt_edc.^2, rms_span));
    env_fds = sqrt(movmean(filt_fds.^2, rms_span));
    
    % Continuous kinematic derivatives
    theta_vel = gradient(theta, 1 / fs_emg);
    theta_acc = gradient(theta_vel, 1 / fs_emg);
    
    records = cell(height(k_table), 28);
    valid_count = 0;
    
    for k = 1:height(k_table)
        m         = k_table.movement(k);
        r         = k_table.repetition(k);
        dir_label = char(k_table.direction(k));
        
        t1 = k_table.t1_takeoff_s(k);
        t2 = k_table.t2_plat_start_s(k);
        tm = k_table.t_max_s(k);
        t3 = k_table.t3_plat_end_s(k);
        t4 = k_table.t4_rest_settle_s(k);
        
        th1 = k_table.theta1_deg(k);
        th2 = k_table.theta2_deg(k);
        thm = k_table.theta_max_deg(k);
        th3 = k_table.theta3_deg(k);
        th4 = k_table.theta4_deg(k);
        
        % Chronological validation
        if isnan(t1) || isnan(t4) || (t1 >= t4) || (t2 < t1) || (t4 < t3)
            continue;
        end
        
        i1 = max(1, round(t1 * fs_emg));
        i2 = max(1, round(t2 * fs_emg));
        im = max(1, round(tm * fs_emg));
        i3 = max(1, round(t3 * fs_emg));
        i4 = min(N_pts, round(t4 * fs_emg));
        
        if i4 <= i1 || i4 > N_pts, continue; end
        
        % Direction-Aware Muscle Allocation
        % M5 (Open / EXT): Agonist = EDC, Antagonist = FDS
        % M6 (Fist / FLX): Agonist = FDS, Antagonist = EDC
        if strcmp(dir_label, 'EXT') || m == 5
            env_ago = env_edc; env_ant = env_fds;
            raw_ago = filt_edc; raw_ant = filt_fds;
        else
            env_ago = env_fds; env_ant = env_edc;
            raw_ago = filt_fds; raw_ant = filt_edc;
        end
        
        % ----------------------------------------------------------------------
        % 1. ELECTROMECHANICAL DELAY (EMD): ONSET & OFFSET
        % ----------------------------------------------------------------------
        % A. Onset EMD (around Takeoff t1: [-200 ms, +50 ms])
        w_on1 = max(1, i1 - round(0.200 * fs_emg));
        w_on2 = min(N_pts, i1 + round(0.200 * fs_emg));
        
        v_on = abs(theta_vel(w_on1:w_on2));
        e_on = gradient(env_ago(w_on1:w_on2), 1 / fs_emg);
        
        v_on_norm = (v_on - mean(v_on)) / (std(v_on) + 1e-8);
        e_on_norm = (e_on - mean(e_on)) / (std(e_on) + 1e-8);
        
        max_lag_on = round(0.200 * fs_emg);
        [r_on, lags_on] = xcorr(v_on_norm, e_on_norm, max_lag_on);
        valid_on = lags_on >= -round(0.200 * fs_emg) & lags_on <= round(0.050 * fs_emg);
        [~, best_on_idx] = max(r_on(valid_on));
        sub_lags_on = lags_on(valid_on);
        emd_onset_ms = (sub_lags_on(best_on_idx) / fs_emg) * 1000.0;
        
        % B. Offset EMD (around Release t3: [-200 ms, +100 ms])
        w_off1 = max(1, i3 - round(0.250 * fs_emg));
        w_off2 = min(N_pts, i3 + round(0.250 * fs_emg));
        
        v_off = abs(theta_vel(w_off1:w_off2));
        e_off = -gradient(env_ago(w_off1:w_off2), 1 / fs_emg); % Envelope collapse
        
        v_off_norm = (v_off - mean(v_off)) / (std(v_off) + 1e-8);
        e_off_norm = (e_off - mean(e_off)) / (std(e_off) + 1e-8);
        
        max_lag_off = round(0.250 * fs_emg);
        [r_off, lags_off] = xcorr(v_off_norm, e_off_norm, max_lag_off);
        valid_off = lags_off >= -round(0.200 * fs_emg) & lags_off <= round(0.100 * fs_emg);
        if any(valid_off)
            [~, best_off_idx] = max(r_off(valid_off));
            sub_lags_off = lags_off(valid_off);
            emd_offset_ms = (sub_lags_off(best_off_idx) / fs_emg) * 1000.0;
        else
            emd_offset_ms = 0.0;
        end
        
        % ----------------------------------------------------------------------
        % 2. KINEMATIC TARGETS
        % ----------------------------------------------------------------------
        rise_dt = max(1e-3, t2 - t1);
        fall_dt = max(1e-3, t4 - t3);
        
        kin_rise_vel      = (th2 - th1) / rise_dt;
        kin_peak_acc      = max(abs(theta_acc(i1:i2)));
        kin_rise_dur_ms   = rise_dt * 1000.0;
        kin_excursion     = abs(thm - th1);
        kin_mean_hold_ang = mean(theta(i2:i3));
        kin_hold_wobble   = std(theta(i2:i3));
        kin_hold_dur_ms   = max(0, t3 - t2) * 1000.0;
        kin_fall_vel      = (th3 - th4) / fall_dt;
        kin_fall_dur_ms   = fall_dt * 1000.0;
        
        % ----------------------------------------------------------------------
        % 3. CAUSAL EMD-SHIFTED RISE PHASE FEATURES
        % ----------------------------------------------------------------------
        % Shift sEMG window backwards by the onset lead to evaluate true neural drive
        shift_pts = round((emd_onset_ms / 1000.0) * fs_emg);
        i1_emg    = max(1, i1 + shift_pts);
        i2_emg    = max(i1_emg + 10, i2 + shift_pts);
        
        rise_env   = env_ago(i1_emg:i2_emg);
        rise_deriv = gradient(rise_env, 1 / fs_emg);
        emg_red_max  = max(rise_deriv);
        emg_red_mean = mean(rise_deriv);
        
        pts_150 = min(length(rise_env), round(0.150 * fs_emg));
        emg_early_iemg = trapz(rise_env(1:pts_150)) / fs_emg;
        emg_rise_wl    = sum(abs(diff(raw_ago(i1_emg:i2_emg))));
        
        % ----------------------------------------------------------------------
        % 4. HOLD PHASE FEATURES (PLATEAU CLAMPING & COORDINATION)
        % ----------------------------------------------------------------------
        hold_ago_env = env_ago(i2:i3);
        hold_ant_env = env_ant(i2:i3);
        hold_ago_raw = raw_ago(i2:i3);
        
        emg_hold_mean_rms = mean(hold_ago_env);
        emg_hold_peak_rms = max(hold_ago_env);
        
        % Spectral Centroid (Mean Power Frequency)
        [pxx, f_ax] = periodogram(hold_ago_raw, [], length(hold_ago_raw), fs_emg);
        sum_pxx = sum(pxx);
        if sum_pxx > 0
            emg_hold_mnf = sum(f_ax .* pxx) / sum_pxx;
        else
            emg_hold_mnf = NaN;
        end
        
        % Hjorth Parameters
        d1 = diff(hold_ago_raw);
        d2 = diff(d1);
        v0 = var(hold_ago_raw); v1 = var(d1); v2 = var(d2);
        if v0 > 1e-12 && v1 > 1e-12
            emg_mobility   = sqrt(v1 / v0);
            emg_complexity = (sqrt(v2 / v1)) / emg_mobility;
        else
            emg_mobility = NaN; emg_complexity = NaN;
        end
        
        % Bounded Coordination Metrics
        denom   = hold_ago_env + hold_ant_env + 1e-8;
        emg_ndi = mean((hold_ago_env - hold_ant_env) ./ denom);
        emg_cci = mean((2.0 * min(hold_ago_env, hold_ant_env)) ./ denom);
        
        % ----------------------------------------------------------------------
        % 5. FALL PHASE FEATURES (DE-RECRUITMENT & ACTIVE BRAKING)
        % ----------------------------------------------------------------------
        fall_ago_env = env_ago(i3:i4);
        fall_ant_env = env_ant(i3:i4);
        fall_deriv   = gradient(fall_ago_env, 1 / fs_emg);
        
        emg_decay_rate       = min(fall_deriv); % Maximum negative rate of collapse
        emg_antagonist_brake = trapz(fall_ant_env) / fs_emg;
        
        valid_count = valid_count + 1;
        records(valid_count, :) = { ...
            sid, m, r, dir_label, emd_onset_ms, emd_offset_ms, ...
            kin_rise_vel, kin_peak_acc, kin_rise_dur_ms, ...
            kin_excursion, kin_mean_hold_ang, kin_hold_wobble, kin_hold_dur_ms, ...
            kin_fall_vel, kin_fall_dur_ms, ...
            emg_red_max, emg_red_mean, emg_early_iemg, emg_rise_wl, ...
            emg_hold_mean_rms, emg_hold_peak_rms, emg_hold_mnf, emg_mobility, ...
            emg_complexity, emg_ndi, emg_cci, emg_decay_rate, emg_antagonist_brake};
    end
    
    if valid_count > 0
        feature_table = cell2table(records(1:valid_count, :), 'VariableNames', { ...
            'subject_id', 'movement', 'repetition', 'direction', ...
            'emd_onset_ms', 'emd_offset_ms', ...
            'kin_rise_vel', 'kin_peak_acc', 'kin_rise_dur_ms', ...
            'kin_excursion', 'kin_mean_hold_ang', 'kin_hold_wobble', 'kin_hold_dur_ms', ...
            'kin_fall_vel', 'kin_fall_dur_ms', ...
            'emg_red_max', 'emg_red_mean', 'emg_early_iemg', 'emg_rise_wl', ...
            'emg_hold_mean_rms', 'emg_hold_peak_rms', 'emg_hold_mnf', 'emg_mobility', ...
            'emg_complexity', 'emg_ndi', 'emg_cci', 'emg_decay_rate', 'emg_antagonist_brake'});
        
        out_csv = fullfile(s_path, sprintf('S%d_E1_A1_feature_metrics.csv', sid));
        writetable(feature_table, out_csv);
        fprintf('  --> Saved %d validated trials to: %s\n', valid_count, out_csv);
    end
end
fprintf('[COMPLETE] Feature extraction finished across all available subjects.\n');