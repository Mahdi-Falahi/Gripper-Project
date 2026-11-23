%% --- Sliding Multi-Window sEMG Feature Extraction Relative to t_kin ---
clear; clc; close all;

% --- Configuration ---
subject_id  = 12;
pruned_file = sprintf('datasets/DB2/DB2_s%d/S%d_E1_A1_pruned.mat', subject_id, subject_id);
if ~isfile(pruned_file)
    pruned_file = sprintf('S%d_E1_A1_pruned.mat', subject_id);
end

fs          = 2000;                      % Sampling frequency (Hz)
win_dur_ms  = [30, 40, 50];              % Window lengths to benchmark
offsets_ms  = -200 : 10 : 200;           % Start offsets relative to t_kin (ms)

[raw_dir, raw_name, ~] = fileparts(pruned_file);
log_file = fullfile(raw_dir, sprintf('S%d_E1_A1_pruned_tkin_log.csv', subject_id));
if ~isfile(log_file)
    log_file = fullfile(raw_dir, sprintf('S%d_E1_A1_kinematic_trapezoid_log.csv', subject_id));
end

% --- 1. Ingestion ---
if ~isfile(log_file)
    error('Log file not found: %s', log_file);
end
tkin_log = readtable(log_file);
load(pruned_file, 'pruned_table');

% Filter target 1-DoF gestures (M5 = Open, M6 = Fist)
tkin_log = tkin_log(ismember(tkin_log.movement, [5, 6]), :);

% Dynamic takeoff column selection
if ismember('tkin_angle', tkin_log.Properties.VariableNames)
    tkin_col = 'tkin_angle';
elseif ismember('t1_takeoff_s', tkin_log.Properties.VariableNames)
    tkin_col = 't1_takeoff_s';
elseif ismember('t_ang', tkin_log.Properties.VariableNames)
    tkin_col = 't_ang';
else
    error('No kinematic takeoff column found in %s', log_file);
end

% Extract sEMG channels (EDC = Ch 9, FDS = Ch 10)
if ismember('emg_edc', pruned_table.Properties.VariableNames)
    emg_edc = double(pruned_table.emg_edc);
    emg_fds = double(pruned_table.emg_fds);
elseif ismember('emg', pruned_table.Properties.VariableNames)
    emg_edc = double(pruned_table.emg(:, 9));
    emg_fds = double(pruned_table.emg(:, 10));
else
    error('Could not locate sEMG columns in pruned_table.');
end

N_total_samples = height(pruned_table);
N_reps          = height(tkin_log);

% --- 2. Sliding Window Extraction ---
feature_rows = {};
row_counter  = 0;

for k = 1:N_reps
    rep_num  = tkin_log.repetition(k);
    mov_type = tkin_log.movement(k);
    t_base   = tkin_log.t_base_start(k);
    t_kin    = tkin_log.(tkin_col)(k);
    
    if isnan(t_kin), continue; end

    for w_ms = win_dur_ms
        N_w = round((w_ms / 1000) * fs);

        % A. Quiet Rest Baseline Epoch
        idx_rest_start = round(t_base * fs) + 1;
        idx_rest_end   = idx_rest_start + N_w - 1;
        
        if idx_rest_start >= 1 && idx_rest_end <= N_total_samples
            seg_e = emg_edc(idx_rest_start:idx_rest_end);
            seg_f = emg_fds(idx_rest_start:idx_rest_end);
            feat_rest = extract_emg_features(seg_e, seg_f, fs);
            
            row_counter = row_counter + 1;
            meta_rest = struct('repetition', rep_num, ...
                               'movement', mov_type, ...
                               'epoch_type', 'rest_gray', ...
                               'offset_ms', NaN, ...
                               'window_ms', w_ms, ...
                               'idx_start', idx_rest_start, ...
                               'idx_end', idx_rest_end);
            feature_rows{row_counter} = cell2struct([struct2cell(meta_rest); struct2cell(feat_rest)], ...
                                                   [fieldnames(meta_rest); fieldnames(feat_rest)], 1);
        end

        % B. Peri-Event Sliding Horizon [-200 ms to +200 ms]
        for tau = offsets_ms
            idx_start = round((t_kin + tau / 1000.0) * fs);
            idx_end   = idx_start + N_w - 1;
            
            if idx_start >= 1 && idx_end <= N_total_samples
                seg_e = emg_edc(idx_start:idx_end);
                seg_f = emg_fds(idx_start:idx_end);
                feat_vec = extract_emg_features(seg_e, seg_f, fs);
            else
                feat_vec = get_empty_feature_struct();
            end
            
            row_counter = row_counter + 1;
            meta_peri = struct('repetition', rep_num, ...
                               'movement', mov_type, ...
                               'epoch_type', 'peri_tkin', ...
                               'offset_ms', tau, ...
                               'window_ms', w_ms, ...
                               'idx_start', idx_start, ...
                               'idx_end', idx_end);
            feature_rows{row_counter} = cell2struct([struct2cell(meta_peri); struct2cell(feat_vec)], ...
                                                   [fieldnames(meta_peri); fieldnames(feat_vec)], 1);
        end
    end
end

% --- 3. Save Structured Tables ---
features_table = struct2table(cell2mat(feature_rows));
output_csv = fullfile(raw_dir, sprintf('S%d_E1_A1_pruned_emg_sliding_features.csv', subject_id));
output_mat = fullfile(raw_dir, sprintf('S%d_E1_A1_pruned_emg_sliding_features.mat', subject_id));

writetable(features_table, output_csv);
save(output_mat, 'features_table', '-v7.3');
fprintf('[SUCCESS] Extracted %d sliding feature rows across Subject %d.\n', height(features_table), subject_id);
fprintf('  CSV: %s\n', output_csv);

%% ========================================================================
%  MATHEMATICAL FEATURE EXTRACTION HELPERS
%  ========================================================================
function f = extract_emg_features(x_edc, x_fds, fs)
    f_edc = compute_channel_metrics(x_edc, fs, 'edc');
    f_fds = compute_channel_metrics(x_fds, fs, 'fds');
    
    eps_val = 1e-8;
    rms_e   = f_edc.edc_RMS;
    rms_f   = f_fds.fds_RMS;
    
    f_rel.antagonist_diff_ratio = (rms_e - rms_f) / (rms_e + rms_f + eps_val);
    f_rel.co_contraction_index  = (2 * min(rms_e, rms_f)) / (rms_e + rms_f + eps_val) * 100;
    
    f = cell2struct([struct2cell(f_edc); struct2cell(f_fds); struct2cell(f_rel)], ...
                    [fieldnames(f_edc); fieldnames(f_fds); fieldnames(f_rel)], 1);
end

function ch = compute_channel_metrics(x, fs, prefix)
    N = length(x);
    eps_val = 1e-8;
    
    ch.([prefix, '_MAV']) = mean(abs(x));
    ch.([prefix, '_RMS']) = sqrt(mean(x.^2));
    ch.([prefix, '_WL'])  = sum(abs(diff(x)));
    ch.([prefix, '_VAR']) = var(x);
    ch.([prefix, '_SSI']) = sum(x.^2);
    ch.([prefix, '_LOG']) = exp(mean(log(abs(x) + eps_val)));
    
    deadband = max(0.01 * std(x), 1e-5);
    zc_count = sum((x(1:N-1) .* x(2:N) < 0) & (abs(x(1:N-1) - x(2:N)) >= deadband));
    ch.([prefix, '_ZC']) = zc_count;
    
    diff1 = x(2:N-1) - x(1:N-2);
    diff2 = x(2:N-1) - x(3:N);
    ssc_count = sum((diff1 .* diff2 > 0) & (abs(diff1) >= deadband | abs(diff2) >= deadband));
    ch.([prefix, '_SSC']) = ssc_count;
    ch.([prefix, '_WAMP']) = sum(abs(diff(x)) >= deadband);
    
    act = var(x);
    dx  = diff(x) * fs;
    mob = sqrt(var(dx) / (act + eps_val));
    ddx = diff(dx) * fs;
    com = (sqrt(var(ddx) / (var(dx) + eps_val))) / (mob + eps_val);
    
    ch.([prefix, '_Hjorth_Activity'])   = act;
    ch.([prefix, '_Hjorth_Mobility'])   = mob;
    ch.([prefix, '_Hjorth_Complexity']) = com;
    
    if N >= 3
        psi = x(2:N-1).^2 - x(1:N-2) .* x(3:N);
        ch.([prefix, '_TKEO_Mean']) = mean(psi);
        ch.([prefix, '_TKEO_Max'])  = max(psi);
    else
        ch.([prefix, '_TKEO_Mean']) = NaN;
        ch.([prefix, '_TKEO_Max'])  = NaN;
    end
    
    X = fft(x);
    P = abs(X(1:floor(N/2)+1)).^2;
    freqs = (0:floor(N/2))' * (fs / N);
    P_tot = sum(P);
    if P_tot > eps_val
        ch.([prefix, '_MNF']) = sum(freqs .* P) / P_tot;
        cum_p = cumsum(P);
        mdf_idx = find(cum_p >= P_tot / 2, 1, 'first');
        ch.([prefix, '_MDF']) = freqs(mdf_idx);
    else
        ch.([prefix, '_MNF']) = NaN;
        ch.([prefix, '_MDF']) = NaN;
    end
    
    try
        ar_coeffs = arburg(x, 4);
        ch.([prefix, '_AR1']) = ar_coeffs(2);
        ch.([prefix, '_AR2']) = ar_coeffs(3);
        ch.([prefix, '_AR3']) = ar_coeffs(4);
        ch.([prefix, '_AR4']) = ar_coeffs(5);
    catch
        ch.([prefix, '_AR1']) = NaN;
        ch.([prefix, '_AR2']) = NaN;
        ch.([prefix, '_AR3']) = NaN;
        ch.([prefix, '_AR4']) = NaN;
    end
end

function empty_f = get_empty_feature_struct()
    prefixes = {'edc', 'fds'};
    fields = {'MAV', 'RMS', 'WL', 'VAR', 'SSI', 'LOG', 'ZC', 'SSC', 'WAMP', ...
              'Hjorth_Activity', 'Hjorth_Mobility', 'Hjorth_Complexity', ...
              'TKEO_Mean', 'TKEO_Max', 'MNF', 'MDF', 'AR1', 'AR2', 'AR3', 'AR4'};
    empty_f = struct();
    for p = 1:numel(prefixes)
        for f = 1:numel(fields)
            empty_f.([prefixes{p}, '_', fields{f}]) = NaN;
        end
    end
    empty_f.antagonist_diff_ratio = NaN;
    empty_f.co_contraction_index  = NaN;
end