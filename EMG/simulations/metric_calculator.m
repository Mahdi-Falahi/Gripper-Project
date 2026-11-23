%% --- Multi-Window sEMG Feature Extraction Around Rest & Kinematic Onsets ---
clear; clc; close all;

% --- Configuration ---
pruned_file = 'datasets/DB2/DB2_s12/S12_E1_A1_pruned.mat'; % Adjust path as needed
fs          = 2000;                                     % Sampling frequency (Hz)
win_dur_ms  = [30, 40, 50];                             % Analysis windows in milliseconds

[raw_dir, raw_name, ~] = fileparts(pruned_file);
log_file = fullfile(raw_dir, [raw_name, '_tkin_log.csv']);

% --- 1. Load Data & Onset Log ---
if ~isfile(log_file)
    error('Log file not found: %s. Run the t_kin detector first.', log_file);
end
tkin_log = readtable(log_file);
load(pruned_file, 'pruned_table');

% Extract antagonist EMG pair (EDC = Ch 9, FDS = Ch 10 in NinaPro DB2)
if ismember('emg', pruned_table.Properties.VariableNames)
    if size(pruned_table.emg, 2) >= 10
        emg_edc = pruned_table.emg(:, 9);
        emg_fds = pruned_table.emg(:, 10);
    else
        emg_edc = pruned_table.emg(:, 1);
        emg_fds = pruned_table.emg(:, 2);
    end
elseif ismember('emg_edc', pruned_table.Properties.VariableNames)
    emg_edc = pruned_table.emg_edc;
    emg_fds = pruned_table.emg_fds;
else
    error('Could not locate sEMG columns in pruned_table.');
end

N_total_samples = height(pruned_table);
N_reps          = height(tkin_log);

% Define target epochs
epoch_names = {'rest_gray', 'pre_tkin_vel', 'post_tkin_vel', 'pre_tkin_angle', 'post_tkin_angle'};

% --- 2. Iterate Over Repetitions, Windows, and Epochs ---
feature_rows = {};
row_counter  = 0;

for k = 1:N_reps
    rep_num   = tkin_log.repetition(k);
    mov_type  = tkin_log.movement(k);
    t_base    = tkin_log.t_base_start(k);
    t_vel     = tkin_log.tkin_vel(k);
    t_ang     = tkin_log.tkin_angle(k);

    for w_ms = win_dur_ms
        N_w = round((w_ms / 1000) * fs);

        for e = 1:numel(epoch_names)
            epoch = epoch_names{e};
            valid_window = true;

            % Determine sample index range [idx_start, idx_end]
            switch epoch
                case 'rest_gray'
                    idx_start = round(t_base * fs) + 1;
                    idx_end   = idx_start + N_w - 1;

                case 'pre_tkin_vel'
                    if isnan(t_vel), valid_window = false; end
                    idx_end   = round(t_vel * fs);
                    idx_start = idx_end - N_w + 1;

                case 'post_tkin_vel'
                    if isnan(t_vel), valid_window = false; end
                    idx_start = round(t_vel * fs) + 1;
                    idx_end   = idx_start + N_w - 1;

                case 'pre_tkin_angle'
                    if isnan(t_ang), valid_window = false; end
                    idx_end   = round(t_ang * fs);
                    idx_start = idx_end - N_w + 1;

                case 'post_tkin_angle'
                    if isnan(t_ang), valid_window = false; end
                    idx_start = round(t_ang * fs) + 1;
                    idx_end   = idx_start + N_w - 1;
            end

            % Boundary checks
            if ~valid_window || idx_start < 1 || idx_end > N_total_samples
                feat_vec = get_empty_feature_struct();
            else
                seg_edc  = emg_edc(idx_start:idx_end);
                seg_fds  = emg_fds(idx_start:idx_end);
                feat_vec = extract_emg_features(seg_edc, seg_fds, fs);
            end

            % Combine metadata with features
            row_counter = row_counter + 1;
            meta_struct = struct('repetition', rep_num, ...
                                 'movement', mov_type, ...
                                 'epoch', epoch, ...
                                 'window_ms', w_ms, ...
                                 'idx_start', idx_start, ...
                                 'idx_end', idx_end);
            
            % Flatten into unified struct
            all_fields = [fieldnames(meta_struct); fieldnames(feat_vec)];
            all_vals   = [struct2cell(meta_struct); struct2cell(feat_vec)];
            feature_rows{row_counter} = cell2struct(all_vals, all_fields, 1);
        end
    end
end

% --- 3. Save Structured Table to CSV & MAT ---
features_table = struct2table(cell2mat(feature_rows));

output_csv = fullfile(raw_dir, [raw_name, '_emg_features.csv']);
output_mat = fullfile(raw_dir, [raw_name, '_emg_features.mat']);

writetable(features_table, output_csv);
save(output_mat, 'features_table', '-v7.3');

fprintf('\nExtracted %d feature rows across %d repetitions.\n', height(features_table), N_reps);
fprintf('  Saved to CSV : %s\n', output_csv);
fprintf('  Saved to MAT : %s\n', output_mat);

% Display rest baseline vs. post-onset snapshot (EDC RMS on Movement 5)
disp('--- Sample Feature Check: Mean EDC RMS (uV/V) during Movement 5 ---');
sub_check = features_table(features_table.movement == 5 & features_table.window_ms == 40, ...
                           {'repetition', 'epoch', 'window_ms', 'edc_RMS', 'fds_RMS', 'antagonist_diff_ratio'});
disp(head(sub_check, 10));


%% ========================================================================
%  HELPER FUNCTIONS: Mathematical Feature Extraction
%  ========================================================================

function f = extract_emg_features(x_edc, x_fds, fs)
    % Extracts mono-channel and bi-channel antagonist metrics
    f_edc = compute_channel_metrics(x_edc, fs, 'edc');
    f_fds = compute_channel_metrics(x_fds, fs, 'fds');
    
    % --- Relational & Bichannel Dynamics ---
    eps_val = 1e-8;
    rms_e   = f_edc.edc_RMS;
    rms_f   = f_fds.fds_RMS;
    
    % Antagonist Differential Ratio: [-1 (pure FDS), +1 (pure EDC)]
    f_rel.antagonist_diff_ratio = (rms_e - rms_f) / (rms_e + rms_f + eps_val);
    
    % Co-contraction Index (CCI, %): Overlap energy between antagonist pair
    f_rel.co_contraction_index = (2 * min(rms_e, rms_f)) / (rms_e + rms_f + eps_val) * 100;
    
    % Merge channel and relational metrics
    f = cell2struct([struct2cell(f_edc); struct2cell(f_fds); struct2cell(f_rel)], ...
                    [fieldnames(f_edc); fieldnames(f_fds); fieldnames(f_rel)], 1);
end

function ch = compute_channel_metrics(x, fs, prefix)
    N = length(x);
    eps_val = 1e-8;
    
    % 1. Time-Domain Amplitude & Energy
    ch.([prefix, '_MAV']) = mean(abs(x));
    ch.([prefix, '_RMS']) = sqrt(mean(x.^2));
    ch.([prefix, '_WL'])  = sum(abs(diff(x)));
    ch.([prefix, '_VAR']) = var(x);
    ch.([prefix, '_SSI']) = sum(x.^2);
    ch.([prefix, '_LOG']) = exp(mean(log(abs(x) + eps_val)));
    
    % 2. Complexity & Morphology
    deadband = max(0.01 * std(x), 1e-5);
    
    % Zero Crossings (ZC)
    zc_count = 0;
    for i = 1:N-1
        if (x(i) * x(i+1) < 0) && abs(x(i) - x(i+1)) >= deadband
            zc_count = zc_count + 1;
        end
    end
    ch.([prefix, '_ZC']) = zc_count;
    
    % Slope Sign Changes (SSC)
    ssc_count = 0;
    for i = 2:N-1
        diff1 = x(i) - x(i-1);
        diff2 = x(i) - x(i+1);
        if (diff1 * diff2 > 0) && (abs(diff1) >= deadband || abs(diff2) >= deadband)
            ssc_count = ssc_count + 1;
        end
    end
    ch.([prefix, '_SSC']) = ssc_count;
    
    % Willison Amplitude (WAMP)
    ch.([prefix, '_WAMP']) = sum(abs(diff(x)) >= deadband);
    
    % Hjorth Parameters
    act = var(x);
    dx  = diff(x) * fs;
    mob = sqrt(var(dx) / (act + eps_val));
    ddx = diff(dx) * fs;
    com = (sqrt(var(ddx) / (var(dx) + eps_val))) / (mob + eps_val);
    
    ch.([prefix, '_Hjorth_Activity'])   = act;
    ch.([prefix, '_Hjorth_Mobility'])   = mob;
    ch.([prefix, '_Hjorth_Complexity']) = com;
    
    % 3. Teager-Kaiser Energy Operator (TKEO)
    % psi[n] = x[n]^2 - x[n-1]*x[n+1]
    if N >= 3
        psi = x(2:N-1).^2 - x(1:N-2) .* x(3:N);
        ch.([prefix, '_TKEO_Mean']) = mean(psi);
        ch.([prefix, '_TKEO_Max'])  = max(psi);
    else
        ch.([prefix, '_TKEO_Mean']) = NaN;
        ch.([prefix, '_TKEO_Max'])  = NaN;
    end
    
    % 4. Spectral Features (MNF & MDF via FFT)
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
    
    % 5. 4th-Order Autoregressive Coefficients (AR)
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
    % Dummy struct with NaNs for out-of-bounds / missing onset windows
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