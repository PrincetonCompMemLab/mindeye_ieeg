import numpy as np 
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def align_channels_across_runs(all_data, all_labels, exclude_channels=['TRIG']):
    """
    Align neural data across runs with potentially different channels.
    
    Parameters
    ----------
    all_data : list of np.ndarray
        List of data arrays, each with shape (trials, channels, time)
    all_labels : list of pd.DataFrame
        List of label dataframes with columns 'channel_name' and 'channel_idx'
    exclude_channels : list of str
        Channel names to exclude (default: ['TRIG'])
    
    Returns
    -------
    data : np.ndarray
        Aligned data with shape (total_trials, n_unique_channels, time)
    channel_info : pd.DataFrame
        DataFrame with channel information, sorted by channel_idx
    """
    # Validate time dimension is the same for all runs
    time_dims = [arr.shape[2] for arr in all_data]
    if len(set(time_dims)) > 1:
        raise ValueError(f"Time dimension varies across runs: {time_dims}")
    n_time = time_dims[0]
    
    # Get all unique channel names across all runs (excluding specified channels)
    all_channel_names = []
    for label_df in all_labels:
        mask = ~label_df['channel_name'].isin(exclude_channels)
        non_excluded = label_df[mask]['channel_name'].values
        all_channel_names.extend(non_excluded)
    
    # Get sorted unique channel names
    unique_channels = sorted(set(all_channel_names))
    n_channels = len(unique_channels)
    
    # Map channel_name to position in aligned array
    channel_name_to_pos = {ch: i for i, ch in enumerate(unique_channels)}
    
    aligned_runs = []
    for run_idx, (data_array, label_df) in enumerate(tqdm(zip(all_data, all_labels), total=len(all_data), desc="Aligning runs")):
        n_trials, n_ch, _ = data_array.shape

        duplicates = label_df['channel_name'].duplicated()
        if duplicates.any():
            dup_names = label_df.loc[duplicates, 'channel_name'].tolist()
            raise ValueError(f"Duplicate channel names found in a single run: {dup_names}")        
        
        assert n_ch == len(label_df), \
            f"Data channels ({n_ch}) do not match label rows ({len(label_df)}) for run {run_idx}"
        
        # Filter out excluded channels
        mask = ~label_df['channel_name'].isin(exclude_channels)
        filtered_labels = label_df[mask].reset_index(drop=True)
        filtered_data = data_array[:, mask.values, :]
        
        # Create aligned array for this run (trials x unique channels x time)
        aligned_data = np.full((n_trials, n_channels, n_time), np.nan)
        
        # Map run-specific label into global label order using "channel_name_to_pos"
        for local_ch_idx, row in filtered_labels.iterrows():
            global_pos = channel_name_to_pos[row['channel_name']]
            aligned_data[:, global_pos, :] = filtered_data[:, local_ch_idx, :]
        
        aligned_runs.append(aligned_data)
    
    # Concatenate all runs along trial dimension
    data = np.vstack(aligned_runs)

    channels_with_missing = []
    for ch_idx, ch_name in enumerate(unique_channels):
        if np.isnan(data[:, ch_idx, :]).any():
            channels_with_missing.append(ch_name)
    
    if channels_with_missing:
        print(f"\nChannels with missing data (not present in all runs): {channels_with_missing}")
    
    # Create aligned channel info dataframe
    labels_concat = pd.concat(all_labels, ignore_index=True)
    channel_info = (labels_concat[~labels_concat['channel_name'].isin(exclude_channels)]
                    .drop_duplicates('channel_name'))
    
    # Sort to match the data array's channel order (sorted by channel_name)
    channel_info = (channel_info
                    .set_index('channel_name')
                    .loc[unique_channels]  # Reindex to match unique_channels order
                    .reset_index())

    return data, channel_info


# NCSNR helper functions
def compute_noise_ceiling(data_in):
    """
    Compute the noise ceiling signal-to-noise ratio (NCSNR) and percentage noise ceiling for each channel.
    
    Parameters:
        data_in (array): Data with shape (channels, conditions, trials), where each channel has more than 1 trial per condition.
    
    Returns:
        noise_ceiling (array): Noise ceiling as a percentage for each channel.
        ncsnr (array): Noise ceiling SNR for each channel.
        signal_var (array): Signal variance for each channel.
        noise_var (array): Noise variance for each channel.
    """    
    # Count number of valid (non-nan) trials per condition
    n_per_cond = np.sum(~np.isnan(data_in), axis=2)
    repeat_mask = n_per_cond >= 2
    
    # Calculate noise variance as mean variance across trials for each channel
    trial_var = np.nanvar(data_in, axis=2, ddof=1)
    noise_var = np.nanmean(np.where(repeat_mask, trial_var, np.nan), axis=1)
    
    # Calculate data variance as variance of the trial means across conditions for each channel
    data_var = np.nanvar(np.nanmean(data_in, axis=2), axis=1, ddof=1)

    # Effective inverse trial count: mean of (1/n_i) across conditions, per channel
    n_eff_inv = np.nanmean(np.where(n_per_cond > 0, 1.0 / n_per_cond, np.nan), axis=1)  # (channels,)
    
    # Calculate signal variance by subtracting noise variance from data variance
    signal_var = np.fmax(data_var - noise_var * n_eff_inv, 0)  # Ensure non-negative variance
    
    # Compute noise ceiling SNR
    ncsnr = np.sqrt(signal_var) / np.sqrt(noise_var)
    
    # Calculate noise ceiling as percentage based on SNR
    noise_ceiling = 100 * (ncsnr ** 2 / (ncsnr ** 2 + n_eff_inv))
    
    return noise_ceiling, ncsnr, signal_var, noise_var


def compute_ncsnr_all_timepoints(data_4d, times):
    """
    Compute NCSNR and NC percentage for every channel at every timepoint.
    
    Parameters:
        data_4d (array): Data with shape (conditions, channels, trials, timepoints)
        times (array): Time array in milliseconds
    
    Returns:
        ncsnr_results (array): NCSNR for each channel and timepoint (channels, timepoints)
        nc_results (array): Noise ceiling percentage for each channel and timepoint (channels, timepoints)
    """
    
    n_conditions, n_channels, n_trials, n_timepoints = data_4d.shape
    
    # Initialize result arrays
    ncsnr_results = np.zeros((n_channels, n_timepoints))
    nc_results = np.zeros((n_channels, n_timepoints))
    
    print(f"Computing NCSNR for {n_channels} channels across {n_timepoints} timepoints...")
    
    # Iterate through each timepoint
    for t in range(n_timepoints):
        if t % 100 == 0:  # Progress indicator
            print(f"Processing timepoint {t+1}/{n_timepoints} ({times[t]:.1f}ms)")
        
        # Extract data for this timepoint: (conditions, channels, trials)
        timepoint_data = data_4d[:, :, :, t]  # (conditions, channels, trials)
        
        # Transpose to (channels, conditions, trials) for compute_noise_ceiling
        reshaped_data = timepoint_data.transpose(1, 0, 2)  # (channels, conditions, trials)
        
        # Compute noise ceiling for this timepoint
        noise_ceiling, ncsnr, _, _ = compute_noise_ceiling(reshaped_data)
        
        # Store results
        ncsnr_results[:, t] = ncsnr
        nc_results[:, t] = noise_ceiling
    
    print("NCSNR computation complete!")
    return ncsnr_results, nc_results


def ncsnr_figs(ncsnr_all, nc_all, times):

    print(f"NCSNR results shape: {ncsnr_all.shape}") 
    print(f"NC results shape: {nc_all.shape}")      

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('NCSNR and Noise Ceiling Analysis Across All Timepoints', fontsize=16)

    # 1. NCSNR heatmap: channels × time
    im1 = axes[0,0].imshow(ncsnr_all, aspect='auto', cmap='viridis', 
                        extent=[times[0], times[-1], nc_all.shape[0], 1], interpolation='none')
    axes[0,0].set_title('NCSNR Heatmap (Channels × Time)')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Channel')
    axes[0,0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.colorbar(im1, ax=axes[0, 0], label='NCSNR')

    # 2. Noise Ceiling heatmap: channels × time
    im2 = axes[0, 1].imshow(nc_all, aspect='auto', cmap='plasma', 
                        extent=[times[0], times[-1], nc_all.shape[0], 1], interpolation='none')
    axes[0, 1].set_title('Noise Ceiling % Heatmap (Channels × Time)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Channel')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.colorbar(im2, ax=axes[0, 1], label='Noise Ceiling %')

    # 3. Mean NCSNR across channels vs time
    mean_ncsnr_time = np.nanmean(ncsnr_all, axis=0)
    axes[1,0].plot(times, mean_ncsnr_time, 'b-', linewidth=2)
    axes[1,0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[1,0].set_title('Mean NCSNR Across Channels vs Time')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('Mean NCSNR')
    axes[1,0].grid(True, alpha=0.3)

    # 4. Mean Noise Ceiling across channels vs time
    mean_nc_time = np.nanmean(nc_all, axis=0)
    axes[1, 1].plot(times, mean_nc_time, 'r-', linewidth=2)
    axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_title('Mean Noise Ceiling % Across Channels vs Time')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Mean Noise Ceiling %')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Mean NCSNR across all channels and time: {np.nanmean(ncsnr_all):.3f}")
    print(f"Max NCSNR: {np.nanmax(ncsnr_all):.3f}")
    print(f"Mean Noise Ceiling across all channels and time: {np.nanmean(nc_all):.1f}%")
    print(f"Max Noise Ceiling: {np.nanmax(nc_all):.1f}%")

    # Find peak times
    peak_ncsnr_time_idx = np.nanargmax(mean_ncsnr_time)
    peak_nc_time_idx = np.nanargmax(mean_nc_time)
    print(f"Peak NCSNR time: {times[peak_ncsnr_time_idx]:.1f}ms")
    print(f"Peak NC time: {times[peak_nc_time_idx]:.1f}ms")

def reshape_electrode_data_by_stimuli(electrode_data, events_df, id_column='nsd_id', stim_column='coco_id', fill_column='filename'):
    """
    Reshape electrode data from (channels, time, trials) to (channels, images, time, repeats)
    based on repeated stimuli in events dataframe.
    
    Parameters:
        electrode_data: array of shape (channels, time, trials) 
        events_df: DataFrame with stimulus information
        stim_column: column name containing stimulus identifiers
    
    Returns:
        reshaped_data: array of shape (channels, n_unique_images, time, max_repeats)
        stimulus_info: dict with stimulus mapping information
    """
    assert id_column not in events_df.columns, print('provide an ID name that isn\'t already a column.')
    
    print(f"Original electrode data shape: {electrode_data.shape}")
    print(f"Events dataframe shape: {events_df.shape}")
    
    events_df_copy = events_df.copy()
    
    # apply extraction function to provided column of the dataframe
    events_df_copy[id_column] = events_df_copy[stim_column]  # by default, use stim_column
    events_df_copy[id_column] = events_df_copy[id_column].fillna(events_df_copy[fill_column])  # for nan values in stim_column, use fill_column
    assert not events_df_copy[id_column].isna().any(), "ID column contains NaN values! Make sure all values are filled."
    
    # Get stimulus repetition information based on NSD ID only
    stim_counts = events_df_copy[id_column].value_counts()
    unique_stimuli = stim_counts.index.tolist()
    n_unique_images = len(unique_stimuli)
    assert n_unique_images == len(pd.unique(events_df['filename']))
    max_repeats = stim_counts.max()
    
    print(f"Number of unique stimuli: {n_unique_images}")
    print(f"Maximum repetitions per stimulus: {max_repeats}")
    print(f"Repetition distribution:")
    print(stim_counts.value_counts().sort_index())
    
    # Initialize reshaped array
    n_channels, n_timepoints, n_trials = electrode_data.shape
    reshaped_data = np.full((n_channels, n_unique_images, n_timepoints, max_repeats), 
                           np.nan, dtype=electrode_data.dtype)
    
    # Create mapping from stimulus NSD ID to trial indices
    stimulus_to_trials = {}
    for trial_idx, nsd_id in enumerate(events_df_copy[id_column]):
        if nsd_id not in stimulus_to_trials:
            stimulus_to_trials[nsd_id] = []
        stimulus_to_trials[nsd_id].append(trial_idx)
    
    # Fill the reshaped array
    for img_idx, stimulus in tqdm(enumerate(unique_stimuli), total=len(unique_stimuli)):
        trial_indices = stimulus_to_trials[stimulus]
        
        # Copy data for all repetitions of this stimulus
        for repeat_idx, trial_idx in enumerate(trial_indices):
            reshaped_data[:, img_idx, :, repeat_idx] = electrode_data[:, :, trial_idx]
            
    # Create stimulus info dictionary
    stimulus_info = {
        'unique_stimuli': unique_stimuli,
        'stimulus_counts': stim_counts.to_dict(),
        'stimulus_to_trials': stimulus_to_trials,
        'max_repeats': max_repeats,
        'n_unique_images': n_unique_images,
        'nsd_id_mapping': dict(zip(events_df_copy[id_column], events_df_copy[stim_column]))
    }
    
    print(f"Final reshaped data shape: {reshaped_data.shape}")
    print(f"Shape interpretation: ({n_channels} channels, {n_unique_images} images, {n_timepoints} timepoints, {max_repeats} max repeats)")
    
    return reshaped_data, stimulus_info


def extract_images_with_n_repeats(reshaped_data, stim_info, n_repeats=6):
    """
    Extract data for images that have exactly n_repeats repetitions.
    
    Parameters:
        reshaped_data: array of shape (channels, n_unique_images, time, max_repeats)
        stim_info: dict with stimulus mapping information from reshape_electrode_data_by_stimuli
        n_repeats: number of repeats to filter for (default=6)
    
    Returns:
        filtered_data: array of shape (channels, n_filtered_images, time, n_repeats)
        filtered_stimuli: list of stimulus IDs that have exactly n_repeats
        filtered_indices: list of original indices in reshaped_data
    """
    
    # Find stimuli with exactly n_repeats
    filtered_stimuli = []
    filtered_indices = []
    
    for i, stimulus in enumerate(stim_info['unique_stimuli']):
        if stim_info['stimulus_counts'][stimulus] == n_repeats:
            filtered_stimuli.append(stimulus)
            filtered_indices.append(i)
    
    print(f"Found {len(filtered_stimuli)} images with exactly {n_repeats} repeats")
    
    if len(filtered_stimuli) == 0:
        print(f"No images found with exactly {n_repeats} repeats")
        return None, [], []
    
    # Extract data for these images
    n_channels, _, n_timepoints, _ = reshaped_data.shape
    filtered_data = np.zeros((n_channels, len(filtered_stimuli), n_timepoints, n_repeats))
    
    for new_idx, orig_idx in enumerate(filtered_indices):
        filtered_data[:, new_idx, :, :] = reshaped_data[:, orig_idx, :, :n_repeats]
    
    print(f"Extracted data shape: {filtered_data.shape}")
    print(f"Shape interpretation: ({n_channels} channels, {len(filtered_stimuli)} images, {n_timepoints} timepoints, {n_repeats} repeats)")
    
    return filtered_data, filtered_stimuli, filtered_indices