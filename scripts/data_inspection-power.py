#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
import pickle

import numpy as np
import scipy
import pandas as pd
from scipy.stats import pearsonr, zscore
import statsmodels

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import r2_score

import imageio.v2 as imageio
import torch
import torch.nn.functional as F
from torchvision import transforms

from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from utils import align_channels_across_runs, compute_ncsnr_all_timepoints, reshape_electrode_data_by_stimuli, extract_images_with_n_repeats, ncsnr_figs
# from PSN.psn import psn

from mne.stats import permutation_cluster_1samp_test
from concurrent.futures import ThreadPoolExecutor

np.random.seed(42)


# In[2]:


is_interactive = 'ipykernel_launcher' in sys.argv[0]

if is_interactive:
    print('running interactively')
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
else:
    print('running non-interactively')


# In[3]:


base_dir = '/home/ri99/ieeg'
ieeg_dir = f'{base_dir}/data/subjects/111'
csv_dir = f'{base_dir}/data/nsd_PTB'
img_dir = f'{base_dir}/data/nsd_ecog'

data_dir = f'{base_dir}/data/hfb'
output_dir = f'{base_dir}/outputs/hfb'
cache_dir = f'{base_dir}/.cache'

os.makedirs(cache_dir, exist_ok=True)
sub = 11
n_blocks = -1  # -1 = use all blocks
n_runs = 56

sampling_rate = 1024  # Hz
baseline_time = 0.2  # seconds of data per trial before stimulus onset
# subject 111 had 5600 images split into 4 distinct sessions over 2 days (day 1: runs 1-12 and 13-24; day 2: runs 25-36 and 37-56)

load_from_file = True


# In[4]:


curric = pd.read_csv(f"{csv_dir}/curriculum_patient{sub}.csv")
curric.head()


# In[5]:


if n_blocks == -1:  # use all available blocks
    block_list = curric['block'].unique().astype(str)
else:  # use n_blocks to specify instead
    block_list = [str(x) for x in range(1, n_blocks+1)]

print(block_list)
image_names = list(curric[curric['block'].isin(block_list)]['filename'])
print(len(image_names))


# In[6]:


# load bipolar-referenced voltage trace data (no frequency decomposition) at 1024Hz
run_list = list(range(1, n_runs+1))

all_data = []
all_labels = []
time_ref = None

for run in tqdm(run_list):
    if load_from_file is False:
        data = np.load(f"{ieeg_dir}/bandpower1024/npy/sub1{sub}_NSD_run{run}_bandpower70-170.npy")
    labels = pd.read_csv(f"{ieeg_dir}/bandpower1024/ch_labels/sub1{sub}_NSD_run{run}_bandpower_channels.csv")
    time = np.load(f"{ieeg_dir}/bandpower1024/time/sub1{sub}_NSD_run{run}_bandpower_time.npy")

    if load_from_file is False:
        assert time.shape[0] == data.shape[2], f"time mismatch in run {run}"
        assert labels.shape[0] == data.shape[1], f"label mismatch in run {run}"
    
    if time_ref is None:
        time_ref = time
    else:
        assert np.allclose(time, time_ref), f"time array differs in run {run}"

    if load_from_file is False:
        all_data.append(data)
    all_labels.append(labels)

if load_from_file is False:
    print(len(all_data))
    del data

# delete to prevent using the wrong ones later; we use channel_info to index channel labels from here on
del labels


# In[7]:


if load_from_file is False:
    # combine data across runs while dealing with some runs not having data from the same channels
    data, channel_info = align_channels_across_runs(all_data, all_labels)
    data = data.transpose(1,2,0).astype(np.float16)  # convert to channels x time x trials
    print(f"Final shape: {data.shape}")
    print(channel_info.head())
    channel_info.to_csv(f'{data_dir}/channel_info.csv', index=False)
else:
    channel_info = pd.read_csv(f'{data_dir}/channel_info.csv')

del all_labels


# # reshape to identify repeated stimuli

# In[8]:


if load_from_file is False:
    # First, reshape your electrode data using the existing function
    reshaped_electrode_data, stim_info = reshape_electrode_data_by_stimuli(
        data,  # Your electrode data (channels, time, trials)
        curric,         # Your events dataframe
        stim_column='coco_id'
    )
    np.save(f'{data_dir}/reshaped_electrode_data', reshaped_electrode_data)
    with open(f'{data_dir}/stim_info.pkl', 'wb') as f:
        pickle.dump(stim_info, f)

else:
    print('loaded reshaped_electrode_data, stim_info from file')
    reshaped_electrode_data = np.load(f'{data_dir}/reshaped_electrode_data.npy')
    with open(f'{data_dir}/stim_info.pkl', 'rb') as f:
        stim_info = pickle.load(f)


# # measure NCSNR and noise ceiling pct over time to identify electrodes with good signal 

# In[9]:


if load_from_file is False:
    data_slim = reshaped_electrode_data.transpose(1, 0, 3, 2)
    print(f"Transposed data shape: {data_slim.shape}")
    print(f"Shape interpretation: ({data_slim.shape[0]} conditions/images, {data_slim.shape[1]} channels, {data_slim.shape[2]} trials/repeats, {data_slim.shape[3]} timepoints)")
    np.save(f'{data_dir}/data_slim', data_slim, allow_pickle=True)
    
    # Compute NCSNR for all timepoints
    ncsnr_all, nc_all = compute_ncsnr_all_timepoints(data_slim, time)
    np.save(f'{data_dir}/ncsnr_all', ncsnr_all, allow_pickle=True)
    np.save(f'{data_dir}/nc_all', nc_all, allow_pickle=True)
else:
    print('loading processed data and ncsnr from file')
    data_slim = np.load(f'{data_dir}/data_slim.npy')
    ncsnr_all = np.load(f'{data_dir}/ncsnr_all.npy')
    nc_all = np.load(f'{data_dir}/nc_all.npy')
    
rankings = np.flip(np.argsort(ncsnr_all.mean(1)))
# Visualize the results
ncsnr_figs(ncsnr_all[rankings[:100]], nc_all[rankings[:100]], time, save_path=f'{output_dir}/ncsnr_bandpower')


# In[10]:


plt.figure(figsize=(8, 4))
plt.hist(ncsnr_all.flatten(), bins=50)
plt.title("Distribution of NCSNR values")
plt.xlabel("NCSNR")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


n_cols = 10
n_rows = int(np.ceil(n_ch / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 32), sharex=True, sharey=True)
fig.suptitle(f'NCSNR per Electrode')
fig.supxlabel('Time (s)')
fig.supylabel('Noise-Ceiling-Corrected SNR')
axes = axes.flatten()

for ch in range(n_ch):
    ax = axes[ch]
    ax.plot(time, ncsnr_all[ch], color='k', alpha=0.7)
    ax.set_title(channel_info.iloc[ch]["channel_name"])
    ax.tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

for ch in range(n_ch, len(axes)):
    axes[ch].axis('off')

ymin = np.nanmin(ncsnr_all)
ymax = np.nanmax(ncsnr_all)
ypad = (ymax - ymin) * 0.1
axes[0].set_ylim(-0.1, ymax + ypad)

plt.tight_layout(rect=[0.01, 0.01, 1, 0.98])
plt.savefig(f'{output_dir}/ncsnr_per_channel')
plt.show()


# In[25]:


perm_path = f"{output_dir}/ncsnr_perm"
os.makedirs(f"{perm_path}", exist_ok=True)
n_perms = 1024

if is_interactive:
    perm_start, perm_end = 0, 2
else:
    task_id    = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    perm_start = task_id * batch_size
    perm_end   = min(perm_start + batch_size, n_perms)

for perm_id in range(perm_start, perm_end):
    path = f"{perm_path}/perm_{perm_id:04d}.npy"
    if os.path.exists(path):
        print(f"Perm {perm_id}: loaded from disk")
        continue

    rng = np.random.default_rng(perm_id)
    data_shuffled = data_slim.copy()
    for img in range(data_slim.shape[0]):
        perm = rng.permutation(data_slim.shape[2])
        data_shuffled[img] = np.take(data_slim[img], perm, axis=1)

    ncsnr_perm, _ = compute_ncsnr_all_timepoints(data_shuffled, time)
    np.save(path, ncsnr_perm)
    print(f"Perm {perm_id}: computed and saved")


# In[33]:


null_mean = null_ncsnr.mean(axis=0)   # (n_channels, n_timepoints)
null_std  = null_ncsnr.std(axis=0)

# Observed z-scores
z_obs = (ncsnr_all - null_mean) / null_std  # (n_channels, n_timepoints)

threshold = 1.645  # p=0.05 one-sided

def find_clusters(z, threshold):
    """Returns list of (indices, mass) for suprathreshold clusters."""
    sig = z > threshold
    clusters = []
    in_cluster = False
    for t in range(len(z)):
        if sig[t] and not in_cluster:
            start = t
            in_cluster = True
        elif not sig[t] and in_cluster:
            idx = np.arange(start, t)
            clusters.append((idx, z[idx].sum()))
            in_cluster = False
    if in_cluster:
        idx = np.arange(start, len(z))
        clusters.append((idx, z[idx].sum()))
    return clusters

# Null distribution of max cluster mass, per channel
null_max_mass = np.zeros((len(null_ncsnr), n_ch))  # (n_perms, n_channels)
for i, null_map in enumerate(null_ncsnr):
    z_null = (null_map - null_mean) / null_std
    for ch in range(n_ch):
        clusters = find_clusters(z_null[ch], threshold)
        null_max_mass[i, ch] = max((m for _, m in clusters), default=0)

# Cluster p-values per channel
cluster_results = []
for ch in range(n_ch):
    clusters = find_clusters(z_obs[ch], threshold)
    ch_results = []
    for idx, mass in clusters:
        pval = (null_max_mass[:, ch] >= mass).mean()
        ch_results.append((idx, mass, pval))
    cluster_results.append(ch_results)


# In[34]:


n_cols = 10
n_rows = int(np.ceil(n_ch / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 32), sharex=True, sharey=True)
fig.suptitle(f'NCSNR per Electrode (Cluster Permutation Test, p<0.05)')
fig.supxlabel('Time (s)')
fig.supylabel('Noise-Ceiling-Corrected SNR')
axes = axes.flatten()

for ch in range(n_ch):
    ax = axes[ch]
    ax.plot(time, ncsnr_all[ch], color='k')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    for idx, mass, pval in cluster_results[ch]:
        if pval < 0.05:
            ax.axvspan(time[idx[0]], time[idx[-1]], alpha=0.3, color='red')

    ax.set_title(channel_info.iloc[ch]["channel_name"])
    ax.tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

for ch in range(n_ch, len(axes)):
    axes[ch].axis('off')

ymin = np.nanmin(ncsnr_all)
ymax = np.nanmax(ncsnr_all)
ypad = (ymax - ymin) * 0.1
axes[0].set_ylim(ymin - ypad, ymax + ypad)

plt.tight_layout(rect=[0.01, 0.01, 1, 0.98])
plt.savefig(f'{output_dir}/significant_ncsnr')
plt.show()


# In[11]:


# calculate average response of each channel
print(reshaped_electrode_data.shape)

# baseline_start_idx = int(np.floor(baseline_time+0.2 * sampling_rate))  # baseline duration (sec; -0.2-0.2sec post stim onset) * sampling rate (1/sec)
# baseline_mean = np.mean(reshaped_electrode_data[:,:,:baseline_start_idx,:], axis=2)
# avg_signal = reshaped_electrode_data - baseline_mean[:,:,np.newaxis,:]  # baseline subtract per trial

avg_signal = np.nanmean(reshaped_electrode_data, axis=(3,1))  # average over repeats (dim 3) and conditions (dim 1)

print(avg_signal.shape)


# In[12]:


# ncols = 10
# nrows = int(np.ceil(len(avg_signal) / ncols))
# fig, axes = plt.subplots(nrows, ncols, figsize=(20, 32), sharex=True, sharey=True)
# fig.suptitle(f'Average Response per Electrode (Power 70-170Hz @ {sampling_rate} Hz, Baseline-Corrected per Trial)')
# fig.supxlabel('Time (s)')
# fig.supylabel('Log Power, 70-170Hz')
# axes_flat = axes.flatten()

# print('Plotting...')
# for i in tqdm(range(len(avg_signal))):
#     ax = axes_flat[i]
#     ax.plot((np.arange(avg_signal.shape[1])/sampling_rate), avg_signal[i], linewidth=1)
#     ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
#     curr_label = channel_info.iloc[i]
#     # if int(curr_label['channel_idx']) in rankings[:20]:
#     #     ax.set_title(f"{curr_label['channel_name']}", fontsize=9, pad=2, color='red', fontweight='bold')
#     # elif int(curr_label['channel_idx']) in rankings[20:50]:
#     #     ax.set_title(f"{curr_label['channel_name']}", fontsize=9, pad=2, color='orange')
#     # else:
#     #     ax.set_title(f"{curr_label['channel_name']}", fontsize=9, pad=2)
#     ax.set_title(f"{curr_label['channel_name']}", fontsize=9, pad=2)
    
#     ax.tick_params(labelsize=7)
    
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

# plt.tight_layout(rect=[0.01, 0.01, 1, 0.98])
# plt.savefig(f'{output_dir}/avg_response_bandpower')
# plt.show()


# In[13]:


print(avg_signal.shape)
print(reshaped_electrode_data.shape)


# In[14]:


# X = np.nanmean(reshaped_electrode_data, axis=3).transpose(1, 2, 0)
# X = np.nan_to_num(X, nan=0.0)
# t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(X, n_jobs=-1)


# In[15]:


# sig_indices = np.where(cluster_pv < 0.05)[0]

# print(f"Found {len(sig_indices)} significant clusters!")

# significant_points = np.zeros(f_obs.shape, dtype=bool)

# for i in sig_indices:
#     significant_points[clusters[i]] = True

# channels_with_sig = np.any(significant_points, axis=0)
# print(f"Channels with significant clusters: {np.where(channels_with_sig)[0]}")

# plt.figure(figsize=(10, 6))
# plt.imshow(t_obs.T, aspect='auto', origin='lower', cmap='RdBu_r')
# plt.colorbar(label='Statistic Value')

# plt.contour(significant_points.T, colors='black', levels=[0.5], linestyles='--')

# plt.xlabel('Timepoints')
# plt.ylabel('Channels')
# plt.title('Significant Clusters (p < 0.05)')
# plt.show()


# In[16]:


def save_result(ch, t_obs, clusters, cluster_pv, H0, save_path):
    np.save(save_path,
            {"t_obs": t_obs, "clusters": clusters, "cluster_pv": cluster_pv, "H0": H0},
            allow_pickle=True)

print(reshaped_electrode_data.shape)
ch_timeseries = np.nanmean(reshaped_electrode_data, axis=3)
print(ch_timeseries.shape)


# In[17]:


perm_path = f"{output_dir}/cluster_perm"
os.makedirs(perm_path, exist_ok=True)

n_ch = ch_timeseries.shape[0]

if is_interactive:  # Jupyter notebook
    ch_start, ch_end = 0, n_ch
else:  # SLURM
    task_id    = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    ch_start   = task_id * batch_size
    ch_end     = min(ch_start + batch_size, n_ch)
    
results = [None] * n_ch
with ThreadPoolExecutor(max_workers=None) as executor:  # None = auto-compute number of available workers
    for ch in range(ch_start, ch_end):
        path = f"{perm_path}/ch_{ch:03d}.npy"
        if os.path.exists(path):
            d = np.load(path, allow_pickle=True).item()
            results[ch] = (d["t_obs"], d["clusters"], d["cluster_pv"], d["H0"])
            print(f"Channel {ch}: loaded from disk")
        else:
            t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
                ch_timeseries[ch],
                tail=0,           # two-sided
                n_permutations=1024,
                adjacency=None,   # linear timepoint adjacency
                seed=42,
                n_jobs=-1
            )
            results[ch] = (t_obs, clusters, cluster_pv, H0)
            executor.submit(save_result, ch, t_obs, clusters, cluster_pv, H0, path)
            print(f"Channel {ch}: computed and saving")


# In[19]:


n_cols = 10
n_rows = int(np.ceil(n_ch / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 32), sharex=True, sharey=True)
fig.suptitle(f'Average Response per Electrode (Power 70-170Hz @ {sampling_rate} Hz, Baseline-Corrected per Trial)')
fig.supxlabel('Time (s)')
fig.supylabel('Log Power, 70-170Hz (High Frequency Broadband)')
axes = axes.flatten()

for ch in range(n_ch):
    ax = axes[ch]
    if results[ch] is None:
        ax.axis('off')
        continue

    ts = ch_timeseries[ch]
    t_obs, clusters, cluster_pv, _ = results[ch]
    mean = np.nanmean(ts, axis=0)
    ci = 1.96 * np.nanstd(ts, axis=0) / np.sqrt(ts.shape[0])

    ax.plot(time, mean, color='k')
    ax.fill_between(time, mean - ci, mean + ci, alpha=0.3, color='k')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    for cluster_idx, pval in zip(clusters, cluster_pv):
        if pval < 0.05:
            start, end = cluster_idx[0][0], cluster_idx[0][-1]
            color = 'red' if t_obs[cluster_idx[0]].mean() > 0 else 'blue'
            ax.axvspan(time[start], time[end], alpha=0.3, color=color)
    ax.set_title(channel_info.iloc[ch]["channel_name"])
    ax.tick_params(labelsize=7)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# hide unused axes
for ch in range(n_ch, len(axes)):
    axes[ch].axis('off')

plt.tight_layout(rect=[0.01, 0.01, 1, 0.98])
plt.savefig(f'{output_dir}/significant_hfb')
plt.show()

