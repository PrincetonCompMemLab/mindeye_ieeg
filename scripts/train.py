#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import os, sys
import pickle
import glob

import numpy as np
import scipy
import pandas as pd
from scipy.stats import pearsonr, zscore
from scipy.ndimage import convolve1d
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

from utils import align_channels_across_runs, compute_ncsnr_all_timepoints, reshape_electrode_data_by_stimuli, ncsnr_figs, do_retrieval
# from PSN.psn import psn

from mne.stats import permutation_cluster_1samp_test
from concurrent.futures import ThreadPoolExecutor

np.random.seed(42)


# In[3]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

assert device == "cuda"


# # load data from a subject

# In[4]:


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
smoothing = True

if smoothing:
    window_size = 25
    stride = 10
    smooth_suffix = f'_window{window_size}_stride{stride}'
else:
    smooth_suffix = ''


# In[5]:


curric = pd.read_csv(f"{csv_dir}/curriculum_patient{sub}.csv")
curric.head()


# In[6]:


if n_blocks == -1:  # use all available blocks
    block_list = curric['block'].unique().astype(str)
else:  # use n_blocks to specify instead
    block_list = [str(x) for x in range(1, n_blocks+1)]

print(block_list)
image_names = list(curric[curric['block'].isin(block_list)]['filename'])
print(len(image_names))


# In[7]:


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


# In[8]:


if load_from_file is False:
    # combine data across runs while dealing with some runs not having data from the same channels
    data, channel_info = align_channels_across_runs(all_data, all_labels)
    data = data.transpose(1,2,0).astype(np.float32)  # convert to channels x time x trials
    print(f"Final shape: {data.shape}")
    print(channel_info.head())
    channel_info.to_csv(f'{data_dir}/channel_info.csv', index=False)
else:
    channel_info = pd.read_csv(f'{data_dir}/channel_info.csv')

del all_labels


# # reshape to identify repeated stimuli

# In[9]:


if load_from_file is False:
    # # First, reshape your electrode data using the existing function
    # reshaped_electrode_data, stim_info = reshape_electrode_data_by_stimuli(
    #     data,  # Your electrode data (channels, time, trials)
    #     curric,         # Your events dataframe
    #     stim_column='coco_id'
    # )
    # np.save(f'{data_dir}/reshaped_electrode_data', reshaped_electrode_data)
    # with open(f'{data_dir}/stim_info.pkl', 'wb') as f:
    #     pickle.dump(stim_info, f)
    raise ValueError
else:
    print('loading reduced_data, stim_info from file')
    reduced_data = np.load(f'{output_dir}/top_ncsnr_electrodes_timepoints{smooth_suffix}.npy')
    with open(f'{data_dir}/stim_info.pkl', 'rb') as f:
        stim_info = pickle.load(f)
    print(reduced_data.shape)


# In[10]:


unique_image_indices = [idx[0] for idx in list(stim_info['stimulus_to_trials'].values())]
assert len(unique_image_indices) == stim_info['n_unique_images']
unique_image_names = pd.unique(curric['filename'])

# sanity checks
assert len(unique_image_indices) == stim_info['n_unique_images']
assert np.all(list(curric['filename'][curric['is_repeat']==0]) == unique_image_names)
assert np.all(list(curric['filename'][curric['repetition_number']==1]) == unique_image_names)


# In[11]:


for i in range(len(unique_image_names)):
    # assert the number of available (not NaN) repetitions in the data matches the expected number from the stimulus ordering
    assert int(np.sum(~np.isnan(reduced_data[0,i,0]))) == len(list(stim_info['stimulus_to_trials'].values())[i])


# In[12]:


cleaned_data = np.nanmean(reduced_data, axis=3)
print('averaged over repeats')
print(cleaned_data.shape)
del reduced_data


# In[13]:


if load_from_file == False:
    images = utils.load_images(unique_image_names, img_dir=img_dir, save_path=f'{output_dir}/images_torch')
else:
    images = torch.load(f'{output_dir}/images_torch', weights_only=True)

print(images.shape, type(images), images.dtype)
print(images.min(), images.max())


# In[14]:


if load_from_file is False:

    # prepare target CLIP embeddings
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=cache_dir
            ).to(device)
    feature_extractor = CLIPImageProcessor(size=224, crop_size=224)
    
    batch_size = 32
    clip_embeds = []
    
    for i in tqdm(range(0, len(images), batch_size), desc="Generating CLIP embeddings"):
        # generate embeddings using GPU in batches, then offload each batch to CPU to prevent GPU memory overflow
        batch = images[i:i + batch_size]
        
        inputs = feature_extractor(
            images=[im.byte().permute(1,2,0).numpy() for im in batch],
            return_tensors='pt', do_rescale=True
        )
        inputs['pixel_values'] = inputs['pixel_values'].to(device)
    
        with torch.inference_mode():
            outputs = image_encoder(**inputs)
            clip_embeds.append(outputs.image_embeds.cpu())
    
    clip_image = torch.cat(clip_embeds, dim=0)
    torch.save(clip_image, f'{output_dir}/images_clipemb')

else:
    clip_image = torch.load(f'{output_dir}/images_clipemb', weights_only=True)

clip_image = clip_image.numpy()


# In[15]:


cleaned_data = cleaned_data.transpose(1,0,2)

print(f"images shape: {list(images.shape)}, {images.dtype}")
print(f"clip emb shape: {list(clip_image.shape)}")
print(f"ieeg data shape: {list(cleaned_data.shape)}")
print(f"ieeg data interpretation: {cleaned_data.shape[0]} conditions, {cleaned_data.shape[1]} channels, {cleaned_data.shape[2]} timepoints")


# In[16]:


# train/test split
x_train, x_test, y_train, y_test, train_idx, test_idx = train_test_split(
    cleaned_data, clip_image, np.arange(cleaned_data.shape[0]), test_size=0.20, random_state=42
)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[17]:


# reshape to flatten channels and timepoints
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

print(x_train_flat.shape, x_test_flat.shape, y_train.shape, y_test.shape)


# In[18]:


do_pca = True

# no z-scoring at all
x_train_scaled = x_train_flat
x_test_scaled = x_test_flat

# baseline mean subtraction per channel
# take each electrode's baseline signal from -200ms to 0ms
# baseline_data = np.nanmean(reshaped_electrode_data[rankings[:n_reliable_channels], :, :int(np.floor(sampling_rate*0.2))], axis=-1)
# baseline_data = np.delete(baseline_data, 2, axis=0)
# baseline_channel_mean = baseline_data[:,train_idx].mean(axis=1).mean(axis=1)  # per-channel baseline mean for train trials

# x_train_scaled = x_train - baseline_channel_mean.reshape(1,-1,1)
# x_test_scaled = x_test - baseline_channel_mean.reshape(1,-1,1)

# x_train_scaled = x_train_scaled.reshape(x_train_scaled.shape[0], -1)
# x_test_scaled = x_test_scaled.reshape(x_test_scaled.shape[0], -1)

# TODO z-score globally per channel
pass

# z-score per channel per trial
# scaler = StandardScaler()
# x_train_scaled = scaler.fit_transform(x_train_flat)
# x_test_scaled = scaler.transform(x_test_flat)

print(x_train_scaled.shape, x_test_scaled.shape, y_train.shape, y_test.shape)


# In[19]:


if do_pca == True:
    from sklearn.decomposition import PCA
    
    # Flatten to (conditions, channels*time)
    pca = PCA().fit(x_train_scaled)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('PC'); plt.ylabel('Cumulative variance')
    
    x = 0.98
    plt.axhline(x, c='r', linestyle=':')
    n_pcs_x = np.searchsorted(np.cumsum(pca.explained_variance_ratio_), x) + 1
    print(f"{n_pcs_x} PCs explain {x} of the variance")
    
    plt.show()


# In[20]:


pca = PCA().fit(y_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('PC'); plt.ylabel('Cumulative variance')

x = 0.8
plt.axhline(x, c='r', linestyle=':')
n_pcs_y = np.searchsorted(np.cumsum(pca.explained_variance_ratio_), x) + 1
print(f"{n_pcs_y} PCs explain {x} of the variance")

plt.show()


# In[21]:


print(x_train_scaled.shape, x_test_scaled.shape, y_train.shape, y_test.shape)

if do_pca == True:
    pca = PCA(n_components=n_pcs_x)
    x_train_scaled = pca.fit_transform(x_train_scaled)
    x_test_scaled = pca.transform(x_test_scaled)

    clip_pca = PCA(n_components=n_pcs_y)
    y_train = clip_pca.fit_transform(y_train)
    y_test = clip_pca.transform(y_test)

print(x_train_scaled.shape, x_test_scaled.shape, y_train.shape, y_test.shape)


# In[22]:


# Set up ridge regression
ridge_model = RidgeCV(alphas=np.logspace(-4, 6, 50), alpha_per_target=True)
ridge_model.fit(x_train_scaled, y_train)
y_pred_train = clip_pca.inverse_transform(ridge_model.predict(x_train_scaled))
y_pred_test  = clip_pca.inverse_transform(ridge_model.predict(x_test_scaled))

coef_of_det = ridge_model.score(x_test_scaled, y_test)
print(f'R^2 (coefficient of determination): {coef_of_det:.2f}')


# In[23]:


# simple retrieval only works if there's no repeats
assert len(cleaned_data) == len(unique_image_names)


# In[31]:


top_k = 5

train_acc, train_sims = do_retrieval(x_train_scaled, clip_pca.inverse_transform(y_train), y_pred_train, top_k=top_k)
test_acc, test_sims = do_retrieval(x_test_scaled, clip_pca.inverse_transform(y_test), y_pred_test, top_k=top_k)
print(f"train fwd acc: {train_acc:.2%}, test fwd acc: {test_acc:.2%}")


# In[32]:


# permutation test: shuffle predictions and recompute accuracy 1000 times
null_accs = []
for _ in tqdm(range(1000)):
    shuffled = y_pred_test[np.random.permutation(len(y_pred_test))]
    acc, _ = do_retrieval(x_test_scaled, clip_pca.inverse_transform(y_test), shuffled, top_k=top_k)
    null_accs.append(acc)

p_val = np.mean(np.array(null_accs) >= test_acc)
print(f"p-value: {p_val:.4f}, null mean: {np.mean(null_accs):.2%}, null 95th pct: {np.percentile(null_accs, 95):.2%}")


# In[33]:


fig, ax = plt.subplots(figsize=(7, 4))

ax.hist(null_accs, bins=10, color='steelblue', edgecolor='white', label='Null distribution')
ax.axvline(test_acc, color='red', linewidth=2, linestyle='--', label=f'True accuracy ({test_acc:.2%})')
ax.axvline(np.percentile(null_accs, 95), color='gray', linewidth=1.5, linestyle=':', label='95th percentile null')

ax.set_xlabel('Top-10 Retrieval Accuracy')
ax.set_ylabel('Count')
ax.set_title(f'Permutation Test (p={p_val:.4f})')
ax.legend()
plt.tight_layout()
plt.show()


# In[26]:


# print('displaying every 20 train images and every 5 test images')

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(train_sims[::20, ::20], vmin=-1, vmax=1, cmap="coolwarm", square=True, ax=axs[0])
axs[0].set_title("Train Cosine Similarity Matrix")
axs[0].set_xlabel("Predicted CLIP embedding")
axs[0].set_ylabel("True CLIP embedding")

sns.heatmap(test_sims[::5, ::5], vmin=-1, vmax=1, cmap="coolwarm", square=True, ax=axs[1])
axs[1].set_title("Test Cosine Similarity Matrix")
axs[1].set_xlabel("Predicted CLIP embedding")
axs[1].set_ylabel("True CLIP embedding")

plt.tight_layout()
plt.show()


# In[ ]:




