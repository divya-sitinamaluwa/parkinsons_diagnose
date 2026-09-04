#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import mne
import pywt as pywt
import os
import tensorflow as tf 
import random
import shutil
import gc

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

from glob import glob
from autoreject import get_rejection_threshold
from random import shuffle
from tensorflow.keras import layers, models


# In[2]:


#EEG of healthy participants
hc_raw = glob("hc/*.bdf")

#EEG of PD patient's on their medication off
pd_raw = glob("pd/ses_off/*.bdf")

# CORRECTED: subject IDs are kept aligned with hc_raw / pd_raw at every step below so that
# splitting can be done per participant instead of per epoch. Without this, epochs from the
# same participant can end up in more than one of train/validation/test.
hc_subject_ids = [os.path.splitext(os.path.basename(f))[0] for f in hc_raw]
pd_subject_ids = [os.path.splitext(os.path.basename(f))[0] for f in pd_raw]


# In[3]:


def loaddata(data_file):
    #Retieve a sample EEG signal for one person
    return mne.io.read_raw_bdf(data_file, preload=True)  


# In[4]:


#Excluded the noisy channels and updated the channel list
ch_names = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz']
ch_types = ['eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg']
    
#Override the info of the filtered sample
new_info = mne.create_info(ch_names=ch_names, sfreq=512, ch_types=ch_types)

    
#Getting the digitized points of the head
montage_kind = "standard_1020"
montage =  mne.channels.make_standard_montage(montage_kind)


# In[5]:


#pre-process method
def preprocessdata(raw_file):
    #Filter the sample
    filtered_raw = raw_file.filter(l_freq = 0.8, h_freq = 30)
    filtered_raw.info = new_info
    filtered_raw.set_montage(montage, match_case=False)
    
    return filtered_raw


# In[6]:


#artifact removing method
def removeartifacts(filtered_raw_file):
    #Make fixed length events
    tstep = 1.0
    events_ica = mne.make_fixed_length_events(filtered_raw_file, duration=tstep)
    epochs_ica = mne.Epochs(filtered_raw_file, events_ica,
                        tmin=0.0, 
                        tmax=tstep,
                        baseline=None,
                        preload=True)
    
    reject = get_rejection_threshold(epochs_ica);
    
    random_state = 42  
    ica_n_components = .99    

    ica = mne.preprocessing.ICA(n_components=ica_n_components, random_state=random_state,)
    ica.fit(epochs_ica, reject=reject, tstep=tstep)
    
    ica_thresh = 1.96 
    eog_indices, eog_scores = ica.find_bads_eog(filtered_raw_file, ch_name=['Fp1', 'Fp2'], threshold=ica_thresh)
    ica.exclude = eog_indices
    
    return ica.apply(filtered_raw_file);


#  

# ------------------------------------------

# In[7]:


#Creating Training, Validation and Testic Dataset folders
subfolder_names = ['Training Data', 'Validation Data', 'Testing Data']
for subfolder_name in subfolder_names:
    os.makedirs(os.path.join('Data', subfolder_name, 'Healthy'))
    os.makedirs(os.path.join('Data', subfolder_name, 'PD'))


# In[8]:


#splittinf epochs of the pre-processed files
def splitting_epochs(preprocessed_file):
    return mne.make_fixed_length_epochs(preprocessed_file, duration=5, preload=False)  


# In[9]:


#reshape the epochs by flattening
# CORRECTED: this used to take the whole list of per-subject epoch objects and merge every
# subject's epochs into one flat list, which erases subject identity before the split happens.
# It now processes ONE subject's epochs at a time and returns that subject's array separately,
# so subject identity is preserved until after the split.
def reshape_epochs_single_subject(epoch_obj):

    epochs_array = []

    for e in epoch_obj.get_data():
        oneD = e.flatten()
        reshaped = np.reshape(oneD, (-1, 1024))

        for r in reshaped:
            epochs_array.append(r)

    return epochs_array

def reshape_epochs(epoch_files):
    """Kept for reference / backward compatibility - NOT used in the corrected pipeline
    below because it discards subject identity. Use reshape_epochs_single_subject per
    subject instead, then keep results grouped in a list-of-lists (one list per subject)."""
    epochs_array = []
    for epoch in epoch_files:
        epochs_array.extend(reshape_epochs_single_subject(epoch))
    return epochs_array


# In[10]:


#creating datasets
healthy_training_data = []
healthy_validation_data = []
healthy_testing_data = []

pd_training_data = []
pd_validation_data = []
pd_testing_data = []

scales = np.arange(1, 33)

# CORRECTED: split assignment now happens once per SUBJECT, not once per epoch. Every epoch
# belonging to a given subject is routed entirely into whichever split that subject was
# assigned to, so no participant's data can appear in more than one of train/val/test.
# A manifest recording which subject went where is written to disk (Data/split_manifest.csv)
# as a deliverable / sanity check.
random.seed(42)  # fixed seed so the split is reproducible and reportable

def subject_level_split(subject_ids, subject_epoch_lists, val_frac=0.1, test_frac=0.1):
    """
    subject_ids: list of subject identifiers (e.g. filenames), one per subject
    subject_epoch_lists: list of lists of epoch arrays, same order/length as subject_ids
    Returns: (train_epochs, val_epochs, test_epochs, train_ids, val_ids, test_ids)
    """
    n_subjects = len(subject_ids)
    order = list(range(n_subjects))
    random.shuffle(order)

    n_test = max(1, round(n_subjects * test_frac))
    n_val = max(1, round(n_subjects * val_frac))
    n_train = n_subjects - n_val - n_test
    if n_train < 1:
        raise ValueError(
            f"Not enough subjects ({n_subjects}) to form a non-empty train split with "
            f"val_frac={val_frac}, test_frac={test_frac}. Consider subject-level k-fold "
            f"instead of a single fixed split for such a small cohort."
        )

    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]

    train_epochs, val_epochs, test_epochs = [], [], []
    for i in train_idx:
        train_epochs.extend(subject_epoch_lists[i])
    for i in val_idx:
        val_epochs.extend(subject_epoch_lists[i])
    for i in test_idx:
        test_epochs.extend(subject_epoch_lists[i])

    train_ids = [subject_ids[i] for i in train_idx]
    val_ids = [subject_ids[i] for i in val_idx]
    test_ids = [subject_ids[i] for i in test_idx]

    return train_epochs, val_epochs, test_epochs, train_ids, val_ids, test_ids


def write_split_manifest(path, group_label, train_ids, val_ids, test_ids):
    """Appends split assignment rows to a manifest CSV so it's auditable/reportable."""
    file_exists = os.path.isfile(path)
    with open(path, 'a') as f:
        if not file_exists:
            f.write("group,subject_id,split\n")
        for sid in train_ids:
            f.write(f"{group_label},{sid},train\n")
        for sid in val_ids:
            f.write(f"{group_label},{sid},validation\n")
        for sid in test_ids:
            f.write(f"{group_label},{sid},test\n")


def create_datasets(subject_ids, subject_epoch_lists, epoch_type, cap_train=4000, cap_val=500, cap_test=500):
    """
    epoch_type: 'h' for healthy, anything else for PD.
    subject_ids / subject_epoch_lists: aligned per-subject data (see subject_level_split).
    cap_* : maximum epochs to keep per split, sampled AFTER the subject-level split so that
    class balance is preserved without ever mixing subjects across splits. If a split has
    fewer epochs than the cap, all available epochs are kept (and this is reported, since
    it affects how much data is actually available downstream).
    """
    global healthy_training_data, healthy_validation_data, healthy_testing_data
    global pd_training_data, pd_validation_data, pd_testing_data

    train_epochs, val_epochs, test_epochs, train_ids, val_ids, test_ids = subject_level_split(
        subject_ids, subject_epoch_lists
    )

    write_split_manifest('Data/split_manifest.csv', 'healthy' if epoch_type == 'h' else 'pd',
                          train_ids, val_ids, test_ids)

    def cap_sample(epochs, cap, split_name, group_label):
        if len(epochs) > cap:
            return random.sample(epochs, cap)
        else:
            print(f"[warning] {group_label} {split_name} split has only {len(epochs)} epochs "
                  f"available (requested cap {cap}) - using all of them.")
            return epochs

    group_label = 'healthy' if epoch_type == 'h' else 'pd'
    train_epochs = cap_sample(train_epochs, cap_train, 'train', group_label)
    val_epochs = cap_sample(val_epochs, cap_val, 'validation', group_label)
    test_epochs = cap_sample(test_epochs, cap_test, 'test', group_label)

    if epoch_type == 'h':
        healthy_training_data.extend(train_epochs)
        healthy_validation_data.extend(val_epochs)
        healthy_testing_data.extend(test_epochs)
    else:
        pd_training_data.extend(train_epochs)
        pd_validation_data.extend(val_epochs)
        pd_testing_data.extend(test_epochs)


# In[11]:


#generating the scalogram images and save
scales = np.arange(1, 33)

def convert_to_scalogram(recieved_sample, sample_status, sample_type):
    
    sample_no = 1
    
    for epoch in recieved_sample:

        coef, freqs = pywt.cwt(epoch, scales, 'gaus2') 
        
        plt.figure(figsize=(8, 5))
        plt.imshow(abs(coef), extent=[0, 1024, 33, 1], interpolation='bilinear', cmap='plasma',
                    aspect='auto', vmax=abs(coef).max(), vmin=abs(coef).min())
        plt.gca().invert_yaxis()
        plt.yticks(ticks=None, labels=None, minor=False)
        plt.xticks(ticks=None, labels=None, minor=False)
        plt.tick_params(left = False, right = False , labelleft = False ,
                        labelbottom = False, bottom = False)
    
        if sample_status == 'h':
            if sample_type == 'training':
                plt.savefig('Data/Training Data/Healthy/h_'+str(sample_no)+'.png', bbox_inches='tight', pad_inches=-0.1)
                sample_no+=1
                plt.close()
                plt.clf()
                gc.collect()
                
            elif sample_type == 'validation':
                plt.savefig('Data/Validation Data/Healthy/h_'+str(sample_no)+'.png', bbox_inches='tight', pad_inches=-0.1)
                sample_no+=1
                plt.close()
                plt.clf()
                gc.collect()
            
            elif sample_type == 'testing':
                plt.savefig('Data/Testing Data/Healthy/h_'+str(sample_no)+'.png', bbox_inches='tight', pad_inches=-0.1)
                sample_no+=1
                plt.close()
                plt.clf()
                gc.collect()
                
        else:
            if sample_type == 'training':
                plt.savefig('Data/Training Data/PD/p_'+str(sample_no)+'.png', bbox_inches='tight', pad_inches=-0.1)
                sample_no+=1
                plt.close()
                plt.clf()
                gc.collect()
                
            elif sample_type == 'validation':
                plt.savefig('Data/Validation Data/PD/p_'+str(sample_no)+'.png', bbox_inches='tight', pad_inches=-0.1)
                sample_no+=1
                plt.close()
                plt.clf()
                gc.collect()
            
            elif sample_type == 'testing':
                plt.savefig('Data/Testing Data/PD/p_'+str(sample_no)+'.png', bbox_inches='tight', pad_inches=-0.1)
                sample_no+=1
                plt.close()
                plt.clf()
                gc.collect()
            


# In[12]:


get_ipython().run_cell_magic('capture', '', 'healthy_read_files = [loaddata(r) for r in hc_raw]\npd_read_files = [loaddata(r) for r in pd_raw]\n')


# In[13]:


get_ipython().run_cell_magic('capture', '', 'healthy_preprocessed_files = [preprocessdata(r) for r in healthy_read_files]\npd_preprocessed_files = [preprocessdata(r) for r in pd_read_files]\n')


# In[14]:


get_ipython().run_cell_magic('capture', '', 'healthy_no_art_file = [removeartifacts(p) for p in healthy_preprocessed_files]\npd_no_art_file = [removeartifacts(p) for p in pd_preprocessed_files]\n')


# In[15]:


get_ipython().run_cell_magic('capture', '', 'healthy_epoch_files = [splitting_epochs(n) for n in healthy_no_art_file]\n')


# In[16]:


get_ipython().run_cell_magic('capture', '', 'pd_epoch_files = [splitting_epochs(n) for n in pd_no_art_file]\n')


# In[17]:


# CORRECTED: reshape each subject's epochs SEPARATELY (list of lists, one entry per subject)
# instead of merging all subjects into one flat list. hc_subject_ids / pd_subject_ids (defined
# near the top of this script) stay aligned index-for-index with these per-subject lists.
get_ipython().run_cell_magic('capture', '', 'healthy_subject_epochs = [reshape_epochs_single_subject(e) for e in healthy_epoch_files]\npd_subject_epochs = [reshape_epochs_single_subject(e) for e in pd_epoch_files]\n')


# In[18]:


# CORRECTED: no global shuffle/sample across subjects here anymore - shuffling now happens
# at the SUBJECT level inside subject_level_split(), and epoch-level sampling happens only
# after subjects have already been routed to train/val/test (inside create_datasets()).
# This cell is intentionally now a no-op check rather than a leakage point.
print(f"Healthy subjects: {len(healthy_subject_epochs)}, PD subjects: {len(pd_subject_epochs)}")


# In[19]:


# (Removed: this used to draw a random 5000-epoch sample from the pooled, subject-agnostic
# list before splitting. Capping/balancing now happens per-split, after subject assignment -
# see cap_train/cap_val/cap_test in create_datasets().)


# In[20]:


if os.path.isfile('Data/split_manifest.csv'):
    os.remove('Data/split_manifest.csv')  # start fresh each run so manifest doesn't accumulate across re-runs

get_ipython().run_cell_magic('capture', '', "create_datasets(hc_subject_ids, healthy_subject_epochs, 'h')\ncreate_datasets(pd_subject_ids, pd_subject_epochs, 'p')\n")


# In[22]:


gc.collect()


# In[23]:


get_ipython().run_cell_magic('capture', '', "convert_to_scalogram(healthy_training_data, 'h', 'training')\n")


# In[24]:


gc.collect()


# In[25]:


get_ipython().run_cell_magic('capture', '', "convert_to_scalogram(pd_training_data, 'p', 'training')\n")


# In[26]:


gc.collect()


# In[27]:


get_ipython().run_cell_magic('capture', '', "convert_to_scalogram(healthy_validation_data, 'h', 'validation')\n")


# In[28]:


gc.collect()


# In[29]:


get_ipython().run_cell_magic('capture', '', "convert_to_scalogram(pd_validation_data, 'p', 'validation')\n")


# In[30]:


gc.collect()


# In[31]:


get_ipython().run_cell_magic('capture', '', "convert_to_scalogram(healthy_testing_data, 'h', 'testing')\n")


# In[32]:


gc.collect()


# In[33]:


get_ipython().run_cell_magic('capture', '', "convert_to_scalogram(pd_testing_data, 'p', 'testing')\n")


# In[ ]:




