# Corrections made to this pipeline

## 1. Subject-level data leakage
(`Create datasets/Data pre-processing and creating full-image dataset.py`)

Previously: EEG epochs from all subjects were pooled into one list, shuffled, then split
80/10/10 **by index**. No subject ID was tracked at any point, so 5-second epochs from the
same participant could land in training, validation, and test simultaneously.

Now: subjects are assigned to train/validation/test **first** (with a fixed seed for
reproducibility), and only that subject's epochs go into that split. A manifest of which
subject went where is written to `Data/split_manifest.csv`.

Class balancing (previously "grab 5000 random epochs before splitting") now happens per split,
after subject assignment, so it can no longer leak subjects across splits. If a split doesn't
have enough epochs to hit the target cap (4000 train / 500 val / 500 test per class), all
available epochs are used and a warning is printed — worth checking given only 31 subjects
total.

## 2. No model ever evaluated on the held-out test set
(all 8 scripts in `Models/`)

Previously: every script loaded only `Training Data` and `Validation Data`. The `Testing Data`
folder created during preprocessing was never read by any model script. Every number in the
manuscript's Tables 1-3 was therefore a **validation-set** metric, and the "best" configuration
in each ablation (pooling, activation, optimiser, neuron count, dropout) was chosen by
repeatedly comparing performance on that same validation set.

Now: every script also loads the corresponding Testing Data folder, and adds one final
evaluation block reporting accuracy, classification report, and confusion matrix on the test
set, clearly labeled "Held-out TEST accuracy (report this, not val_acc)". Validation accuracy
is still used internally for model/hyperparameter selection — only the final reported number
changes.

## 3. Copy-paste bug (`Hybrid model with split H2.py` and `V2.py`)

Group 2's validation-data loader was pointing at `Validation Data/Group 1` instead of
`Group 2`, so `val_data2` was silently a duplicate of `val_data1` rather than genuine Group-2
data. Fixed to point at Group 2.

## 4. Output folder path mismatch (found via a small end-to-end dry run)

Previously: the preprocessing script wrote scalogram images to `Data/Training Data/...`,
`Data/Validation Data/...`, `Data/Testing Data/...`, but every model script in `Models/`
reads from `Dataset/Full Image/Training Data/...` (and the split-image script reads from
the same `Dataset/Full Image/...` path to build its split-image variants). This meant an
undocumented manual step - renaming/moving the output folder - was required between running
preprocessing and running any model script. That step wasn't captured anywhere in the repo.

Now: the preprocessing script writes directly to `Dataset/Full Image/...`, matching what
every downstream script already expects. The split manifest also moved accordingly, from
`Data/split_manifest.csv` to `Dataset/split_manifest.csv`. No manual folder move is needed
anymore - run preprocessing, then the split-image script, then any model script, in that
order, from the same working directory.

## 5. Worth checking with the manuscript before re-reporting

The `H4`/`V4` scripts use `Dropout(0.8)` in the final classifier, but Table 3 in the manuscript
records the "best" configuration as using 0.5 dropout. Please confirm which value was actually
used to produce the originally reported numbers before treating the corrected H4 run as the
like-for-like replacement for Table 3.

## What's NOT changed

- The preprocessing/filtering/ICA steps themselves
- Model architectures
- All the ablation variants (pooling, activation, optimiser, neuron count) — only the data
  split and test evaluation were touched
- `Creating split image datasets.py` — no changes needed; it already reads from whichever
  Training/Validation/Testing folders the preprocessing script produces, so it inherits the
  corrected subject-level split automatically
