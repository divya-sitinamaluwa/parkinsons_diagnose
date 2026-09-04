# Corrected Parkinson's EEG Pipeline — Start Here

This package contains the corrected version of `github.com/divya-sitinamaluwa/parkinsons_diagnose`,
plus everything needed to get from raw data to a genuine, trustworthy test-set result.

## Step 1 — Get the raw dataset

The dataset is public: OpenNeuro **ds002778**, "UC San Diego Resting State EEG Data from
Patients with Parkinson's Disease" (Rockhill, Jackson, George, Aron & Swann, 2021).

```bash
pip install awscli --break-system-packages   # if not already installed
aws s3 sync s3://openneuro.org/ds002778/ ./ds002778/ --no-sign-request --region us-east-1
```

This downloads ~545 MB, all 31 subjects (16 HC, 15 PD) in BIDS format.

**Important — dataset usage condition:** the OpenNeuro page for this dataset states:
> "Please email arockhil@uoregon.edu before submitting a manuscript to be published in a
> peer-reviewed journal using this data, we wish to ensure that the data to be analyzed and
> interpreted with scientific integrity."

This should happen before submission, regardless of which framing the paper ends up taking.

## Step 2 — Reorganize into the folder layout the scripts expect

The raw download is in BIDS format (`sub-hc1/ses-hc/eeg/...`), but the preprocessing script
expects a flat `hc/*.bdf` and `pd/ses_off/*.bdf` layout. Run:

```bash
python reorganize_ds002778.py ./ds002778 ./ds002778_flat
```

This produces `./ds002778_flat/hc/` (16 files) and `./ds002778_flat/pd/ses_off/` (15 files).
Run the preprocessing script from inside `./ds002778_flat` (or adjust its `hc_raw`/`pd_raw`
glob paths to point there).

## Step 3 — Run the corrected preprocessing script

`Create datasets/Data pre-processing and creating full-image dataset.py` now:
- Splits subjects into train/validation/test **before** pooling any epochs, so no participant's
  data can appear in more than one split (previously epochs were pooled and shuffled across
  all subjects before splitting — see CHANGES.md for detail).
- Writes `Data/split_manifest.csv` recording which subject went to which split — please share
  this file back, it's worth checking given only 31 subjects total.
- Balances/caps epoch counts per split *after* subject assignment (previously this happened
  before splitting, which was part of the leakage).

Then run `Create datasets/Creating split image datasets.py` as before — no changes needed there,
it inherits the corrected split automatically since it just reads from the Training/Validation/
Testing folders the previous script produces.

## Step 4 — Run the corrected model scripts

All 8 scripts in `Models/` now load the `Testing Data` folder (previously created but never
read) and print a final held-out test evaluation clearly labeled "Held-out TEST accuracy
(report this, not val_acc)". The validation accuracy is still used internally for
model/hyperparameter selection, exactly as before — only the final reported number changes.

**Please prioritise these four, matching the manuscript's Tables 1–3:**
1. `CNN model with full-image dataset.py`
2. `VGG16 model with full-image dataset.py` (this fills in the blank VGG-16 result in Table 1)
3. `Hybrid model with full-image dataset.py`
4. `Hybrid model with split H4.py` (Table 3's "best" configuration — **please double check**:
   this script currently uses `Dropout(0.8)`, but Table 3 records the best config as using
   0.5 dropout. Confirm which is correct before reporting this number.)

## What to send back

- The four test-set results above (accuracy, precision, recall, F1 macro + weighted)
- `Data/split_manifest.csv`
- The original (leaky) validation numbers already in the manuscript — please don't discard
  them, we may want to report both as a before/after comparison
- Anything unexpected (e.g. a split ending up with very few epochs, or accuracy swinging a
  lot given how few subjects are in the test fold)

See `CHANGES.md` for the full technical detail of what was changed and why.
