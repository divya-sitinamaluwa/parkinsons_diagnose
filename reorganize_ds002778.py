"""
Reorganizes the raw ds002778 BIDS download into the flat hc/ and pd/ses_off/
folder layout that "Data pre-processing and creating full-image dataset.py"
expects (glob("hc/*.bdf") and glob("pd/ses_off/*.bdf")).

Usage: python reorganize_ds002778.py /path/to/ds002778 /path/to/output
"""
import shutil
import sys
from pathlib import Path

def main(src_root, dst_root):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    hc_dst = dst_root / "hc"
    pd_dst = dst_root / "pd" / "ses_off"
    hc_dst.mkdir(parents=True, exist_ok=True)
    pd_dst.mkdir(parents=True, exist_ok=True)

    n_hc, n_pd = 0, 0
    for sub_dir in sorted(src_root.glob("sub-*")):
        subject = sub_dir.name  # e.g. "sub-hc1" or "sub-pd3"
        if subject.startswith("sub-hc"):
            bdf_files = list(sub_dir.glob("ses-hc/eeg/*_eeg.bdf"))
            for f in bdf_files:
                dest = hc_dst / f"{subject}.bdf"
                shutil.copy2(f, dest)
                n_hc += 1
        elif subject.startswith("sub-pd"):
            # Only the OFF-medication session, matching the repo's pd_raw = glob("pd/ses_off/*.bdf")
            bdf_files = list(sub_dir.glob("ses-off/eeg/*_eeg.bdf"))
            for f in bdf_files:
                dest = pd_dst / f"{subject}.bdf"
                shutil.copy2(f, dest)
                n_pd += 1

    print(f"Copied {n_hc} healthy-control files to {hc_dst}")
    print(f"Copied {n_pd} PD (off-medication) files to {pd_dst}")
    print("Expect 16 HC and 15 PD if the full dataset downloaded correctly.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
