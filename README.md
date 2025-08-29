# REDCap RDS Tree Automata

Interactive Streamlit app for visualizing respondent-driven sampling (RDS) recruitment trees, cleaning/imputing network sizes, and (optionally) computing Gile’s weights via an R script.

> **Important modeling rule**  
> A node (participant) is kept **only if** its code appears in the incoming coupon column (incoupon) **or** it is a seed.  
> Coupons that appear **only** in outpon columns (never in incoupon/seed) are **not** treated as participants and are excluded from the graph and all calculations.

---

## ✨ Features

- **Tidy Wave Layout** – nodes in the same wave are perfectly horizontal; seeds sit on the top wave. Optional small **jitter** reduces label overlap; edges can be toggled on/off.  
- **Rich hover tooltips** – ID (integer-formatted), `is_seed`, `networksize_in_tree`, `reported_networksize`; seeds also show `<Wave, N>`.  
- **Seed focus & wave depth** – focus a seed and use a slider to show up to a given wave.  
- **Site-level views** – build per-site subtrees from a site code prefix; explore per-site waves.  
- **Cleaning toolkit** – under-report fix, capping, NA/0 imputation, percentiles, CSV export.  
- **Export trees** – download the current interactive Plotly tree as a standalone HTML file.  
- **Weights (optional)** – run **Gile’s weights** via `compute_rds_weights.R` **per site-level dataset** (not the full tree).

---

## Expected Columns

Configurable in the UI; typical names:

- **Incoming coupon** (e.g., `inpon_number`) — participant’s own coupon (incoupon)  
- **Seed** (e.g., `seed_id`) — seed code (fills identity when incoupon is empty)  
- **Out coupons** (e.g., `outpons_1 ... outpons_k`) — coupons given to recruits  
- **Network size** (e.g., `networksize`) — self-reported degree

> We keep nodes only if they appear in incoupon or are seeds. Coupons that appear only in outpon columns are excluded.

---

## Local Setup

### 1) Clone
```bash
git clone https://github.com/Yuanqi-Mi/RDS-Automatic.git
cd RDS-Automatic

