REDCap RDS Tree Automata

Interactive Streamlit app for visualizing respondent-driven sampling (RDS) recruitment trees, cleaning/imputing network sizes, and (optionally) computing Gile’s weights via an R script.

✨ Features

Tidy Wave Layout: nodes in the same wave are perfectly horizontal; seeds sit on top.
Small jitter option reduces label overlap; edges can be toggled on/off.

Rich hover: see ID (integer-formatted), is_seed, networksize_in_tree, and reported_networksize.

Site-level views: filter by site prefix; explore per-seed subtrees and wave depth sliders.

Cleaning toolkit: under-report fix, capping, NA/0 imputation, percentiles table, and CSV export.

Export: download any tree as a standalone HTML (interactive Plotly).

Weights (optional): run Gile’s weights via compute_rds_weights.R per site-level dataset; export CSV.

🗂️ Expected Columns

Your data should contain (names are configurable in the UI):

Incoming coupon (e.g., inpon_number) — participant’s own coupon (incoupon)

Seed (e.g., seed_id) — seed code (for rows where incoupon is empty, seed fills identity)

Out coupons (e.g., outpons_1 ... outpons_k) — coupons given to recruits

Network size (e.g., networksize) — self-reported degree

Important: We only keep nodes that appear in incoupon or as a seed.
Recruits who never became participants (coupon only shows in outpon columns) are not counted as nodes.
