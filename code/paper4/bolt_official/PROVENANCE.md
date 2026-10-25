# Vendored official BOLT code (unmodified)

- Upstream: https://github.com/sming256/BOLT ("BOLT: Boost Large Vision-Language Model
  Without Training for Long-form Video Understanding", CVPR 2025,
  kg:paper:W4413146798).
- Pinned commit: 8a9ba455e8c7668ae4443b3273bbf283b700cc7f (branch main, 2026-02-05,
  "Update repository URL in README").
- Obtained through this deployment's own dataset channel (dataset_fetch, provider url,
  allowed host codeload.github.com):
  https://codeload.github.com/sming256/BOLT/tar.gz/8a9ba455e8c7668ae4443b3273bbf283b700cc7f
  tarball bytes sha256 fd86933ae83f515fad9fb9b036e64cf9ef3a662e0c6ef2defad8dadf2b06ad35
  (platform dataset record plan-dataset-bolt-code-8a9ba45). The device itself
  cannot reach github over https; nothing was downloaded from any other address.
- Files vendored VERBATIM (byte-exact, sha256 pinned and re-checked at the start of
  every run that imports them):
  - select_frames.py  1e2297a0ae33dee9183cc30d71cc1dee566ff2acf5ae05d3d337ee809be2ce5c
    (contains the official inverse_transform_sampling, Eq. 5-7 of the paper)
  - extract_feature.py  23156ebd4433bc542106ba3525e2668b1836a3ce75c7f626fe153a7cc6ef4c12
    (contains the official CLIP class: openai/clip-vit-large-patch14, fp16, autocast,
     get_image_features / get_text_features, cosine x logit_scale similarity)
  - LICENSE  a1d77e793203297eba6aac941a1040b4b119fa82608cb42d9dff7df80321c965
    (MIT, Copyright (c) 2025 BOLT Project Contributors)
- Official LongVideoBench parameters used by the exploratory arm (from
  scripts/frame_selection.sh of the same commit): model clip (ViT-L/14),
  power 2.5, num_frames 16, features at fps 1.
- What this project uses from the vendored files: the CLIP class and
  inverse_transform_sampling, unmodified. What it does NOT use: the decord-based
  fps=1 feature-extraction main(), the dataset parquet loaders, the lmms-eval
  integration (the device has no decord and the selection cell's registered frame
  universe is its own 32-frame uniform midpoint candidate pool; see
  plan-exploratory-bolt-timing setup for the registered deviations).
- The only adaptation is OUTSIDE these files (paper4/exploratory_bolt.py): a subclass
  whose __init__ loads the same weights from the local device directory instead of the
  hub id, because the run is offline. Class methods are inherited unmodified.
- paper4/offline_shims/ (ours, not upstream): decord.py and pandas.py shims that raise
  NotImplementedError on any use. The official files import both at module top for
  paths this arm never executes (fps=1 decord extraction, parquet dataset loaders);
  the shims let the byte-exact files import offline and fail loudly if a declared-
  unused path is ever reached.
