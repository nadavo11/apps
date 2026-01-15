# Vacation Runner - Next Steps

## Unmapped Arguments (Require Manual Widget Creation)

The following `train_edge_head_e2e.py` arguments are available via "Custom Flags" but don't have dedicated widgets:

### Training Script
- `--sam_checkpoint` - Path to SAM checkpoint (rarely changed)
- `--prefetch_factor` - DataLoader prefetch factor
- `--seed` - Random seed (add if reproducibility matters)
- `--augment` - Data augmentation flag
- `--label_smoothing_sigma` - GT label relaxation
- `--lr_schedule` - LR scheduler type (cosine/step)
- `--lr_step_epochs`, `--lr_step_gamma` - Step scheduler params
- `--wandb_run_name`, `--wandb_run_id` - W&B run management
- `--thresholds`, `--max_dist` - Edge eval thresholds
- `--apply_thinning`, `--apply_nms` - Post-processing
- `--nproc`, `--preview_limit` - Eval parallelism

### Eval Script  
- `--sam_variant`, `--sam_checkpoint` - Override SAM config
- `--max_dist` - Edge matching distance
- `--preview_limit` - Number of preview images

## Hardcoded Paths

The following paths may need adjustment for different setups:

1. **MODELS_DIR** - Currently auto-detected from script location
2. **Default seams_root** - Placeholder `/path/to/seams_dataset`
3. **Default out_dir** - `runs/new_run` relative to models dir

### Suggested Fix
Add a config file (`vacation_runner_config.yaml`) for site-specific defaults:
```yaml
defaults:
  seams_root: /data/seams_v2
  out_dir_prefix: /data/experiments
  sam_checkpoint: /models/sam3_base.pt
```

## Authentication Suggestions

If exposing publicly (not recommended), consider:

### Option 1: streamlit-authenticator
```bash
pip install streamlit-authenticator
```

```python
import streamlit_authenticator as stauth

authenticator = stauth.Authenticate(
    credentials,
    "vacation_runner",
    "random_key",
    cookie_expiry_days=7
)
```

### Option 2: Streamlit Secrets + Password
Add to `.streamlit/secrets.toml`:
```toml
password = "your_secure_password"
```

```python
if st.text_input("Password", type="password") != st.secrets["password"]:
    st.stop()
```

### Option 3: Tailscale ACLs (Recommended)
Keep the app on Tailscale-only network and use Tailscale ACLs to restrict access.

## Future Improvements

1. **GPU Monitoring** - Add nvidia-smi stats via `pynvml`
2. **Multiple Runs** - Support concurrent experiments with separate log files  
3. **Checkpoint Browser** - List and compare checkpoints in runs/
4. **W&B Integration** - Embed W&B dashboard iframe
5. **Config Presets** - Save/load frequently used configurations
