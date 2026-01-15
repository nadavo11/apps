# 🏖️ Vacation Runner

Mobile-first remote control for DL training and evaluation experiments.

## Setup

```bash
# Install dependencies
pip install streamlit psutil

# Or with conda
conda install -c conda-forge streamlit psutil
```

## Usage

```bash
# Launch the app from the apps/vacation_runner directory
cd /home/nada/PycharmProjects/apps/vacation_runner
streamlit run app.py --server.port 8501
```

Access at: `http://localhost:8501` or via Tailscale IP.

## Tailscale Access

1. Install Tailscale on your server and mobile device
2. Start Tailscale: `sudo tailscale up`
3. Get server IP: `tailscale ip -4`
4. Access from mobile: `http://<tailscale-ip>:8501`

## Adding New Scripts

1. Open `vacation_runner_app.py`
2. Add new script path in the Configuration section:
   ```python
   NEW_SCRIPT = MODELS_DIR / "your_script.py"
   ```
3. Create a command builder function:
   ```python
   def build_new_command(config: Dict) -> List[str]:
       cmd = ["python", str(NEW_SCRIPT)]
       # Add arguments...
       return cmd
   ```
4. Add a new tab in the UI section
5. Add widgets for script arguments
6. Add command preview and run button

## Files

- `vacation_runner_app.py` - Main Streamlit application
- `vacation_runner_data/` - Runtime data directory (auto-created)
  - `latest_run.log` - Redirected stdout/stderr from processes
  - `.running_pid` - PID file for process tracking
