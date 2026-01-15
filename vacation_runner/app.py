"""
Vacation Runner - Mobile-First Remote Control for DL Training

A lightweight Streamlit app to run, monitor, and tweak experiments remotely.
Designed for mobile access via Tailscale.

Usage:
    streamlit run app.py --server.port 8501
"""
import os
import subprocess
import signal
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

import streamlit as st

# Try to import psutil for process monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Try to import pyngrok for remote access
try:
    from pyngrok import ngrok, conf
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Path configuration - explicitly set the contrastive_texture_learning path
APP_DIR = Path(__file__).resolve().parent
# The apps repo is separate from contrastive_texture_learning, so we use absolute paths
PROJECT_ROOT = Path("/home/nada/PycharmProjects/contrastive_texture_learning")
MODELS_DIR = PROJECT_ROOT / "models"
TRAIN_SCRIPT = MODELS_DIR / "train_edge_head_e2e.py"
EVAL_SCRIPT = MODELS_DIR / "eval_edge_head_e2e.py"
# Conda environment for the edge detection training
CONDA_ENV = MODELS_DIR / "conda_edgehead"
DATA_DIR = APP_DIR / "data"  # Local app data
LOG_FILE = DATA_DIR / "latest_run.log"
PID_FILE = DATA_DIR / ".running_pid"
NGROK_URL_FILE = DATA_DIR / ".ngrok_url"

# Ensure app data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Ngrok Management
# -----------------------------------------------------------------------------
def get_ngrok_url() -> Optional[str]:
    """Get the current ngrok tunnel URL if active."""
    if not NGROK_AVAILABLE:
        return None
    try:
        tunnels = ngrok.get_tunnels()
        for tunnel in tunnels:
            if "8501" in tunnel.public_url or "localhost" in str(tunnel.config.get("addr", "")):
                return tunnel.public_url
        # Also check saved URL file
        if NGROK_URL_FILE.exists():
            return NGROK_URL_FILE.read_text().strip()
    except Exception:
        pass
    return None


def start_ngrok_tunnel(port: int = 8501) -> Optional[str]:
    """Start an ngrok tunnel to the specified port."""
    if not NGROK_AVAILABLE:
        return None
    try:
        # Kill any existing tunnels first
        ngrok.kill()
        # Start new tunnel
        tunnel = ngrok.connect(port, "http")
        url = tunnel.public_url
        # Save URL to file
        NGROK_URL_FILE.write_text(url)
        return url
    except Exception as e:
        st.error(f"❌ Failed to start ngrok: {e}")
        return None


def stop_ngrok_tunnel():
    """Stop all ngrok tunnels."""
    if not NGROK_AVAILABLE:
        return
    try:
        ngrok.kill()
        NGROK_URL_FILE.unlink(missing_ok=True)
    except Exception:
        pass

# -----------------------------------------------------------------------------
# Process Management
# -----------------------------------------------------------------------------
def get_running_process() -> Optional[Dict]:
    """Check if a training/eval process is running."""
    if not PID_FILE.exists():
        return None
    
    try:
        pid = int(PID_FILE.read_text().strip())
        if PSUTIL_AVAILABLE:
            try:
                proc = psutil.Process(pid)
                if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                    return {
                        "pid": pid,
                        "name": proc.name(),
                        "status": proc.status(),
                        "create_time": datetime.fromtimestamp(proc.create_time()).strftime("%Y-%m-%d %H:%M:%S"),
                        "cpu": f"{proc.cpu_percent():.1f}%",
                        "memory": f"{proc.memory_percent():.1f}%",
                    }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        else:
            # Fallback: check if process exists via /proc
            if Path(f"/proc/{pid}").exists():
                return {"pid": pid, "name": "unknown", "status": "running"}
    except (ValueError, FileNotFoundError):
        pass
    
    # Clean up stale PID file
    PID_FILE.unlink(missing_ok=True)
    return None


def start_process(command: List[str], script_type: str) -> bool:
    """Start a training/eval process in the background."""
    if get_running_process():
        st.error("⚠️ A process is already running! Stop it first.")
        return False
    
    try:
        # Ensure log file parent exists
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        with open(LOG_FILE, "w") as log_f:
            log_f.write(f"=== {script_type} started at {datetime.now().isoformat()} ===\n")
            log_f.write(f"Command: {' '.join(command)}\n")
            log_f.write("=" * 60 + "\n\n")
        
        with open(LOG_FILE, "a") as log_f:
            proc = subprocess.Popen(
                command,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=str(MODELS_DIR),
                start_new_session=True,  # Detach from parent
            )
        
        # Save PID
        PID_FILE.write_text(str(proc.pid))
        return True
    
    except Exception as e:
        st.error(f"❌ Failed to start process: {e}")
        return False


def stop_process() -> bool:
    """Stop the running process."""
    proc_info = get_running_process()
    if not proc_info:
        st.warning("No process is currently running.")
        return False
    
    try:
        pid = proc_info["pid"]
        os.killpg(os.getpgid(pid), signal.SIGINT)  # Graceful shutdown
        PID_FILE.unlink(missing_ok=True)
        st.success(f"✅ Sent SIGINT to process {pid}")
        return True
    except Exception as e:
        st.error(f"❌ Failed to stop process: {e}")
        return False


def read_log_tail(n_lines: int = 50) -> str:
    """Read the last n lines from the log file."""
    if not LOG_FILE.exists():
        return "(No log file yet)"
    
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
            return "".join(lines[-n_lines:])
    except Exception as e:
        return f"(Error reading log: {e})"


# -----------------------------------------------------------------------------
# Command Builders
# -----------------------------------------------------------------------------
def build_train_command(config: Dict) -> List[str]:
    """Build the training command from config."""
    # Use conda run to execute in the correct environment
    cmd = ["conda", "run", "--prefix", str(CONDA_ENV), "--no-capture-output", "python", str(TRAIN_SCRIPT)]
    
    # Required paths
    cmd.extend(["--seams_root", config["seams_root"]])
    cmd.extend(["--out_dir", config["out_dir"]])
    
    # Hyperparameters
    cmd.extend(["--lr", f"{config['lr']:.1e}"])
    cmd.extend(["--epochs", str(config["epochs"])])
    cmd.extend(["--batch_size", str(config["batch_size"])])
    cmd.extend(["--weight_decay", str(config["weight_decay"])])
    
    # Model config
    cmd.extend(["--model", config["model"]])
    cmd.extend(["--sam_variant", config["sam_variant"]])
    cmd.extend(["--lateral_dim", str(config["lateral_dim"])])
    cmd.extend(["--dropout", str(config["dropout"])])
    
    # Data config
    cmd.extend(["--val_split", config["val_split"]])
    cmd.extend(["--image_size", str(config["image_size"])])
    cmd.extend(["--workers", str(config["workers"])])
    
    # Loss settings
    if config["use_loc_loss"]:
        cmd.append("--use_loc_loss")
        cmd.extend(["--loc_loss_weight", str(config["loc_loss_weight"])])
    if config["use_dice_loss"]:
        cmd.append("--use_dice_loss")
        cmd.extend(["--dice_weight", str(config["dice_weight"])])
    
    # Eval settings
    cmd.extend(["--eval_every", str(config["eval_every"])])
    cmd.extend(["--eval_mode", config["eval_mode"]])
    
    # Advanced
    if config["amp"]:
        cmd.append("--amp")
    if config["compile"]:
        cmd.append("--compile")
    if config["resume"]:
        cmd.append("--resume")
    if config["wandb"]:
        cmd.append("--wandb")
        if config.get("wandb_project"):
            cmd.extend(["--wandb_project", config["wandb_project"]])
    
    # Custom flags
    if config.get("custom_flags"):
        custom = config["custom_flags"].strip()
        if custom:
            cmd.extend(custom.split())
    
    return cmd


def build_eval_command(config: Dict) -> List[str]:
    """Build the evaluation command from config."""
    # Use conda run to execute in the correct environment
    cmd = ["conda", "run", "--prefix", str(CONDA_ENV), "--no-capture-output", "python", str(EVAL_SCRIPT)]
    
    # Required
    cmd.extend(["--data_root", config["data_root"]])
    cmd.extend(["--ckpt", config["ckpt"]])
    cmd.extend(["--out_dir", config["out_dir"]])
    
    # Dataset
    cmd.extend(["--dataset", config["dataset"]])
    cmd.extend(["--split", config["split"]])
    
    # Runtime
    cmd.extend(["--batch_size", str(config["batch_size"])])
    cmd.extend(["--workers", str(config["workers"])])
    cmd.extend(["--image_size", str(config["image_size"])])
    
    # Eval settings
    cmd.extend(["--eval_mode", config["eval_mode"]])
    cmd.extend(["--thresholds", str(config["thresholds"])])
    cmd.extend(["--nproc", str(config["nproc"])])
    
    if config["apply_thinning"]:
        cmd.append("--apply_thinning")
    if config["apply_nms"]:
        cmd.append("--apply_nms")
    
    # W&B
    if config["wandb"]:
        cmd.append("--wandb")
    
    # Custom flags
    if config.get("custom_flags"):
        custom = config["custom_flags"].strip()
        if custom:
            cmd.extend(custom.split())
    
    return cmd


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="🏖️ Vacation Runner",
        page_icon="🏖️",
        layout="centered",  # Mobile-friendly
    )
    
    st.title("🏖️ Vacation Runner")
    st.caption("Remote control for DL experiments")
    
    # -------------------------------------------------------------------------
    # Sidebar: Remote Access (ngrok)
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("🌐 Remote Access")
        
        if not NGROK_AVAILABLE:
            st.warning("pyngrok not installed. Run: `pip install pyngrok`")
        else:
            ngrok_url = get_ngrok_url()
            
            if ngrok_url:
                st.success("🟢 **Tunnel Active**")
                st.code(ngrok_url, language=None)
                st.caption("Access this URL from anywhere!")
                if st.button("🛑 Stop Tunnel", use_container_width=True):
                    stop_ngrok_tunnel()
                    st.rerun()
            else:
                st.info("⚪ No tunnel active")
                st.caption("Start a tunnel to access from outside your network")
                if st.button("🚀 Start ngrok Tunnel", type="primary", use_container_width=True):
                    url = start_ngrok_tunnel(8501)
                    if url:
                        st.success(f"✅ Tunnel started!")
                        st.rerun()
        
        st.divider()
        st.caption("💡 Tip: ngrok provides a public URL to access this app from your phone while on vacation!")
    
    # Process status indicator
    proc = get_running_process()
    if proc:
        st.success(f"🟢 **Process Running** (PID: {proc['pid']})")
        with st.expander("Process Details", expanded=False):
            for k, v in proc.items():
                st.text(f"{k}: {v}")
        if st.button("🛑 Stop Process", type="primary", use_container_width=True):
            stop_process()
            st.rerun()
    else:
        st.info("⚪ No process running")
    
    st.divider()
    
    # Tab selection
    tab_train, tab_eval, tab_logs = st.tabs(["🚂 Train", "📊 Eval", "📜 Logs"])
    
    # -------------------------------------------------------------------------
    # Training Tab
    # -------------------------------------------------------------------------
    with tab_train:
        train_config = {}
        
        # Required paths
        with st.expander("📁 **Data Paths** (required)", expanded=True):
            train_config["seams_root"] = st.text_input(
                "Seams Root",
                value="/path/to/seams_dataset",
                help="Root directory containing train/val/test folders"
            )
            train_config["out_dir"] = st.text_input(
                "Output Directory",
                value=str(MODELS_DIR / "runs" / "new_run"),
                help="Where to save checkpoints and logs"
            )
        
        # Hyperparameters
        with st.expander("⚙️ **Hyperparameters**", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                train_config["lr"] = st.number_input(
                    "Learning Rate",
                    min_value=1e-6, max_value=1e-1,
                    value=3e-4, format="%.1e", step=1e-5
                )
                train_config["epochs"] = st.number_input(
                    "Epochs", min_value=1, max_value=1000, value=100
                )
            with col2:
                train_config["batch_size"] = st.number_input(
                    "Batch Size", min_value=1, max_value=32, value=4
                )
                train_config["weight_decay"] = st.number_input(
                    "Weight Decay", min_value=0.0, max_value=0.1,
                    value=5e-4, format="%.1e", step=1e-5
                )
        
        # Model config
        with st.expander("🧠 **Model Configuration**"):
            train_config["model"] = st.selectbox(
                "Model Architecture",
                options=["ced", "linear_probe", "attn_lite"],
                index=0
            )
            train_config["sam_variant"] = st.selectbox(
                "SAM Variant",
                options=["sam3_tiny", "sam3_small", "sam3_base", "sam3_large"],
                index=2
            )
            col1, col2 = st.columns(2)
            with col1:
                train_config["lateral_dim"] = st.number_input(
                    "Lateral Dim", min_value=16, max_value=256, value=64
                )
            with col2:
                train_config["dropout"] = st.number_input(
                    "Dropout", min_value=0.0, max_value=0.9, value=0.3, step=0.1
                )
        
        # Data config
        with st.expander("📊 **Data Configuration**"):
            train_config["val_split"] = st.selectbox(
                "Validation Split", options=["val", "test"], index=1
            )
            train_config["image_size"] = st.number_input(
                "Image Size", min_value=256, max_value=2048, value=1008
            )
            train_config["workers"] = st.number_input(
                "DataLoader Workers", min_value=0, max_value=16, value=2
            )
        
        # Loss settings
        with st.expander("📉 **Loss Settings**"):
            col1, col2 = st.columns(2)
            with col1:
                train_config["use_loc_loss"] = st.checkbox("Use Loc Loss", value=True)
                train_config["loc_loss_weight"] = st.number_input(
                    "Loc Weight", min_value=0.0, max_value=10.0, value=1.0
                )
            with col2:
                train_config["use_dice_loss"] = st.checkbox("Use Dice Loss", value=True)
                train_config["dice_weight"] = st.number_input(
                    "Dice Weight", min_value=0.0, max_value=10.0, value=0.1
                )
        
        # Eval settings
        with st.expander("📏 **Evaluation Settings**"):
            train_config["eval_every"] = st.number_input(
                "Eval Every N Epochs", min_value=1, max_value=50, value=5
            )
            train_config["eval_mode"] = st.selectbox(
                "Eval Mode", options=["edge", "binary"], index=0
            )
        
        # Advanced
        with st.expander("🔧 **Advanced Options**"):
            col1, col2 = st.columns(2)
            with col1:
                train_config["amp"] = st.checkbox("Use AMP (Mixed Precision)", value=False)
                train_config["compile"] = st.checkbox("Use torch.compile", value=False)
            with col2:
                train_config["resume"] = st.checkbox("Resume from Checkpoint", value=False)
                train_config["wandb"] = st.checkbox("Enable W&B Logging", value=False)
            if train_config["wandb"]:
                train_config["wandb_project"] = st.text_input(
                    "W&B Project", value="sam3-edge-e2e"
                )
        
        # Custom flags
        with st.expander("🎛️ **Custom Flags**"):
            train_config["custom_flags"] = st.text_input(
                "Additional Arguments",
                placeholder="--seed 42 --augment",
                help="Space-separated extra flags to append to the command"
            )
        
        # Command preview
        st.subheader("📋 Command Preview")
        train_cmd = build_train_command(train_config)
        st.code(" ".join(train_cmd), language="bash")
        
        # Run button
        if st.button("🚀 Start Training", type="primary", use_container_width=True, disabled=proc is not None):
            if start_process(train_cmd, "Training"):
                st.success("✅ Training started! Check the Logs tab.")
                st.rerun()
    
    # -------------------------------------------------------------------------
    # Evaluation Tab
    # -------------------------------------------------------------------------
    with tab_eval:
        eval_config = {}
        
        # Required paths
        with st.expander("📁 **Paths** (required)", expanded=True):
            eval_config["data_root"] = st.text_input(
                "Data Root",
                value="/path/to/dataset",
                key="eval_data_root"
            )
            eval_config["ckpt"] = st.text_input(
                "Checkpoint Path",
                value=str(MODELS_DIR / "runs" / "best.pt"),
                help="Path to model checkpoint (.pt file)"
            )
            eval_config["out_dir"] = st.text_input(
                "Output Directory",
                value=str(MODELS_DIR / "runs" / "eval_output"),
                key="eval_out_dir"
            )
        
        # Dataset config
        with st.expander("📊 **Dataset Configuration**", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                eval_config["dataset"] = st.selectbox(
                    "Dataset Type",
                    options=["seams", "rwtd"],
                    index=0
                )
            with col2:
                eval_config["split"] = st.selectbox(
                    "Split",
                    options=["train", "val", "test"],
                    index=2
                )
        
        # Runtime
        with st.expander("⚙️ **Runtime Settings**"):
            col1, col2 = st.columns(2)
            with col1:
                eval_config["batch_size"] = st.number_input(
                    "Batch Size", min_value=1, max_value=32, value=4, key="eval_batch"
                )
                eval_config["workers"] = st.number_input(
                    "Workers", min_value=0, max_value=16, value=4, key="eval_workers"
                )
            with col2:
                eval_config["image_size"] = st.number_input(
                    "Image Size", min_value=256, max_value=2048, value=1008, key="eval_img_size"
                )
        
        # Eval settings
        with st.expander("📏 **Evaluation Metrics**"):
            eval_config["eval_mode"] = st.selectbox(
                "Mode", options=["edge", "binary"], index=0, key="eval_mode_select"
            )
            eval_config["thresholds"] = st.number_input(
                "Thresholds", min_value=1, max_value=999, value=99
            )
            eval_config["nproc"] = st.number_input(
                "Parallel Workers", min_value=1, max_value=16, value=4
            )
            col1, col2 = st.columns(2)
            with col1:
                eval_config["apply_thinning"] = st.checkbox("Apply Thinning", value=False)
            with col2:
                eval_config["apply_nms"] = st.checkbox("Apply NMS", value=False)
        
        # W&B
        with st.expander("📊 **Logging**"):
            eval_config["wandb"] = st.checkbox("Enable W&B Logging", value=False, key="eval_wandb")
        
        # Custom flags
        with st.expander("🎛️ **Custom Flags**"):
            eval_config["custom_flags"] = st.text_input(
                "Additional Arguments",
                placeholder="--preview_limit 32",
                key="eval_custom"
            )
        
        # Command preview
        st.subheader("📋 Command Preview")
        eval_cmd = build_eval_command(eval_config)
        st.code(" ".join(eval_cmd), language="bash")
        
        # Run button
        if st.button("🚀 Start Evaluation", type="primary", use_container_width=True, disabled=proc is not None):
            if start_process(eval_cmd, "Evaluation"):
                st.success("✅ Evaluation started! Check the Logs tab.")
                st.rerun()
    
    # -------------------------------------------------------------------------
    # Logs Tab
    # -------------------------------------------------------------------------
    with tab_logs:
        st.subheader("📜 Live Logs")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            n_lines = st.number_input("Lines to show", min_value=10, max_value=500, value=50)
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        # Auto-refresh option
        auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)
        if auto_refresh:
            import time
            time.sleep(0.1)  # Small delay to allow UI to render
            st.rerun()
        
        # Log content
        log_content = read_log_tail(n_lines)
        st.code(log_content, language="log")
        
        # Log file info
        if LOG_FILE.exists():
            size_kb = LOG_FILE.stat().st_size / 1024
            mtime = datetime.fromtimestamp(LOG_FILE.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            st.caption(f"📄 Log file: {LOG_FILE.name} | Size: {size_kb:.1f} KB | Modified: {mtime}")
        
        # Download log
        if LOG_FILE.exists():
            with open(LOG_FILE, "r") as f:
                st.download_button(
                    "📥 Download Full Log",
                    data=f.read(),
                    file_name="training_log.txt",
                    mime="text/plain",
                    use_container_width=True
                )


if __name__ == "__main__":
    main()
