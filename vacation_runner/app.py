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
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

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
CONFIG_FILE = DATA_DIR / "user_preferences.json"

# Ensure app data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Default Training Configuration (user's preferred defaults)
# -----------------------------------------------------------------------------
DEFAULT_TRAIN_CONFIG = {
    "seams_root": str(PROJECT_ROOT / "data_generators" / "inpainted" / "seams2"),
    "out_dir": str(MODELS_DIR / "attn_head"),
    "lr": 3e-4,
    "epochs": 100,
    "batch_size": 8,
    "weight_decay": 5e-4,
    "model": "attn_lite",
    "sam_variant": "sam3_base",
    "lateral_dim": 32,
    "dropout": 0.3,
    "val_split": "test",
    "image_size": 1008,
    "workers": 2,
    "use_loc_loss": True,
    "loc_loss_weight": 1.0,
    "use_dice_loss": True,
    "dice_weight": 0.1,
    "eval_every": 5,
    "eval_mode": "edge",
    "amp": True,
    "compile": False,
    "resume": False,
    "wandb": True,
    "wandb_project": "sam3-edge",
    "custom_flags": "--label_smoothing_sigma 1.5",
}

DEFAULT_EVAL_CONFIG = {
    "data_root": "/path/to/dataset",
    "ckpt": str(MODELS_DIR / "runs" / "best.pt"),
    "out_dir": str(MODELS_DIR / "runs" / "eval_output"),
    "dataset": "seams",
    "split": "test",
    "batch_size": 4,
    "workers": 4,
    "image_size": 1008,
    "eval_mode": "edge",
    "thresholds": 99,
    "nproc": 4,
    "apply_thinning": False,
    "apply_nms": False,
    "wandb": False,
    "custom_flags": "",
}


# -----------------------------------------------------------------------------
# Preferences Management
# -----------------------------------------------------------------------------
def load_preferences() -> Dict[str, Any]:
    """Load user preferences from JSON file."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"train": DEFAULT_TRAIN_CONFIG.copy(), "eval": DEFAULT_EVAL_CONFIG.copy()}


def save_preferences(train_config: Dict, eval_config: Dict) -> bool:
    """Save user preferences to JSON file."""
    try:
        prefs = {"train": train_config, "eval": eval_config}
        with open(CONFIG_FILE, "w") as f:
            json.dump(prefs, f, indent=2)
        return True
    except IOError:
        return False


def get_pref(prefs: Dict, section: str, key: str, default: Any) -> Any:
    """Get a preference value with fallback to default."""
    return prefs.get(section, {}).get(key, default)

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


def stop_process(force: bool = False) -> bool:
    """Stop the running process. Use force=True for SIGKILL instead of SIGINT."""
    proc_info = get_running_process()
    if not proc_info:
        # Check if there's a stale PID file anyway
        if PID_FILE.exists():
            PID_FILE.unlink(missing_ok=True)
            st.info("🧹 Cleaned up stale PID file")
        else:
            st.warning("No process is currently running.")
        return False
    
    try:
        pid = proc_info["pid"]
        sig = signal.SIGKILL if force else signal.SIGINT
        sig_name = "SIGKILL" if force else "SIGINT"
        
        try:
            # Try to kill the process group first
            os.killpg(os.getpgid(pid), sig)
        except (ProcessLookupError, PermissionError):
            # Fallback to killing just the process
            os.kill(pid, sig)
        
        PID_FILE.unlink(missing_ok=True)
        st.success(f"✅ Sent {sig_name} to process {pid}")
        return True
    except Exception as e:
        st.error(f"❌ Failed to stop process: {e}")
        # Still try to clean up PID file
        PID_FILE.unlink(missing_ok=True)
        return False


def force_kill_by_pid(pid: int) -> bool:
    """Force kill a process by PID - resilient kill for orphaned processes."""
    try:
        # First try to kill the process group
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            # Fallback to killing just the process
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        
        # Clean up PID file if it matches
        if PID_FILE.exists():
            try:
                stored_pid = int(PID_FILE.read_text().strip())
                if stored_pid == pid:
                    PID_FILE.unlink(missing_ok=True)
            except (ValueError, IOError):
                pass
        
        return True
    except Exception:
        return False


def find_training_processes() -> List[Dict]:
    """Find any running training/eval processes by command pattern."""
    processes = []
    if not PSUTIL_AVAILABLE:
        return processes
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline:
                    cmd_str = ' '.join(cmdline)
                    # Look for our training/eval scripts
                    if 'train_edge_head_e2e.py' in cmd_str or 'eval_edge_head_e2e.py' in cmd_str:
                        processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cmdline': cmd_str[:100] + '...' if len(cmd_str) > 100 else cmd_str,
                            'create_time': datetime.fromtimestamp(proc.info['create_time']).strftime("%H:%M:%S")
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass
    
    return processes


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
        col_stop, col_kill = st.columns(2)
        with col_stop:
            if st.button("🛑 Stop (SIGINT)", use_container_width=True):
                stop_process(force=False)
                st.rerun()
        with col_kill:
            if st.button("💀 Force Kill (SIGKILL)", type="primary", use_container_width=True):
                stop_process(force=True)
                st.rerun()
    else:
        st.info("⚪ No process running")
    
    # Resilient kill section - find orphaned processes
    with st.expander("🔧 **Process Manager** (orphaned processes)", expanded=False):
        st.caption("Find and kill training processes that may have been orphaned")
        if st.button("🔍 Scan for Training Processes", use_container_width=True):
            st.session_state["scan_processes"] = True
        
        if st.session_state.get("scan_processes"):
            orphaned = find_training_processes()
            if orphaned:
                st.warning(f"Found {len(orphaned)} training process(es):")
                for p in orphaned:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(f"PID {p['pid']} ({p['create_time']})")
                        st.caption(p['cmdline'])
                    with col2:
                        if st.button(f"💀 Kill", key=f"kill_{p['pid']}", use_container_width=True):
                            if force_kill_by_pid(p['pid']):
                                st.success(f"Killed PID {p['pid']}")
                                st.rerun()
                            else:
                                st.error(f"Failed to kill PID {p['pid']}")
            else:
                st.success("✅ No training processes found")
        
        # Manual PID kill
        st.divider()
        st.caption("Or enter a PID manually:")
        manual_pid = st.number_input("PID to kill", min_value=1, value=1, step=1, key="manual_pid")
        if st.button("💀 Force Kill PID", use_container_width=True):
            if force_kill_by_pid(int(manual_pid)):
                st.success(f"✅ Sent SIGKILL to PID {manual_pid}")
            else:
                st.error(f"❌ Failed to kill PID {manual_pid}")
    
    st.divider()
    
    # Tab selection
    tab_train, tab_eval, tab_logs = st.tabs(["🚂 Train", "📊 Eval", "📜 Logs"])
    
    # -------------------------------------------------------------------------
    # Training Tab
    # -------------------------------------------------------------------------
    with tab_train:
        # Load preferences
        prefs = load_preferences()
        train_config = {}
        
        # Save preferences button at the top
        col_save, col_reset = st.columns(2)
        with col_save:
            if st.button("💾 Save Current Settings", use_container_width=True):
                # We'll collect the config at the end and save
                st.session_state["save_train_prefs"] = True
        with col_reset:
            if st.button("🔄 Reset to Defaults", use_container_width=True):
                if CONFIG_FILE.exists():
                    CONFIG_FILE.unlink()
                st.success("Reset to defaults!")
                st.rerun()
        
        # Required paths
        with st.expander("📁 **Data Paths** (required)", expanded=True):
            train_config["seams_root"] = st.text_input(
                "Seams Root",
                value=get_pref(prefs, "train", "seams_root", DEFAULT_TRAIN_CONFIG["seams_root"]),
                help="Root directory containing train/val/test folders"
            )
            train_config["out_dir"] = st.text_input(
                "Output Directory",
                value=get_pref(prefs, "train", "out_dir", DEFAULT_TRAIN_CONFIG["out_dir"]),
                help="Where to save checkpoints and logs"
            )
        
        # Hyperparameters
        with st.expander("⚙️ **Hyperparameters**", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                train_config["lr"] = st.number_input(
                    "Learning Rate",
                    min_value=1e-6, max_value=1e-1,
                    value=float(get_pref(prefs, "train", "lr", DEFAULT_TRAIN_CONFIG["lr"])),
                    format="%.1e", step=1e-5
                )
                train_config["epochs"] = st.number_input(
                    "Epochs", min_value=1, max_value=1000,
                    value=int(get_pref(prefs, "train", "epochs", DEFAULT_TRAIN_CONFIG["epochs"]))
                )
            with col2:
                train_config["batch_size"] = st.number_input(
                    "Batch Size", min_value=1, max_value=32,
                    value=int(get_pref(prefs, "train", "batch_size", DEFAULT_TRAIN_CONFIG["batch_size"]))
                )
                train_config["weight_decay"] = st.number_input(
                    "Weight Decay", min_value=0.0, max_value=0.1,
                    value=float(get_pref(prefs, "train", "weight_decay", DEFAULT_TRAIN_CONFIG["weight_decay"])),
                    format="%.1e", step=1e-5
                )
        
        # Model config
        model_options = ["ced", "linear_probe", "attn_lite"]
        sam_options = ["sam3_tiny", "sam3_small", "sam3_base", "sam3_large"]
        with st.expander("🧠 **Model Configuration**"):
            default_model = get_pref(prefs, "train", "model", DEFAULT_TRAIN_CONFIG["model"])
            train_config["model"] = st.selectbox(
                "Model Architecture",
                options=model_options,
                index=model_options.index(default_model) if default_model in model_options else 2
            )
            default_sam = get_pref(prefs, "train", "sam_variant", DEFAULT_TRAIN_CONFIG["sam_variant"])
            train_config["sam_variant"] = st.selectbox(
                "SAM Variant",
                options=sam_options,
                index=sam_options.index(default_sam) if default_sam in sam_options else 2
            )
            col1, col2 = st.columns(2)
            with col1:
                train_config["lateral_dim"] = st.number_input(
                    "Lateral Dim", min_value=16, max_value=256,
                    value=int(get_pref(prefs, "train", "lateral_dim", DEFAULT_TRAIN_CONFIG["lateral_dim"]))
                )
            with col2:
                train_config["dropout"] = st.number_input(
                    "Dropout", min_value=0.0, max_value=0.9,
                    value=float(get_pref(prefs, "train", "dropout", DEFAULT_TRAIN_CONFIG["dropout"])),
                    step=0.1
                )
        
        # Data config
        val_options = ["val", "test"]
        with st.expander("📊 **Data Configuration**"):
            default_val = get_pref(prefs, "train", "val_split", DEFAULT_TRAIN_CONFIG["val_split"])
            train_config["val_split"] = st.selectbox(
                "Validation Split", options=val_options,
                index=val_options.index(default_val) if default_val in val_options else 1
            )
            train_config["image_size"] = st.number_input(
                "Image Size", min_value=256, max_value=2048,
                value=int(get_pref(prefs, "train", "image_size", DEFAULT_TRAIN_CONFIG["image_size"]))
            )
            train_config["workers"] = st.number_input(
                "DataLoader Workers", min_value=0, max_value=16,
                value=int(get_pref(prefs, "train", "workers", DEFAULT_TRAIN_CONFIG["workers"]))
            )
        
        # Loss settings
        with st.expander("📉 **Loss Settings**"):
            col1, col2 = st.columns(2)
            with col1:
                train_config["use_loc_loss"] = st.checkbox(
                    "Use Loc Loss",
                    value=bool(get_pref(prefs, "train", "use_loc_loss", DEFAULT_TRAIN_CONFIG["use_loc_loss"]))
                )
                train_config["loc_loss_weight"] = st.number_input(
                    "Loc Weight", min_value=0.0, max_value=10.0,
                    value=float(get_pref(prefs, "train", "loc_loss_weight", DEFAULT_TRAIN_CONFIG["loc_loss_weight"]))
                )
            with col2:
                train_config["use_dice_loss"] = st.checkbox(
                    "Use Dice Loss",
                    value=bool(get_pref(prefs, "train", "use_dice_loss", DEFAULT_TRAIN_CONFIG["use_dice_loss"]))
                )
                train_config["dice_weight"] = st.number_input(
                    "Dice Weight", min_value=0.0, max_value=10.0,
                    value=float(get_pref(prefs, "train", "dice_weight", DEFAULT_TRAIN_CONFIG["dice_weight"]))
                )
        
        # Eval settings
        eval_mode_options = ["edge", "binary"]
        with st.expander("📏 **Evaluation Settings**"):
            train_config["eval_every"] = st.number_input(
                "Eval Every N Epochs", min_value=1, max_value=50,
                value=int(get_pref(prefs, "train", "eval_every", DEFAULT_TRAIN_CONFIG["eval_every"]))
            )
            default_eval_mode = get_pref(prefs, "train", "eval_mode", DEFAULT_TRAIN_CONFIG["eval_mode"])
            train_config["eval_mode"] = st.selectbox(
                "Eval Mode", options=eval_mode_options,
                index=eval_mode_options.index(default_eval_mode) if default_eval_mode in eval_mode_options else 0
            )
        
        # Advanced
        with st.expander("🔧 **Advanced Options**"):
            col1, col2 = st.columns(2)
            with col1:
                train_config["amp"] = st.checkbox(
                    "Use AMP (Mixed Precision)",
                    value=bool(get_pref(prefs, "train", "amp", DEFAULT_TRAIN_CONFIG["amp"]))
                )
                train_config["compile"] = st.checkbox(
                    "Use torch.compile",
                    value=bool(get_pref(prefs, "train", "compile", DEFAULT_TRAIN_CONFIG["compile"]))
                )
            with col2:
                train_config["resume"] = st.checkbox(
                    "Resume from Checkpoint",
                    value=bool(get_pref(prefs, "train", "resume", DEFAULT_TRAIN_CONFIG["resume"]))
                )
                train_config["wandb"] = st.checkbox(
                    "Enable W&B Logging",
                    value=bool(get_pref(prefs, "train", "wandb", DEFAULT_TRAIN_CONFIG["wandb"]))
                )
            if train_config["wandb"]:
                train_config["wandb_project"] = st.text_input(
                    "W&B Project",
                    value=get_pref(prefs, "train", "wandb_project", DEFAULT_TRAIN_CONFIG["wandb_project"])
                )
        
        # Custom flags
        with st.expander("🎛️ **Custom Flags**"):
            train_config["custom_flags"] = st.text_input(
                "Additional Arguments",
                value=get_pref(prefs, "train", "custom_flags", DEFAULT_TRAIN_CONFIG["custom_flags"]),
                placeholder="--seed 42 --augment",
                help="Space-separated extra flags to append to the command"
            )
        
        # Handle save preferences
        if st.session_state.get("save_train_prefs"):
            if save_preferences(train_config, prefs.get("eval", DEFAULT_EVAL_CONFIG)):
                st.success("✅ Preferences saved!")
            else:
                st.error("❌ Failed to save preferences")
            st.session_state["save_train_prefs"] = False
        
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
        # Load preferences (reuse from training tab scope or reload)
        eval_prefs = load_preferences()
        eval_config = {}
        
        # Save preferences button at the top
        col_save_eval, col_reset_eval = st.columns(2)
        with col_save_eval:
            if st.button("💾 Save Eval Settings", use_container_width=True, key="save_eval_btn"):
                st.session_state["save_eval_prefs"] = True
        with col_reset_eval:
            if st.button("🔄 Reset Eval Defaults", use_container_width=True, key="reset_eval_btn"):
                # Only reset eval portion
                new_prefs = load_preferences()
                new_prefs["eval"] = DEFAULT_EVAL_CONFIG.copy()
                save_preferences(new_prefs.get("train", DEFAULT_TRAIN_CONFIG), DEFAULT_EVAL_CONFIG)
                st.success("Reset eval to defaults!")
                st.rerun()
        
        # Required paths
        with st.expander("📁 **Paths** (required)", expanded=True):
            eval_config["data_root"] = st.text_input(
                "Data Root",
                value=get_pref(eval_prefs, "eval", "data_root", DEFAULT_EVAL_CONFIG["data_root"]),
                key="eval_data_root"
            )
            eval_config["ckpt"] = st.text_input(
                "Checkpoint Path",
                value=get_pref(eval_prefs, "eval", "ckpt", DEFAULT_EVAL_CONFIG["ckpt"]),
                help="Path to model checkpoint (.pt file)"
            )
            eval_config["out_dir"] = st.text_input(
                "Output Directory",
                value=get_pref(eval_prefs, "eval", "out_dir", DEFAULT_EVAL_CONFIG["out_dir"]),
                key="eval_out_dir"
            )
        
        # Dataset config
        dataset_options = ["seams", "rwtd"]
        split_options = ["train", "val", "test"]
        with st.expander("📊 **Dataset Configuration**", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                default_dataset = get_pref(eval_prefs, "eval", "dataset", DEFAULT_EVAL_CONFIG["dataset"])
                eval_config["dataset"] = st.selectbox(
                    "Dataset Type",
                    options=dataset_options,
                    index=dataset_options.index(default_dataset) if default_dataset in dataset_options else 0
                )
            with col2:
                default_split = get_pref(eval_prefs, "eval", "split", DEFAULT_EVAL_CONFIG["split"])
                eval_config["split"] = st.selectbox(
                    "Split",
                    options=split_options,
                    index=split_options.index(default_split) if default_split in split_options else 2
                )
        
        # Runtime
        with st.expander("⚙️ **Runtime Settings**"):
            col1, col2 = st.columns(2)
            with col1:
                eval_config["batch_size"] = st.number_input(
                    "Batch Size", min_value=1, max_value=32,
                    value=int(get_pref(eval_prefs, "eval", "batch_size", DEFAULT_EVAL_CONFIG["batch_size"])),
                    key="eval_batch"
                )
                eval_config["workers"] = st.number_input(
                    "Workers", min_value=0, max_value=16,
                    value=int(get_pref(eval_prefs, "eval", "workers", DEFAULT_EVAL_CONFIG["workers"])),
                    key="eval_workers"
                )
            with col2:
                eval_config["image_size"] = st.number_input(
                    "Image Size", min_value=256, max_value=2048,
                    value=int(get_pref(eval_prefs, "eval", "image_size", DEFAULT_EVAL_CONFIG["image_size"])),
                    key="eval_img_size"
                )
        
        # Eval settings
        eval_mode_options_eval = ["edge", "binary"]
        with st.expander("📏 **Evaluation Metrics**"):
            default_eval_mode_eval = get_pref(eval_prefs, "eval", "eval_mode", DEFAULT_EVAL_CONFIG["eval_mode"])
            eval_config["eval_mode"] = st.selectbox(
                "Mode", options=eval_mode_options_eval,
                index=eval_mode_options_eval.index(default_eval_mode_eval) if default_eval_mode_eval in eval_mode_options_eval else 0,
                key="eval_mode_select"
            )
            eval_config["thresholds"] = st.number_input(
                "Thresholds", min_value=1, max_value=999,
                value=int(get_pref(eval_prefs, "eval", "thresholds", DEFAULT_EVAL_CONFIG["thresholds"]))
            )
            eval_config["nproc"] = st.number_input(
                "Parallel Workers", min_value=1, max_value=16,
                value=int(get_pref(eval_prefs, "eval", "nproc", DEFAULT_EVAL_CONFIG["nproc"]))
            )
            col1, col2 = st.columns(2)
            with col1:
                eval_config["apply_thinning"] = st.checkbox(
                    "Apply Thinning",
                    value=bool(get_pref(eval_prefs, "eval", "apply_thinning", DEFAULT_EVAL_CONFIG["apply_thinning"]))
                )
            with col2:
                eval_config["apply_nms"] = st.checkbox(
                    "Apply NMS",
                    value=bool(get_pref(eval_prefs, "eval", "apply_nms", DEFAULT_EVAL_CONFIG["apply_nms"]))
                )
        
        # W&B
        with st.expander("📊 **Logging**"):
            eval_config["wandb"] = st.checkbox(
                "Enable W&B Logging",
                value=bool(get_pref(eval_prefs, "eval", "wandb", DEFAULT_EVAL_CONFIG["wandb"])),
                key="eval_wandb"
            )
        
        # Custom flags
        with st.expander("🎛️ **Custom Flags**"):
            eval_config["custom_flags"] = st.text_input(
                "Additional Arguments",
                value=get_pref(eval_prefs, "eval", "custom_flags", DEFAULT_EVAL_CONFIG["custom_flags"]),
                placeholder="--preview_limit 32",
                key="eval_custom"
            )
        
        # Handle save preferences
        if st.session_state.get("save_eval_prefs"):
            if save_preferences(eval_prefs.get("train", DEFAULT_TRAIN_CONFIG), eval_config):
                st.success("✅ Eval preferences saved!")
            else:
                st.error("❌ Failed to save eval preferences")
            st.session_state["save_eval_prefs"] = False
        
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
