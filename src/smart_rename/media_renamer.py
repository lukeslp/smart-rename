
import os
import sys
import json
import base64
import re
import cv2
import tempfile
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

try:
    from openai import OpenAI
    from rich.panel import Panel
    from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
    from rich.table import Table
    from rich.prompt import Confirm
except ImportError as e:
    print(f"Missing required package: {e}")
    sys.exit(1)

from .utils import console, get_ai_config, LOG_DIR

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.heic', '.mp4'}
RENAME_LOG_FILE = LOG_DIR / "media_rename_log.json"

def get_ai_client():
    """Get configured AI client with vision model (OpenAI or Ollama)."""
    config = get_ai_config(vision=True)
    if not config["api_key"]:
        return None, None
    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
    return client, config["model"]

# --- Helper Functions ---

def load_rename_log() -> Dict:
    """Load the rename log for undo functionality."""
    if RENAME_LOG_FILE.exists():
        with open(RENAME_LOG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"renames": [], "last_operation": None}

def save_rename_log(log_data: Dict) -> None:
    """Save the rename log to file."""
    log_data["last_updated"] = datetime.now().isoformat()
    RENAME_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RENAME_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

def extract_video_frame(video_path: Path, frame_position: float = 0.3) -> Optional[Path]:
    """Extract a frame from video file."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None
        target_frame = int(total_frames * frame_position)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            temp_path = Path(tmp_file.name)
            cv2.imwrite(str(temp_path), frame)
        return temp_path
    except Exception:
        return None

def get_image_base64(image_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Read and encode image/video frame to base64."""
    try:
        frame_path = None
        if image_path.suffix.lower() == '.mp4':
            frame_path = extract_video_frame(image_path)
            if not frame_path:
                return None, None
            read_path = frame_path
            mime_type = 'jpeg'
        else:
            read_path = image_path
            ext = image_path.suffix.lower()
            mime_map = {'.jpg': 'jpeg', '.jpeg': 'jpeg', '.png': 'png', '.gif': 'gif', '.webp': 'webp'}
            mime_type = mime_map.get(ext, 'jpeg')

        with open(read_path, 'rb') as img_file:
            img_data = img_file.read()
            encoded = base64.b64encode(img_data).decode('utf-8')
        
        if frame_path and frame_path.exists():
            frame_path.unlink()
            
        return encoded, mime_type
    except Exception as e:
        console.print(f"[red]Error encoding {image_path.name}: {e}[/red]")
        return None, None

def generate_filename(image_path: Path) -> Optional[str]:
    """Use vision model to generate a descriptive filename."""
    client, model = get_ai_client()
    if not client:
        console.print("[red]Error: AI not configured. Set OPENAI_API_KEY or use Ollama.[/red]")
        return None

    media_type = "video" if image_path.suffix.lower() == '.mp4' else "image"
    console.print(f"[cyan]Analyzing {media_type}:[/cyan] {image_path.name}")

    base64_img, mime_type = get_image_base64(image_path)
    if not base64_img:
        return None

    media_url = f"data:image/{mime_type};base64,{base64_img}"
    prompt = "Analyze this image and generate a descriptive filename (5 words or less, lowercase, underscores). Return ONLY the filename."

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": media_url, "detail": "high"}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            temperature=0.3,
            max_tokens=50,
        )
        filename = response.choices[0].message.content.strip()
        filename = re.sub(r'[^a-z0-9_]', '', filename.lower().replace(' ', '_').replace('-', '_'))
        filename = re.sub(r'_+', '_', filename).strip('_')
        console.print(f"[green]Generated:[/green] {filename}")
        return filename
    except Exception as e:
        console.print(f"[red]Error calling AI: {e}[/red]")
        return None

def rename_media(image_path: Path, dry_run: bool = False) -> Optional[Dict]:
    """Rename a single media file."""
    if not image_path.exists() or image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return None
    
    new_name = generate_filename(image_path)
    if not new_name:
        return None
    
    new_name_ext = new_name + image_path.suffix.lower()
    new_path = image_path.parent / new_name_ext
    
    counter = 1
    while new_path.exists() and new_path != image_path:
        new_path = image_path.parent / f"{new_name}_{counter}{image_path.suffix.lower()}"
        counter += 1
        
    if new_path == image_path:
        return None

    console.print(f"  [cyan]To:[/cyan]   {new_path.name}")
    
    if dry_run:
        return {
            "original_path": str(image_path.absolute()),
            "new_path": str(new_path.absolute()),
            "dry_run": True
        }
    
    try:
        image_path.rename(new_path)
        return {
            "original_path": str(image_path.absolute()),
            "new_path": str(new_path.absolute()),
            "dry_run": False
        }
    except Exception as e:
        console.print(f"[red]Error renaming: {e}[/red]")
        return None

def process_directory_media(directory: Path, recursive: bool = False, dry_run: bool = False):
    """Process all media in a directory."""
    pattern = "**/*" if recursive else "*"
    files = list(directory.glob(pattern))
    media_files = [f for f in files if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
    
    if not media_files:
        console.print("[yellow]No media files found[/yellow]")
        return

    log_data = load_rename_log()
    operation_id = datetime.now().isoformat()
    operations = []

    with Progress(console=console) as progress:
        task = progress.add_task("Processing...", total=len(media_files))
        for f in media_files:
            op = rename_media(f, dry_run=dry_run)
            if op:
                op["operation_id"] = operation_id
                operations.append(op)
            progress.update(task, advance=1)
    
    if operations and not dry_run:
        log_data["renames"].extend(operations)
        log_data["last_operation"] = operation_id
        save_rename_log(log_data)
        console.print(f"[green]Renamed {len(operations)} files[/green]")

def undo_renames(undo_all: bool = False):
    """Undo rename operations."""
    log_data = load_rename_log()
    if not log_data["renames"]:
        console.print("[yellow]No renames to undo[/yellow]")
        return

    renames = reversed(log_data["renames"]) if undo_all else reversed([
        r for r in log_data["renames"] 
        if r.get("operation_id") == log_data["last_operation"]
    ])
    
    count = 0
    for op in renames:
        if op.get("dry_run"): continue
        orig = Path(op["original_path"])
        curr = Path(op["new_path"])
        if curr.exists():
            curr.rename(orig)
            count += 1
            if not undo_all:
                log_data["renames"].remove(op)
    
    save_rename_log(log_data)
    console.print(f"[green]Restored {count} files[/green]")
