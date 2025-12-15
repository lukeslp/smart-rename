
import os
import sys
import json
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import Counter

try:
    import fitz  # PyMuPDF
    from openai import OpenAI
    import requests
    from rich.panel import Panel
    from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.prompt import Confirm
except ImportError as e:
    print(f"Missing required package: {e}")
    sys.exit(1)

from .utils import console, get_ai_config, get_unpaywall_email, LOG_DIR, CACHE_DIR

# Paths
RENAME_LOG_FILE = LOG_DIR / "pdf_rename_log.json"
UNPAYWALL_CACHE_FILE = CACHE_DIR / "unpaywall_cache.json"

# API Configuration
UNPAYWALL_API_BASE = "https://api.unpaywall.org/v2"

# Processing Configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
TEXT_EXTRACT_CHARS = 1000
DOI_PAGES_TO_CHECK = 2

def get_ai_client():
    """Get configured AI client (OpenAI or Ollama)."""
    config = get_ai_config(vision=False)
    if not config["api_key"]:
        return None, None
    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
    return client, config["model"]

# --- Helper Functions ---

def load_rename_log() -> Dict:
    """Load the rename log for undo functionality."""
    if RENAME_LOG_FILE.exists():
        try:
            with open(RENAME_LOG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            console.print("[yellow]Warning: Corrupt rename log, creating new one[/yellow]")
    return {"renames": [], "last_operation": None, "stats": {}}

def save_rename_log(log_data: Dict) -> None:
    """Save the rename log to file."""
    log_data["last_updated"] = datetime.now().isoformat()
    RENAME_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RENAME_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

def load_unpaywall_cache() -> Dict:
    """Load cached Unpaywall API responses."""
    if UNPAYWALL_CACHE_FILE.exists():
        try:
            with open(UNPAYWALL_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {}

def save_unpaywall_cache(cache: Dict) -> None:
    """Save Unpaywall API cache."""
    UNPAYWALL_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(UNPAYWALL_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def clean_filename(text: str, max_words: int = 5) -> str:
    """Clean and format text for use in filename."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\-]', '', text)
    text = re.sub(r'[\s\-]+', ' ', text)
    words = text.strip().split()[:max_words]
    return '_'.join(words)

def extract_year(text: str) -> Optional[str]:
    """Extract 4-digit year from text."""
    years = re.findall(r'\b(19\d{2}|20[0-3]\d)\b', text)
    if years:
        for year in years:
            year_int = int(year)
            if 1990 <= year_int <= 2030:
                return year
        return years[0]
    return None

def extract_doi(text: str) -> Optional[str]:
    """Extract DOI from text."""
    doi_patterns = [
        r'doi:\s*([10]\.\d{4,}/[^\s]+)',
        r'DOI:\s*([10]\.\d{4,}/[^\s]+)',
        r'https?://doi\.org/([10]\.\d{4,}/[^\s]+)',
        r'https?://dx\.doi\.org/([10]\.\d{4,}/[^\s]+)',
    ]
    for pattern in doi_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            doi = match.group(1)
            doi = re.sub(r'[,;.\s]+$', '', doi)
            return doi
    return None

def query_unpaywall(doi: str, cache: Dict) -> Optional[Dict]:
    """Query Unpaywall API for metadata by DOI."""
    if doi in cache:
        return cache[doi]

    email = get_unpaywall_email()
    try:
        url = f"{UNPAYWALL_API_BASE}/{doi}"
        params = {"email": email}
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            metadata = {
                'title': data.get('title'),
                'year': data.get('year'),
                'authors': data.get('z_authors', []),
                'source': 'unpaywall'
            }
            cache[doi] = metadata
            save_unpaywall_cache(cache)
            return metadata
        else:
            cache[doi] = None
            save_unpaywall_cache(cache)
            return None
    except Exception as e:
        console.print(f"[yellow]Unpaywall API error: {e}[/yellow]")
        return None

def extract_pdf_metadata(pdf_path: Path) -> Optional[Dict]:
    """Extract metadata from PDF file using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        if not metadata:
            return None

        title = metadata.get('title', '').strip()
        author = metadata.get('author', '').strip()
        subject = metadata.get('subject', '').strip()

        year = None
        for field in [metadata.get('creationDate', ''), metadata.get('modDate', ''), subject]:
            if field:
                year = extract_year(field)
                if year:
                    break

        if title or author:
            return {
                'title': title,
                'author': author,
                'year': year,
                'source': 'pdf_metadata'
            }
        return None
    except Exception as e:
        console.print(f"[yellow]Error extracting PDF metadata from {pdf_path.name}: {e}[/yellow]")
        return None

def extract_text_from_pdf(pdf_path: Path, max_chars: int = TEXT_EXTRACT_CHARS, max_pages: int = 2) -> Optional[str]:
    """Extract text from first few pages of PDF."""
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        for page_num in range(min(max_pages, len(doc))):
            page = doc[page_num]
            text_parts.append(page.get_text())
            if len(''.join(text_parts)) >= max_chars:
                break
        text = ''.join(text_parts)[:max_chars]
        return text.strip() if text.strip() else None
    except Exception:
        return None

def extract_doi_from_pdf(pdf_path: Path, max_pages: int = DOI_PAGES_TO_CHECK) -> Optional[str]:
    """Extract DOI from PDF text."""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(min(max_pages, len(doc))):
            page = doc[page_num]
            text = page.get_text()
            doi = extract_doi(text)
            if doi:
                return doi
        return None
    except Exception:
        return None

def generate_filename_with_ai(text: str, original_filename: str) -> Optional[str]:
    """Use AI to generate standardized filename from text excerpt."""
    client, model = get_ai_client()
    if not client:
        return None

    prompt = f"""Analyze this academic paper excerpt and generate a standardized filename.
Original filename: {original_filename}
Text excerpt:
{text[:1000]}
Generate a filename in this EXACT format: AuthorLastName_YYYY_Keyword_Keyword_Keyword
Rules:
- Use the first author's last name (lowercase)
- Include the publication year (YYYY)
- Add 3-5 descriptive keywords from the title/topic (lowercase, underscores)
- Use ONLY lowercase letters, numbers, and underscores
- Return ONLY the filename without .pdf extension
- Maximum 5 words after the year
"""
    try:
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=100,
                )
                filename = response.choices[0].message.content.strip()
                filename = filename.lower().strip('"\'`.')
                filename = re.sub(r'\.pdf$', '', filename)
                filename = re.sub(r'[^a-z0-9_]', '', filename)
                while '__' in filename:
                    filename = filename.replace('__', '_')
                filename = filename.strip('_')

                if len(filename.split('_')) >= 3:
                     # Basic validation, ensure it's not empty
                    return filename
                return filename if filename else None
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    raise
        return None
    except Exception as e:
        console.print(f"[bold red]Error generating filename with AI: {e}[/bold red]")
        return None

def extract_metadata_multi_strategy(pdf_path: Path, unpaywall_cache: Dict) -> Tuple[Optional[Dict], str]:
    """Extract metadata using multiple strategies."""
    # Strategy 1: PDF Metadata
    metadata = extract_pdf_metadata(pdf_path)
    if metadata and metadata.get('title'):
        return metadata, 'pdf_metadata'

    # Strategy 2: DOI + Unpaywall
    doi = extract_doi_from_pdf(pdf_path)
    if doi:
        unpaywall_data = query_unpaywall(doi, unpaywall_cache)
        if unpaywall_data:
            return unpaywall_data, 'unpaywall'

    # Strategy 3: AI Analysis
    text = extract_text_from_pdf(pdf_path)
    if text:
        year_from_text = extract_year(text)
        return {
            'text': text,
            'year': year_from_text,
            'source': 'ai_analysis'
        }, 'ai_analysis'

    return None, 'failed'

def metadata_to_filename(metadata: Dict, original_filename: str) -> Optional[str]:
    """Convert metadata to standardized filename."""
    if metadata['source'] == 'ai_analysis':
        return generate_filename_with_ai(metadata.get('text', ''), original_filename)

    year = metadata.get('year')
    title = metadata.get('title', '')
    author_last = None

    author = metadata.get('author', '')
    if author:
        parts = author.split()
        if parts:
            author_last = parts[-1]
    elif metadata.get('authors'):
        authors = metadata['authors']
        if authors and len(authors) > 0:
            first_author = authors[0]
            if isinstance(first_author, dict):
                author_last = first_author.get('family', first_author.get('given', ''))
            else:
                author_last = str(first_author).split()[-1] if first_author else None

    parts = []
    if author_last:
        parts.append(clean_filename(author_last, max_words=1))
    if year:
        parts.append(str(year))
    if title:
        title_keywords = clean_filename(title, max_words=5)
        if title_keywords:
            parts.append(title_keywords)

    if len(parts) >= 2:
        return '_'.join(parts)
    return None

def generate_unique_filename(base_path: Path, base_name: str, extension: str) -> Path:
    """Generate unique filename."""
    new_path = base_path / f"{base_name}{extension}"
    counter = 2
    while new_path.exists():
        new_path = base_path / f"{base_name}_{counter}{extension}"
        counter += 1
    return new_path

def rename_pdf(pdf_path: Path, unpaywall_cache: Dict, dry_run: bool = True) -> Optional[Dict]:
    """Rename a single PDF file."""
    if not pdf_path.exists() or pdf_path.suffix.lower() != '.pdf':
        return None

    console.print(f"\n[cyan]Processing:[/cyan] {pdf_path.name}")
    metadata, strategy = extract_metadata_multi_strategy(pdf_path, unpaywall_cache)

    if not metadata:
        console.print(f"[red]✗ Failed to extract metadata[/red]")
        return {'status': 'failed', 'original_path': str(pdf_path), 'strategy': strategy}

    console.print(f"[green]Strategy:[/green] {strategy}")
    new_name = metadata_to_filename(metadata, pdf_path.stem)

    if not new_name:
        console.print(f"[red]✗ Failed to generate filename[/red]")
        return {'status': 'failed', 'original_path': str(pdf_path), 'strategy': strategy}

    new_path = generate_unique_filename(pdf_path.parent, new_name, pdf_path.suffix.lower())
    if new_path == pdf_path:
        console.print(f"[yellow]Filename unchanged[/yellow]")
        return None

    console.print(f"[green]New name:[/green] {new_path.name}")

    if dry_run:
        console.print(f"[yellow][DRY RUN][/yellow]")
        return {
            'original_path': str(pdf_path.absolute()),
            'new_path': str(new_path.absolute()),
            'status': 'success',
            'dry_run': True,
            'strategy': strategy
        }

    try:
        pdf_path.rename(new_path)
        console.print(f"[bold green]✓ Successfully renamed![/bold green]")
        return {
            'original_path': str(pdf_path.absolute()),
            'new_path': str(new_path.absolute()),
            'status': 'success',
            'dry_run': False,
            'strategy': strategy
        }
    except Exception as e:
        console.print(f"[bold red]✗ Error: {e}[/bold red]")
        return {'status': 'error', 'error': str(e), 'original_path': str(pdf_path)}

def process_directory_pdfs(directory: Path, batch_size: int = 100, dry_run: bool = True, skip_confirm: bool = False):
    """Process all PDFs in a directory."""
    if not directory.exists():
        console.print(f"[bold red]Directory not found:[/bold red] {directory}")
        return

    pdf_files = sorted([f for f in directory.glob("*.pdf") if f.is_file()])
    if not pdf_files:
        console.print("[yellow]No PDF files found[/yellow]")
        return

    unpaywall_cache = load_unpaywall_cache()
    
    # Confirm
    if not dry_run and not skip_confirm:
        if not Confirm.ask(f"Process {len(pdf_files)} files (AI calls may incur cost)?"):
            return

    stats = Counter()
    with Progress(console=console) as progress:
        task = progress.add_task("Processing PDFs...", total=len(pdf_files))
        for pdf_file in pdf_files:
            op = rename_pdf(pdf_file, unpaywall_cache, dry_run=dry_run)
            if op:
                stats[op.get('strategy', 'unknown')] += 1
                if op.get('status') == 'success':
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
            progress.update(task, advance=1)
            time.sleep(0.1)

    save_unpaywall_cache(unpaywall_cache)
    console.print(f"\n[bold]Finished:[/bold] {stats['success']} renamed, {stats['failed']} failed.")
