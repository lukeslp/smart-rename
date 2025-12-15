# Smart Rename

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AI-powered intelligent file renaming tools for academic PDFs and media files.

## Features

- **PDF Renamer**: Extract metadata from academic papers using PDF parsing, Unpaywall (DOI), and AI analysis.
  - Renames to: `author_year_keywords.pdf`
- **Media Renamer**: Uses vision models to analyze images and videos and generate descriptive filenames.
  - Renames to: `descriptive_content_5_words.jpg`
- **Undo Support**: Full undo capability for media renames

## Installation

```bash
pip install smart-rename
```

Or install from source:

```bash
git clone https://github.com/lukeslp/smart-rename.git
cd smart-rename
pip install -e .
```

## Configuration

### AI Provider

The tool supports multiple AI providers:

**OpenAI** (default):
```bash
export OPENAI_API_KEY="sk-..."
```

**Ollama** (local, no API key needed):
```bash
export AI_PROVIDER="ollama"
# Requires Ollama running locally: ollama serve
# For vision: ollama pull llava
```

Or create `~/.config/smart-rename/config.json`:

```json
{
  "ai_provider": "openai",
  "openai_api_key": "sk-...",
  "unpaywall_email": "your@email.com"
}
```

### Unpaywall (Optional)

For better academic PDF metadata via DOI lookup:
```bash
export UNPAYWALL_EMAIL="your@email.com"
```

## AI Providers

| Provider | Text Model | Vision Model | API Key Required |
|----------|------------|--------------|------------------|
| OpenAI | gpt-4o-mini | gpt-4o | Yes |
| Ollama | llama3.2 | llava | No (local) |

Set `AI_PROVIDER` environment variable to switch providers.

## Usage

### Rename PDFs

```bash
# Dry run (preview changes)
smart-rename pdf --dir ~/Documents/Papers

# Execute renames
smart-rename pdf --dir ~/Documents/Papers --execute

# Skip confirmation prompts
smart-rename pdf --dir ~/Documents/Papers --execute --skip-confirm
```

### Rename Media (Images/Videos)

```bash
# Dry run
smart-rename media --dir ~/Photos

# Execute renames
smart-rename media --dir ~/Photos --execute

# Recursive (include subdirectories)
smart-rename media --dir ~/Photos --recursive --execute
```

### Undo Media Renames

```bash
# Undo last operation
smart-rename undo

# Undo all recorded operations
smart-rename undo --all
```

## How It Works

### PDF Renaming Strategy

1. **PDF Metadata**: Extract author, title, year from embedded PDF metadata
2. **DOI + Unpaywall**: Find DOI in PDF text, query Unpaywall API for metadata
3. **AI Analysis**: Send text excerpt to AI for analysis and filename generation

### Media Renaming

1. Extract representative frame from videos (or use image directly)
2. Send to vision model for content analysis
3. Generate descriptive 5-word filename

## Development

```bash
# Clone the repo
git clone https://github.com/lukeslp/smart-rename.git
cd smart-rename

# Install in development mode
pip install -e .

# Run linting
ruff check src/
black src/
```

## Requirements

- Python 3.8+
- pymupdf (PDF parsing)
- openai (AI API client)
- rich (terminal output)
- requests (HTTP client)
- opencv-python-headless (video frame extraction)
- pillow (image processing)
- python-dotenv (environment variables)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Luke Steuber**
- Website: [lukesteuber.com](https://lukesteuber.com)
- GitHub: [lukeslp](https://github.com/lukeslp)
