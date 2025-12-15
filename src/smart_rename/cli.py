
import argparse
import sys
from pathlib import Path
from rich.console import Console
from .pdf_renamer import process_directory_pdfs
from .media_renamer import process_directory_media, undo_renames

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Smart Rename - AI-powered file renaming tools")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # PDF Command
    pdf_parser = subparsers.add_parser("pdf", help="Rename academic PDFs")
    pdf_parser.add_argument("--dir", "-d", required=True, type=str, help="Directory containing PDFs")
    pdf_parser.add_argument("--execute", action="store_true", help="Execute renames (default is dry-run)")
    pdf_parser.add_argument("--skip-confirm", action="store_true", help="Skip confirmation prompts")

    # Media Command
    media_parser = subparsers.add_parser("media", help="Rename images and videos")
    media_parser.add_argument("--dir", "-d", required=True, type=str, help="Directory containing media")
    media_parser.add_argument("--recursive", "-r", action="store_true", help="Process recursively")
    media_parser.add_argument("--execute", action="store_true", help="Execute renames (default is dry-run)")

    # Undo Command
    undo_parser = subparsers.add_parser("undo", help="Undo last media rename")
    undo_parser.add_argument("--all", action="store_true", help="Undo all recorded operations")

    args = parser.parse_args()

    if args.command == "pdf":
        process_directory_pdfs(
            Path(args.dir), 
            dry_run=not args.execute, 
            skip_confirm=args.skip_confirm
        )
    elif args.command == "media":
        process_directory_media(
            Path(args.dir), 
            recursive=args.recursive, 
            dry_run=not args.execute
        )
    elif args.command == "undo":
        undo_renames(undo_all=args.all)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
