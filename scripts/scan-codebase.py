#!/usr/bin/env python3
"""
Codebase Scanner for Cleanup Agent
Scans entire codebase and updates cleanup tracking file with file status.
"""

import yaml
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import argparse
import subprocess
from pathlib import Path


def load_cleanup_tracker(file_path: str) -> Dict[str, Any]:
    """Load the cleanup tracker YAML file."""
    if not os.path.exists(file_path):
        # Create initial structure if file doesn't exist
        return {
            'metadata': {
                'last_run': {
                    'files_edited': 0,
                    'lines_added': 0,
                    'lines_deleted': 0,
                    'timestamp': None
                },
                'running_average_last_5': {
                    'files_edited': 0.0,
                    'lines_added': 0.0,
                    'lines_deleted': 0.0,
                    'cycles_included': 0
                },
                'last_5_cleanup_runs': [],
                'last_full_scan': None,
                'next_scan_due': None,
                'scan_interval_hours': 24,
                'thresholds': {
                    'max_lines_per_file': 500,
                    'max_complexity_score': 10,
                    'min_test_coverage': 80
                }
            },
            'files': {}
        }
    
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def save_cleanup_tracker(data: Dict[str, Any], file_path: str) -> None:
    """Save the cleanup tracker to YAML file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, indent=2)


def get_all_source_files(root_dir: str = '.') -> List[str]:
    """Get all source code files in the project."""
    source_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.rb', '.go', '.rs', '.php'}
    ignore_dirs = {'.git', '__pycache__', 'node_modules', '.pytest_cache', 'venv', 'env', '.venv'}
    ignore_files = {'__pycache__', '.pyc', '.pyo', '.pyd'}
    
    source_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # Remove ignored directories from traversal
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, root_dir)
            
            # Skip ignored files
            if any(ignore in file for ignore in ignore_files):
                continue
                
            # Check if it's a source file
            if any(file.endswith(ext) for ext in source_extensions):
                source_files.append(rel_path.replace('\\', '/'))  # Normalize path separators
    
    return sorted(source_files)


def get_file_stats(file_path: str) -> Dict[str, Any]:
    """Get basic statistics about a file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Basic metrics
        total_lines = len(lines)
        non_empty_lines = len([line for line in lines if line.strip()])
        comment_lines = len([line for line in lines if line.strip().startswith('#')])
        
        # Get file modification time
        stat = os.stat(file_path)
        last_modified = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
        
        # Simple complexity score (very basic)
        complexity_indicators = ['if ', 'for ', 'while ', 'try:', 'except:', 'elif ', 'def ', 'class ']
        complexity_score = sum(line.count(indicator) for line in lines for indicator in complexity_indicators)
        
        return {
            'size_lines': total_lines,
            'non_empty_lines': non_empty_lines,
            'comment_lines': comment_lines,
            'complexity_score': min(complexity_score, 20),  # Cap at 20
            'last_modified': last_modified,
        }
    
    except Exception as e:
        return {
            'size_lines': 0,
            'non_empty_lines': 0,
            'comment_lines': 0,
            'complexity_score': 0,
            'last_modified': datetime.now(timezone.utc).isoformat(),
            'error': str(e)
        }


def update_cleanup_stats(tracker: Dict[str, Any], files_edited: int, lines_added: int, lines_deleted: int) -> None:
    """Update cleanup statistics with new run data."""
    metadata = tracker['metadata']
    
    # Update last_run
    metadata['last_run'] = {
        'files_edited': files_edited,
        'lines_added': lines_added,
        'lines_deleted': lines_deleted,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    
    # Add to last_5_cleanup_runs history
    run_data = {
        'files_edited': files_edited,
        'lines_added': lines_added,
        'lines_deleted': lines_deleted,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    
    if 'last_5_cleanup_runs' not in metadata:
        metadata['last_5_cleanup_runs'] = []
    
    metadata['last_5_cleanup_runs'].append(run_data)
    if len(metadata['last_5_cleanup_runs']) > 5:
        metadata['last_5_cleanup_runs'] = metadata['last_5_cleanup_runs'][-5:]
    
    # Calculate running averages
    valid_runs = [run for run in metadata['last_5_cleanup_runs'] if run['timestamp'] is not None]
    if valid_runs:
        metadata['running_average_last_5'] = {
            'files_edited': sum(r['files_edited'] for r in valid_runs) / len(valid_runs),
            'lines_added': sum(r['lines_added'] for r in valid_runs) / len(valid_runs),
            'lines_deleted': sum(r['lines_deleted'] for r in valid_runs) / len(valid_runs),
            'cycles_included': len(valid_runs)
        }
    
    # Update scan timestamps
    metadata['last_full_scan'] = datetime.now(timezone.utc).isoformat()
    next_scan = datetime.now(timezone.utc).timestamp() + (metadata['scan_interval_hours'] * 3600)
    metadata['next_scan_due'] = datetime.fromtimestamp(next_scan, timezone.utc).isoformat()


def scan_codebase(tracker_file: str, root_dir: str = '.') -> None:
    """Scan the entire codebase and update tracking file."""
    print("[SCAN] Scanning codebase for cleanup tracking...")
    
    # Load existing tracker
    tracker = load_cleanup_tracker(tracker_file)
    
    # Get all source files
    source_files = get_all_source_files(root_dir)
    print(f"Found {len(source_files)} source files")
    
    # Track changes
    files_added = 0
    files_updated = 0
    files_removed = 0
    
    # Update file tracking
    current_files = set(source_files)
    tracked_files = set(tracker['files'].keys())
    
    # Remove files that no longer exist
    for file_path in tracked_files - current_files:
        del tracker['files'][file_path]
        files_removed += 1
        print(f"[REMOVED]  Removed deleted file: {file_path}")
    
    # Add or update existing files
    for file_path in source_files:
        file_stats = get_file_stats(os.path.join(root_dir, file_path))
        
        if file_path in tracker['files']:
            # Update existing file
            existing = tracker['files'][file_path]
            # Only update if file was modified since last review
            if existing.get('last_modified') != file_stats['last_modified']:
                existing.update(file_stats)
                existing['status'] = 'pending'  # Mark for review
                files_updated += 1
        else:
            # Add new file
            tracker['files'][file_path] = {
                'last_reviewed': None,
                'status': 'pending',
                'lines_cleaned': 0,
                'notes': 'Newly discovered file, pending initial review',
                **file_stats
            }
            files_added += 1
    
    # Update metadata with scan results
    update_cleanup_stats(tracker, files_added + files_updated, 0, 0)
    
    # Save updated tracker
    save_cleanup_tracker(tracker, tracker_file)
    
    print(f"[OK] Scan complete!")
    print(f"   Added: {files_added} files")
    print(f"   Updated: {files_updated} files") 
    print(f"   Removed: {files_removed} files")
    print(f"   Total tracked: {len(tracker['files'])} files")


def display_cleanup_stats(tracker_file: str) -> None:
    """Display cleanup statistics."""
    tracker = load_cleanup_tracker(tracker_file)
    metadata = tracker['metadata']
    
    print("\n[CLEANUP] Cleanup Agent Statistics")
    print("=" * 40)
    
    # Last run stats
    if metadata['last_run']['timestamp']:
        print("[LAST-RUN] Last Run:")
        last = metadata['last_run']
        print(f"  Files edited: {last['files_edited']}")
        print(f"  Lines added: {last['lines_added']}")
        print(f"  Lines deleted: {last['lines_deleted']}")
        print(f"  Timestamp: {last['timestamp']}")
    
    # Running averages
    print("\n[AVERAGE] Running Average (Last 5 Cleanup Cycles):")
    avg = metadata['running_average_last_5']
    print(f"  Files edited: {avg['files_edited']:.1f}")
    print(f"  Lines added: {avg['lines_added']:.1f}")
    print(f"  Lines deleted: {avg['lines_deleted']:.1f}")
    print(f"  Cycles included: {avg['cycles_included']}")
    
    # File status summary
    files = tracker['files']
    status_counts = {}
    for file_info in files.values():
        status = file_info['status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print(f"\n[FILES] File Status Summary (Total: {len(files)} files):")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    
    # Files needing attention
    needs_review = [path for path, info in files.items() if info['status'] in ['pending', 'needs_review']]
    if needs_review:
        print(f"\n[WARNING]  Files needing review ({len(needs_review)}):")
        for file_path in needs_review[:10]:  # Show first 10
            info = files[file_path]
            print(f"   {file_path} ({info['size_lines']} lines, complexity: {info['complexity_score']})")
        if len(needs_review) > 10:
            print(f"   ... and {len(needs_review) - 10} more")


def main():
    parser = argparse.ArgumentParser(description='Scan codebase for cleanup tracking')
    parser.add_argument('--tracker-file', default='suggestions/cleanup-tracker.yaml', 
                       help='Path to cleanup tracker file')
    parser.add_argument('--root-dir', default='.', help='Root directory to scan')
    parser.add_argument('--stats', action='store_true', help='Display statistics only')
    
    args = parser.parse_args()
    
    if args.stats:
        display_cleanup_stats(args.tracker_file)
        return
    
    scan_codebase(args.tracker_file, args.root_dir)


if __name__ == '__main__':
    main()