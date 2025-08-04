#!/usr/bin/env python3
"""
Agent Activity Logger
Logs agent activities and maintains running statistics in YAML format.
"""

import yaml
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import argparse


def load_activity_log(file_path: str) -> Dict[str, Any]:
    """Load the agent activity log YAML file."""
    if not os.path.exists(file_path):
        # Create initial structure if file doesn't exist
        return {
            'metadata': {
                'project': 'Multi-Agent Project',
                'created': datetime.now(timezone.utc).isoformat(),
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'log_version': '1.0'
            },
            'statistics': {
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
                'last_5_runs': [],
                'totals': {
                    'total_agents_run': 0,
                    'total_duration': 0,
                    'total_lines_added': 0,
                    'total_lines_removed': 0,
                    'total_files_modified': 0,
                    'unique_files_touched': 0
                }
            },
            'activities': []
        }
    
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def save_activity_log(data: Dict[str, Any], file_path: str) -> None:
    """Save the agent activity log to YAML file."""
    data['metadata']['last_updated'] = datetime.now(timezone.utc).isoformat()
    
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, indent=2)


def update_running_averages(data: Dict[str, Any], new_run: Dict[str, Any]) -> None:
    """Update running averages with new run data."""
    stats = data['statistics']
    
    # Update last_run
    stats['last_run'] = {
        'files_edited': new_run.get('files_modified', 0),
        'lines_added': new_run.get('lines_added', 0),
        'lines_deleted': new_run.get('lines_removed', 0),
        'timestamp': new_run.get('timestamp')
    }
    
    # Add to last_5_runs history
    run_data = {
        'files_edited': new_run.get('files_modified', 0),
        'lines_added': new_run.get('lines_added', 0),
        'lines_deleted': new_run.get('lines_removed', 0),
        'timestamp': new_run.get('timestamp')
    }
    
    # Maintain only last 5 runs
    if 'last_5_runs' not in stats:
        stats['last_5_runs'] = []
    
    stats['last_5_runs'].append(run_data)
    if len(stats['last_5_runs']) > 5:
        stats['last_5_runs'] = stats['last_5_runs'][-5:]
    
    # Calculate running averages
    valid_runs = [run for run in stats['last_5_runs'] if run['timestamp'] is not None]
    if valid_runs:
        stats['running_average_last_5'] = {
            'files_edited': sum(r['files_edited'] for r in valid_runs) / len(valid_runs),
            'lines_added': sum(r['lines_added'] for r in valid_runs) / len(valid_runs),
            'lines_deleted': sum(r['lines_deleted'] for r in valid_runs) / len(valid_runs),
            'cycles_included': len(valid_runs)
        }


def update_totals(data: Dict[str, Any], new_activity: Dict[str, Any]) -> None:
    """Update total statistics."""
    totals = data['statistics']['totals']
    
    totals['total_agents_run'] += 1
    totals['total_duration'] += new_activity.get('duration', 0)
    totals['total_lines_added'] += new_activity.get('lines_added', 0)
    totals['total_lines_removed'] += new_activity.get('lines_removed', 0)
    totals['total_files_modified'] += new_activity.get('files_modified', 0)
    
    # Update unique files touched
    all_artifacts = set()
    for activity in data['activities']:
        all_artifacts.update(activity.get('artifacts', []))
    all_artifacts.update(new_activity.get('artifacts', []))
    totals['unique_files_touched'] = len(all_artifacts)


def log_agent_activity(
    log_file: str,
    agent: str,
    agent_type: str,
    instruction: str,
    result: str,
    duration: int,
    lines_added: int = 0,
    lines_removed: int = 0,
    files_modified: int = 0,
    artifacts: Optional[List[str]] = None,
    status: str = "success",
    triggered_by: Optional[str] = None,
    triggers: Optional[List[str]] = None,
    phase: Optional[str] = None,
    task: Optional[str] = None,
    percent_complete: Optional[float] = None,
    critical_path: bool = False,
    notes: Optional[str] = None
) -> None:
    """Log a new agent activity."""
    
    # Load existing log
    data = load_activity_log(log_file)
    
    # Create new activity entry
    activity = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'agent': agent,
        'agent_type': agent_type,
        'phase': phase,
        'task': task,
        'instruction': instruction,
        'result': result,
        'duration': duration,
        'lines_added': lines_added,
        'lines_removed': lines_removed,
        'files_modified': files_modified,
        'artifacts': artifacts or [],
        'status': status,
        'triggered_by': triggered_by,
        'triggers': triggers or [],
        'percent_complete': percent_complete,
        'critical_path': critical_path,
        'notes': notes
    }
    
    # Add to activities list
    data['activities'].append(activity)
    
    # Update statistics
    update_running_averages(data, activity)
    update_totals(data, activity)
    
    # Save updated log
    save_activity_log(data, log_file)
    
    print(f"[OK] Logged activity: {agent} - {instruction[:50]}...")
    print(f"   Duration: {duration}s, +{lines_added}/-{lines_removed} lines, {files_modified} files")


def display_stats(log_file: str) -> None:
    """Display current statistics."""
    data = load_activity_log(log_file)
    stats = data['statistics']
    
    print("\n[STATS] Agent Activity Statistics")
    print("=" * 40)
    
    print(f"Total agents run: {stats['totals']['total_agents_run']}")
    print(f"Total duration: {stats['totals']['total_duration']}s")
    print(f"Total lines added: {stats['totals']['total_lines_added']}")
    print(f"Total lines removed: {stats['totals']['total_lines_removed']}")
    print(f"Unique files touched: {stats['totals']['unique_files_touched']}")
    
    print("\n[AVERAGE] Running Average (Last 5 Runs):")
    avg = stats['running_average_last_5']
    print(f"  Files edited: {avg['files_edited']:.1f}")
    print(f"  Lines added: {avg['lines_added']:.1f}")
    print(f"  Lines deleted: {avg['lines_deleted']:.1f}")
    print(f"  Cycles included: {avg['cycles_included']}")
    
    if stats['last_run']['timestamp']:
        print(f"\n[LAST-RUN] Last Run:")
        last = stats['last_run']
        print(f"  Files edited: {last['files_edited']}")
        print(f"  Lines added: {last['lines_added']}")
        print(f"  Lines deleted: {last['lines_deleted']}")
        print(f"  Timestamp: {last['timestamp']}")


def main():
    parser = argparse.ArgumentParser(description='Log agent activities')
    parser.add_argument('--log-file', default='logs/agent-activity.yaml', 
                       help='Path to activity log file')
    parser.add_argument('--agent', help='Agent name')
    parser.add_argument('--agent-type', help='Agent type (implementation, validation, etc.)')
    parser.add_argument('--instruction', help='Instruction given to agent')
    parser.add_argument('--result', help='Result from agent')
    parser.add_argument('--duration', type=int, help='Duration in seconds')
    parser.add_argument('--lines-added', type=int, default=0, help='Lines added')
    parser.add_argument('--lines-removed', type=int, default=0, help='Lines removed')
    parser.add_argument('--files-modified', type=int, default=0, help='Files modified')
    parser.add_argument('--artifacts', nargs='*', help='List of artifacts created')
    parser.add_argument('--status', default='success', help='Status (success, failed, partial)')
    parser.add_argument('--triggered-by', help='What triggered this agent')
    parser.add_argument('--triggers', nargs='*', help='What this agent triggers')
    parser.add_argument('--phase', help='Development phase')
    parser.add_argument('--task', help='Specific task ID')
    parser.add_argument('--percent-complete', type=float, help='Percent of project complete')
    parser.add_argument('--critical-path', action='store_true', help='Is this on critical path')
    parser.add_argument('--notes', help='Additional notes')
    parser.add_argument('--stats', action='store_true', help='Display statistics only')
    
    args = parser.parse_args()
    
    if args.stats:
        display_stats(args.log_file)
        return
    
    if not all([args.agent, args.agent_type, args.instruction, args.result, args.duration]):
        print("[ERROR] Missing required arguments: --agent, --agent-type, --instruction, --result, --duration")
        parser.print_help()
        return
    
    log_agent_activity(
        log_file=args.log_file,
        agent=args.agent,
        agent_type=args.agent_type,
        instruction=args.instruction,
        result=args.result,
        duration=args.duration,
        lines_added=args.lines_added,
        lines_removed=args.lines_removed,
        files_modified=args.files_modified,
        artifacts=args.artifacts,
        status=args.status,
        triggered_by=args.triggered_by,
        triggers=args.triggers,
        phase=args.phase,
        task=args.task,
        percent_complete=args.percent_complete,
        critical_path=args.critical_path,
        notes=args.notes
    )


if __name__ == '__main__':
    main()