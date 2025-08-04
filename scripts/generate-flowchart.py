#!/usr/bin/env python3
"""
Generate Mermaid flowcharts from agent activity logs.
"""

import yaml
import argparse
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone


def load_activity_log(file_path: str) -> Dict[str, Any]:
    """Load the agent activity log YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def format_duration(seconds: int) -> str:
    """Format duration in human readable format."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds//60}m {seconds%60}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def generate_agent_flowchart(
    activities: List[Dict[str, Any]], 
    condensed: bool = False,
    last_n: Optional[int] = None
) -> str:
    """Generate a Mermaid flowchart from activities."""
    
    if last_n:
        activities = activities[-last_n:]
    
    # Build flowchart
    mermaid = ["graph TD"]
    
    if condensed:
        # Condensed view - group by agent type and show major changes only
        agent_summary = {}
        for activity in activities:
            agent = activity['agent']
            if agent not in agent_summary:
                agent_summary[agent] = {
                    'total_duration': 0,
                    'total_lines_added': 0,
                    'total_lines_removed': 0,
                    'total_files': set(),
                    'activities': 0,
                    'phases': set()
                }
            
            summary = agent_summary[agent]
            summary['total_duration'] += activity.get('duration', 0)
            summary['total_lines_added'] += activity.get('lines_added', 0)
            summary['total_lines_removed'] += activity.get('lines_removed', 0)
            summary['total_files'].update(activity.get('artifacts', []))
            summary['activities'] += 1
            if activity.get('phase'):
                summary['phases'].add(activity['phase'])
        
        # Create condensed nodes
        for i, (agent, summary) in enumerate(agent_summary.items()):
            duration_str = format_duration(summary['total_duration'])
            lines_net = summary['total_lines_added'] - summary['total_lines_removed']
            lines_str = f"+{lines_net}" if lines_net >= 0 else str(lines_net)
            phases_str = ",".join(sorted(summary['phases']))
            
            node_label = f"{agent.title()}<br/>{duration_str}<br/>{lines_str} lines<br/>{len(summary['total_files'])} files<br/>Phase {phases_str}"
            mermaid.append(f'    {agent}["{node_label}"]')
            
            if i > 0:
                prev_agent = list(agent_summary.keys())[i-1]
                mermaid.append(f'    {prev_agent} --> {agent}')
    
    else:
        # Verbose view - show each activity
        prev_node = None
        
        for i, activity in enumerate(activities):
            agent = activity['agent']
            task = activity.get('task', f"task-{i}")
            duration_str = format_duration(activity.get('duration', 0))
            lines_added = activity.get('lines_added', 0)
            lines_removed = activity.get('lines_removed', 0)
            lines_net = lines_added - lines_removed
            lines_str = f"+{lines_net}" if lines_net >= 0 else str(lines_net)
            
            # Create unique node ID
            node_id = f"{agent}_{i}"
            
            # Create node label
            instruction = activity.get('instruction', 'Unknown task')[:30]
            if len(activity.get('instruction', '')) > 30:
                instruction += "..."
            
            node_label = f"{agent.title()}<br/>{instruction}<br/>{duration_str}, {lines_str} lines"
            
            mermaid.append(f'    {node_id}["{node_label}"]')
            
            # Add connection from previous node
            if prev_node:
                mermaid.append(f'    {prev_node} --> {node_id}')
            
            prev_node = node_id
    
    return "\n".join(mermaid)


def display_summary_stats(activities: List[Dict[str, Any]]) -> None:
    """Display summary statistics for the flowchart."""
    if not activities:
        print("No activities to display")
        return
    
    total_duration = sum(a.get('duration', 0) for a in activities)
    total_lines_added = sum(a.get('lines_added', 0) for a in activities)
    total_lines_removed = sum(a.get('lines_removed', 0) for a in activities)
    total_files = len(set(f for a in activities for f in a.get('artifacts', [])))
    
    agents = list(set(a['agent'] for a in activities))
    phases = list(set(a.get('phase') for a in activities if a.get('phase')))
    
    print(f"\n[SUMMARY] Flowchart Summary")
    print(f"Activities: {len(activities)}")
    print(f"Agents: {', '.join(sorted(agents))}")
    print(f"Phases: {', '.join(sorted(phases))}")
    print(f"Total duration: {format_duration(total_duration)}")
    print(f"Lines added: +{total_lines_added}")
    print(f"Lines removed: -{total_lines_removed}")
    print(f"Net lines: {total_lines_added - total_lines_removed:+d}")
    print(f"Files touched: {total_files}")


def main():
    parser = argparse.ArgumentParser(description='Generate Mermaid flowcharts from agent activities')
    parser.add_argument('--log-file', default='logs/agent-activity.yaml', 
                       help='Path to activity log file')
    parser.add_argument('--condensed', action='store_true', 
                       help='Generate condensed view (group by agent)')
    parser.add_argument('--last-n', type=int, 
                       help='Show only last N activities')
    parser.add_argument('--output', help='Output file for mermaid chart')
    parser.add_argument('--stats', action='store_true', 
                       help='Display summary statistics')
    
    args = parser.parse_args()
    
    # Load activity log
    try:
        data = load_activity_log(args.log_file)
        activities = data.get('activities', [])
    except FileNotFoundError:
        print(f"[ERROR] Log file not found: {args.log_file}")
        return
    except Exception as e:
        print(f"[ERROR] Error loading log file: {e}")
        return
    
    if not activities:
        print("No activities found in log file")
        return
    
    # Filter activities if requested
    if args.last_n:
        activities = activities[-args.last_n:]
        print(f"Showing last {len(activities)} activities")
    
    # Display stats if requested
    if args.stats:
        display_summary_stats(activities)
        return
    
    # Generate flowchart
    mermaid_chart = generate_agent_flowchart(activities, args.condensed, args.last_n)
    
    # Output flowchart
    if args.output:
        with open(args.output, 'w') as f:
            f.write(mermaid_chart)
        print(f"[OK] Flowchart saved to {args.output}")
    else:
        print("\n[FLOWCHART] Mermaid Flowchart:")
        print("=" * 50)
        print(mermaid_chart)
        print("=" * 50)
    
    # Display summary
    display_summary_stats(activities)


if __name__ == '__main__':
    main()