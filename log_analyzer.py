#!/usr/bin/env python3
"""
log_analyzer.py

Analyzes all pipeline log files to extract:
1. Execution time for each stage
2. Progress patterns within each stage
3. File I/O patterns
4. Memory/time bottlenecks
5. Detailed statistics
"""

import re
import json
import time
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import statistics

def find_all_logs(project_path):
    """Find all log files in the project"""
    project_path = Path(project_path)
    
    # Look for logs directory
    if (project_path / "logs").exists():
        log_dir = project_path / "logs"
    elif project_path.name == "logs":
        log_dir = project_path
    else:
        log_dir = project_path
    
    # Find all .log files
    log_files = list(log_dir.glob("*.log"))
    
    # Also look in subdirectories
    for subdir in log_dir.rglob("*.log"):
        if subdir not in log_files:
            log_files.append(subdir)
    
    return log_files, log_dir

def extract_timestamps(line):
    """Extract timestamp from log line"""
    # Pattern: [2024-01-01 12:00:00]
    timestamp_pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]'
    match = re.search(timestamp_pattern, line)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
        except:
            return None
    return None

def analyze_stage_log(log_file):
    """Analyze a single stage log file"""
    analysis = {
        'file': str(log_file.name),
        'total_lines': 0,
        'start_time': None,
        'end_time': None,
        'duration': 0,
        'progress_indicators': [],
        'file_operations': [],
        'errors': [],
        'warnings': [],
        'key_events': [],
        'line_patterns': Counter()
    }
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            analysis['total_lines'] = len(lines)
            
            if not lines:
                return analysis
            
            # Get start and end times
            first_timestamp = extract_timestamps(lines[0])
            last_timestamp = extract_timestamps(lines[-1])
            
            if first_timestamp and last_timestamp:
                analysis['start_time'] = first_timestamp
                analysis['end_time'] = last_timestamp
                analysis['duration'] = (last_timestamp - first_timestamp).total_seconds()
            
            # Analyze each line
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Track line patterns (first 3 words)
                words = line.split()[:3]
                if words:
                    pattern = ' '.join(words)
                    analysis['line_patterns'][pattern] += 1
                
                # Detect progress indicators
                progress_patterns = [
                    r'(\d+)\s*/\s*(\d+)',  # X/Y format
                    r'progress.*?(\d+)%',  # X% progress
                    r'processed.*?(\d+)\s*(images|points|matches)',
                    r'complete.*?(\d+)%',
                    r'(\d+)\s*of\s*(\d+)',
                ]
                
                for pattern in progress_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        analysis['progress_indicators'].append({
                            'line': i,
                            'content': line,
                            'pattern': pattern
                        })
                        break
                
                # Detect file operations
                file_patterns = [
                    r'file.*?(saved|written|loaded|read|created):\s*(.+)',
                    r'writing.*?to\s*(.+)',
                    r'reading.*?from\s*(.+)',
                    r'saved.*?to\s*(.+)',
                    r'output.*?to\s*(.+)',
                ]
                
                for pattern in file_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        analysis['file_operations'].append({
                            'line': i,
                            'operation': match.group(1) if match.groups() > 1 else 'unknown',
                            'file': match.group(2) if match.groups() > 1 else match.group(1)
                        })
                        break
                
                # Detect errors and warnings
                if 'ERROR' in line or 'Error' in line:
                    analysis['errors'].append({
                        'line': i,
                        'content': line
                    })
                elif 'WARNING' in line or 'Warning' in line:
                    analysis['warnings'].append({
                        'line': i,
                        'content': line
                    })
                
                # Detect key events
                key_events = [
                    'starting', 'completed', 'finished', 'done',
                    'initialized', 'loaded', 'saved', 'exported',
                    'time:', 'elapsed:', 'duration:'
                ]
                
                for event in key_events:
                    if event.lower() in line.lower():
                        analysis['key_events'].append({
                            'line': i,
                            'event': event,
                            'content': line
                        })
                        break
            
            # Extract explicit duration if mentioned
            duration_patterns = [
                r'time.*?[:=]\s*([\d.]+)\s*s',
                r'elapsed.*?[:=]\s*([\d.]+)\s*s',
                r'duration.*?[:=]\s*([\d.]+)\s*s',
                r'took\s*([\d.]+)\s*seconds',
                r'in\s*([\d.]+)\s*s',
            ]
            
            for line in lines[-10:]:  # Check last 10 lines for duration
                for pattern in duration_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        analysis['explicit_duration'] = float(match.group(1))
                        break
            
    except Exception as e:
        analysis['error'] = str(e)
    
    return analysis

def analyze_pipeline_performance(log_analyses):
    """Analyze overall pipeline performance"""
    stages = {}
    total_duration = 0
    
    for analysis in log_analyses:
        stage_name = Path(analysis['file']).stem
        duration = analysis.get('explicit_duration') or analysis['duration'] or 0
        
        stages[stage_name] = {
            'duration': duration,
            'lines': analysis['total_lines'],
            'progress_points': len(analysis['progress_indicators']),
            'file_ops': len(analysis['file_operations']),
            'errors': len(analysis['errors']),
            'warnings': len(analysis['warnings']),
            'key_events': len(analysis['key_events'])
        }
        
        if duration > 0:
            total_duration += duration
    
    # Calculate percentages
    for stage_name, data in stages.items():
        if total_duration > 0:
            data['percentage'] = (data['duration'] / total_duration) * 100
        else:
            data['percentage'] = 0
    
    # Sort by duration
    sorted_stages = sorted(stages.items(), key=lambda x: x[1]['duration'], reverse=True)
    
    return {
        'total_duration': total_duration,
        'stages': dict(sorted_stages),
        'stage_count': len(stages)
    }

def detect_progress_patterns(log_analyses):
    """Detect common progress reporting patterns"""
    patterns = defaultdict(list)
    
    for analysis in log_analyses:
        stage_name = Path(analysis['file']).stem
        
        for progress in analysis['progress_indicators']:
            # Extract numeric progress
            numbers = re.findall(r'\d+', progress['content'])
            if numbers:
                patterns[stage_name].append({
                    'value': int(numbers[0]),
                    'max_value': int(numbers[1]) if len(numbers) > 1 else 100,
                    'line': progress['line']
                })
    
    return patterns

def generate_report(project_path, log_analyses, performance, progress_patterns):
    """Generate detailed analysis report"""
    report = {
        'project': str(project_path),
        'analysis_date': datetime.now().isoformat(),
        'log_files_analyzed': len(log_analyses),
        'performance_summary': performance,
        'stage_details': {},
        'recommendations': []
    }
    
    # Detailed stage information
    for analysis in log_analyses:
        stage_name = Path(analysis['file']).stem
        report['stage_details'][stage_name] = {
            'duration': analysis.get('explicit_duration') or analysis['duration'] or 0,
            'lines_of_logs': analysis['total_lines'],
            'progress_points': len(analysis['progress_indicators']),
            'file_operations': analysis['file_operations'][:5],  # First 5
            'common_line_patterns': analysis['line_patterns'].most_common(5),
            'errors': len(analysis['errors']),
            'key_events': [e['event'] for e in analysis['key_events'][:5]]
        }
    
    # Generate recommendations
    if performance['total_duration'] > 0:
        # Identify bottlenecks
        bottlenecks = [(name, data) for name, data in performance['stages'].items() 
                      if data['percentage'] > 20]  # Stages taking >20% of time
        
        for stage_name, data in bottlenecks:
            report['recommendations'].append(
                f"Bottleneck detected: {stage_name} takes {data['percentage']:.1f}% of total time "
                f"({data['duration']:.1f}s)"
            )
    
    # Check for stages with many errors
    for stage_name, details in report['stage_details'].items():
        if details['errors'] > 0:
            report['recommendations'].append(
                f"Stage {stage_name} has {details['errors']} errors - needs investigation"
            )
    
    # Check progress reporting
    for stage_name, patterns in progress_patterns.items():
        if len(patterns) >= 3:  # Good progress reporting
            report['recommendations'].append(
                f"Stage {stage_name} has good progress reporting ({len(patterns)} points)"
            )
        elif len(patterns) == 0:
            report['recommendations'].append(
                f"Stage {stage_name} has NO progress reporting - add progress indicators"
            )
    
    return report

def main():
    print("="*70)
    print("MARK-2 PIPELINE LOG ANALYZER")
    print("="*70)
    
    # Get project path
    project_path = input("\nEnter project folder path: ").strip().strip('"')
    
    # Find all logs
    log_files, log_dir = find_all_logs(project_path)
    
    if not log_files:
        print(f"\n❌ No log files found in: {log_dir}")
        return
    
    print(f"\n📁 Found {len(log_files)} log files in: {log_dir}")
    
    # Analyze each log file
    print("\n🔍 Analyzing log files...")
    log_analyses = []
    
    for log_file in log_files:
        print(f"  Analyzing: {log_file.name}")
        analysis = analyze_stage_log(log_file)
        log_analyses.append(analysis)
    
    # Analyze performance
    print("\n📊 Analyzing performance...")
    performance = analyze_pipeline_performance(log_analyses)
    progress_patterns = detect_progress_patterns(log_analyses)
    
    # Generate report
    print("\n📝 Generating report...")
    report = generate_report(project_path, log_analyses, performance, progress_patterns)
    
    # Display summary
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\n📅 Analysis date: {report['analysis_date']}")
    print(f"📁 Project: {Path(project_path).name}")
    print(f"📊 Log files analyzed: {report['log_files_analyzed']}")
    print(f"⏱️  Total pipeline duration: {performance['total_duration']:.1f}s")
    
    print("\n" + "-"*70)
    print("STAGE PERFORMANCE (sorted by duration)")
    print("-"*70)
    
    for stage_name, data in performance['stages'].items():
        print(f"{stage_name:25} {data['duration']:7.1f}s ({data['percentage']:5.1f}%) | "
              f"Progress points: {data['progress_points']:2d} | "
              f"Errors: {data['errors']}")
    
    print("\n" + "-"*70)
    print("PROGRESS REPORTING ANALYSIS")
    print("-"*70)
    
    for stage_name, patterns in progress_patterns.items():
        if patterns:
            # Calculate progress interval
            lines = [p['line'] for p in patterns]
            avg_interval = statistics.mean([lines[i+1]-lines[i] for i in range(len(lines)-1)]) if len(lines) > 1 else 0
            
            print(f"{stage_name:25} {len(patterns):2d} progress points | "
                  f"Avg interval: {avg_interval:.0f} lines")
    
    print("\n" + "-"*70)
    print("RECOMMENDATIONS")
    print("-"*70)
    
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"{i:2d}. {rec}")
    
    # Save detailed report
    output_file = log_dir / "pipeline_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n💾 Detailed report saved to: {output_file}")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Review the bottlenecks above")
    print("2. Check the detailed JSON report")
    print("3. Use this data to create an improved monitor.py")
    print("4. Add progress reporting to stages with none")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()