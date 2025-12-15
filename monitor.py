#!/usr/bin/env python3
"""
smart_monitor.py

Intelligent monitor based on actual pipeline timing data
Optimized for your specific pipeline characteristics
"""

import time
import os
import re
from pathlib import Path
from collections import defaultdict

# BASED ON YOUR ACTUAL DATA
STAGE_TIMINGS = {
    'init': 0,
    'ingest': 0,
    'image_filter': 7,
    'pre_processing': 9,
    'database_builder': 3,
    'matcher': 3,
    'sparse_reconstruction': 56,
    'sparse_evaluation': 0,
    'dense_reconstruction': 3182,  # MAJOR BOTTLENECK
    'dense_evaluation': 4,
    'gen_mesh': 14,
    'mesh_postprocess': 12,
    'texture_mapping': 0,
    'mesh_evaluation': 1,
    'evaluation_aggregator': 0,
    'visualization': 3
}

# Pipeline order
PIPELINE_ORDER = [
    'init',
    'ingest',
    'image_filter',
    'pre_processing',
    'database_builder',
    'matcher',
    'sparse_reconstruction',
    'sparse_evaluation',
    'dense_reconstruction',  # MAJOR BOTTLENECK
    'dense_evaluation',
    'gen_mesh',
    'mesh_postprocess',
    'texture_mapping',
    'mesh_evaluation',
    'evaluation_aggregator',
    'visualization'
]

def find_log_directory():
    """Find the logs directory"""
    print("📁 Looking for log files...")
    
    # Try current directory
    cwd = Path.cwd()
    if (cwd / "logs").exists():
        return cwd / "logs"
    
    # Try parent directory (common when running from project root)
    parent = cwd.parent
    if (parent / "logs").exists():
        return parent / "logs"
    
    # Ask user
    while True:
        log_path = input("\nEnter logs directory path: ").strip().strip('"')
        log_dir = Path(log_path)
        if log_dir.exists():
            return log_dir
        print(f"Directory not found: {log_path}")

def detect_stage_completion(log_file):
    """Check if a stage log indicates completion"""
    if not log_file.exists():
        return False, 0
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            if not lines:
                return False, 0
            
            # Check last line for completion
            last_line = lines[-1].strip().lower()
            completion_keywords = ['complete', 'finished', 'done', 'saved', 
                                 'exported', 'time:', 'elapsed:', 'duration:']
            
            for keyword in completion_keywords:
                if keyword in last_line:
                    return True, len(lines)
            
            # Check if file has grown significantly (stage is running)
            return False, len(lines)
            
    except Exception as e:
        return False, 0

def extract_progress_from_stage(log_file, stage_name):
    """Extract progress percentage from stage log"""
    if not log_file.exists():
        return 0
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # Different patterns for different stages
            if stage_name == 'database_builder':
                # Look for "X/Y" patterns
                matches = re.findall(r'(\d+)\s*/\s*(\d+)', content)
                if matches:
                    current, total = map(int, matches[-1])
                    return (current / total) * 100 if total > 0 else 0
            
            elif stage_name == 'dense_reconstruction':
                # Look for percentage patterns
                matches = re.findall(r'(\d+)%', content)
                if matches:
                    return float(matches[-1])
                # Look for "X/Y" patterns
                matches = re.findall(r'(\d+)\s*/\s*(\d+)', content)
                if matches:
                    current, total = map(int, matches[-1])
                    return (current / total) * 100 if total > 0 else 0
            
            elif stage_name == 'sparse_reconstruction':
                # Look for percentage or X/Y patterns
                matches = re.findall(r'(\d+)%', content)
                if matches:
                    return float(matches[-1])
                matches = re.findall(r'(\d+)\s*/\s*(\d+)', content)
                if matches:
                    current, total = map(int, matches[-1])
                    return (current / total) * 100 if total > 0 else 0
            
    except:
        pass
    
    return 0

def calculate_remaining_time(completed_stages, current_stage, current_progress):
    """Calculate remaining time based on actual timings"""
    total_estimated = sum(STAGE_TIMINGS.values())
    
    # Time for completed stages
    completed_time = 0
    for stage in completed_stages:
        completed_time += STAGE_TIMINGS.get(stage, 0)
    
    # Time for current stage (adjusted by progress)
    if current_stage and current_stage in STAGE_TIMINGS:
        stage_total = STAGE_TIMINGS[current_stage]
        stage_remaining = stage_total * (1 - current_progress / 100)
        completed_time += (stage_total * current_progress / 100)
    else:
        stage_remaining = 0
    
    # Time for remaining stages
    remaining_stages = []
    found_current = current_stage is None
    for stage in PIPELINE_ORDER:
        if stage == current_stage:
            found_current = True
            continue
        elif stage in completed_stages:
            continue
        elif found_current:
            remaining_stages.append(stage)
    
    remaining_time = sum(STAGE_TIMINGS.get(stage, 0) for stage in remaining_stages)
    total_remaining = stage_remaining + remaining_time
    
    return total_remaining, len(remaining_stages)

def format_time(seconds):
    """Format seconds into human readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def display_progress(completed_stages, current_stage, stage_progress, 
                    stage_line_count, total_lines_read):
    """Display the progress monitor"""
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                    MARK-2 PIPELINE MONITOR                   ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    
    # Overall progress
    completed_count = len(completed_stages)
    total_stages = len(PIPELINE_ORDER)
    overall_progress = (completed_count / total_stages) * 100
    
    # Calculate remaining time
    remaining_time, stages_remaining = calculate_remaining_time(
        completed_stages, current_stage, stage_progress
    )
    
    print(f"║ 📊 Overall: {completed_count}/{total_stages} stages")
    print(f"║ ⏱️  Estimated remaining: {format_time(remaining_time)}")
    print(f"║ 📈 Stages remaining: {stages_remaining}")
    print("╠══════════════════════════════════════════════════════════════╣")
    
    # Current stage info
    if current_stage:
        stage_time = STAGE_TIMINGS.get(current_stage, 0)
        print(f"║ 🔄 CURRENT: {current_stage}")
        print(f"║    Expected time: {format_time(stage_time)}")
        
        if stage_progress > 0:
            # Stage progress bar
            bar_width = 40
            filled = int(bar_width * stage_progress / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"║    [{bar}] {stage_progress:.1f}%")
        
        if stage_line_count > 0:
            print(f"║    Log lines: {stage_line_count}")
    else:
        print("║ ⏳ Waiting for pipeline to start...")
    
    print("╠══════════════════════════════════════════════════════════════╣")
    
    # Stage completion status
    print("║ 📋 Stage Status:")
    
    # Group stages
    for i in range(0, len(PIPELINE_ORDER), 4):
        batch = PIPELINE_ORDER[i:i+4]
        line = "║    "
        for stage in batch:
            if stage in completed_stages:
                line += f"✓ {stage:20} "
            elif stage == current_stage:
                line += f"▶ {stage:20} "
            else:
                line += f"□ {stage:20} "
        print(line)
    
    print("╠══════════════════════════════════════════════════════════════╣")
    
    # Bottleneck warnings
    if current_stage in ['dense_reconstruction', 'sparse_reconstruction']:
        print("║ ⚠️  NOTE: This is a long-running stage")
        print("║    (Expected: " + format_time(STAGE_TIMINGS.get(current_stage, 0)) + ")")
    
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"   Total log lines processed: {total_lines_read}")
    print()

def monitor_pipeline():
    """Main monitoring function"""
    log_dir = find_log_directory()
    
    print(f"📁 Monitoring: {log_dir}")
    print("Press Ctrl+C to stop monitoring")
    print("-" * 60)
    
    # Tracking variables
    completed_stages = []
    current_stage = None
    stage_progress = 0
    stage_line_counts = defaultdict(int)
    total_lines_read = 0
    last_update = 0
    
    # Stage log files
    stage_logs = {}
    for stage in PIPELINE_ORDER:
        log_file = log_dir / f"{stage}.log"
        stage_logs[stage] = log_file
    
    try:
        while True:
            current_time = time.time()
            
            # Update display at most once per second
            if current_time - last_update >= 1.0:
                # Check each stage
                for stage in PIPELINE_ORDER:
                    if stage in completed_stages:
                        continue
                    
                    log_file = stage_logs[stage]
                    is_complete, line_count = detect_stage_completion(log_file)
                    
                    # Track line count
                    if line_count > stage_line_counts[stage]:
                        stage_line_counts[stage] = line_count
                        total_lines_read += (line_count - stage_line_counts[stage])
                    
                    if is_complete and stage not in completed_stages:
                        # Stage completed
                        completed_stages.append(stage)
                        current_stage = None
                        stage_progress = 0
                        print(f"\n✅ Stage completed: {stage}")
                    elif line_count > 0 and stage not in completed_stages:
                        # Stage is running
                        if current_stage != stage:
                            current_stage = stage
                            stage_progress = 0
                            print(f"\n🔄 Stage started: {stage}")
                        
                        # Extract progress for this stage
                        stage_progress = extract_progress_from_stage(log_file, stage)
                
                # Check if all stages are complete
                if len(completed_stages) == len(PIPELINE_ORDER):
                    display_progress(completed_stages, None, 0, 0, total_lines_read)
                    print("\n" + "="*60)
                    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
                    print("="*60)
                    break
                
                # Display progress
                current_line_count = stage_line_counts.get(current_stage, 0)
                display_progress(completed_stages, current_stage, 
                               stage_progress, current_line_count, total_lines_read)
                
                last_update = current_time
            
            # Sleep to save CPU
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
    
    except Exception as e:
        print(f"\nError: {e}")
    
    finally:
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    monitor_pipeline()