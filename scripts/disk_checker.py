#!/usr/bin/env python
"""
Debug script to identify what's causing disk quota exceeded errors.
Run this to understand where the problem is coming from.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
import psutil
import shutil

def run_command(cmd):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return f"Error: {e}", ""

def check_disk_quotas():
    """Check disk quotas using various methods."""
    print("=== DISK QUOTA ANALYSIS ===\n")
    
    # Method 1: Basic df command
    print("1. Basic disk usage (df -h):")
    stdout, stderr = run_command("df -h")
    print(stdout)
    if stderr:
        print(f"Stderr: {stderr}")
    print()
    
    # Method 2: Check user quotas
    print("2. User disk quotas (quota -u):")
    stdout, stderr = run_command("quota -u")
    print(stdout if stdout else "No quota information available")
    if stderr:
        print(f"Stderr: {stderr}")
    print()
    
    # Method 3: Check group quotas
    print("3. Group disk quotas (quota -g):")
    stdout, stderr = run_command("quota -g")
    print(stdout if stdout else "No group quota information available")
    if stderr:
        print(f"Stderr: {stderr}")
    print()
    
    # Method 4: Check specific filesystem quotas
    print("4. Filesystem quotas (repquota -a):")
    stdout, stderr = run_command("repquota -a")
    print(stdout if stdout else "No filesystem quota information available")
    if stderr:
        print(f"Stderr: {stderr}")
    print()

def check_filesystem_details():
    """Check detailed filesystem information."""
    print("=== FILESYSTEM DETAILS ===\n")
    
    # Get current working directory info
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    
    # Check mount points
    print("\n5. Mount points and filesystem types:")
    stdout, stderr = run_command("mount | grep -E '(scratch|leonardo|home|tmp)'")
    print(stdout if stdout else "No special mounts found")
    print()
    
    # Check inode usage (sometimes the issue is inodes, not space)
    print("6. Inode usage (df -i):")
    stdout, stderr = run_command("df -i")
    print(stdout)
    print()
    
    # Check for specific filesystem quotas
    print("7. Filesystem type for current directory:")
    stdout, stderr = run_command(f"stat -f -c %T {cwd}")
    print(f"Filesystem type: {stdout}")
    print()

def check_temp_directories():
    """Check temporary directory locations and usage."""
    print("=== TEMPORARY DIRECTORY ANALYSIS ===\n")
    
    temp_locations = [
        os.environ.get('TMPDIR', '/tmp'),
        '/tmp',
        '/var/tmp',
        tempfile.gettempdir(),
        os.path.expanduser('~/.cache'),
        os.path.expanduser('~/.cache/huggingface'),
    ]
    
    print("8. Temporary directory locations and usage:")
    for i, temp_dir in enumerate(temp_locations, 8):
        if os.path.exists(temp_dir):
            try:
                # Get size of directory
                stdout, stderr = run_command(f"du -sh {temp_dir}")
                print(f"  {temp_dir}: {stdout.split()[0] if stdout else 'Unknown size'}")
                
                # Check if we can write to it
                test_file = os.path.join(temp_dir, f"test_write_{os.getpid()}")
                try:
                    with open(test_file, 'w') as f:
                        f.write("test")
                    os.remove(test_file)
                    print(f"    ✓ Writable")
                except Exception as e:
                    print(f"    ✗ Not writable: {e}")
                    
            except Exception as e:
                print(f"  {temp_dir}: Error checking - {e}")
        else:
            print(f"  {temp_dir}: Does not exist")
    print()

def check_huggingface_cache():
    """Check HuggingFace cache locations and sizes."""
    print("=== HUGGINGFACE CACHE ANALYSIS ===\n")
    
    # Common HF cache locations
    hf_cache_locations = [
        os.environ.get('HF_DATASETS_CACHE'),
        os.environ.get('HF_HOME'),
        os.environ.get('TRANSFORMERS_CACHE'),
        os.path.expanduser('~/.cache/huggingface/datasets'),
        os.path.expanduser('~/.cache/huggingface/transformers'),
        os.path.expanduser('~/.cache/huggingface/hub'),
    ]
    
    print("9. HuggingFace cache locations:")
    for cache_dir in hf_cache_locations:
        if cache_dir and os.path.exists(cache_dir):
            try:
                stdout, stderr = run_command(f"du -sh {cache_dir}")
                size = stdout.split()[0] if stdout else 'Unknown'
                print(f"  {cache_dir}: {size}")
                
                # Count number of files
                stdout, stderr = run_command(f"find {cache_dir} -type f | wc -l")
                file_count = stdout if stdout else 'Unknown'
                print(f"    Files: {file_count}")
                
            except Exception as e:
                print(f"  {cache_dir}: Error - {e}")
        elif cache_dir:
            print(f"  {cache_dir}: Does not exist")
    print()

def test_file_creation():
    """Test file creation in various locations."""
    print("=== FILE CREATION TESTS ===\n")
    
    test_locations = [
        '.',  # Current directory
        os.environ.get('TMPDIR', '/tmp'),
        '/tmp',
        os.path.expanduser('~'),
        os.path.expanduser('~/scratch') if os.path.exists(os.path.expanduser('~/scratch')) else None,
    ]
    
    print("10. Testing file creation in various locations:")
    for location in test_locations:
        if location is None:
            continue
            
        try:
            test_file = os.path.join(location, f"quota_test_{os.getpid()}.tmp")
            
            # Try to create a small file
            with open(test_file, 'w') as f:
                f.write("test data for quota checking")
            
            # Try to create a larger file (1MB)
            with open(test_file + "_large", 'wb') as f:
                f.write(b'0' * (1024 * 1024))
            
            # Clean up
            os.remove(test_file)
            os.remove(test_file + "_large")
            
            print(f"  ✓ {location}: Can create files")
            
        except OSError as e:
            if "Disk quota exceeded" in str(e):
                print(f"  ✗ {location}: QUOTA EXCEEDED - {e}")
            elif "No space left on device" in str(e):
                print(f"  ✗ {location}: NO SPACE LEFT - {e}")
            else:
                print(f"  ✗ {location}: Other error - {e}")
        except Exception as e:
            print(f"  ✗ {location}: Unexpected error - {e}")
    print()

def check_process_info():
    """Check current process and system information."""
    print("=== PROCESS AND SYSTEM INFO ===\n")
    
    print("11. Current process info:")
    print(f"  PID: {os.getpid()}")
    print(f"  User: {os.getenv('USER', 'unknown')}")
    print(f"  UID: {os.getuid()}")
    print(f"  GID: {os.getgid()}")
    print(f"  Working directory: {os.getcwd()}")
    print()
    
    print("12. Environment variables related to storage:")
    storage_vars = ['TMPDIR', 'TMP', 'TEMP', 'HF_DATASETS_CACHE', 'HF_HOME', 'TRANSFORMERS_CACHE']
    for var in storage_vars:
        value = os.environ.get(var)
        print(f"  {var}: {value if value else 'Not set'}")
    print()

def check_leonardo_specific():
    """Check Leonardo HPC specific quota information."""
    print("=== LEONARDO HPC SPECIFIC CHECKS ===\n")
    
    print("13. Leonardo HPC quota commands:")
    
    # Leonardo specific quota commands
    leonardo_commands = [
        "saldo -b",  # Check storage quota
        "cineca-quota",  # Alternative quota command
        "lfs quota -u $USER /leonardo_scratch",  # Lustre filesystem quota
        "lfs quota -g $(id -gn) /leonardo_scratch",  # Group quota
    ]
    
    for cmd in leonardo_commands:
        print(f"\nRunning: {cmd}")
        stdout, stderr = run_command(cmd)
        if stdout:
            print(f"Output: {stdout}")
        if stderr:
            print(f"Error: {stderr}")
    
    # Check Leonardo scratch directories
    leonardo_dirs = [
        '/leonardo_scratch',
        '/leonardo_work',
        os.path.expanduser('~/scratch'),
    ]
    
    print("\n14. Leonardo directory status:")
    for dir_path in leonardo_dirs:
        if os.path.exists(dir_path):
            try:
                stdout, stderr = run_command(f"ls -la {dir_path}")
                print(f"\n{dir_path} contents (first 10 lines):")
                print('\n'.join(stdout.split('\n')[:10]))
            except Exception as e:
                print(f"{dir_path}: Error - {e}")
        else:
            print(f"{dir_path}: Does not exist")

def main():
    """Run all diagnostic checks."""
    print("DISK QUOTA DEBUGGING TOOL")
    print("=" * 50)
    print()
    
    try:
        check_disk_quotas()
        check_filesystem_details()
        check_temp_directories()
        check_huggingface_cache()
        test_file_creation()
        check_process_info()
        check_leonardo_specific()
        
        print("\n" + "=" * 50)
        print("DEBUGGING COMPLETE")
        print("\nKey things to check:")
        print("1. Look for any location showing 'QUOTA EXCEEDED'")
        print("2. Check if any cache directories are using excessive space")
        print("3. Note which filesystem type you're using")
        print("4. Check if it's a user quota vs. group quota issue")
        print("5. Look for inode limitations (100% inode usage)")
        
    except Exception as e:
        print(f"Error running diagnostics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()