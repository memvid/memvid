#!/usr/bin/env python3
# Save this as: docker/resource_monitor.py

import psutil
import json
import argparse
import sys
from datetime import datetime

def health_check():
    """Simple health check for container."""
    try:
        # Check if system is responsive
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        if cpu_percent > 95:
            print("WARNING: CPU usage very high")
            return False

        if memory.percent > 95:
            print("WARNING: Memory usage very high")
            return False

        print("✓ System healthy")
        return True

    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Resource monitor for H.265 encoding')
    parser.add_argument('--health-check', action='store_true', help='Run health check and exit')

    args = parser.parse_args()

    if args.health_check:
        healthy = health_check()
        sys.exit(0 if healthy else 1)
    else:
        # Simple stats output
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        stats = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / 1024**3,
            'memory_total_gb': memory.total / 1024**3
        }

        print(json.dumps(stats, indent=2))

if __name__ == '__main__':
    main()