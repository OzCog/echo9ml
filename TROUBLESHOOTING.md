# Deep Tree Echo System - Troubleshooting Guide

This guide provides solutions to common issues when working with the Deep Tree Echo multi-language persona system.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Component-Specific Issues](#component-specific-issues)
3. [Communication Issues](#communication-issues)
4. [Performance Issues](#performance-issues)
5. [Error Messages](#error-messages)
6. [Diagnostic Tools](#diagnostic-tools)

---

## Installation Issues

### Missing Executables

**Problem:** Error message "Missing executable: ./deep-tree-echo" or "./hyper-echo"

**Solution:**
```bash
# Compile C++ orchestrator
g++ -std=c++17 -O2 -pthread -o deep-tree-echo deep-tree-echo.cpp

# Compile Go engine
go build -o hyper-echo hyper-echo.go

# Make them executable
chmod +x deep-tree-echo hyper-echo
```

### Build Failures

**Problem:** C++ compilation errors

**Solution:**
- Ensure you have a C++17-compatible compiler (GCC 7+, Clang 5+)
- Install required development tools:
  ```bash
  sudo apt-get install build-essential cmake
  ```

**Problem:** Go build errors

**Solution:**
- Install Go 1.16 or later
- Install required dependencies:
  ```bash
  go mod init
  go mod tidy
  go get github.com/gorilla/websocket
  ```

---

## Component-Specific Issues

### C++ Orchestrator Issues

**Problem:** Segmentation fault when running deep-tree-echo

**Solution:**
- This has been fixed in the current version
- Ensure you're using the latest compiled version
- If issue persists, check system memory:
  ```bash
  free -h
  ```

**Problem:** LLAMA inference not working

**Solution:**
- The system includes inference hooks but requires model files
- Check if node-llama-cpp directory exists
- Ensure model files are in the correct location

### Go Engine Issues

**Problem:** "Address already in use" error on port 8080

**Solution:**
```bash
# Find process using port 8080
lsof -i :8080

# Kill the process
kill <PID>

# Or configure a different port in deep_tree_echo_config.json
```

**Problem:** Go engine exits immediately

**Solution:**
- Check logs for error messages
- Ensure gorilla/websocket is installed:
  ```bash
  go get github.com/gorilla/websocket
  ```
- Run with verbose output to see startup messages

### Crystal Component Issues

**Problem:** Crystal executable not available

**Solution:**
- Crystal component requires Crystal runtime which may not be available
- The system is fully functional without the Crystal component
- C++ and Go components provide all core functionality

---

## Communication Issues

### WebSocket Connection Failures

**Problem:** Cannot connect to WebSocket server

**Solution:**
1. Verify Go engine is running:
   ```bash
   ps aux | grep hyper-echo
   ```

2. Check if port 8080 is accessible:
   ```bash
   netstat -tulpn | grep 8080
   ```

3. Test WebSocket connection:
   ```bash
   curl http://localhost:8080/status
   ```

### Component Communication Timeout

**Problem:** Components not communicating

**Solution:**
1. Check all components are running:
   ```bash
   python3 launch_unified_system.py --mode demo
   ```

2. Review configuration:
   ```bash
   python3 config_manager.py --show
   ```

3. Verify network settings allow localhost connections

---

## Performance Issues

### High CPU Usage

**Problem:** System using excessive CPU

**Solution:**
1. Monitor component usage:
   ```bash
   python3 monitoring_system.py --interval 5
   ```

2. Check for runaway processes:
   ```bash
   top -u $USER
   ```

3. Adjust worker count in configuration:
   ```json
   {
     "go": {
       "worker_count": 2
     }
   }
   ```

### High Memory Usage

**Problem:** System using too much memory

**Solution:**
1. Check memory usage:
   ```bash
   free -h
   ps aux --sort=-%mem | head -10
   ```

2. Adjust memory limits in configuration:
   ```json
   {
     "performance": {
       "memory_limit_mb": 2048
     }
   }
   ```

3. Reduce history size in monitoring system

### Slow Performance

**Problem:** System responding slowly

**Solution:**
1. Run benchmarks to identify bottleneck:
   ```bash
   python3 benchmark_suite.py --export
   ```

2. Check system resources:
   ```bash
   htop
   ```

3. Optimize compilation flags:
   ```bash
   g++ -std=c++17 -O3 -march=native -o deep-tree-echo deep-tree-echo.cpp
   ```

---

## Error Messages

### "Failed to start component"

**Cause:** Component executable missing or not executable

**Solution:**
```bash
# Check if files exist
ls -lh deep-tree-echo hyper-echo

# Make executable if needed
chmod +x deep-tree-echo hyper-echo

# Test individual components
./deep-tree-echo
./hyper-echo
```

### "Configuration validation failed"

**Cause:** Invalid configuration file

**Solution:**
```bash
# Create default configuration
python3 config_manager.py --create-default

# Validate configuration
python3 config_manager.py --validate
```

### "Process terminated unexpectedly"

**Cause:** Component crashed

**Solution:**
1. Check logs for error messages
2. Run component individually to see output:
   ```bash
   ./deep-tree-echo
   ./hyper-echo
   ```
3. Check system resources
4. Review recent code changes

---

## Diagnostic Tools

### System Status Check

```bash
# Run end-to-end tests
python3 test_end_to_end.py

# Check component status
python3 launch_unified_system.py --mode demo --verbose
```

### Log Analysis

```bash
# View monitoring logs
ls -lh logs/
tail -f logs/monitor_*.log

# Check for errors
grep ERROR logs/monitor_*.log
```

### Performance Profiling

```bash
# Run comprehensive benchmarks
python3 benchmark_suite.py --export

# Compare with baseline
python3 benchmark_suite.py --baseline benchmark_results.json

# Monitor real-time performance
python3 monitoring_system.py --interval 1 --export
```

### Configuration Debugging

```bash
# Show current configuration
python3 config_manager.py --show

# Validate configuration
python3 config_manager.py --validate

# Reset to defaults
mv deep_tree_echo_config.json deep_tree_echo_config.json.backup
python3 config_manager.py --create-default
```

---

## Advanced Troubleshooting

### Debugging C++ Issues

```bash
# Compile with debug symbols
g++ -std=c++17 -g -o deep-tree-echo deep-tree-echo.cpp

# Run with debugger
gdb ./deep-tree-echo
```

### Debugging Go Issues

```bash
# Run with race detector
go run -race hyper-echo.go

# Enable verbose logging
# Add to hyper-echo.go:
log.SetFlags(log.LstdFlags | log.Lshortfile)
```

### Network Debugging

```bash
# Monitor network connections
netstat -tulpn | grep -E "8080|5000"

# Test WebSocket with wscat
npm install -g wscat
wscat -c ws://localhost:8080/ws

# Check firewall settings
sudo iptables -L -n
```

---

## Getting Help

If you encounter issues not covered in this guide:

1. **Check Documentation:**
   - `DEEP_TREE_ECHO_IMPLEMENTATION_COMPLETE.md`
   - `README.md`
   - Source code comments

2. **Run Diagnostics:**
   ```bash
   python3 test_end_to_end.py
   python3 benchmark_suite.py
   python3 monitoring_system.py
   ```

3. **Collect Information:**
   - System specifications (OS, CPU, RAM)
   - Component versions
   - Error messages and logs
   - Steps to reproduce the issue

4. **Create Issue:**
   - Include diagnostic information
   - Attach log files
   - Describe expected vs actual behavior

---

## Quick Reference

### Start System
```bash
python3 launch_unified_system.py --mode demo
```

### Run Tests
```bash
python3 test_end_to_end.py
```

### Check Status
```bash
python3 monitoring_system.py --interval 5
```

### Performance Check
```bash
python3 benchmark_suite.py
```

### View Configuration
```bash
python3 config_manager.py --show
```

---

## Version Information

- Deep Tree Echo System: v1.0.0
- C++ Standard: C++17
- Go Version: 1.16+
- Python Version: 3.7+
- Last Updated: 2025-12-13
