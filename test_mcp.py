# -*- coding: utf-8 -*-
"""Test MCP JSON-RPC call to verify server responds correctly."""
import subprocess, json, sys, time

server_cmd = [
    r"F:\claude-tools\akari-mem-mcp\.venv\Scripts\python.exe", r"F:\claude-tools\akari-mem-mcp\server.py"
]
env = {
    "HF_HOME": r"F:\models",
    "HF_ENDPOINT": "https://hf-mirror.com",
    "PYTHONPATH": r"F:\python-libs",
    "PATH": "",  # will inherit
    "SystemRoot": r"C:\Windows",
}
import os
full_env = os.environ.copy()
full_env.update(env)

# Start server process
proc = subprocess.Popen(
    server_cmd,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env=full_env,
)

def send_rpc(obj):
    line = json.dumps(obj) + "\n"
    proc.stdin.write(line.encode("utf-8"))
    proc.stdin.flush()

def read_response(timeout=30):
    import select
    start = time.time()
    data = b""
    while time.time() - start < timeout:
        byte = proc.stdout.read(1)
        if byte == b"\n" or byte == b"":
            break
        data += byte
    if data:
        return json.loads(data.decode("utf-8"))
    return None

# Step 1: Initialize
print("[1] Sending initialize...")
t0 = time.time()
send_rpc({
    "jsonrpc": "2.0",
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1.0"}
    },
    "id": 1,
})
resp = read_response()
print(f"    Response in {time.time()-t0:.1f}s: OK" if resp else "    FAILED")

# Step 2: Notify initialized
send_rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})

# Step 3: Call list_memories
print("[2] Calling list_memories...")
t0 = time.time()
send_rpc({
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {"name": "list_memories", "arguments": {"limit": 3}},
    "id": 2,
})
resp = read_response()
dt = time.time() - t0
if resp and "result" in resp:
    content = resp["result"]["content"][0]["text"]
    print(f"    Response in {dt:.1f}s:")
    for line in content.split("\n")[:5]:
        print(f"      {line}")
else:
    print(f"    FAILED after {dt:.1f}s: {resp}")

# Step 4: Call quick_search
print("[3] Calling quick_search('明日方舟')...")
t0 = time.time()
send_rpc({
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {"name": "quick_search", "arguments": {"query": "明日方舟"}},
    "id": 3,
})
resp = read_response()
dt = time.time() - t0
if resp and "result" in resp:
    content = resp["result"]["content"][0]["text"]
    print(f"    Response in {dt:.1f}s:")
    for line in content.split("\n")[:3]:
        print(f"      {line}")
else:
    print(f"    FAILED after {dt:.1f}s: {resp}")

# Step 5: Call get_stats
print("[4] Calling get_stats...")
t0 = time.time()
send_rpc({
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {"name": "get_stats", "arguments": {}},
    "id": 4,
})
resp = read_response()
dt = time.time() - t0
if resp and "result" in resp:
    content = resp["result"]["content"][0]["text"]
    print(f"    Response in {dt:.1f}s:")
    for line in content.split("\n"):
        print(f"      {line}")
else:
    print(f"    FAILED after {dt:.1f}s: {resp}")

proc.terminate()
print("\n[DONE] All tests completed.")
