#!/usr/bin/env python3
"""
smoke_test.py — reproduce the free-portal fallback proof.

Best run as a script so it can orchestrate start/stop reliably. Windows:
use subprocess to launch the gateway, then curl from Python.

Usage:
    python smoke_test.py

What it does:
    1. Starts the gateway on a port.
    2. Verifies healthz.
    3. Registers glass + owner keys.
    4. Chats as glass (local Ollama).
    5. Kills Ollama, chats as glass again (should fall back to free portal).
    6. Chats as owner (should 503).
    7. Teardown.
"""
import subprocess, sys, os, time, json, socket, urllib.request, urllib.error, signal, platform

GATEWAY_PORT = 11597
GATEWAY_DIR = os.path.dirname(os.path.abspath(__file__))
GATEWAY_SCRIPT = os.path.join(GATEWAY_DIR, "reference", "gateway.py")
HOME = os.path.expanduser("~")
PYTHON = sys.executable

def port_open(port, host="127.0.0.1"):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    try:
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        return False
    finally:
        s.close()

def curl_json(url, method="GET", data=None, headers=None, timeout=10):
    hdrs = headers or {}
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=body, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())
    except Exception as e:
        return None, {"error": str(e)}

def main():
    print("=== llm-toolkit smoke test ===\n")
    print("1) Starting gateway on :%d ..." % GATEWAY_PORT)
    log_f = open(os.path.join(GATEWAY_DIR, "smoke.log"), "wb")
    gw = subprocess.Popen(
        [PYTHON, GATEWAY_SCRIPT, "server", "--port", str(GATEWAY_PORT)],
        cwd=GATEWAY_DIR,
        stdout=log_f, stderr=subprocess.STDOUT,
    )
    # Wait for port
    for _ in range(20):
        if port_open(GATEWAY_PORT):
            break
        time.sleep(0.5)
    else:
        print("FAIL: gateway did not start on port %d" % GATEWAY_PORT)
        gw.terminate()
        return 1
    print("   gateway up")
    base = "http://127.0.0.1:%d" % GATEWAY_PORT

    # Healthz
    code, health = curl_json(base + "/healthz")
    print("2) healthz: %s\n   %s" % (code, json.dumps(health, indent=2)))

    # Register keys
    print("3) Registering keys...")
    code, glass = curl_json(base + "/v1/chat/completions", method="POST",
                            data={"messages": [{"role": "user", "content": "register"}], "model": "llama3.2:3b"},
                            headers={"Authorization": "Bearer placeholder"})
    # Key registration is done via the CLI, not the HTTP API. Use subprocess.
    gk = subprocess.check_output([PYTHON, GATEWAY_SCRIPT, "issue", "glass user", "glass"],
                                 cwd=GATEWAY_DIR).decode().strip().split()[0]
    ok = subprocess.check_output([PYTHON, GATEWAY_SCRIPT, "issue", "owner", "owner"],
                                 cwd=GATEWAY_DIR).decode().strip().split()[0]
    print("   glass key: %s" % gk)
    print("   owner key: %s" % ok)

    # Chat as glass (local Ollama)
    print("\n4) Chat as glass (local Ollama)...")
    code, resp = curl_json(base + "/v1/chat/completions", method="POST",
                           data={"model": "qwen3.8-uncensored:27b",
                                 "messages": [{"role": "user", "content": "Say exactly: SMOKE-NS"}],
                                 "max_tokens": 5, "temperature": 0},
                           headers={"Authorization": "Bearer " + gk},
                           timeout=30)
    print("   status: %s" % code)
    print("   %s" % json.dumps(resp, indent=2))

    # Kill Ollama
    print("\n5) Killing Ollama...")
    killed = False
    if platform.system() == "Windows":
        try:
            out = subprocess.check_output(["tasklist", "/FI", "IMAGENAME eq ollama.exe", "/FO", "CSV"],
                                          stderr=subprocess.DEVNULL).decode()
            for line in out.splitlines():
                if "ollama.exe" in line:
                    pid = line.split(",")[1].strip()
                    subprocess.call(["taskkill", "/F", "/PID", pid],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    killed = True
                    print("   killed ollama PID %s" % pid)
                    break
        except Exception as e:
            print("   taskkill failed: %s" % e)
    else:
        try:
            subprocess.call(["pkill", "-9", "ollama"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            killed = True
            print("   pkill ollama")
        except Exception as e:
            print("   pkill failed: %s" % e)
    if not killed:
        print("   WARN: could not kill Ollama — fallback test may not trigger")
    time.sleep(2)
    code, health = curl_json(base + "/healthz")
    print("   healthz after kill: ollama=%s" % health.get("ollama"))

    # Chat as glass (should fall back)
    print("\n6) Chat as glass (Ollama down — should fall back to free portal)...")
    code, resp = curl_json(base + "/v1/chat/completions", method="POST",
                           data={"model": "qwen3.8-uncensored:27b",
                                 "messages": [{"role": "user", "content": "Say exactly: SMOKE-FB"}],
                                 "max_tokens": 5, "temperature": 0},
                           headers={"Authorization": "Bearer " + gk},
                           timeout=45)
    print("   status: %s" % code)
    model = resp.get("model", "?")
    content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "?")
    print("   model: %s" % model)
    print("   content: %s" % content)
    if model == "nvidia/nemotron-3-super-120b-a12b:free" or model == "openai/gpt-5.6-luna":
        print("   ✓ free-portal fallback WORKED")
    else:
        print("   ✗ expected free-portal model, got: %s" % model)

    # Chat as owner (should 503)
    print("\n7) Chat as owner (Ollama down — should 503)...")
    code, resp = curl_json(base + "/v1/chat/completions", method="POST",
                           data={"model": "qwen3.8-uncensored:27b",
                                 "messages": [{"role": "user", "content": "Say exactly: SMOKE-OWNER"}],
                                 "max_tokens": 5, "temperature": 0},
                           headers={"Authorization": "Bearer " + ok},
                           timeout=30)
    print("   status: %s" % code)
    print("   %s" % json.dumps(resp, indent=2))
    if code == 503:
        print("   ✓ owner correctly got 503 (no free-portal fallback)")
    else:
        print("   ✗ expected 503 for owner")

    # Teardown
    print("\n8) Teardown...")
    gw.terminate()
    try:
        gw.wait(timeout=5)
    except Exception:
        gw.kill()
    print("   done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
