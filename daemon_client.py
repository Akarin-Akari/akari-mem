# -*- coding: utf-8 -*-
"""
akari-mem daemon HTTP client (thin client side).

Responsibilities:
==================
1. Auto-spawn daemon (api_server.py) as Windows DETACHED_PROCESS if not running.
2. Provide convenient methods for each MCP tool's HTTP call.
3. Handle daemon restart / recovery (retry once on connection failure).

Used by server.py (thin MCP client). Pure stdlib (urllib + subprocess).
"""
import os
import sys
import json
import time
import socket
import logging
import subprocess
import urllib.request
import urllib.parse
import urllib.error
from typing import Any, Optional

logger = logging.getLogger("akari-mem.client")

# Windows process creation flags
_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_CREATE_NO_WINDOW = 0x08000000


def _get_daemon_python_exe() -> str:
    """
    返回启动 daemon 用的 Python 可执行文件路径。

    Windows: 优先用 pythonw.exe (GUI 子系统, 完全无控制台窗口)。
    python.exe 是控制台子系统, 即使配合 DETACHED_PROCESS 也可能短暂闪现
    或保留一个空 console。pythonw.exe 从根本上就不创建 console。
    """
    if sys.platform == "win32":
        exe_dir = os.path.dirname(sys.executable)
        pythonw = os.path.join(exe_dir, "pythonw.exe")
        if os.path.exists(pythonw):
            return pythonw
    return sys.executable

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 37800
DEFAULT_SPAWN_TIMEOUT = 60.0           # daemon ready 等待
DEFAULT_REQUEST_TIMEOUT = 300.0        # 深度搜索可能 30s+
DEFAULT_HEALTH_TIMEOUT = 2.0           # /health 超时


class DaemonError(Exception):
    """daemon 调用错误的统一异常。"""


class DaemonClient:
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        spawn_timeout: float = DEFAULT_SPAWN_TIMEOUT,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        daemon_script: Optional[str] = None,
    ):
        self.host = host
        self.port = int(port)
        self.base_url = f"http://{self.host}:{self.port}"
        self.spawn_timeout = float(spawn_timeout)
        self.request_timeout = float(request_timeout)
        # daemon 脚本绝对路径; 默认就是同目录的 api_server.py
        if daemon_script is None:
            daemon_script = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "api_server.py"
            )
        self.daemon_script = daemon_script
        self._spawned = False  # 标记本进程是否曾尝试 spawn

    # ──────────────────────────────────────────────────────────
    # 健康检查 + spawn
    # ──────────────────────────────────────────────────────────
    def is_running(self, timeout: float = DEFAULT_HEALTH_TIMEOUT) -> bool:
        """快速检查 daemon 是否响应 /health。"""
        # 先做 TCP 层 connect 检查 (省 HTTP 开销)
        try:
            with socket.create_connection((self.host, self.port), timeout=timeout):
                pass
        except (ConnectionRefusedError, OSError, socket.timeout):
            return False
        # TCP 通了再做 HTTP /health 确认
        try:
            self._get("/health", timeout=timeout)
            return True
        except Exception:
            return False

    def ensure_running(self, auto_spawn: bool = True) -> bool:
        """
        确保 daemon 在运行。如果不在，spawn 一个新的（DETACHED_PROCESS）。

        多个 thin client 同时 spawn 也安全：第一个抢占 port 成功的 daemon 存活，
        其他 daemon 进程 bind 失败立即退出（见 api_server.py main 函数）。
        """
        if self.is_running():
            return True
        if not auto_spawn:
            return False
        self._spawn()
        return self._wait_until_ready(self.spawn_timeout)

    def _spawn(self) -> None:
        """以 DETACHED_PROCESS + pythonw.exe 静默后台启动 daemon (Windows 无窗口)。"""
        if not os.path.exists(self.daemon_script):
            raise DaemonError(f"daemon script not found: {self.daemon_script}")

        python_exe = _get_daemon_python_exe()
        cmd = [python_exe, self.daemon_script, "--daemon", "--port", str(self.port)]
        logger.info(f"Spawning daemon: {' '.join(cmd)}")

        kwargs = dict(
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(self.daemon_script),
            close_fds=True,
        )
        if sys.platform == "win32":
            # 三重保险阻止任何窗口/console 弹出:
            # 1. pythonw.exe (GUI 子系统, 根本不要 console)
            # 2. DETACHED_PROCESS (不继承父进程 console)
            # 3. CREATE_NO_WINDOW (即使是 console 进程, 也别显示窗口)
            # 4. STARTUPINFO.SW_HIDE (双重保险, 隐藏任何可能弹出的窗口)
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = subprocess.SW_HIDE
            kwargs["startupinfo"] = startup
            kwargs["creationflags"] = (
                _DETACHED_PROCESS | _CREATE_NEW_PROCESS_GROUP | _CREATE_NO_WINDOW
            )
        else:
            # POSIX: setsid 让子进程独立 session
            kwargs["start_new_session"] = True

        try:
            subprocess.Popen(cmd, **kwargs)
            self._spawned = True
        except Exception as e:
            raise DaemonError(f"Failed to spawn daemon: {e}") from e

    def _wait_until_ready(self, timeout: float) -> bool:
        """轮询 /health 直到 daemon ready 或超时。"""
        deadline = time.monotonic() + timeout
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            if self.is_running(timeout=1.0):
                logger.info(f"Daemon ready after {attempt} attempts")
                return True
            time.sleep(0.5)
        logger.error(f"Daemon did not become ready within {timeout:.0f}s")
        return False

    # ──────────────────────────────────────────────────────────
    # HTTP 调用底层
    # ──────────────────────────────────────────────────────────
    def _url(self, path: str, params: Optional[dict] = None) -> str:
        url = self.base_url + path
        if params:
            # 跳过 None / 空字符串值
            cleaned = {k: v for k, v in params.items() if v is not None and v != ""}
            if cleaned:
                url += "?" + urllib.parse.urlencode(cleaned)
        return url

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        body: Optional[dict] = None,
        timeout: Optional[float] = None,
        _retry: bool = True,
    ) -> Any:
        """通用 HTTP 调用; 失败时一次 retry (重启 daemon)。"""
        timeout = timeout if timeout is not None else self.request_timeout
        url = self._url(path, params)
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                if not raw:
                    return None
                return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as e:
            # 4xx / 5xx: 读 body 拿错误细节
            try:
                detail = json.loads(e.read().decode("utf-8"))
            except Exception:
                detail = {"error": str(e)}
            raise DaemonError(f"HTTP {e.code} on {method} {path}: {detail}") from e
        except (urllib.error.URLError, ConnectionRefusedError, socket.timeout, OSError) as e:
            if _retry:
                logger.warning(f"Daemon call failed ({e}), attempting to respawn...")
                if self.ensure_running(auto_spawn=True):
                    return self._request(
                        method, path, params=params, body=body,
                        timeout=timeout, _retry=False,
                    )
            raise DaemonError(f"Daemon unreachable on {method} {path}: {e}") from e

    def _get(self, path: str, params: Optional[dict] = None, timeout: Optional[float] = None):
        return self._request("GET", path, params=params, timeout=timeout)

    def _post(self, path: str, body: Optional[dict] = None, timeout: Optional[float] = None):
        return self._request("POST", path, body=body, timeout=timeout)

    def _delete(self, path: str, params: Optional[dict] = None, timeout: Optional[float] = None):
        return self._request("DELETE", path, params=params, timeout=timeout)

    # ──────────────────────────────────────────────────────────
    # MCP tool 对应的高层 API
    # ──────────────────────────────────────────────────────────
    def health(self) -> dict:
        return self._get("/health", timeout=DEFAULT_HEALTH_TIMEOUT)

    def quick_search(
        self, query: str, limit: int = 5,
        project: str = "", tags: str = "",
    ) -> list:
        result = self._get("/search", params={
            "q": query, "limit": limit,
            "project": project, "tags": tags,
        })
        return result if isinstance(result, list) else []

    def deep_search(
        self, query: str, limit: int = 5,
        project: str = "", tags: str = "",
    ) -> list:
        result = self._get("/deep", params={
            "q": query, "limit": limit,
            "project": project, "tags": tags,
        }, timeout=self.request_timeout)
        return result if isinstance(result, list) else []

    def get_memory(self, memory_id: int) -> Optional[dict]:
        try:
            return self._get("/memory", params={"id": memory_id})
        except DaemonError as e:
            if "404" in str(e):
                return None
            raise

    def save_memory(
        self, title: str, text: str,
        tags: str = "", project: str = "", source: str = "mcp",
    ) -> dict:
        return self._post("/save", body={
            "title": title, "text": text,
            "tags": tags, "project": project, "source": source,
        })

    def list_memories(self, limit: int = 10) -> list:
        result = self._get("/list", params={"limit": limit})
        return result if isinstance(result, list) else []

    def delete_memory(self, memory_id: int) -> dict:
        return self._delete("/delete", params={"id": memory_id})

    def get_stats(self) -> dict:
        return self._get("/stats")

    def shutdown(self) -> dict:
        return self._post("/shutdown")


# ──────────────────────────────────────────────────────────────
# 模块级单例 (供 server.py 用)
# ──────────────────────────────────────────────────────────────
_client: Optional[DaemonClient] = None


def get_client(config: Optional[dict] = None) -> DaemonClient:
    """获取/构造全局 DaemonClient。配置可来自 config.json 的 daemon section。"""
    global _client
    if _client is not None:
        return _client
    cfg = (config or {}).get("daemon", {}) if isinstance(config, dict) else {}
    _client = DaemonClient(
        host=cfg.get("host", DEFAULT_HOST),
        port=int(cfg.get("port", DEFAULT_PORT)),
        spawn_timeout=float(cfg.get("spawn_timeout_seconds", DEFAULT_SPAWN_TIMEOUT)),
        request_timeout=float(cfg.get("request_timeout_seconds", DEFAULT_REQUEST_TIMEOUT)),
    )
    return _client
