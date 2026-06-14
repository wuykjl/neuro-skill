"""
Auto-start wrapper — one import, no manual server lifecycle.

Three modes, pick one:

1. LazySingleton (simplest — first query builds, stays hot in-process)
2. BackgroundServer (subprocess — survives Python restarts, for external consumers)
3. SessionStart hook (Claude Code native — auto-starts when CC launches)

Usage examples in neuro_skill/__init__.py:
    from neuro_skill import query  # auto-starts on first call
    results = query("Go build error")
"""

import os, json, time, atexit, subprocess, threading, logging
from pathlib import Path

logger = logging.getLogger("neuro_skill.autostart")


# ── Mode 1: Lazy Singleton (first query builds, lives until process exits) ──

class _LazyRouter:
    """One import, zero config. First .query() call builds the index.
    Subsequent calls are hot (5ms). Process exit cleans up."""

    def __init__(self, directories=None):
        self._dirs = directories or _default_dirs()
        self._router = None
        self._lock = threading.Lock()

    def _ensure_built(self):
        if self._router is None:
            with self._lock:
                if self._router is None:  # double-check
                    from neuro_skill import SkillRouter
                    self._router = SkillRouter()
                    self._router.build(self._dirs)

    def query(self, text, top_k=5, method="hybrid"):
        self._ensure_built()
        return self._router.query(text, top_k=top_k, method=method)

    @property
    def built(self):
        return self._router is not None

    def rebuild(self):
        self._router = None
        self._ensure_built()


# ── Mode 2: Background HTTP Server (survives Python restarts) ──

_BG_PID_FILE = Path.home() / ".neuro-skill-server.pid"
_BG_PORT = 8765


def _is_server_running(port=_BG_PORT):
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.3)
        s.connect(("127.0.0.1", port))
        s.close()
        return True
    except (ConnectionRefusedError, socket.timeout, OSError):
        return False


def start_background(directories=None):
    """Start a background HTTP server. Idempotent — skips if already running."""
    if _is_server_running():
        return {"status": "already_running", "port": _BG_PORT}

    dirs = directories or _default_dirs()
    script = str(Path(__file__).parent / "server.py")

    proc = subprocess.Popen(
        ["python", "-c", f"""
import sys; sys.path.insert(0, {str(Path(__file__).parent.parent)!r})
from neuro_skill.server import _SkillServer
from http.server import HTTPServer
svr = _SkillServer({dirs!r})
httpd = HTTPServer(("127.0.0.1", {_BG_PORT}), __import__('neuro_skill.server', fromlist=['_QueryHandler'])._QueryHandler)
print("READY")
httpd.serve_forever()
"""],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for READY signal
    try:
        proc.stdout.readline(timeout=10)
    except Exception as e:
        logger.warning("Background server startup timed out: %s", e)

    _BG_PID_FILE.write_text(str(proc.pid))
    atexit.register(lambda: stop_background())

    return {"status": "started", "port": _BG_PORT, "pid": proc.pid}


def stop_background():
    """Stop the background HTTP server."""
    if _BG_PID_FILE.exists():
        try:
            pid = int(_BG_PID_FILE.read_text().strip())
            import signal
            os.kill(pid, signal.SIGTERM)
        except Exception as e:
            logger.debug("Failed to stop background server: %s", e)
        _BG_PID_FILE.unlink(missing_ok=True)


def query_via_server(text, top_k=5, method="hybrid"):
    """Query via the background HTTP server (5ms, no import overhead)."""
    import urllib.request, urllib.parse
    params = urllib.parse.urlencode({"q": text, "k": top_k, "method": method})
    url = f"http://127.0.0.1:{_BG_PORT}/query?{params}"
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        # Fallback: start server and retry
        start_background()
        with urllib.request.urlopen(url, timeout=2) as resp:
            return json.loads(resp.read().decode("utf-8"))


# ── Helpers ──

def _default_dirs():
    return [
        os.path.expanduser("~/.claude/skills/"),
        os.path.expanduser("~/.claude/agents/"),
        os.path.expanduser("~/.claude/.agents/skills/"),
    ]


# ── Module-level lazy singleton ──
_lazy = None


def _get_lazy():
    global _lazy
    if _lazy is None:
        _lazy = _LazyRouter()
    return _lazy


# Public API
def query(text, top_k=5, method="hybrid"):
    """Auto-starting lazy query. First call builds index (~500ms), then 5ms."""
    return _get_lazy().query(text, top_k=top_k, method=method)
