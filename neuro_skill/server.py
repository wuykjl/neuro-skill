"""
Memory-resident server — keeps index hot for sub-6ms queries.

Usage:
  neuro-skill serve --dirs ./skills/ ./agents/ --port 8765

Then query via HTTP:
  curl "http://localhost:8765/query?q=Go+build+error&k=5&method=hybrid"

Or via stdin pipe:
  echo "Go build error" | nc localhost 8765

The index stays in memory. No disk I/O per query. No import overhead.
First cold build ~0.5s. Subsequent queries ~4-7ms.
"""

import json, time, os, sys, socket
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from neuro_skill import SkillRouter
from neuro_skill.features import BROAD, PRECISE


class _SkillServer:
    def __init__(self, directories: list[str]):
        t0 = time.time()
        self.router = SkillRouter()
        self.stats = self.router.build(directories)
        self.build_ms = (time.time() - t0) * 1000
        self.hits = 0
        self.total_ms = 0.0

    def query(self, text: str, top_k: int = 5, method: str = "hybrid") -> dict:
        t0 = time.time()
        results = self.router.query(text, top_k=top_k, method=method)
        elapsed_ms = (time.time() - t0) * 1000
        self.hits += 1
        self.total_ms += elapsed_ms
        return {
            "skills": [{"name": name, "score": round(score, 4)} for name, score in results],
            "query": text,
            "time_ms": round(elapsed_ms, 1),
            "n_total": self.router.skill_count,
        }

    def health(self) -> dict:
        return {
            "n_skills": self.stats["n_skills"],
            "n_features": self.stats["n_features"],
            "build_ms": round(self.build_ms, 1),
            "hits": self.hits,
            "avg_ms": round(self.total_ms / max(self.hits, 1), 1),
            "graph_density": self.stats.get("graph_density", 0),
        }


_shared: _SkillServer | None = None


def _get_server(dirs: list[str]) -> _SkillServer:
    global _shared
    if _shared is None:
        _shared = _SkillServer(dirs)
    return _shared


# ── HTTP handler ──

class _QueryHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # silence HTTP logs

    def _send_json(self, data: dict, code: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/health":
            self._send_json(_get_server([]).health())
            return

        if path == "/query":
            q = params.get("q", [""])[0]
            if not q:
                self._send_json({"error": "missing ?q= parameter"}, 400)
                return
            top_k = int(params.get("k", ["5"])[0])
            method = params.get("method", ["hybrid"])[0]
            result = _get_server([]).query(q, top_k=top_k, method=method)
            self._send_json(result)
            return

        if path == "/stats":
            self._send_json(_get_server([]).health())
            return

        self._send_json({"error": "not found", "paths": ["/query?q=...", "/health", "/stats"]}, 404)

    def do_POST(self):
        if self.path == "/query":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body) if body else {}
            except json.JSONDecodeError:
                data = {"q": body}
            q = data.get("q", data.get("query", ""))
            if not q:
                self._send_json({"error": "missing 'q' in POST body"}, 400)
                return
            top_k = data.get("k", data.get("top_k", 5))
            method = data.get("method", "hybrid")
            result = _get_server([]).query(q, top_k=top_k, method=method)
            self._send_json(result)
            return

        self._send_json({"error": "POST only at /query"}, 404)


# ── Socket mode (stdin, no HTTP overhead) ──

def _socket_serve(dirs: list[str], sock_path: str):
    """Unix domain socket or named pipe — for sub-2ms IPC."""
    server = _SkillServer(dirs)
    print(f"neuro-skill socket server: {server.stats['n_skills']} skills, "
          f"{server.build_ms:.0f}ms build")
    print(f"Listening on {sock_path}")

    # Remove stale socket
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(sock_path)
    sock.listen(5)

    while True:
        conn, _ = sock.accept()
        try:
            data = conn.recv(4096).decode("utf-8").strip()
            if not data:
                continue
            if data == "HEALTH":
                resp = json.dumps(server.health(), ensure_ascii=False)
            elif data == "SHUTDOWN":
                conn.sendall(b'{"status":"shutting_down"}')
                conn.close()
                break
            else:
                result = server.query(data)
                resp = json.dumps(result, ensure_ascii=False)
            conn.sendall(resp.encode("utf-8"))
        except Exception as e:
            conn.sendall(json.dumps({"error": str(e)}).encode("utf-8"))
        finally:
            conn.close()

    sock.close()
    os.unlink(sock_path)


# ── CLI ──

def cmd_serve():
    import argparse
    p = argparse.ArgumentParser(description="neuro-skill memory-resident server")
    p.add_argument("--dirs", "-d", nargs="+",
                   default=[
                       os.path.expanduser("~/.claude/skills/"),
                       os.path.expanduser("~/.claude/agents/"),
                       os.path.expanduser("~/.claude/.agents/skills/"),
                   ],
                   help="Skill directories")
    p.add_argument("--port", "-p", type=int, default=8765, help="HTTP port")
    p.add_argument("--socket", "-s", help="Unix socket path (faster than HTTP)")
    args = p.parse_args()

    if args.socket:
        _socket_serve(args.dirs, args.socket)
        return

    # HTTP mode
    server = _SkillServer(args.dirs)
    print(f"neuro-skill server: {server.stats['n_skills']} skills, "
          f"{server.build_ms:.0f}ms build")
    print(f"Listening on http://localhost:{args.port}")
    print(f"  curl 'http://localhost:{args.port}/query?q=Go+build+error&k=5'")
    print(f"  curl 'http://localhost:{args.port}/health'")

    httpd = HTTPServer(("127.0.0.1", args.port), _QueryHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\nShutting down. {server.hits} queries served, "
              f"avg {server.total_ms / max(server.hits, 1):.1f}ms/q")
        httpd.shutdown()
