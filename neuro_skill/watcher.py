"""
File watcher — auto-rebuild index when skills change.

Inspired by CodeGraph's `sales` module: OS-level file events
→ debounce 2s → incremental reindex → ready for next query.

Usage:
  from neuro_skill.watcher import watch

  router = SkillRouter(); router.build(dirs)
  watch(dirs, router)  # returns immediately, watcher runs in background thread

CLI:
  neuro-skill watch --dirs ./skills/ ./agents/
"""

from __future__ import annotations

import os, time, threading
from pathlib import Path
from typing import Callable


class _Debouncer:
    """Accumulate events, fire only after quiet period."""

    def __init__(self, callback: Callable[[], None], delay: float = 2.0):
        self._callback = callback
        self._delay = delay
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def trigger(self):
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._delay, self._callback)
            self._timer.start()

    def stop(self):
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


def _polling_watch(directories: list[str], on_change: Callable[[], None],
                   interval: float = 3.0):
    """Cross-platform polling fallback. Works everywhere, no dependencies."""
    dirs = [Path(d).expanduser().resolve() for d in directories]
    mtimes: dict[str, float] = {}

    # Initial snapshot
    for d in dirs:
        if not d.exists():
            continue
        for f in d.rglob("*"):
            if f.is_file():
                mtimes[str(f)] = f.stat().st_mtime

    while True:
        time.sleep(interval)
        changed = False
        for d in dirs:
            if not d.exists():
                continue
            for f in d.rglob("*"):
                if not f.is_file():
                    continue
                key = str(f)
                current = f.stat().st_mtime
                if key not in mtimes:
                    mtimes[key] = current
                    changed = True
                elif abs(current - mtimes[key]) > 0.1:
                    mtimes[key] = current
                    changed = True
        # Also detect deletions
        removed = [k for k in mtimes if not Path(k).exists()]
        if removed:
            for k in removed:
                del mtimes[k]
            changed = True
        if changed:
            on_change()


def watch(directories: list[str], router: "SkillRouter", debounce_s: float = 2.0):
    """
    Start background file watcher. Auto-rebuilds router when skills change.

    Args:
      directories: skill/agent directories to watch
      router:      SkillRouter instance to rebuild
      debounce_s:  quiet period before rebuild (default 2s)
    """
    # Check if router has build()
    if not hasattr(router, 'build'):
        raise TypeError("router must be a SkillRouter instance with .build() method")

    def rebuild():
        try:
            dirs = [str(Path(d).expanduser().resolve()) for d in directories]
            router.build(dirs)
        except Exception as e:
            pass  # Silently ignore — next query will surface the issue

    debouncer = _Debouncer(rebuild, delay=debounce_s)

    def on_change():
        debouncer.trigger()

    # Use polling — zero dependencies, works on all platforms
    t = threading.Thread(target=_polling_watch, args=(directories, on_change),
                         daemon=True, name="neuro-skill-watcher")
    t.start()

    return debouncer  # caller can .stop() later

# ── Watchdog-based fast path (optional, pip install watchdog) ──

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    class _WatchdogHandler(FileSystemEventHandler):
        def __init__(self, on_change):
            self.on_change = on_change
        def on_modified(self, event):
            if not event.is_directory:
                self.on_change()
        def on_created(self, event):
            if not event.is_directory:
                self.on_change()
        def on_deleted(self, event):
            if not event.is_directory:
                self.on_change()

    def watch_fast(directories: list[str], router: "SkillRouter",
                   debounce_s: float = 2.0):
        """Kernel-level file watching (needs: pip install watchdog)."""
        dirs = [str(Path(d).expanduser().resolve()) for d in directories]

        def rebuild():
            try:
                router.build(dirs)
            except Exception:
                pass

        debouncer = _Debouncer(rebuild, delay=debounce_s)
        handler = _WatchdogHandler(lambda: debouncer.trigger())
        observer = Observer()
        for d in dirs:
            if Path(d).exists():
                observer.schedule(handler, str(d), recursive=True)
        observer.start()
        return observer, debouncer  # observer.stop() later

    _HAS_WATCHDOG = True
except ImportError:
    _HAS_WATCHDOG = False
    def watch_fast(directories, router, debounce_s=2.0):
        raise ImportError("watchdog not installed. pip install watchdog")


# ── CLI ──

def cmd_watch():
    import argparse
    p = argparse.ArgumentParser(description="Watch skill dirs and auto-rebuild index")
    p.add_argument("--dirs", "-d", nargs="+",
                   default=[
                       os.path.expanduser("~/.claude/skills/"),
                       os.path.expanduser("~/.claude/agents/"),
                       os.path.expanduser("~/.claude/.agents/skills/"),
                   ],
                   help="Skill directories to watch")
    args = p.parse_args()

    from neuro_skill import SkillRouter
    router = SkillRouter()
    stats = router.build(args.dirs)
    print(f"Watching {stats['n_skills']} skills across {len(args.dirs)} directories")
    print("Auto-rebuild on file change. Ctrl+C to stop.")

    try:
        if _HAS_WATCHDOG:
            print("Using watchdog (kernel-level events)")
            observer, debouncer = watch_fast(args.dirs, router)
            observer.join()
        else:
            print("Using polling (3s interval, pip install watchdog for kernel events)")
            watch(args.dirs, router, debounce_s=3.0)
            while True:
                time.sleep(60)
    except KeyboardInterrupt:
        print("\nStopped.")
