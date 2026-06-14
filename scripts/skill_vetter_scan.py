"""Security scan: check recent skill changes for malicious patterns."""
import re, time, os
from pathlib import Path

now = time.time()
cutoff = now - 48 * 3600

dirs = [
    Path.home() / ".claude" / "skills",
    Path.home() / ".claude" / ".agents" / "skills",
    Path.home() / ".claude" / "agents",
]

recent = []
for d in dirs:
    if not d.exists():
        continue
    for f in d.rglob("*.md"):
        if f.stat().st_mtime > cutoff:
            recent.append(f)

print("Scanning {} files modified in last 48h...".format(len(recent)))

suspicious = []
for f in sorted(recent):
    text = f.read_text(encoding="utf-8", errors="ignore")
    issues = []

    # Malicious patterns
    if re.search(r"(?i)(curl|wget)\s+.*\|.*(bash|sh|python)", text):
        issues.append("SHELL_PIPE")
    if re.search(r"(?i)eval\s*\(|exec\s*\(|__import__\s*\(", text):
        issues.append("CODE_EXEC")
    if re.search(r"(?i)rm\s+-rf\s+/|sudo\s+rm", text):
        issues.append("DESTRUCTIVE_CMD")
    if re.search(r"(?i)(github\.com|gitlab\.com).*raw.*\.(sh|py|exe)", text):
        issues.append("REMOTE_EXEC")
    if re.search(r"[a-zA-Z0-9+/]{40,}={0,2}", text):
        issues.append("BASE64_BLOB")
    if re.search(r"(?i)(password|secret|token|api_key|apikey)\s*[:=]\s*[\"']?\w{8,}", text):
        issues.append("HARDCODED_SECRET")

    if issues:
        suspicious.append((f.relative_to(Path.home()), issues))

if suspicious:
    for path, iss in suspicious:
        print("  [!] {} : {}".format(path, ", ".join(iss)))
else:
    print("  [OK] No security issues found")

print("  Files: {} checked, {} suspicious".format(len(recent), len(suspicious)))
