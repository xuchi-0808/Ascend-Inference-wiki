"""mkdocs hook: 取每页的作者（首次提交者）和最后修改人（最后提交者）。

通过 git log 取每页首尾提交的 SHA，调 GitHub API 拿 author 的 login/avatar/url，
注入 page_first_author / page_last_author 到页面上下文，供 content.html 渲染。
token 复用环境变量 MKDOCS_GIT_COMMITTERS_APIKEY（可选，提高 API 限流）。
"""
import os
import subprocess
import requests

REPO = "xuchi-0808/Ascend-Inference-wiki"

_session = requests.Session()
_token = os.environ.get("MKDOCS_GIT_COMMITTERS_APIKEY", "")
if _token:
    _session.headers["Authorization"] = f"token {_token}"

_cache = {}


def _commit_author(sha):
    if sha in _cache:
        return _cache[sha]
    try:
        r = _session.get(
            f"https://api.github.com/repos/{REPO}/commits/{sha}", timeout=10
        )
        data = r.json()
        author = data.get("author") or {}
        commit_author = data.get("commit", {}).get("author", {}) or {}
        info = {
            "name": author.get("login") or commit_author.get("name") or "unknown",
            "url": author.get("html_url") or "",
            "avatar": author.get("avatar_url") or "",
        }
    except Exception:
        info = {"name": "", "url": "", "avatar": ""}
    _cache[sha] = info
    return info


def _git(*args, cwd):
    return subprocess.check_output(
        ["git", *args], cwd=cwd, stderr=subprocess.DEVNULL
    ).decode().strip()


def on_page_context(context, page, config, **kwargs):
    try:
        src = page.file.abs_src_path
        repo_root = os.path.dirname(config["docs_dir"])
        rel = os.path.relpath(src, repo_root)

        first = _git("log", "--diff-filter=A", "--follow", "--format=%H", "--", rel, cwd=repo_root)
        first_sha = first.split("\n")[-1] if first else ""

        last_sha = _git("log", "-1", "--format=%H", "--", rel, cwd=repo_root)
        last_date = _git("log", "-1", "--format=%ad", "--date=short", "--", rel, cwd=repo_root) if last_sha else ""

        if first_sha:
            context["page_first_author"] = _commit_author(first_sha)
        if last_sha:
            context["page_last_author"] = _commit_author(last_sha)
        if last_date:
            context["last_commit_date"] = last_date
    except Exception:
        pass
    return context
