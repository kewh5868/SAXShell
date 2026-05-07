from __future__ import annotations

_GLOBAL_NOTICE = """
!!! warning "Documentation status"
    This page was generated and edited with the assistance of an LLM and is still in development.
    It has not been fully vetted by the developer. Verify commands, UI labels, file paths,
    workflow descriptions, and scientific claims against the current code and your local
    workflow before relying on it.

    If you notice an error, omission, or outdated guidance, please open an issue on
    [GitHub](https://github.com/kewh5868/SAXSShell/issues).
""".strip()

_SECTION_NOTICES = {
    "tutorials/": """
!!! note "Tutorials section status"
    The Tutorials section is still being built out. Treat this page as a draft scaffold
    rather than a complete end-to-end tutorial.
""".strip(),
    "api/": """
!!! note "API section status"
    The API section has not been fully created yet. Use this page as a provisional pointer
    to likely workflow classes, not as a complete or stable API reference.
""".strip(),
    "development/": """
!!! note "Development section status"
    The Development section is still incomplete. Current pages are working notes for
    contributors rather than a fully vetted maintenance guide.
""".strip(),
}


def _section_notice(src_path: str) -> str | None:
    normalized = str(src_path).strip().replace("\\", "/")
    for prefix, notice in _SECTION_NOTICES.items():
        if normalized.startswith(prefix):
            return notice
    return None


def on_page_markdown(markdown, page, config, files):
    notices = [_GLOBAL_NOTICE]
    section_notice = _section_notice(page.file.src_path)
    if section_notice is not None:
        notices.append(section_notice)
    notices.append(str(markdown).lstrip())
    return "\n\n".join(notices)
