# Repository Instructions

## Secret Handling

- Never store literal `GH_TOKEN` values or other credentials in `.claude/settings.local.json`, `.claude/claude-progress-log.jsonl`, or any tracked file.
- If a secret appears in repo-local settings or logs, redact or remove it immediately and prefer external shell auth such as `gh auth login`.
- Treat `.claude/` as local-only workspace state and keep it out of commits.

## GitHub Process Flow

- When the user says `run the github process flow`, treat that as a specific publish-prep workflow for this repository.
- Inspect the current branch and working tree, analyze the recent changes, and split the work into logical separate commits.
- Use clear commit messages with enough detail to explain each commit's purpose. Prefer a short subject plus a descriptive body when the diff is non-trivial.
- If the worktree contains unrelated changes or there are multiple reasonable ways to split the commits, pause and confirm scope before staging.
- Run the local commit flow and handle pre-commit fallout. If hooks reformat files or require fixes, review the resulting edits, stage the updated files, and retry the commit until it succeeds or a real blocker remains.
- Finish the workflow with local commits only unless the user explicitly asks for a push or PR creation.
- Do not push automatically. Instead, give the user the exact `git push` command to run manually.
- Do not open the pull request automatically. Instead, provide a copyable PR body/comment for the user to paste into the GitHub PR description.
- The PR body should summarize what changed, why it changed, how it was validated, and include issue-closing lines such as `Resolves #123` for every related issue number.
- If the related GitHub issue numbers are not already clear from the conversation or branch context, ask for them before drafting the final PR body.
