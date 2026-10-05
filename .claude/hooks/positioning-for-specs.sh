#!/usr/bin/env bash
# PreToolUse: перед первой записью в docs/superpowers/specs/ за сессию подкладывает
# в контекст .claude/POSITIONING.md — спека сверяется с тем, что продукт обещает игроку.
# Срабатывает на Write/Edit по пути спеки и на Bash, чья команда упоминает этот каталог.
set -euo pipefail

input=$(cat)
tool=$(jq -r '.tool_name // ""' <<<"$input")
session=$(jq -r '.session_id // "nosession"' <<<"$input")

case "$tool" in
  Write|Edit) target=$(jq -r '.tool_input.file_path // ""' <<<"$input") ;;
  Bash)       target=$(jq -r '.tool_input.command // ""' <<<"$input") ;;
  *)          exit 0 ;;
esac

[[ "$target" == *docs/superpowers/specs/* ]] || exit 0

marker="${TMPDIR:-/tmp}/claude-positioning-${session}"
[[ -e "$marker" ]] && exit 0

file="${CLAUDE_PROJECT_DIR:-$(pwd)}/.claude/POSITIONING.md"
[[ -f "$file" ]] || exit 0
touch "$marker"

jq -n --rawfile doc "$file" '{
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    additionalContext: ("Пишется спека. Сверь фичу с позиционированием продукта (.claude/POSITIONING.md):\n\n" + $doc)
  }
}'
