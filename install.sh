#!/usr/bin/env bash
# Install the robotics skills for Claude Code and Antigravity (Agent Skills standard: <name>/SKILL.md).
#
#   ./install.sh /path/to/project     # symlink into <project>/.claude/skills and <project>/.agents/skills
#   ./install.sh --global             # ~/.claude/skills and ~/.gemini/config/skills (all projects)
#   add --copy to copy instead of symlink (Windows, containers, or to vendor a snapshot into a repo)
#   Symlinks point back to this clone, so `git pull` here updates every project.
#
# Discovery locations (Oct 2026):
#   Claude Code:  <project>/.claude/skills/<name>/   ~/.claude/skills/<name>/
#   Antigravity:  <project>/.agents/skills/<name>/   ~/.gemini/config/skills/<name>/  (CLI: ~/.gemini/antigravity-cli/skills/)
#   Gemini CLI also reads <project>/.agents/skills/.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SKILLS=(gemini-robotics-er mujoco-physics threejs-rendering)
MODE=link; TARGET=""; GLOBAL=0
for a in "$@"; do
  case "$a" in
    --copy) MODE=copy ;;
    --global) GLOBAL=1 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) TARGET="$a" ;;
  esac
done

if [[ $GLOBAL == 0 && -z "$TARGET" ]]; then sed -n '2,11p' "$0"; exit 1; fi

if [[ $GLOBAL == 1 ]]; then
  DESTS=("$HOME/.claude/skills" "$HOME/.gemini/config/skills")
else
  ROOT="$(cd "$TARGET" && pwd)"
  DESTS=("$ROOT/.claude/skills" "$ROOT/.agents/skills")
fi

for dest in "${DESTS[@]}"; do
  mkdir -p "$dest"
  for s in "${SKILLS[@]}"; do
    rm -rf "${dest:?}/$s"
    if [[ $MODE == copy ]]; then
      cp -R "$SRC/$s" "$dest/$s"
    else
      ln -s "$SRC/$s" "$dest/$s"            # absolute: survives moving the target project
    fi
    echo "$MODE  $dest/$s"
  done
done
echo "Done. Restart the agent session so it re-indexes skills."
