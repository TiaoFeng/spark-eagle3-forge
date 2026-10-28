#!/usr/bin/env bash
# Install a pre-commit hook that runs scripts/check_public.sh.
#
#   scripts/install_hooks.sh          install
#   scripts/install_hooks.sh --remove uninstall
#
# Hooks live in .git/hooks (not versioned), so every clone must opt in.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
hook="$root/.git/hooks/pre-commit"

if [[ "${1:-}" == "--remove" ]]; then
  rm -f "$hook"
  echo "removed $hook"
  exit 0
fi

cat > "$hook" <<'HOOK'
#!/usr/bin/env bash
# Installed by scripts/install_hooks.sh — blocks commits that leak local info.
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
[[ -x "$root/scripts/check_public.sh" ]] || exit 0
exec "$root/scripts/check_public.sh"
HOOK

chmod +x "$hook"
echo "installed $hook"
