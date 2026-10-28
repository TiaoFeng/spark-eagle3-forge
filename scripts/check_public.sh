#!/usr/bin/env bash
# Check tracked files for machine-specific / personally identifying strings.
#
# Usage: scripts/check_public.sh
# Exit:  0 = clean, 1 = findings
#
# Machine-specific words that are unique to YOUR setup (GPU model, disk model,
# username, env name, private hostnames) belong in AGENTS.local.md, not in any
# tracked file. Add your own terms to EXTRA below — but keep them OUT of git,
# e.g. put them in a local, untracked file and source it:
#
#     [[ -f .check_public.extra ]] && source .check_public.extra
#
set -uo pipefail

cd "$(git rev-parse --show-toplevel)"

EXTRA=()
[[ -f .check_public.extra ]] && source .check_public.extra

PATTERNS=(
  # absolute home paths
  '/home/[^ /]+'
  '/Users/[^ /]+'
  '~/'
  # e-mail addresses
  '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
  # private / link-local IPs
  '\b(10|127)\.[0-9]+\.[0-9]+\.[0-9]+'
  '\b192\.168\.[0-9]+\.[0-9]+'
  '\b172\.(1[6-9]|2[0-9]|3[01])\.[0-9]+\.[0-9]+'
  # GPU / disk model names
  '(RTX|GTX|Radeon RX|Arc) ?[0-9]{3,4}'
  '[0-9]+\.[0-9]+ ?TFLOP'
  # this machine's hostname and login name
  "\b$(hostname -s)\b"
  "\b$(id -un)\b"
)

# local-only terms (never committed; see header)
for t in "${EXTRA[@]}"; do
  PATTERNS+=("$t")
done

fail=0
for p in "${PATTERNS[@]}"; do
  # skip the checker itself so its own patterns do not self-match
  hits=$(git grep -nIE -- "$p" -- . ':(exclude)scripts/check_public.sh' 2>/dev/null)
  if [[ -n "$hits" ]]; then
    echo "⚠️  pattern: $p"
    echo "$hits" | sed 's/^/    /'
    fail=1
  fi
done

if [[ "$fail" -eq 0 ]]; then
  echo "✅ no machine-specific strings found in tracked files"
else
  echo
  echo "→ Move these to AGENTS.local.md (gitignored) or replace with \$ENV_VAR placeholders."
fi
exit "$fail"
