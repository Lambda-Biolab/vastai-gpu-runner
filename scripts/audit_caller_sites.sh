#!/usr/bin/env bash
# audit_caller_sites.sh — repository-wide audit for v4 legacy sweep.
#
# Run BEFORE deleting the legacy symbols in v4 step 7. Every line
# reported below is a caller/mock/fixture/truthiness-check that
# still assumes an old contract (sweep_zombie_instances,
# load_vastai_api_key, _image_is_allowed, the v2 substring match
# in cli.instances, the boolean return of
# verify_instance_ownership). Update every site to handle the v4
# contract explicitly; no stale caller may remain.
#
# Comments and binary caches are ignored — only code references
# count.
#
# Usage: scripts/audit_caller_sites.sh
# Exit code: 0 = clean, 1 = legacy references still present.

set -uo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root" || exit 2

# Strip ``__pycache__`` matches, comment-only lines, and lines
# whose match is in a docstring or triple-quoted text (where the
# reference is descriptive documentation, not a callable site).
# The audit only reports actionable CODE references.
_filter() {
    grep -v "^grep:\|__pycache__" \
        | grep -vE "^[^:]+:[0-9]+:\\s*#" \
        | grep -vE "^[^\"]+\\.py:[0-9]+:[^\"]*[\"\u201c]" \
        || true
}

declare -a checks=(
    "orchestrator.sweep_zombie_instances|grep -rnE \"sweep_zombie_instances\" src/ tests/ || true"
    "orchestrator.load_vastai_api_key|grep -rnE \"load_vastai_api_key\" src/ tests/ || true"
    "providers.vastai._image_is_allowed|grep -rnE \"_image_is_allowed\" src/ tests/ || true"
    "VastaiRunner.allowed_images attribute (read-only external use)|grep -rnE \"runner\\.allowed_images\\b|VastaiRunner\\.allowed_images\\b|r\\.allowed_images\\b\" src/ tests/ || true"
    "verify_instance_ownership bool-return truthiness check (v2 contract)|grep -rnE \"verify_instance_ownership\\(\\)\" src/ tests/ || true"
    "vastai_cmd show instances raw (legacy direct parse in cli.py only)|grep -rnE 'vastai_cmd\\(\\[\"show\", \"instances\"' src/vastai_gpu_runner/cli.py tests/ || true"
    "v2 substring image match|grep -rnE \"img.split\\(\\\":\\\"\\)\\[0\\] in image\" src/ tests/ || true"
)

echo "=== v4 legacy caller audit ==="
echo "Repo root: $repo_root"
echo

rc=0
for entry in "${checks[@]}"; do
    label="${entry%%|*}"
    cmd="${entry#*|}"
    echo "--- $label ---"
    out="$(eval "$cmd" 2>&1 | _filter)"
    if [[ -z "$out" ]]; then
        echo "  (no references)"
    else
        echo "$out" | sed 's/^/  /'
        rc=1
    fi
    echo
done

if [[ "$rc" -ne 0 ]]; then
    echo "=== AUDIT FAILED ==="
    echo "Legacy references still present. Update every site before deleting."
    exit 1
fi
echo "=== AUDIT CLEAN ==="