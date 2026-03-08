#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CARGO_TOML="$WORKSPACE_ROOT/Cargo.toml"
MODE="${1:---check}"

if [[ "$MODE" != "--check" && "$MODE" != "--write" ]]; then
    echo "usage: $0 [--check|--write]" >&2
    exit 2
fi

workspace_version="$(
    awk '
        /^\[workspace\.package\]/ { in_section = 1; next }
        /^\[/ && in_section { exit }
        in_section && $1 == "version" {
            gsub(/"/, "", $3)
            print $3
            exit
        }
    ' "$CARGO_TOML"
)"

if [[ -z "$workspace_version" ]]; then
    echo "error: failed to read workspace.package.version from $CARGO_TOML" >&2
    exit 1
fi

mapfile -t internal_crates < <(
    awk '
        /^\[workspace\.dependencies\]/ { in_section = 1; next }
        /^\[/ && in_section { exit }
        in_section && $1 ~ /^tirea[a-z0-9-]*$/ && $2 == "=" {
            print $1
        }
    ' "$CARGO_TOML"
)

if [[ "${#internal_crates[@]}" -eq 0 ]]; then
    echo "error: failed to discover internal crates from workspace.dependencies" >&2
    exit 1
fi

mkdir -p "$WORKSPACE_ROOT/target"

mapfile -t markdown_files < <(find "$WORKSPACE_ROOT/docs/book/src" -type f -name '*.md' | sort)

status=0

for file in "${markdown_files[@]}"; do
    tmp_file="$(mktemp "$WORKSPACE_ROOT/target/doc-version-sync.XXXXXX")"
    cp "$file" "$tmp_file"

    for crate in "${internal_crates[@]}"; do
        CRATE_NAME="$crate" WORKSPACE_VERSION="$workspace_version" \
            perl -0pi -e '
                s/\b\Q$ENV{CRATE_NAME}\E(\s*=\s*\{\s*version\s*=\s*")([^"]+)(")/$ENV{CRATE_NAME}$1$ENV{WORKSPACE_VERSION}$3/g;
                s/\b\Q$ENV{CRATE_NAME}\E(\s*=\s*")([^"]+)(")/$ENV{CRATE_NAME}$1$ENV{WORKSPACE_VERSION}$3/g;
            ' "$tmp_file"
    done

    if ! cmp -s "$file" "$tmp_file"; then
        if [[ "$MODE" == "--write" ]]; then
            mv "$tmp_file" "$file"
            echo "updated $file"
        else
            rm -f "$tmp_file"
            echo "outdated crate version references in $file" >&2
            status=1
        fi
    else
        rm -f "$tmp_file"
    fi
done

if [[ "$MODE" == "--check" && "$status" -ne 0 ]]; then
    echo "run scripts/sync-doc-crate-versions.sh --write to synchronize documentation versions" >&2
fi

exit "$status"
