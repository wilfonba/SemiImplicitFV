#!/usr/bin/env bash
# List available simulation cases.
# Sourced by sifv.sh — do not execute directly.

list_cases() {
    echo "Cases in cases/:"
    echo ""

    declare -A cases
    for dir in "$CASES_DIR"/*/; do
        [[ -d "$dir" ]] || continue
        name="$(basename "$dir")"
        for ext in jsonc json; do
            if [[ -f "$dir/$name.$ext" ]]; then
                cases[$name]="${cases[$name]:-}json "
                break
            fi
        done
        if compgen -G "$dir"/*.cpp > /dev/null 2>&1; then
            cases[$name]="${cases[$name]:-}compiled "
        fi
    done

    for name in $(echo "${!cases[@]}" | tr ' ' '\n' | sort); do
        types="${cases[$name]}"
        tags=""
        [[ "$types" == *json* ]] && tags="${tags}json"
        [[ "$types" == *compiled* ]] && { [[ -n "$tags" ]] && tags="${tags}, "; tags="${tags}compiled"; }
        printf "  %-30s [%s]\n" "$name" "$tags"
    done
}
