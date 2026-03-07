#!/usr/bin/env bash
# Batch job submission helpers.
# Sourced by sifv.sh — do not execute directly.

TEMPLATES_DIR="$TOOLS_DIR/templates"

# Auto-detect template by finding a compute.conf entry whose submit command is available.
_detect_template() {
    if [[ ! -f "$COMPUTE_CONF" ]]; then
        echo ""
        return
    fi
    local section="" submit_cmd=""
    while IFS= read -r line || [[ -n "$line" ]]; do
        if [[ "$line" =~ ^\[([^]]+)\]$ ]]; then
            section="${BASH_REMATCH[1]}"
        elif [[ -n "$section" && "$line" =~ ^[[:space:]]*submit[[:space:]]*=[[:space:]]*(.+)$ ]]; then
            submit_cmd="${BASH_REMATCH[1]}"
            if command -v "$submit_cmd" &>/dev/null; then
                echo "$section"
                return
            fi
        fi
    done < "$COMPUTE_CONF"
    echo ""
}

# Render a template file by substituting {{KEY}} placeholders.
# Usage: _render_template <template_file> <output_file> KEY=VALUE ...
_render_template() {
    local template="$1" output="$2"
    shift 2

    local content
    content="$(cat "$template")"

    local kv key val
    for kv in "$@"; do
        key="${kv%%=*}"
        val="${kv#*=}"
        content="${content//\{\{${key}\}\}/${val}}"
    done

    printf '%s\n' "$content" > "$output"
}

# Generate and submit a batch job script.
# Arguments are passed as named variables from cmd_run.
_submit_batch() {
    local BARE_NAME="$1"
    local NODES="$2"
    local PPN="$3"
    local TOTAL_RANKS="$4"
    local ACCOUNT="$5"
    local WALLTIME="$6"
    local PARTITION="$7"
    local QOS="$8"
    local TEMPLATE_NAME="$9"
    local OUTPUT_DIR="${10}"
    local RUN_COMMAND="${11}"

    # --- Resolve template ---
    if [[ -z "$TEMPLATE_NAME" ]]; then
        TEMPLATE_NAME="$(_detect_template)"
        if [[ -z "$TEMPLATE_NAME" ]]; then
            echo "Error: could not auto-detect a compute environment." >&2
            echo "  Specify a template with --template <name>." >&2
            exit 1
        fi
    fi

    local TEMPLATE_FILE="$TEMPLATES_DIR/${TEMPLATE_NAME}.sh"
    if [[ ! -f "$TEMPLATE_FILE" ]]; then
        echo "Error: template '$TEMPLATE_NAME' not found at $TEMPLATE_FILE" >&2
        echo "  Available templates:" >&2
        for f in "$TEMPLATES_DIR"/*.sh; do
            [[ -f "$f" ]] && echo "    $(basename "${f%.sh}")" >&2
        done
        exit 1
    fi

    # --- Build optional directive lines ---
    local PARTITION_LINE=""
    if [[ -n "$PARTITION" ]]; then
        PARTITION_LINE="#SBATCH --partition=${PARTITION}"
    fi

    local QOS_LINE=""
    if [[ -n "$QOS" ]]; then
        QOS_LINE="#SBATCH --qos=${QOS}"
    fi

    # --- Generate job script ---
    local JOB_SCRIPT="${OUTPUT_DIR}/${BARE_NAME}.sh"
    mkdir -p "$OUTPUT_DIR"

    _render_template "$TEMPLATE_FILE" "$JOB_SCRIPT" \
        "JOB_NAME=${BARE_NAME}" \
        "NODES=${NODES}" \
        "PPN=${PPN}" \
        "TOTAL_RANKS=${TOTAL_RANKS}" \
        "ACCOUNT=${ACCOUNT}" \
        "WALLTIME=${WALLTIME}" \
        "PARTITION_LINE=${PARTITION_LINE}" \
        "QOS_LINE=${QOS_LINE}" \
        "OUTPUT_DIR=${OUTPUT_DIR}" \
        "RUN_COMMAND=${RUN_COMMAND}" \
        "SIFV_DIR=${ROOT_DIR}" \
        "TEMPLATE_NAME=${TEMPLATE_NAME}"

    chmod +x "$JOB_SCRIPT"

    echo ""
    echo "=== Generated job script: $JOB_SCRIPT ==="
    echo "--- begin ---"
    cat "$JOB_SCRIPT"
    echo "--- end ---"
    echo ""

    # --- Submit ---
    local SUBMIT_CMD
    if SUBMIT_CMD="$(_get_compute_value "$TEMPLATE_NAME" submit)"; then
        echo "Submitting with $SUBMIT_CMD..."
        $SUBMIT_CMD "$JOB_SCRIPT"
    else
        echo "Job script written to $JOB_SCRIPT." >&2
        echo "No 'submit' command found in compute.conf for '$TEMPLATE_NAME'." >&2
        echo "Submit manually or add 'submit = <command>' to the [$TEMPLATE_NAME] section." >&2
    fi
}
