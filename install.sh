#!/usr/bin/env bash
# Install microGEM and (optionally) register it as an AI skill.
#
# Two independent things happen here, and either can be skipped:
#
#   1. the `microgem` command line tool  (pip install, into a venv or --user)
#   2. the `microgem` agent skill        (a SKILL.md directory the assistant
#      reads so it knows when and how to call the CLI)
#
# The skill half is deliberately dumb: it copies a directory into the agent's
# skill root and prints where it went. Every host has a different root, so the
# script asks the right question instead of guessing -- see --agent.
#
# Usage
#   ./install.sh                     # CLI + skill, auto-detected agent roots
#   ./install.sh --cli-only
#   ./install.sh --skill-only --agent dsh
#   ./install.sh --venv ~/.venvs/microgem
#   ./install.sh --dry-run
#
# Exit codes: 0 success, 1 usage error, 2 install failure, 3 nothing to install.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_SRC="${REPO_ROOT}/skill"

CLI_ONLY=0
SKILL_ONLY=0
DRY_RUN=0
ASSUME_YES=0
VENV=""
AGENT="auto"
DOWNLOAD_MODEL=0
LINK_TO=""
PYTHON="${PYTHON:-python3}"

usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options
  --cli-only            install the `microgem` command, skip the agent skill
  --skill-only          install the agent skill, skip the command
  --agent NAME          dsh | claude | agents | <path>   (default: auto)
  --venv PATH           create/use a virtualenv at PATH (default: pip --user)
  --link-to DIR         also symlink the command into DIR (e.g. a PATH dir of
                        a service that cannot read ~/.local/bin)
  --python BIN          interpreter to use (default: $PYTHON or python3)
  --download-model      also fetch the 2.6 GB ESM-2 checkpoint now
  --with-torch          install torch + fair-esm into the venv (fast GPU path)
  --yes                 do not prompt
  --dry-run             print what would happen, change nothing
  -h, --help            this message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cli-only) CLI_ONLY=1 ;;
        --skill-only) SKILL_ONLY=1 ;;
        --agent) AGENT="${2:?--agent needs a value}"; shift ;;
        --venv) VENV="${2:?--venv needs a path}"; shift ;;
        --python) PYTHON="${2:?--python needs a binary}"; shift ;;
        --link-to) LINK_TO="${2:?--link-to needs a directory}"; shift ;;
        --download-model) DOWNLOAD_MODEL=1 ;;
        --with-torch) WITH_TORCH=1 ;;
        --yes|-y) ASSUME_YES=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
    shift
done

WITH_TORCH="${WITH_TORCH:-0}"

if [[ ${CLI_ONLY} -eq 1 && ${SKILL_ONLY} -eq 1 ]]; then
    echo "--cli-only and --skill-only are mutually exclusive" >&2
    exit 1
fi

run() {
    if [[ ${DRY_RUN} -eq 1 ]]; then
        echo "  [dry-run] $*"
    else
        "$@"
    fi
}

say() { printf '\033[1m%s\033[0m\n' "$*"; }
note() { printf '  %s\n' "$*"; }

# ── 1. the command ─────────────────────────────────────────────────────────

install_cli() {
    say "Installing the microgem command"

    local pip=()
    if [[ -n "${VENV}" ]]; then
        if [[ ! -x "${VENV}/bin/python" ]]; then
            note "creating virtualenv at ${VENV}"
            run "${PYTHON}" -m venv "${VENV}"
        fi
        pip=("${VENV}/bin/python" -m pip)
        note "target: ${VENV}"
    else
        note "target: ${PYTHON} (user site)"
        pip=("${PYTHON}" -m pip install --user)
    fi

    if [[ -z "${VENV}" ]]; then
        run "${PYTHON}" -m pip install --user --upgrade "${REPO_ROOT}"
        MICROGEM_BIN="${HOME}/.local/bin/microgem"
    else
        run "${VENV}/bin/python" -m pip install --upgrade "${REPO_ROOT}"
        MICROGEM_BIN="${VENV}/bin/microgem"
    fi

    if [[ ${WITH_TORCH} -eq 1 ]]; then
        note "installing the PyTorch backend (large; ~2 GB)"
        if [[ -n "${VENV}" ]]; then
            run "${VENV}/bin/python" -m pip install "torch>=2.0" "fair-esm>=2.0"
        else
            run "${PYTHON}" -m pip install --user "torch>=2.0" "fair-esm>=2.0"
        fi
    fi

    expose_on_path
    note "verify with: microgem info"
}

# `--user` installs land the executable in ~/.local/bin, which is on an
# interactive login shell's PATH but NOT on the PATH of a process started from a
# minimal environment -- which is how agent tool shells inherit theirs. Two
# things handle that, and neither of them writes outside the user's home:
#
#   1. the skill resolves the binary itself (see the `$MG` preamble in SKILL.md)
#   2. this function reports the situation instead of silently "fixing" it into
#      some unrelated writable directory
expose_on_path() {
    [[ -x "${MICROGEM_BIN}" ]] || return 0
    [[ -e "${HOME}/.local/bin/microgem" ]] || {
        run mkdir -p "${HOME}/.local/bin"
        run ln -sf "${MICROGEM_BIN}" "${HOME}/.local/bin/microgem"
    }
    if command -v microgem >/dev/null 2>&1; then
        return 0
    fi
    note "! microgem is installed at ${MICROGEM_BIN}"
    note "  but ~/.local/bin is not on this shell's PATH."
    note "  Login shells get it via ~/.profile; agent shells do not."
    note "  The skill works anyway (it resolves the path itself)."
    note "  For a plain shell, either:"
    note "    export PATH="\$HOME/.local/bin:\$PATH"    # add to ~/.bashrc"
    note "    or run:  ${MICROGEM_BIN}"
    if [[ -n "${LINK_TO}" ]]; then
        run mkdir -p "${LINK_TO}"
        run ln -sf "${MICROGEM_BIN}" "${LINK_TO}/microgem"
        note "  linked into ${LINK_TO} (--link-to)"
    fi
}

# ── 2. the skill ───────────────────────────────────────────────────────────

# Candidate skill roots, highest precedence first. The DSH and Claude/agents
# conventions differ only in directory name, so one list covers both.
skill_roots() {
    case "${AGENT}" in
        dsh)     echo "${DSH_HOME:-$HOME/.dsh}/skills"; return ;;
        claude)  echo "$HOME/.claude/skills"; return ;;
        agents)  echo "${DSH_AGENTS_HOME:-$HOME/.agents}/skills"; return ;;
        auto)    ;;
        *)       echo "${AGENT}"; return ;;
    esac
    [[ -d "${DSH_HOME:-$HOME/.dsh}/skills" ]] && echo "${DSH_HOME:-$HOME/.dsh}/skills"
    [[ -d "$HOME/.claude/skills" ]] && echo "$HOME/.claude/skills"
    [[ -d "${DSH_AGENTS_HOME:-$HOME/.agents}/skills" ]] && echo "${DSH_AGENTS_HOME:-$HOME/.agents}/skills"
    return 0
}

install_skill() {
    say "Installing the agent skill"

    local roots=()
    while IFS= read -r line; do
        [[ -n "${line}" ]] && roots+=("${line}")
    done < <(skill_roots)

    if [[ ${#roots[@]} -eq 0 ]]; then
        note "no agent skill root found (looked for ~/.dsh/skills, ~/.claude/skills, ~/.agents/skills)"
        note "create one, or rerun with:  --agent /path/to/skills"
        note "installing to the first candidate anyway: ${HOME}/.dsh/skills"
        roots=("${HOME}/.dsh/skills")
    fi

    for root in "${roots[@]}"; do
        local target="${root}/microgem"
        note "→ ${target}"
        run mkdir -p "${root}"
        if [[ ${DRY_RUN} -eq 0 ]]; then
            rm -rf "${target}"
            cp -R "${SKILL_SRC}" "${target}"
        else
            echo "  [dry-run] cp -R ${SKILL_SRC} ${target}"
        fi
    done
    note "restart the agent session (or reload skills) to pick it up, then type: /microgem"
}

# ── 3. model ───────────────────────────────────────────────────────────────

maybe_model() {
    [[ ${DOWNLOAD_MODEL} -eq 1 ]] || return 0
    say "Fetching the ESM-2 checkpoint (2.6 GB)"
    if [[ -n "${VENV}" ]]; then
        run "${VENV}/bin/microgem" download
    else
        run microgem download
    fi
}

# ── main ───────────────────────────────────────────────────────────────────

say "microGEM installer"
note "repo: ${REPO_ROOT}"
[[ ${DRY_RUN} -eq 1 ]] && note "DRY RUN — nothing will be written"

if [[ ${SKILL_ONLY} -eq 0 && ${CLI_ONLY} -eq 0 ]] || [[ ${SKILL_ONLY} -eq 0 ]]; then
    install_cli
fi
if [[ ${CLI_ONLY} -eq 0 ]]; then
    install_skill
fi
maybe_model

say "Done"
cat <<EOF
  microgem info                     # backend + model locations
  microgem panels                   # the 17 panels and their performance
  microgem score proteins.faa       # score

The model is not bundled (2.6 GB). First run:
  microgem download                 # needs torch + fair-esm
  microgem download --convert       # or: also write the torch-free weights
EOF
