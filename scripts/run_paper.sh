#!/usr/bin/env bash
# Every generated table, in-text number and figure of the paper, from the analysis outputs (into $CHAINUQ_PAPER_DIR).
set -euo pipefail
source "$(dirname "$0")/common.sh"
mkdir -p "$CHAINUQ_PAPER_DIR/table" "$CHAINUQ_PAPER_DIR/fig"
cpu reports
stamp "paper outputs written to $CHAINUQ_PAPER_DIR"
