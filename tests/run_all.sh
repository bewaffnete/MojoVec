#!/usr/bin/env bash
set -euo pipefail

# Determine repository root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ANSI Color codes
BOLD="\033[1m"
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
CYAN="\033[0;36m"
NC="\033[0m"

echo -e "${BOLD}${CYAN}===========================================${NC}"
echo -e "${BOLD}${CYAN}       MojoVec - Running All Tests         ${NC}"
echo -e "${BOLD}${CYAN}===========================================${NC}\n"

cd "$REPO_ROOT"

PASSED=0
FAILED=0
FAILED_FILES=()

# 1. Mojo Tests
echo -e "${BOLD}${YELLOW}>>> Running Mojo Unit Tests (tests/*.mojo)...${NC}\n"

for test_file in tests/*.mojo; do
    if [ ! -f "$test_file" ]; then
        continue
    fi

    echo -e "${BOLD}Running ${test_file}...${NC}"
    if mojo run -I . "$test_file"; then
        echo -e "${GREEN}✓ PASS: ${test_file}${NC}\n"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗ FAIL: ${test_file}${NC}\n"
        FAILED=$((FAILED + 1))
        FAILED_FILES+=("$test_file")
    fi
done

# Summary
echo -e "${BOLD}${CYAN}===========================================${NC}"
echo -e "${BOLD}${CYAN}               Test Summary                ${NC}"
echo -e "${BOLD}${CYAN}===========================================${NC}"
echo -e "${GREEN}Passed: ${PASSED}${NC}"

if [ "$FAILED" -ne 0 ]; then
    echo -e "${RED}Failed: ${FAILED}${NC}"
    echo -e "${RED}Failed files:${NC}"
    for failed_file in "${FAILED_FILES[@]}"; do
        echo -e "  ${RED}- ${failed_file}${NC}"
    done
    exit 1
else
    echo -e "${GREEN}All Mojo tests passed successfully!${NC}"
    exit 0
fi
