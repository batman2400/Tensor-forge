#!/bin/sh
# Enable the repo's git hooks (secret scan on commit). Run once after cloning.
set -e
cd "$(git rev-parse --show-toplevel)"
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit .githooks/commit-msg 2>/dev/null || true
echo "Git hooks enabled (core.hooksPath=$(git config core.hooksPath))"
