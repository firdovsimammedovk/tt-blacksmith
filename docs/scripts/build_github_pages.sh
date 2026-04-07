#!/usr/bin/env bash
# Build the full GitHub Pages tree: "latest" at site root, each git tag under /<tag>/.
# Run from repo root after: pip install -r docs/requirements.txt
set -euo pipefail

ROOT="${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$ROOT"

PAGES="${ROOT}/pages"
rm -rf "$PAGES"
mkdir -p "$PAGES"

LIMIT="${DOCS_VERSION_TAG_LIMIT:-20}"
BASE_SHA="$(git rev-parse HEAD)"

export DOC_SITE_BASE_URL="${DOC_SITE_BASE_URL:-https://firdovsimammedovk.github.io/tt-blacksmith}"

echo "::group::Build latest (main) at site root"
export DOCS_VERSION="latest"
cd docs
rm -rf output
make build
cp -a output/. "$PAGES/"
cd "$ROOT"
echo "::endgroup::"

if [[ -n "${DOCS_SEARCH_INGEST_API_KEY:-}" ]]; then
  echo "::group::Index search (latest only)"
  export DOCS_OUTPUT_DIR="$PAGES"
  export DOCS_INDEX_VERSION="latest"
  if ! python docs/scripts/index_remote_search.py; then
    echo "::warning::Search indexing failed (non-fatal)"
  fi
  echo "::endgroup::"
fi

echo "::group::Build tagged releases under /<tag>/"
TAGLIST="$(git tag -l 'v*' --sort=-version:refname | head -n "$LIMIT")"
if [[ -z "$TAGLIST" ]]; then
  TAGLIST="$(git tag -l --sort=-version:refname | head -n "$LIMIT")"
fi

while IFS= read -r tag; do
  [[ -z "${tag:-}" ]] && continue
  [[ "$tag" == "latest" ]] && continue

  if ! git checkout -q "$tag"; then
    echo "::warning::Could not checkout tag $tag — skip"
    continue
  fi
  if [[ ! -f docs/Makefile ]] || [[ ! -f docs/conf.py ]]; then
    echo "::warning::No docs/ at $tag — skip"
    git checkout -q "$BASE_SHA"
    continue
  fi

  echo "Building docs for tag $tag"
  cd docs
  rm -rf output
  export DOCS_VERSION="$tag"
  if ! make build; then
    echo "::warning::sphinx-build failed for $tag — skip"
    cd "$ROOT"
    git checkout -q "$BASE_SHA"
    continue
  fi
  mkdir -p "$PAGES/$tag"
  cp -a output/. "$PAGES/$tag/"
  cd "$ROOT"
  git checkout -q "$BASE_SHA"
done <<< "$(printf '%s\n' "$TAGLIST")"

echo "::endgroup::"
echo "Pages output: $PAGES"
find "$PAGES" -maxdepth 2 -type f -name 'index.html' 2>/dev/null | head -20 || true
