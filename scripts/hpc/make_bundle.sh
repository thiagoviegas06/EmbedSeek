#!/bin/bash
# Package the whole project for the cluster: code, docs, git history, local notes, data, results.
# Leaves out the virtualenv (rebuilt on the cluster), caches, and incomplete embedding folders
# (no meta.json = the run never finished). Writes data/MANIFEST.sha256 so the copy can be verified.
#
# Usage (from anywhere): bash scripts/hpc/make_bundle.sh [output.tar.gz]
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
out=${1:-$(dirname "$root")/embedseek_bundle_$(date +%Y%m%d).tar.gz}
cd "$root"

excludes=(--exclude=.venv --exclude=__pycache__ --exclude=.pytest_cache --exclude='*.egg-info'
          --exclude=.DS_Store --exclude=logs)
for d in data/*/emb/*/; do
  if [ ! -f "$d/meta.json" ]; then
    echo "skipping incomplete embeddings: $d"
    excludes+=(--exclude="${d%/}")
  fi
done

# Checksums for every data file that ships, relative to data/ (verified by setup_env.sh).
(cd data && find . -type f ! -name MANIFEST.sha256 ! -name .DS_Store | sort \
   | grep -v -F -f <(printf '%s\n' "${excludes[@]}" | sed -n 's#^--exclude=data/#./#p') \
   | xargs shasum -a 256 > MANIFEST.sha256)
echo "checksummed $(wc -l < data/MANIFEST.sha256 | tr -d ' ') data files"

# COPYFILE_DISABLE / --no-mac-metadata: no macOS ._ files or xattrs in the archive.
COPYFILE_DISABLE=1 tar --no-mac-metadata "${excludes[@]}" -s ',^\.,EmbedSeek,' -czf "$out" .
echo "wrote $out ($(du -h "$out" | cut -f1))"
