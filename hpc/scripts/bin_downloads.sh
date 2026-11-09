#!/bin/bash
# =============================================================================
# bin_downloads.sh: put Relate v1.2.4 (static), SINGER 0.1.9-beta and POLEGON 0.1.3-alpha in
# bin/ of the repo.
# Run: bash hpc/scripts/bin_downloads.sh
# =============================================================================
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)   # the repo root, 2 levels above this file
mkdir -p "$ROOT/bin" && cd "$ROOT/bin" || exit 1

RELATE_URL=https://myersgroup.github.io/relate/download/relate_v1.2.4_x86_64_static.tgz
RELATE_SHA=9d6fe0f25846da9a846099b0d7c8d1f9f529aee0042e92803aef72b82e771438
SINGER_URL=https://raw.githubusercontent.com/popgenmethods/SINGER/f88d687/releases/singer-0.1.9-beta-linux-x86_64.tar.gz
SINGER_SHA=6f0644a94efc077d2fcc23be96a5ef60f526576bb39c8791e0079d2d0c48f265
POLEGON_URL=https://raw.githubusercontent.com/YunDeng98/POLEGON/0c088b3/releases/polegon-0.1.3-alpha-linux-x86_64.tar.gz
POLEGON_SHA=75d6cd868a59c5e4df8a4aedc1837986832c5729c6a88a2097ec4d408aee50e9

fetch() {                                   # download $1 into bin/ unless the tarball is already here
    local f
    f=$(basename "$1")
    [ -s "$f" ] && return 0
    echo "downloading $f"
    wget -q -O "$f.part" "$1" && mv "$f.part" "$f"
}

fetch_checked() {                           # fetch $1, check against sha $2, download again once on mismatch
    local f
    f=$(basename "$1")
    fetch "$1" || return 1
    echo "$2  $f" | sha256sum -c - && return 0
    echo "checksum failed for $f: downloading again"
    rm -f "$f"
    fetch "$1" || return 1
    echo "$2  $f" | sha256sum -c - || { echo "checksum still fails for $f: NOT extracting"; return 1; }
}

extract() {                                 # replace bin/$2 with a fresh copy from tarball $1
    rm -rf "${ROOT:?}/bin/${2:?}"
    tar -xzf "$1" || { echo "tar failed for $1"; return 1; }
}

# ── Relate ──
fetch_checked "$RELATE_URL" "$RELATE_SHA" || exit 1
extract relate_v1.2.4_x86_64_static.tgz relate_v1.2.4_x86_64_static || exit 1

# ── SINGER ──
fetch_checked "$SINGER_URL" "$SINGER_SHA" || exit 1
extract singer-0.1.9-beta-linux-x86_64.tar.gz singer-0.1.9-beta-linux-x86_64 || exit 1
# arggs.ploidy -> args.ploidy at singer_master:302.
sed -i.orig '302s/arggs\.ploidy/args.ploidy/' singer-0.1.9-beta-linux-x86_64/singer_master || exit 1
if grep -q arggs singer-0.1.9-beta-linux-x86_64/singer_master; then
    echo "SINGER patch failed: singer_master still contains 'arggs'"; exit 1
fi

# ── POLEGON ──
fetch_checked "$POLEGON_URL" "$POLEGON_SHA" || exit 1
extract polegon-0.1.3-alpha-linux-x86_64.tar.gz polegon-0.1.3-alpha-linux-x86_64 || exit 1

# ── check that every tool file is there ──
missing=0
for f in relate_v1.2.4_x86_64_static/bin/Relate \
         relate_v1.2.4_x86_64_static/bin/RelateFileFormats \
         relate_v1.2.4_x86_64_static/scripts/EstimatePopulationSize/EstimatePopulationSize.sh \
         singer-0.1.9-beta-linux-x86_64/singer_master \
         singer-0.1.9-beta-linux-x86_64/convert_to_tskit \
         singer-0.1.9-beta-linux-x86_64/singer \
         polegon-0.1.3-alpha-linux-x86_64/polegon_master \
         polegon-0.1.3-alpha-linux-x86_64/polegon; do
    if [ -e "$f" ]; then ls -l "$f"; else echo "MISSING: $ROOT/bin/$f"; missing=1; fi
done
exit $missing
