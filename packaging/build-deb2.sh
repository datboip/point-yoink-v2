#!/usr/bin/env bash
# Build the PointYoink 2.0 .deb. Small on purpose: it carries the app's own files (tracked in git, so no local junk)
# and a one-time setup window. On first launch that window shows each part 2.0 runs on (Qt, Open3D, PyMeshLab...),
# its size and why it is needed, and downloads the picked parts into ~/.local/share/pointyoink/venv. Nothing big
# ships in the .deb, and nothing is installed system-wide besides the app files.
#   packaging/build-deb2.sh [version]      default: VERSION in app2/version.py
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VER="${1:-$(grep -oP '^VERSION = "\K[0-9A-Za-z.-]+' "$ROOT/app2/version.py")}"
VER="$(printf '%s' "$VER" | sed 's/-/~/')"     # a pre-release such as 2.0.0-pre.1 becomes 2.0.0~pre.1 and sorts before 2.0.0
BUILD="$ROOT/packaging/build2"
NAME="point-yoink"
LIB="usr/lib/point-yoink"
PKG="$BUILD/${NAME}_${VER}_amd64"

rm -rf "$BUILD"
mkdir -p "$PKG/DEBIAN" "$PKG/$LIB" "$PKG/usr/bin" "$PKG/usr/share/applications" \
         "$PKG/usr/share/icons/hicolor/512x512/apps" "$PKG/usr/share/doc/point-yoink"

# --- app files: what git tracks, minus tests, design prototypes and internal notes ---
cd "$ROOT"
git ls-files -- app2 scanner assets '*.py' icon.png \
  | grep -vE '^(tests/|scanner/bench_)' \
  | grep -E '^(app2/|scanner/|assets/|[^/]+\.py$|icon\.png$)' \
  | while read -r f; do [ -e "$f" ] && install -D -m 644 "$f" "$PKG/$LIB/$f"; done
install -m 644 packaging/setup_window.py packaging/requirements.txt "$PKG/$LIB/"
git rev-parse --short HEAD > "$PKG/$LIB/app2/BUILD"          # About and the title show which build this is
cp icon.png "$PKG/usr/share/icons/hicolor/512x512/apps/point-yoink.png"
cp LICENSE README.md CHANGELOG.md THIRD_PARTY_NOTICES.md "$PKG/usr/share/doc/point-yoink/"

# --- ufw profile: `sudo ufw allow point-yoink` opens just the Wi-Fi receiver port ---
mkdir -p "$PKG/etc/ufw/applications.d"
cat > "$PKG/etc/ufw/applications.d/point-yoink" <<'EOF'
[point-yoink]
title=PointYoink
description=Receive Revopoint MIRACO scans shared over Wi-Fi
ports=9706/tcp|9706/udp
EOF

# --- launcher: setup window until every required part is in, then the app ---
cat > "$PKG/usr/bin/point-yoink" <<'EOF'
#!/bin/sh
# point-yoink [--setup] [app options]      --setup reopens the setup window (add or change parts)
APP="${POINTYOINK_APP_DIR:-/usr/lib/point-yoink}"
ENV="${XDG_DATA_HOME:-$HOME/.local/share}/pointyoink"
if [ "$1" = "--setup" ]; then shift; python3 "$APP/setup_window.py" --env "$ENV" || exit 1
elif ! python3 -c "import sys; sys.path.insert(0, '$APP'); import setup_window as s; sys.exit(0 if s.ready('$ENV') else 1)" 2>/dev/null; then
  python3 "$APP/setup_window.py" --env "$ENV" || exit 1
fi
cd "$APP" && exec env XMODIFIERS=@im=none "$ENV/venv/bin/python" -m app2 "$@"
EOF
chmod 755 "$PKG/usr/bin/point-yoink"

# --- menu entries ---
cat > "$PKG/usr/share/applications/point-yoink.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=PointYoink
GenericName=3D Scan Builder
Comment=Build and clean up 3D scans from a Revopoint MIRACO
Exec=point-yoink
Icon=point-yoink
Terminal=false
Categories=Graphics;3DGraphics;
Keywords=3d;scan;scanner;revopoint;miraco;stl;ply;mesh;
StartupWMClass=point-yoink
Actions=setup;

[Desktop Action setup]
Name=Add or change parts
Exec=point-yoink --setup
EOF

# --- control ---
cat > "$PKG/DEBIAN/control" <<EOF
Package: point-yoink
Version: ${VER}
Architecture: amd64
Maintainer: datboip <datboip@users.noreply.github.com>
Depends: python3 (>= 3.11), python3-venv, python3-tk, libgl1, libegl1, libxkbcommon-x11-0, libxcb-cursor0, libxcb-icccm4, libxcb-keysyms1
Recommends: systemd, jmtpfs, rsync, fuse3 | fuse, xdg-utils
Conflicts: pointyoink
Replaces: pointyoink
Section: graphics
Priority: optional
Homepage: https://github.com/datboip/point-yoink-v2
Description: Scan, build and clean up 3D models from a Revopoint MIRACO
 PointYoink 2.0 imports scans from a Revopoint MIRACO / MIRACO Pro over USB or
 Wi-Fi and builds them into one clean, printable model, step by step, on Linux.
 .
 On first launch a setup window lists the parts it runs on (Qt, Open3D,
 PyMeshLab), their sizes, and downloads them once into your home folder.
EOF

chmod -R u+rwX,go+rX "$PKG"
dpkg-deb --root-owner-group -Zxz --build "$PKG" >/dev/null
echo "built: $(ls "$BUILD"/*.deb) ($(du -h "$BUILD"/*.deb | cut -f1))"
