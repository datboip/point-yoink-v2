# Third-party notices

PointYoink 2.0 and later are licensed under the GNU Affero General Public License v3.0 or later (see LICENSE). PointYoink stands on the following open-source work; third-party components retain their respective licenses.

## Dependencies

The `.deb` carries none of these. The setup window installs them with pip into your home folder, or your distribution provides them. All are compatible with AGPL-3.0.

| Package | Version tested | License |
|---|---|---|
| PySide6 / shiboken6 (Qt for Python) | 6.11.2 | LGPL-3.0 (or GPL) |
| NumPy | 2.4.2 | BSD-3-Clause |
| SciPy | 1.17.1 | BSD-3-Clause |
| Open3D | 0.19.0 | MIT |
| OpenCV (opencv-python-headless) | 5.0.0.93 | Apache-2.0 |
| PyMeshLab | 2025.7.post1 | GPL-3.0 |
| trimesh | 5.1.0 | MIT |
| fast-simplification | 0.2.0 | MIT |
| PyOpenGL | 3.1.10 | BSD |
| Pillow | 12.3.0 | MIT-CMU |
| lz4 | 4.4.5 | BSD-3-Clause |
| threadpoolctl | 3.6.0 | BSD-3-Clause |
| psutil | 7.2.2 | BSD-3-Clause |
| NetworkX (3MF export) | 3.6.1 | BSD-3-Clause |
| lxml (3MF export) | 5.2.1 | BSD-3-Clause |

## Used at runtime, installed from your distribution

Not bundled; the `.deb` depends on Python 3 with venv, Tk (`python3-tk`, only for the setup window), and the OpenGL / EGL and X libraries Qt needs. USB import from the scanner uses `jmtpfs` / `libmtp` and FUSE, and timelapse clips use `ffmpeg` (both optional, from your distribution).

## Fonts

The wordmark uses the Ubuntu font when it is installed on your system (Ubuntu Font Licence).

## Trademarks

Revopoint and MIRACO are trademarks of their respective owners. PointYoink is an independent project, not affiliated with or endorsed by Revopoint, and contains no Revopoint software. It works with the scanner through its USB camera interface, its Wi-Fi share and its project files, learned by observing the author's own device.

## Icons

The line icons in `assets/icons/` were made for this project (grid, stroke and colors in `assets/icons/manifest.json`). No third-party icon set is used.
