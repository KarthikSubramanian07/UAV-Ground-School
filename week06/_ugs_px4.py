"""Make ros2_ws/src/ugs_px4 importable: its takeoff logic and PX4 model have no ROS in them.

The ROS package is the one source of that code; week06 uses it from the repository
checkout, found beside this package (an editable install) or from the working
directory (a regular install run from the repository root).
"""

import sys
from pathlib import Path

for root in (Path(__file__).resolve().parents[1], Path.cwd()):
    package = root / "ros2_ws" / "src" / "ugs_px4"
    if (package / "ugs_px4" / "takeoff.py").is_file():
        if str(package) not in sys.path:
            sys.path.insert(0, str(package))
        break
else:
    raise ImportError("week06 needs ros2_ws/src/ugs_px4: run it from the repository checkout")
