import sys
from pathlib import Path

# Allow importing existing app modules without moving desktop code.
ROOT_DIR = Path(__file__).resolve().parent
APP_DIR = ROOT_DIR / "anatomy-teacher-app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from webapp import main  # noqa: E402


if __name__ == "__main__":
    main()
