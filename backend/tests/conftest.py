import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

# Fix imports
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep pytest runs offline from LangSmith even when the developer .env enables it.
# The application bootstrap loads .env later with override=False, so this setting wins.
os.environ["LANGSMITH_TRACING"] = "false"
