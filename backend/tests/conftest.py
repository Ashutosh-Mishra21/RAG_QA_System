import sys
from pathlib import Path
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]

# Fix imports
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 🔥 Load .env explicitly
load_dotenv(ROOT / ".env", override=True)