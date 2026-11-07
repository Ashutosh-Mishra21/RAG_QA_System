
"""Application package bootstrap.

Load the repository environment once, before importing settings or services.
Existing process environment variables keep precedence over values in .env.
"""

from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)
