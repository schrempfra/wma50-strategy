import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
API_PASSPHRASE = os.getenv('API_PASSPHRASE')

# Trading parameters
DEFAULT_SYMBOL = 'BTCUSDT'
DEFAULT_QUANTITY = 0.0001  # Default position size (small for testing)

# WMA strategy parameters
TAKE_PROFIT_PERCENT = 8.0  # Take profit at +8%
STOP_LOSS_PERCENT = 2.0    # Stop loss at -2%
EXIT_HOURS = 48            # Exit after 48 hours if neither TP nor SL hit