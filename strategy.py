import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import config
from signature import SignatureHandler
import logging
from datetime import datetime, timedelta
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WMAStrategy")

class BitgetTradingStrategy:
    def __init__(self, api_key, api_secret, passphrase, symbol='BTCUSDT', quantity=0.0001, check_interval=2):
        """
        Initialize the Bitget trading strategy with API credentials and trading parameters
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.symbol = symbol
        self.base_url = "https://api.bitget.com"
        
        # Strategy parameters
        self.support_resistance_period = 10  # Shorter period for more frequent levels
        self.volume_ma_period = 10          # Shorter period for volume
        self.atr_period = 14                # ATR period
        self.atr_multiplier = 1.2           # Tighter ATR multiplier
        self.min_volume_ratio = 0.8         # Lower volume requirement
        self.min_price_change = 0.3         # Lower price change requirement
        self.max_positions = 3
        self.leverage = 1
        self.exit_hours = 8                 # Shorter hold time
        self.check_interval = check_interval
        self.quantity = quantity
        
        # Risk Management
        self.take_profit_percent = 3.0         # Take profit at 3%
        self.stop_loss_percent = 1.5          # Stop loss at 1.5%
        self.use_trailing_stop = True          # Use trailing stop
        self.trailing_activation_percent = 1.0  # Activate trailing stop at 1% profit
        self.trailing_distance_percent = 0.8    # 0.8% trailing distance
        
        # Performance tracking
        self.active_positions = []
        self.trailing_stops = {}

        # Initialize signature handler
        self.signature_handler = SignatureHandler(api_secret, passphrase)
    
    def check_trailing_stop(self, symbol, entry_price, current_price, order_id, direction):
        """
        Check and update trailing stop if needed (works for both long and short positions)
        """
        if direction == 'long':
            # Long: Profit when current_price > entry_price
            profit_pct = ((current_price - entry_price) / entry_price) * 100
            # Trailing stop moves up with price
            trailing_stop_price = current_price * (1 - self.trailing_distance_percent/100)
            original_stop = entry_price * (1 - self.stop_loss_percent/100)
            activation_condition = profit_pct >= self.trailing_activation_percent
            update_condition = current_price > self.trailing_stops.get(order_id, {}).get('activation_price', -float('inf'))
            raise_condition = trailing_stop_price > self.trailing_stops.get(order_id, {}).get('stop_price', -float('inf'))
            # For long: Stop must never go below original_stop
        else:
            # Short: Profit when current_price < entry_price
            profit_pct = ((entry_price - current_price) / entry_price) * 100
            # Trailing stop moves down with price
            trailing_stop_price = current_price * (1 + self.trailing_distance_percent/100)
            original_stop = entry_price * (1 + self.stop_loss_percent/100)
            activation_condition = profit_pct >= self.trailing_activation_percent
            update_condition = current_price < self.trailing_stops.get(order_id, {}).get('activation_price', float('inf'))
            raise_condition = trailing_stop_price < self.trailing_stops.get(order_id, {}).get('stop_price', float('inf'))
            # For short: Stop must never go above original_stop

        # Activate trailing stop
        if order_id not in self.trailing_stops:
            if activation_condition:
                # Only activate if new stop is "better" than original
                if (direction == 'long' and trailing_stop_price > original_stop) or (direction == 'short' and trailing_stop_price < original_stop):
                    self.trailing_stops[order_id] = {
                        'stop_price': trailing_stop_price,
                        'activation_price': current_price
                    }
                    logger.info(f"Activated trailing stop for {symbol} at {trailing_stop_price:.2f} ({direction})")
                    return trailing_stop_price
        # Update trailing stop
        elif order_id in self.trailing_stops:
            if update_condition:
                if raise_condition:
                    self.trailing_stops[order_id]['stop_price'] = trailing_stop_price
                    self.trailing_stops[order_id]['activation_price'] = current_price
                    logger.info(f"Updated trailing stop for {symbol} to {trailing_stop_price:.2f} ({direction})")
                    return trailing_stop_price
            # Return current stop price
            return self.trailing_stops[order_id]['stop_price']

        # Return original stop loss if not activated
        return original_stop

    def monitor_positions(self):
        """
        Monitor and manage open positions
        """
        if not self.active_positions:
            return
            
        # Make a copy to avoid modifying during iteration
        positions_to_check = self.active_positions.copy()
        
        for position in positions_to_check:
            try:
                # Get current price
                symbol = position['symbol']
                
                # For live trading, get latest ticker price using v2 API
                params = {
                    'symbol': symbol,
                    'productType': 'USDT-FUTURES'
                }
                ticker_data = self._make_request('GET', "/api/v2/mix/market/ticker", params=params)
                
                if not ticker_data or ticker_data.get('code') != '00000' or not ticker_data.get('data'):
                    logger.warning(f"Failed to get ticker data for {symbol}")
                    continue
                
                # Extract price from the first item in the data array
                market_data = ticker_data['data'][0]
                current_price = float(market_data['lastPr'])
                
                # Update trailing stop if needed
                updated_stop = self.check_trailing_stop(
                    symbol, 
                    position['entry_price'], 
                    current_price, 
                    position['order_id'],
                    position['direction']
                )
                
                # Check exit conditions based on position direction
                if position['direction'] == 'long':
                    # Long position exit conditions
                    # 1. Take profit hit
                    if current_price >= position['take_profit']:
                        self.close_position(position, current_price, "Take Profit")
                        continue
                        
                    # 2. Stop loss hit (possibly trailing stop)
                    if current_price <= updated_stop:
                        self.close_position(position, current_price, "Stop Loss")
                        continue
                else:
                    # Short position exit conditions
                    # 1. Take profit hit
                    if current_price <= position['take_profit']:
                        self.close_position(position, current_price, "Take Profit")
                        continue
                        
                    # 2. Stop loss hit (possibly trailing stop)
                    if current_price >= updated_stop:
                        self.close_position(position, current_price, "Stop Loss")
                        continue
                    
                # 3. Time-based exit (applies to both long and short)
                if datetime.now() >= position['exit_time']:
                    self.close_position(position, current_price, "Time Exit")
                    continue
            except Exception as e:
                logger.error(f"Error monitoring position {position['order_id']}: {str(e)}")

    def get_account_balance(self):
        """
        Get the available balance from the exchange
        """
        try:
            # Get account balance from API
            endpoint = "/api/v2/mix/account/accounts"
            params = {
                'productType': 'USDT-FUTURES'
            }
            
            response = self._make_request('GET', endpoint, params=params)
            if not response or response.get('code') != '00000':
                logger.error(f"Failed to get account balance: {response}")
                return 0
                
            # Extract available balance
            for account in response.get('data', []):
                if account.get('marginCoin') == 'USDT':
                    available_balance = float(account.get('available', 0))
                    logger.info(f"Available balance: {available_balance} USDT")
                    return available_balance
                    
            logger.warning("No USDT account found")
            return 0
        except Exception as e:
            logger.error(f"Error getting account balance: {str(e)}")
            return 0

    def close_position(self, position, exit_price, exit_reason):
        """
        Close a position and record the trade
        Works for both long and short positions
        """
        try:
            direction = position.get('direction', 'long')  # Default to long if not specified
            logger.info(f"Attempting to close {direction} position for {position['symbol']} at {exit_price:.2f} ({exit_reason})")
            
            # First check if we still have a position
            position_response = self._make_request('GET', "/api/v2/mix/position/single-position", params={
                "symbol": self.symbol,
                "productType": "USDT-FUTURES",
                "marginCoin": "USDT"
            })
            
            # Process position data more carefully to detect empty positions
            has_active_position = False
            if (position_response and 
                position_response.get('code') == '00000' and 
                position_response.get('data')):
                
                position_data = position_response.get('data', [])
                for pos in position_data:
                    # Check if we have the same direction and a non-zero size
                    if (pos.get('holdSide', '').lower() == direction.lower() and 
                        float(pos.get('total', '0')) > 0):
                        has_active_position = True
                        break
            
            if not has_active_position:
                logger.warning(f"No active {direction} position found to close for {self.symbol}")
                # Remove from active positions to avoid repeated closing attempts
                self.active_positions = [p for p in self.active_positions if p['order_id'] != position['order_id']]
                if position['order_id'] in self.trailing_stops:
                    del self.trailing_stops[position['order_id']]
                return
            
            # Initialize success flag
            order_success = False
            
            # Cancel pending orders first
            logger.info(f"Canceling pending orders before closing position")
            self._make_request('POST', "/api/v2/mix/order/cancel-all-orders", data={
                "symbol": self.symbol,
                "productType": "USDT-FUTURES",
                "marginCoin": "USDT"
            })
            
            # Give a small delay for cancellations to process
            time.sleep(0.5)
            
            # Try close-positions endpoint first (most reliable)
            logger.info(f"Trying to close position with close-positions endpoint")
            close_response = self._make_request('POST', "/api/v2/mix/order/close-positions", data={
                "symbol": self.symbol,
                "productType": "USDT-FUTURES",
                "marginCoin": "USDT",
                "holdSide": direction
            })
            
            if close_response and close_response.get('code') == '00000':
                order_success = True
                logger.info(f"Successfully closed {direction} position with close-positions endpoint")
            
            # Remove position from active positions if successfully closed or max retries reached
            if order_success:
                logger.info(f"Successfully closed {direction} position for {position['symbol']}")
                self.active_positions = [p for p in self.active_positions if p['order_id'] != position['order_id']]
                if position['order_id'] in self.trailing_stops:
                    del self.trailing_stops[position['order_id']]
                
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            # Don't remove from active positions on error so it will be retried

    def emergency_close_all(self):
        """
        Emergency function to close all open positions
        Uses a multi-layered approach to reliably close all positions
        """
        logger.warning("EMERGENCY: Closing all positions")
        
        try:
            # Cancel pending orders
            self._make_request('POST', "/api/v2/mix/order/cancel-all-orders", data={
                "symbol": self.symbol,
                "productType": "USDT-FUTURES",
                "marginCoin": "USDT"
            })

            # Get current positions
            position_response = self._make_request('GET', "/api/v2/mix/position/single-position", params={
                "symbol": self.symbol,
                "productType": "USDT-FUTURES",
                "marginCoin": "USDT"
            })
            
            if not position_response or position_response.get('code') != '00000':
                logger.error(f"Failed to get positions: {position_response}")
                return
            
            if position_response.get('data'):
                position_data = position_response.get('data', [])[0]
                holdSide = position_data.get('holdSide', '')
                total_size = position_data.get('available', '0')
                
                if holdSide and float(total_size) > 0:
                    # Try close-positions endpoint first
                    close_response = self._make_request('POST', "/api/v2/mix/order/close-positions", data={
                        "symbol": self.symbol,
                        "productType": "USDT-FUTURES",
                        "marginCoin": "USDT",
                        "holdSide": holdSide
                    })
                    
                    if close_response and close_response.get('code') == '00000':
                        return
                    
                    # Fallback to market order
                    market_response = self._make_request('POST', "/api/v2/mix/order/place-order", data={
                        "symbol": self.symbol,
                        "productType": "USDT-FUTURES",
                        "marginCoin": "USDT",
                        "size": total_size,
                        "side": "sell" if holdSide == "long" else "buy",
                        "tradeSide": "close",
                        "orderType": "market",
                        "marginMode": "isolated",
                        "force": "gtc",
                        "clientOid": f"emergency_{int(time.time())}"
                    })
                    
                    # Final fallback to limit order
                    if not market_response or market_response.get('code') != '00000':
                        ticker_data = self._make_request('GET', "/api/v2/mix/market/ticker", params={
                            "symbol": self.symbol,
                            "productType": "USDT-FUTURES"
                        })
                        
                        if ticker_data and ticker_data.get('code') == '00000' and ticker_data.get('data'):
                            current_price = float(ticker_data['data'][0]['lastPr'])
                            execution_price = current_price * 0.99 if holdSide == "long" else current_price * 1.01
                            
                            self._make_request('POST', "/api/v2/mix/order/place-order", data={
                                "symbol": self.symbol,
                                "productType": "USDT-FUTURES",
                                "marginCoin": "USDT",
                                "size": total_size,
                                "side": "sell" if holdSide == "long" else "buy",
                                "tradeSide": "close",
                                "orderType": "limit",
                                "price": str(execution_price),
                                "marginMode": "isolated",
                                "force": "gtc",
                                "clientOid": f"emergency_limit_{int(time.time())}"
                            })

            # Remove position from active positions
            self.active_positions = []
            # Remove trailing stop data if exists
            self.trailing_stops = {}
        except Exception as e:
            logger.error(f"Error in emergency close: {str(e)}")
            raise

    def _make_request(self, method, endpoint, params=None, data=None):
        """
        Make an authenticated request to the Bitget API
        """
        url = f"{self.base_url}{endpoint}"
        timestamp = str(self.signature_handler.get_timestamp())
        
        # Handle query parameters for GET requests
        if method == 'GET' and params:
            query_string = self.signature_handler.parse_params_to_str(params)
            endpoint = f"{endpoint}{query_string}"
            url = f"{self.base_url}{endpoint}"
        
        # Create request body for POST requests
        body = ''
        if method == 'POST' and data:
            body = json.dumps(data)
        
        # Generate signature
        message = self.signature_handler.pre_hash(timestamp, method, endpoint, body)
        signature = self.signature_handler.sign(message)
        
        # Set headers
        headers = {
            'ACCESS-KEY': self.api_key,
            'ACCESS-SIGN': signature,
            'ACCESS-TIMESTAMP': timestamp,
            'ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
        
        # Make the request
        if method == 'GET':
            response = requests.get(url, headers=headers)
        elif method == 'POST':
            response = requests.post(url, headers=headers, data=body)
        
        # Check for errors
        if response.status_code != 200:
            print(f"API Error: {response.status_code} - {response.text}")
            return None
        
        return response.json()
    
    def get_klines(self, interval='4H', limit=200):
        """
        Get historical candlestick data from Bitget
        
        Valid intervals: 1m, 3m, 5m, 15m, 30m, 1H, 4H, 6H, 12H, 1D, 1W, 1M, 
                        6Hutc, 12Hutc, 1Dutc, 3Dutc, 1Wutc, 1Mutc
        """
        endpoint = "/api/v2/mix/market/candles"
        params = {
            'symbol': self.symbol,
            'productType': 'USDT-FUTURES',
            'granularity': interval,
            'limit': limit
        }
        
        response = self._make_request('GET', endpoint, params=params)
        if not response or response.get('code') != '00000':
            print(f"Failed to get klines: {response}")
            return None
        
        # Process the candlestick data
        candles = response.get('data', [])
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'usdtVolume'])
        
        # Convert string values to numeric
        for col in ['open', 'high', 'low', 'close', 'volume', 'usdtVolume']:
            df[col] = pd.to_numeric(df[col])
        
        # Convert timestamp to datetime - fix the deprecation warning
        # Explicitly convert to numeric type first
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Sort by timestamp in ascending order
        df = df.sort_values(by='timestamp')
        
        return df
    
    def calculate_indicators(self, df):
        """
        Calculate technical indicators for the strategy
        """
        # Calculate Support and Resistance levels
        df['support'] = df['low'].rolling(window=self.support_resistance_period).min()
        df['resistance'] = df['high'].rolling(window=self.support_resistance_period).max()
        
        # Calculate Volume Profile
        df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Calculate ATR
        df['tr'] = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': abs(df['high'] - df['close'].shift(1)),
            'lc': abs(df['low'] - df['close'].shift(1))
        }).max(axis=1)
        df['atr'] = df['tr'].rolling(window=self.atr_period).mean()
        
        # Calculate Price Action
        df['price_change'] = df['close'].pct_change() * 100
        df['higher_high'] = df['high'] > df['high'].shift(1)
        df['lower_low'] = df['low'] < df['low'].shift(1)
        
        # Calculate distance to S/R levels
        df['dist_to_support'] = (df['close'] - df['support']) / df['close'] * 100
        df['dist_to_resistance'] = (df['resistance'] - df['close']) / df['close'] * 100
        
        # Calculate momentum
        df['momentum'] = df['close'].pct_change(3) * 100  # Shorter momentum period
        
        # Calculate trend
        df['trend'] = df['close'].rolling(window=5).mean() > df['close'].rolling(window=20).mean()
        
        # Debug logging
        logger.info(f"Indicators calculation completed. Sample values:")
        logger.info(f"Latest Support: {df['support'].iloc[-1]:.2f}")
        logger.info(f"Latest Resistance: {df['resistance'].iloc[-1]:.2f}")
        logger.info(f"Latest Volume Ratio: {df['volume_ratio'].iloc[-1]:.2f}")
        logger.info(f"Latest ATR: {df['atr'].iloc[-1]:.2f}")
        logger.info(f"Latest Price Change: {df['price_change'].iloc[-1]:.2f}%")
        logger.info(f"Latest Momentum: {df['momentum'].iloc[-1]:.2f}%")
        
        return df
    
    def find_trade_setup(self):
        """
        Find trade setups using price action and support/resistance levels
        """
        try:
            df = self.get_klines()
            if df is None or len(df) < self.support_resistance_period:
                return None
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            # Get latest data point
            current = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Check if we have too many positions
            if len(self.active_positions) >= self.max_positions:
                return None
            
            # Long setup conditions
            long_conditions = (
                (current['dist_to_support'] < 2.0 or current['trend']) and  # Price near support or in uptrend
                current['price_change'] > self.min_price_change and  # Price moving up
                current['volume_ratio'] > self.min_volume_ratio and  # Good volume
                current['momentum'] > -0.5  # Not strongly negative momentum
            )
            
            # Short setup conditions
            short_conditions = (
                (current['dist_to_resistance'] < 2.0 or not current['trend']) and  # Price near resistance or in downtrend
                current['price_change'] < -self.min_price_change and  # Price moving down
                current['volume_ratio'] > self.min_volume_ratio and  # Good volume
                current['momentum'] < 0.5  # Not strongly positive momentum
            )
            
            if long_conditions:
                logger.info(f"Long setup found at {current['timestamp']}")
                return {
                    'symbol': self.symbol,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'timeframe': '4h',
                    'exit_time': datetime.now() + timedelta(hours=self.exit_hours),
                    'direction': 'long'
                }
            elif short_conditions:
                logger.info(f"Short setup found at {current['timestamp']}")
                return {
                    'symbol': self.symbol,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'timeframe': '4h',
                    'exit_time': datetime.now() + timedelta(hours=self.exit_hours),
                    'direction': 'short'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding trade setup: {str(e)}")
            return None
    
    def execute_trade(self, setup, quantity):
        """
        Execute a trade based on the provided setup
        """
        # Place limit order
        order_result = self.place_market_order(
                    size=quantity,
                    side="buy" if setup['direction'] == 'long' else "sell",
                    trade_side="open",
                )

        if not order_result:
            print("Failed to place limit order")
            return False

        # Get the filled price
        filled_price = self.get_order_filled_price(order_result)
        if not filled_price:
            print("Could not determine filled price, using planned entry price")
            filled_price = setup['entry_price']
        filled_price = float(filled_price)

        # Recalculate stop loss and take profit based on actual filled price and direction
        if setup['direction'] == 'long':
            stop_loss = filled_price * (1 - float(self.stop_loss_percent)/100)
            take_profit = filled_price * (1 + float(self.take_profit_percent)/100)
            sl_side = "sell"
            tp_side = "buy"
        else:  # short
            stop_loss = filled_price * (1 + float(self.stop_loss_percent)/100)
            take_profit = filled_price * (1 - float(self.take_profit_percent)/100)
            sl_side = "buy"
            tp_side = "sell"
       
        # Place stop loss order
        sl_result = self.place_stop_order(quantity, sl_side, "open", stop_loss)
        if not sl_result:
            print("Failed to place stop loss order")
      
        # Place take profit order
        tp_result = self.place_take_profit_order(quantity, tp_side, "close", take_profit)
        if not tp_result:
            print("Failed to place take profit order")
       
        print(f"Trade execution completed for {setup['symbol']}")

        position = {
            'symbol': self.symbol,
            'entry_time': datetime.now(),
            'entry_price': filled_price,
            'order_id': order_result.get('orderId'),
            'direction': setup['direction'],
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'exit_time': setup['exit_time'],
            'position_size': quantity
        }
        
        self.active_positions.append(position)
        
        return True
    
    def place_market_order(self, size, side, trade_side):
        """
        Place a market order
        """
        # First, set the leverage
        leverage_endpoint = "/api/v2/mix/account/set-leverage"
        leverage_data = {
            "symbol": self.symbol,
            "productType": "USDT-FUTURES",
            "marginCoin": "USDT",
            "leverage": str(self.leverage),
            "holdSide": "long" if side == "buy" else "short" 
        }
       
        leverage_response = self._make_request('POST', leverage_endpoint, data=leverage_data)
        if not leverage_response or leverage_response.get('code') != '00000':
            print(f"Failed to set leverage: {leverage_response}")
            return None
    
        # Then place the order
        endpoint = "/api/v2/mix/order/place-order"
        data = {
            "symbol": self.symbol,
            "productType": "USDT-FUTURES",
            "marginCoin": "USDT",
            "size": str(size),
            "side": side,  # buy or sell
            "tradeSide": trade_side,  # open or close
            "orderType": "market",
            "marginMode": "isolated",
            "force": "gtc",
            "clientOid": f"strat_{int(time.time())}"
        }
        
        response = self._make_request('POST', endpoint, data=data)
        if not response or response.get('code') != '00000':
            print(f"Failed to place market order: {response}")
            return None
        
        print(f"Market order placed: {response.get('data', {})}")
        return response.get('data')
    
    def place_stop_order(self, size, side, trade_side, stop_price):
        """
        Place a stop loss order
        """
        endpoint = "/api/v2/mix/order/place-order"
        data = {
            "symbol": self.symbol,
            "productType": "USDT-FUTURES",
            "marginCoin": "USDT",
            "size": str(size),
            "side": "buy" if side == "sell" else "sell",  # Reverse the side for stop loss
            "tradeSide": trade_side,  # Reverse trade side for shorts
            "orderType": "market",
            "marginMode": "isolated",
            "force": "gtc",
            "clientOid": f"sl_{int(time.time())}",
            "presetStopLossPrice": str(self.round_price(float(stop_price)))
        }

        response = self._make_request('POST', endpoint, data=data)
        if not response or response.get('code') != '00000':
            print(f"Failed to place stop loss order: {response}")
            return None
        print(f"Stop loss order placed at {stop_price:.2f}")
        return response.get('data')
    
    def place_take_profit_order(self, size, side, trade_side, price):
        """
        Place a limit order for take profit
        """  
        # Then place the limit order
        endpoint = "/api/v2/mix/order/place-order"
        rounded_price = self.round_price(float(price))
        data = {
            "symbol": self.symbol,
            "productType": "USDT-FUTURES",
            "marginCoin": "USDT",
            "size": str(size),
            "side": side,  # buy or sell
            "tradeSide": trade_side,  # open or close
            "orderType": "limit",
            "marginMode": "isolated",
            "force": "gtc",
            "clientOid": f"tp_{int(time.time())}",
            "price": str(rounded_price),
            "presetTakeProfitPrice": str(rounded_price)
        }

        response = self._make_request('POST', endpoint, data=data)
        if not response or response.get('code') != '00000':
            print(f"Failed to place limit order: {response}")
            return None
        
        print(f"Limit order placed: {response.get('data', {})}")
        return response.get('data')
    
    def place_limit_order(self, size, side, trade_side, price):
        """
        Place a limit order
        """
        # First, set the leverage
        leverage_endpoint = "/api/v2/mix/account/set-leverage"
        leverage_data = {
            "symbol": self.symbol,
            "productType": "USDT-FUTURES",
            "marginCoin": "USDT",
            "leverage": str(self.leverage),
            "holdSide": "long" if side == "buy" else "short"
        }
        
        leverage_response = self._make_request('POST', leverage_endpoint, data=leverage_data)
        if not leverage_response or leverage_response.get('code') != '00000':
            print(f"Failed to set leverage: {leverage_response}")
            return None
            
        # Then place the limit order
        endpoint = "/api/v2/mix/order/place-order"
        rounded_price = self.round_price(float(price))
        data = {
            "symbol": self.symbol,
            "productType": "USDT-FUTURES",
            "marginCoin": "USDT",
            "size": str(size),
            "side": side,  # buy or sell
            "tradeSide": trade_side,  # open or close
            "orderType": "limit",
            "marginMode": "isolated",
            "force": "gtc",
            "clientOid": f"strat_{int(time.time())}",
            "price": str(rounded_price)
        }
        
        response = self._make_request('POST', endpoint, data=data)
        if not response or response.get('code') != '00000':
            print(f"Failed to place limit order: {response}")
            return None
        
        print(f"Limit order placed: {response.get('data', {})}")
        return response.get('data')
    
    def get_order_filled_price(self, order_data):
        """
        Get the filled price of an order
        """
        if not order_data or 'orderId' not in order_data:
            return None
        
        order_id = order_data['orderId']
        endpoint = "/api/v2/mix/order/fills"
        params = {
            'symbol': self.symbol,
            'productType': 'USDT-FUTURES',
            'orderId': order_id
        }
        
        response = self._make_request('GET', endpoint, params=params)
        if not response or response.get('code') != '00000' or not response.get('data'):
            return None
        
        # Extract filled price from order fills
        fills = response.get('data', {})
        fill_list = fills.get('fillList', [])
        if not fill_list:
            print(f"No fills found for order {order_id}")
            return None
     
        if fill_list:
            price = float(fill_list[0].get('price', 0))
            return price
        else:
            print(f"Order fills data is not a non-empty list: {fills}")
            return None

    def run_strategy(self):
        """
        Main strategy loop
        """
        try:
            while True:
                
                # Get current price and print it
                params = {
                    'symbol': self.symbol,
                    'productType': 'USDT-FUTURES'
                }
                ticker_data = self._make_request('GET', "/api/v2/mix/market/ticker", params=params)
                
                if ticker_data and ticker_data.get('code') == '00000' and ticker_data.get('data'):
                    market_data = ticker_data['data'][0]
                    print(f"\nCurrent Market Data:")
                    print(f"Last Price: ${float(market_data['lastPr']):.2f}")
                    print(f"Bid Price: ${float(market_data['bidPr']):.2f}")
                    print(f"Ask Price: ${float(market_data['askPr']):.2f}")
                    print(f"24h Change: {float(market_data['change24h']) * 100:.2f}%")
                    print(f"24h Volume: {float(market_data['baseVolume']):.2f} BTC")
                else:
                    print("\nFailed to fetch current market data")
                
                # Check for available capital
                balance = self.get_account_balance()
                if balance is not None:
                    print(f"\nAccount Balance: ${balance:.2f}")
                
                # Monitor existing positions
                self.monitor_positions()
                
                # Look for new trade setups
                setup = self.find_trade_setup()
                if setup:
                    self.execute_trade(setup, self.quantity)
                
                print(f"\nWaiting for next check...")
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            self.emergency_close_all()
        except Exception as e:
            self.emergency_close_all()
            raise

    def round_price(self, price, tick_size=0.1):
        return round(round(price / tick_size) * tick_size, 1)

def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Optimized WMA Trading Strategy')
    parser.add_argument('--symbol', type=str, default=config.DEFAULT_SYMBOL, 
                        help=f'Trading pair symbol (default: {config.DEFAULT_SYMBOL})')
    parser.add_argument('--live', action='store_true', 
                        help='Run in live trading mode (place real orders)')
    parser.add_argument('--check-interval', type=int, default=60,
                        help='Time in seconds between strategy checks (default: 60)')
    
    args = parser.parse_args()

    # Initialize the optimized strategy
    strategy = BitgetTradingStrategy(
        api_key=config.API_KEY, 
        api_secret=config.API_SECRET, 
        passphrase=config.API_PASSPHRASE,
        symbol=args.symbol,
        check_interval=args.check_interval
    )
    
    # Run the strategy
    strategy.run_strategy()

if __name__ == "__main__":
    main()