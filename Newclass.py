from datetime import datetime
import pytz

# Set the correct time zone
utc_now = datetime.utcnow()
sast = pytz.timezone('Africa/Johannesburg')
sast_now = utc_now.replace(tzinfo=pytz.utc).astimezone(sast)

print("Current time in SAST:", sast_now.strftime('%Y-%m-%d %H:%M:%S'))



class SMCBacktester:
    def __init__(self, api_key="", api_secret="", test=True, symbol="SOL-USD",
                 timeframe="15m", risk_per_trade=0.02, lookback_periods=100):
        """
        Initialize the BitMEXLiveTrader class

        Parameters:
        api_key (str): BitMEX API key
        api_secret (str): BitMEX API secret
        test (bool): Whether to use testnet (True) or live (False)
        symbol (str): Trading symbol (default XBTUSD)
        timeframe (str): Candle timeframe (1m, 5m, 1h, etc.)
        risk_per_trade (float): Risk percentage per trade (0.02 = 2%)
        lookback_periods (int): Number of candles to fetch for analysis
        
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.test = test
        self.symbol = symbol
        self.timeframe = timeframe
        self.risk_per_trade = risk_per_trade
        self.lookback_periods = lookback_periods

        # Initialize BitMEX API client
        self.api = BitMEXTestAPI(
            api_key=self.api_key,
            api_secret=self.api_secret,
            test=self.test
        )

        # Trading state
        """
        self.in_trade = False
        self.trade_type = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.position_size = 0
        self.trades = []
        self.equity_curve = []
        self.initial_balance = 0
        self.current_balance = 0
        
        """

        logger.info(f"BitMEXLiveTrader initialized for {symbol} on {timeframe} timeframe")

        """
       
        """
        logger.info(f"Initializing SMC Strategy Backtester with {len(df)} candles")
        print(f"Initializing SMC Strategy Backtester with {len(df)} candles")

        
        # Convert column names to lowercase for consistency with original code
        #self.df.columns = [col.lower() for col in self.df.columns]
        self.initial_balance = 0
        self.balance = 0
        self.risk_per_trade = risk_per_trade

        # Add columns for SMC analysis
        self.df['higher_high'] = False
        self.df['lower_low'] = False
        self.df['bos_up'] = False
        self.df['bos_down'] = False
        self.df['choch_up'] = False
        self.df['choch_down'] = False
        self.df['bullish_fvg'] = False
        self.df['bearish_fvg'] = False

        # Trading variables
        self.in_trade = False
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.position_size = 0
        self.trade_type = None  # 'long' or 'short'

        # Performance tracking
        self.trades = []
        self.equity_curve = [initial_balance]

        logger.info(f"Strategy initialized with {initial_balance} account balance and {risk_per_trade*100}% risk per trade")
        print(f"Strategy initialized with {initial_balance} account balance and {risk_per_trade*100}% risk per trade")
       
    
    def get_market_data(self):
        """
        Fetch market data from BitMEX API or fallback to yfinance
        """
        try:
            # Try to get data from BitMEX API
            logger.info(f"Fetching {self.symbol} market data from BitMEX")
            print(f"Fetching {self.symbol} market data from BitMEX")

            # This is a placeholder - you'll need to implement BitMEX data fetching
            # based on your actual API implementation
            data = self.api.get_candles(
                symbol=self.symbol,
                timeframe=self.timeframe,
                count=self.lookback_periods
            )

            # Convert to DataFrame format expected by your SMC logic
            df = pd.DataFrame(data)

            #logger.info(f"Retrieved {len(df)} candles from BitMEX")
            print(f"Retrieved {len(df)} candles from BitMEX")

            return df

        except Exception as e:
            # Fallback to yfinance for data (mostly for testing purposes)
            #logger.warning(f"Failed to get data from BitMEX API: {str(e)}. Falling back to yfinance.")
            print(f"Failed to get data from BitMEX API: {str(e)}. Falling back to yfinance.")
            
            # For crypto we'll use a different ticker format
            crypto_ticker = self.symbol.replace('USD', '-USD')
            end_date = sast_now.strftime('%Y-%m-%d %H:%M:%S')
            start_date = end_date - timedelta(days=76t)  # Adjust as needed

            data = yf.download(
                crypto_ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=self.timeframe
            )

            logger.info(f"Retrieved {len(data)} candles from yfinance")
            print(f"Retrieved {len(data)} candles from yfinance")

            return data



    def identify_structure(self):
        """Identify market structure including highs, lows, BOS and CHoCH"""
        logger.info("Identifying market structure")
        print("Identifying market structure")
        df = self.df

        # Identify Higher Highs and Lower Lows (using a 5-candle lookback)
        window = 5
        for i in range(window, len(df)):
            # Higher High
            if df.iloc[i]['high'] > max(df.iloc[i-window:i]['high']):
                df.loc[df.index[i], 'higher_high'] = True

            # Lower Low
            if df.iloc[i]['low'] < min(df.iloc[i-window:i]['low']):
                df.loc[df.index[i], 'lower_low'] = True

        # Identify Break of Structure (BOS)
        prev_structure_high = df.iloc[0]['high']
        prev_structure_low = df.iloc[0]['low']
        structure_points_high = []
        structure_points_low = []

        for i in range(1, len(df)):
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']

            # Track significant structure points
            if df.iloc[i]['higher_high']:
                structure_points_high.append((i, current_high))

            if df.iloc[i]['lower_low']:
                structure_points_low.append((i, current_low))

            # BOS Up: Price breaks above recent structure high
            if len(structure_points_high) >= 2:
                last_high_idx, last_high = structure_points_high[-1]
                prev_high_idx, prev_high = structure_points_high[-2]

                if current_low > prev_high and i > last_high_idx + 1:
                    df.loc[df.index[i], 'bos_up'] = True
                    logger.info(f"Bullish BOS detected at index {i}, price: {current_low}")
                    print(f"Bullish BOS detected at index {i}, price: {current_low}")

            # BOS Down: Price breaks below recent structure low
            if len(structure_points_low) >= 2:
                last_low_idx, last_low = structure_points_low[-1]
                prev_low_idx, prev_low = structure_points_low[-2]

                if current_high < prev_low and i > last_low_idx + 1:
                    df.loc[df.index[i], 'bos_down'] = True
                    logger.info(f"Bearish BOS detected at index {i}, price: {current_high}")
                    print(f"Bearish BOS detected at index {i}, price: {current_high}")

        # Identify Change of Character (CHoCH)
        for i in range(window+1, len(df)):
            # Bullish CHoCH: After BOS up, creates higher low
            if df.iloc[i-1]['bos_up']:
                recent_lows = df.iloc[i-window:i]['low'].tolist()
                if min(recent_lows[:-1]) < recent_lows[-1]:
                    df.loc[df.index[i], 'choch_up'] = True
                    logger.info(f"Bullish CHoCH detected at index {i}")
                    print(f"Bullish CHoCH detected at index {i}")

            # Bearish CHoCH: After BOS down, creates lower high
            if df.iloc[i-1]['bos_down']:
                recent_highs = df.iloc[i-window:i]['high'].tolist()
                if max(recent_highs[:-1]) > recent_highs[-1]:
                    df.loc[df.index[i], 'choch_down'] = True
                    logger.info(f"Bearish CHoCH detected at index {i}")
                    print(f"Bearish CHoCH detected at index {i}")

        return df

    def identify_fvg(self):
        """Identify Fair Value Gaps (FVGs)"""
        logger.info("Identifying Fair Value Gaps")
        print("Identifying Fair Value Gaps")
        df = self.df

        # Create separate columns for FVG values to avoid tuple storage issues
        if 'bullish_fvg_low' not in df.columns:
            df['bullish_fvg_low'] = np.nan
            df['bullish_fvg_high'] = np.nan
            df['bullish_fvg_sl_index'] = np.nan

        if 'bearish_fvg_low' not in df.columns:
            df['bearish_fvg_low'] = np.nan
            df['bearish_fvg_high'] = np.nan
            df['bearish_fvg_sl_index'] = np.nan

        # Add mitigation tracking columns
        if 'bullish_fvg_mitigated' not in df.columns:
            df['bullish_fvg_mitigated'] = False

        if 'bearish_fvg_mitigated' not in df.columns:
            df['bearish_fvg_mitigated'] = False

        for i in range(2, len(df)):
            # Bullish FVG: Previous candle's low > Current candle's high
            if df.iloc[i-2]['low'] > df.iloc[i]['high']:
                # Check Gann Box (0-0.5 range for bullish FVGs)
                high_point = df.iloc[i-2]['high']
                low_point = df.iloc[i]['low']
                price_range = high_point - low_point
                fvg_low = df.iloc[i]['high']
                fvg_high = df.iloc[i-2]['low']

                # Calculate relative position in Gann Box (0 to 1)
                if price_range > 0:
                    relative_pos = (fvg_high - low_point) / price_range

                    # Only mark valid FVGs within 0-0.5 Gann range
                    if 0 <= relative_pos <= 0.5:
                        df.loc[df.index[i], 'bullish_fvg'] = True
                        # Store FVG values in separate columns
                        df.loc[df.index[i], 'bullish_fvg_low'] = fvg_low
                        df.loc[df.index[i], 'bullish_fvg_high'] = fvg_high
                        df.loc[df.index[i], 'bullish_fvg_sl_index'] = i

                        logger.info(f"Bullish FVG detected at index {i}, range: {fvg_low}-{fvg_high}")
                        print(f"Bullish FVG detected at index {i}, range: {fvg_low}-{fvg_high}")

            # Bearish FVG: Previous candle's high < Current candle's low
            if df.iloc[i-2]['high'] < df.iloc[i]['low']:
                # Check Gann Box (0.5-1 range for bearish FVGs)
                high_point = df.iloc[i]['high']
                low_point = df.iloc[i-2]['low']
                price_range = high_point - low_point
                fvg_low = df.iloc[i-2]['high']
                fvg_high = df.iloc[i]['low']

                # Calculate relative position in Gann Box (0 to 1)
                if price_range > 0:
                    relative_pos = (fvg_high - low_point) / price_range

                    # Only mark valid FVGs within 0.5-1 Gann range
                    if 0.5 <= relative_pos <= 1:
                        df.loc[df.index[i], 'bearish_fvg'] = True
                        # Store FVG values in separate columns
                        df.loc[df.index[i], 'bearish_fvg_low'] = fvg_low
                        df.loc[df.index[i], 'bearish_fvg_high'] = fvg_high
                        df.loc[df.index[i], 'bearish_fvg_sl_index'] = i

                        logger.info(f"Bearish FVG detected at index {i}, range: {fvg_low}-{fvg_high}")
                        print(f"Bearish FVG detected at index {i}, range: {fvg_low}-{fvg_high}")

        return df

    def check_fvg_mitigation(self, current_idx):
        """Check if any previously identified FVGs have been mitigated"""
        df = self.df

        # Loop through all previous candles
        for i in range(current_idx):
            # Check if the candle had a bullish FVG
            if df.iloc[i].get('bullish_fvg', False) and pd.notna(df.iloc[i].get('bullish_fvg_low')):
                fvg_low = df.iloc[i]['bullish_fvg_low']
                fvg_high = df.iloc[i]['bullish_fvg_high']

                # Check if price revisited the FVG area
                for j in range(i+1, current_idx+1):
                    if df.iloc[j]['low'] <= fvg_high and df.iloc[j]['high'] >= fvg_low:
                        # FVG has been mitigated
                        df.loc[df.index[i], 'bullish_fvg_mitigated'] = True
                        logger.info(f"Bullish FVG at index {i} has been mitigated at index {j}")
                        print(f"Bullish FVG at index {i} has been mitigated at index {j}")
                        break

            # Check if the candle had a bearish FVG
            if df.iloc[i].get('bearish_fvg', False) and pd.notna(df.iloc[i].get('bearish_fvg_low')):
                fvg_low = df.iloc[i]['bearish_fvg_low']
                fvg_high = df.iloc[i]['bearish_fvg_high']

                # Check if price revisited the FVG area
                for j in range(i+1, current_idx+1):
                    if df.iloc[j]['low'] <= fvg_high and df.iloc[j]['high'] >= fvg_low:
                        # FVG has been mitigated
                        df.loc[df.index[i], 'bearish_fvg_mitigated'] = True
                        logger.info(f"Bearish FVG at index {i} has been mitigated at index {j}")
                        print(f"Bearish FVG at index {i} has been mitigated at index {j}")
                        break

        return df

    def execute_trades(self):
        """Execute trades based on SMC signals"""
        logger.info("Starting trade execution backtesting")
        print("Starting trade execution backtesting")
        df = self.df

        # Iterate through each candle for backtesting
        for i in range(5, len(df)):
            current_price = df.iloc[i]['close']

            # Check if we're in a trade
            if self.in_trade:
                # Check if stop loss hit
                if (self.trade_type == 'long' and df.iloc[i]['low'] <= self.stop_loss) or \
                   (self.trade_type == 'short' and df.iloc[i]['high'] >= self.stop_loss):
                    # Stop loss hit
                    signal = {
                                #'price':  current_price,#set to actual price to enter or exit trade
                               'stop_loss': stop_loss,
                               #'take_profit': take_profit,
                                'action': "exit",
                                'reason': 'stop loss',
                               }
                     self.execute_signal(signal)
                    #self.exit_trade(i, self.stop_loss, 'stop_loss')

                # Check if take profit hit
                elif (self.trade_type == 'long' and df.iloc[i]['high'] >= self.take_profit) or \
                     (self.trade_type == 'short' and df.iloc[i]['low'] <= self.take_profit):
                    # Take profit hit
                    signal = {
                                #'price':  current_price,#set to actual price to enter or exit trade
                               # 'stop_loss': stop_loss,
                               'take_profit': take_profit,
                                'action': "exit",
                                'reason': 'takeprofit,
                               }
                     self.execute_signal(signal)
                    #self.exit_trade(i, self.take_profit, 'take_profit')

            else:
                # Update FVG mitigation status
                self.check_fvg_mitigation(i)

                # Check for new trade setups

                # Bullish setup: BOS up + CHoCH up + unmitigated bullish FVG
                if df.iloc[i-1]['bos_up'] and df.iloc[i]['choch_up']:
                    # Look back for unmitigated bullish FVGs
                    for j in range(i-10, i):
                        if j >= 0 and df.iloc[j].get('bullish_fvg', False) and not df.iloc[j].get('bullish_fvg_mitigated', False):
                            fvg_low = df.iloc[j]['bullish_fvg_low']
                            fvg_high = df.iloc[j]['bullish_fvg_high']
                            sl_idx = int(df.iloc[j]['bullish_fvg_sl_index'])

                            # Check if price is near the FVG
                            if fvg_low <= current_price <= fvg_high:
                                # Setup stop loss at the low of the FVG-forming candle
                                stop_loss = df.iloc[sl_idx]['low']
                               
                                # Find recent structure low for take profit
                                recent_lows = df.iloc[i-20:i]['low'].tolist()
                                min_idx = recent_lows.index(min(recent_lows))
                                take_profit = df.iloc[i-20+min_idx]['low']
                                 signal = {
                                'price':  current_price,#set to actual price to enter or exit trade
                                'stop_loss': stop_loss,
                               'take_profit': take_profit,
                                'action': "entry",
                                'side': 'long',
                               }
                               self.execute_signal(signal)
                                # Enter long trade
                                #self.enter_trade(i, current_price, stop_loss, take_profit, 'long')
                                
                                break

                # Bearish setup: BOS down + CHoCH down + unmitigated bearish FVG
                if df.iloc[i-1]['bos_down'] and df.iloc[i]['choch_down']:
                    # Look back for unmitigated bearish FVGs
                    for j in range(i-10, i):
                        if j >= 0 and df.iloc[j].get('bearish_fvg', False) and not df.iloc[j].get('bearish_fvg_mitigated', False):
                            fvg_low = df.iloc[j]['bearish_fvg_low']
                            fvg_high = df.iloc[j]['bearish_fvg_high']
                            sl_idx = int(df.iloc[j]['bearish_fvg_sl_index'])

                            # Check if price is near the FVG
                            if fvg_low <= current_price <= fvg_high:
                                # Setup stop loss at the high of the FVG-forming candle
                                stop_loss = df.iloc[sl_idx]['high']

                                # Find recent structure high for take profit
                                recent_highs = df.iloc[i-20:i]['high'].tolist()
                                max_idx = recent_highs.index(max(recent_highs))
                                take_profit = df.iloc[i-20+max_idx]['high']

                                # Enter short trade
                                #self.enter_trade(i, current_price, stop_loss, take_profit, 'short')
                                signal = {
                                'price':  current_price,#set to actual price to enter or exit trade
                                'stop_loss': stop_loss,
                               'take_profit': take_profit,
                                'action': "entry",
                                'side': 'short',
                                }
                                self.execute_signal(signal)
                                break

        # Close any open trade at the end of testing
        if self.in_trade:
            self.exit_trade(len(df) - 1, df.iloc[-1]['close'], 'end_of_test')

        logger.info(f"Trade execution completed with {len(self.trades)} trades")
        print(f"Trade execution completed with {len(self.trades)} trades")

        return self.trades, self.equity_curve

    

    

    def calculate_performance(self):
        """Calculate and return performance metrics"""
        if not self.trades:
            logger.warning("No trades to calculate performance metrics")
            print("No trades to calculate performance metrics")
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_return_pct': 0,
                'max_drawdown_pct': 0
            }

        # Calculate win rate
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(self.trades)

        # Calculate profit factor
        gross_profit = sum([t['pnl'] for t in self.trades if t['pnl'] > 0])
        gross_loss = abs(sum([t['pnl'] for t in self.trades if t['pnl'] <= 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Calculate total return
        total_return = (self.balance - self.initial_balance) / self.initial_balance

        # Calculate maximum drawdown
        peak = self.initial_balance
        max_drawdown = 0

        for balance in self.equity_curve:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak
            max_drawdown = max(max_drawdown, drawdown)

        performance = {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(self.trades) - len(winning_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': self.balance - self.initial_balance,
            'total_return_pct': total_return * 100,
            'max_drawdown_pct': max_drawdown * 100,
            'final_balance': self.balance
        }

        logger.info(f"Performance metrics: Win rate: {win_rate:.2%}, Profit factor: {profit_factor:.2f}, " +
                     f"Return: {total_return:.2%}, Max drawdown: {max_drawdown:.2%}")
        print(f"Performance metrics: Win rate: {win_rate:.2%}, Profit factor: {profit_factor:.2f}, " +
              f"Return: {total_return:.2%}, Max drawdown: {max_drawdown:.2%}")

        return performance

    def visualize_results(self, start_idx=0, end_idx=None):
        """Visualize backtesting results with trades and SMC patterns"""
        if end_idx is None:
            end_idx = len(self.df)

        # Create figure with subplots
        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot price data
        subset = self.df.iloc[start_idx:end_idx]
        ax.plot(subset.index, subset['close'], label='Close Price', color='black', linewidth=1)

        # Plot FVGs
        for i in range(start_idx, min(end_idx, len(self.df))):
            if self.df.iloc[i].get('bullish_fvg', False) and pd.notna(self.df.iloc[i].get('bullish_fvg_low')):
                fvg_low = self.df.iloc[i]['bullish_fvg_low']
                fvg_high = self.df.iloc[i]['bullish_fvg_high']
                mitigated = self.df.iloc[i].get('bullish_fvg_mitigated', False)
                color = 'lightgreen' if not mitigated else 'darkgreen'
                rect = patches.Rectangle((i-0.5, fvg_low), 1, fvg_high-fvg_low, linewidth=1,
                                        edgecolor=color, facecolor=color, alpha=0.3)
                ax.add_patch(rect)

            if self.df.iloc[i].get('bearish_fvg', False) and pd.notna(self.df.iloc[i].get('bearish_fvg_low')):
                fvg_low = self.df.iloc[i]['bearish_fvg_low']
                fvg_high = self.df.iloc[i]['bearish_fvg_high']
                mitigated = self.df.iloc[i].get('bearish_fvg_mitigated', False)
                color = 'lightcoral' if not mitigated else 'darkred'
                rect = patches.Rectangle((i-0.5, fvg_low), 1, fvg_high-fvg_low, linewidth=1,
                                        edgecolor=color, facecolor=color, alpha=0.3)
                ax.add_patch(rect)

        # Plot BOS and CHoCH
        bos_up_idx = subset[subset['bos_up'] == True].index
        bos_down_idx = subset[subset['bos_down'] == True].index
        choch_up_idx = subset[subset['choch_up'] == True].index
        choch_down_idx = subset[subset['choch_down'] == True].index

        ax.scatter(bos_up_idx, subset.loc[bos_up_idx, 'low'], color='green', marker='^', s=100, label='BOS Up')
        ax.scatter(bos_down_idx, subset.loc[bos_down_idx, 'high'], color='red', marker='v', s=100, label='BOS Down')
        ax.scatter(choch_up_idx, subset.loc[choch_up_idx, 'low'], color='blue', marker='^', s=80, label='CHoCH Up')
        ax.scatter(choch_down_idx, subset.loc[choch_down_idx, 'high'], color='purple', marker='v', s=80, label='CHoCH Down')

        # Plot trades
        for trade in self.trades:
            if start_idx <= trade['entry_index'] < end_idx:
                # Entry point
                color = 'green' if trade['type'] == 'long' else 'red'
                marker = '^' if trade['type'] == 'long' else 'v'
                ax.scatter(trade['entry_index'], trade['entry_price'], color=color, marker=marker, s=120, zorder=5)

                # Exit point
                if trade['exit_index'] < end_idx:
                    color = 'green' if trade['pnl'] > 0 else 'red'
                    ax.scatter(trade['exit_index'], trade['exit_price'], color=color, marker='o', s=120, zorder=5)

                    # Connect entry and exit
                    ax.plot([trade['entry_index'], trade['exit_index']],
                           [trade['entry_price'], trade['exit_price']],
                           color=color, linewidth=1, linestyle='--')

                    # Annotate PnL
                    ax.annotate(f"{trade['pnl']:.2f}",
                              (trade['exit_index'], trade['exit_price']),
                              textcoords="offset points",
                              xytext=(0,10),
                              ha='center')

        ax.set_title('SMC Backtest Results')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Plot equity curve
        fig2, ax2 = plt.subplots(figsize=(15, 5))
        ax2.plot(self.equity_curve, label='Account Balance', color='blue')
        ax2.set_title('Equity Curve')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        return fig, fig2
        
        
        
    def execute_signal(self, signal):
        """
        Execute the trading signal via BitMEX API
        """
        if signal is None:
            logger.info("No trading signal detected")
            print("No trading signal detected")
            return

        if signal['action'] == 'entry':
            self.execute_entry(signal)
        elif signal['action'] == 'exit':
            self.execute_exit(signal)




    def execute_entry(self, signal):
        """
        Execute an entry order based on the signal
        """
        side = signal['side']
        price = signal['price']
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']

        logger.info(f"=== OPENING {side.upper()} POSITION ===")
        print(f"=== OPENING {side.upper()} POSITION ===")

        # Calculate position size based on risk
        account_info = self.api.get_profile_info()
        self.current_balance = account_info['balance']  # Adjust based on your API's response format

        risk_amount = self.current_balance * self.risk_per_trade
        risk_per_unit = abs(price - stop_loss)
        position_size = risk_amount / risk_per_unit

        # Round position size to appropriate precision
        position_size = max(1, round(position_size))  # Minimum size of 1

        logger.info(f"Entry Price: {price}, Stop Loss: {stop_loss}, Take Profit: {take_profit}")
        logger.info(f"Position Size: {position_size} contracts")
        print(f"Entry Price: {price}, Stop Loss: {stop_loss}, Take Profit: {take_profit}")
        print(f"Position Size: {position_size} contracts")

        try:
            # Execute order via BitMEX API
            order_side = "Buy" if side == "long" else "Sell"
            order_result = self.api.open_test_position(side=order_side, quantity=position_size)

            # Update trading state
            self.in_trade = True
            self.trade_type = side
            self.entry_price = price
            self.stop_loss = stop_loss
            self.take_profit = take_profit
            self.position_size = position_size

            logger.info(f"{side.capitalize()} position opened successfully")
            print(f"{side.capitalize()} position opened successfully")

            # Record trade
            trade = {
                'entry_time': sast_now.strftime('%Y-%m-%d %H:%M:%S'),
                'entry_price': price,
                'side': side,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_amount': risk_amount
            }
            self.trades.append(trade)

        except Exception as e:
            logger.error(f"Error opening {side} position: {str(e)}")
            print(f"Error opening {side} position: {str(e)}")

    def execute_exit(self, signal):
        """
        Execute an exit order based on the signal
        """
        reason = signal['reason']
        price = signal['price']

        logger.info(f"=== CLOSING {self.trade_type.upper()} POSITION ({reason}) ===")
        print(f"=== CLOSING {self.trade_type.upper()} POSITION ({reason}) ===")

        try:
            # Close position via BitMEX API
            self.api.close_all_positions()

            # Calculate profit/loss
            if self.trade_type == 'long':
                pnl = (price - self.entry_price) * self.position_size
            else:  # short
                pnl = (self.entry_price - price) * self.position_size

            # Update the latest trade record
            current_trade = self.trades[-1]
            current_trade['exit_time'] = sast_now.strftime('%Y-%m-%d %H:%M:%S')
            current_trade['exit_price'] = price
            current_trade['exit_reason'] = reason
            current_trade['pnl'] = pnl

            # Update account balance
            self.api.get_profile_info()  # Refresh account info

            logger.info(f"Position closed with P&L: {pnl}")
            print(f"Position closed with P&L: {pnl}")

            # Reset trading state
            self.in_trade = False
            self.trade_type = None
            self.entry_price = None
            self.stop_loss = None
            self.take_profit = None
            self.position_size = 0

        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            print(f"Error closing position: {str(e)}")

    def run_backtest(self):
        """Run the full backtest process"""
        logger.info("Starting SMC backtest")
        print("Starting SMC backtest")

        # Identify market structure
        self.identify_structure()

        # Identify Fair Value Gaps
        self.identify_fvg()

        # Execute trades based on signals
        self.execute_trades()

        # Calculate performance metrics
        performance = self.calculate_performance()

        logger.info("Backtest completed successfully")
        print("Backtest completed successfully")

        return {
            'trades': self.trades,
            'performance': performance,
            'equity_curve': self.equity_curve,
            'processed_data': self.df
        }
    def run(self, scan_interval=60):
        """
        Main loop for live trading

        Parameters:
        scan_interval (int): Seconds between market scans
        """
        logger.info("Starting BitMEXLiveTrader")
        print("Starting BitMEXLiveTrader")

        # Display initial profile info
        logger.info("=== INITIAL PROFILE ===")
        print("=== INITIAL PROFILE ===")
        profile = self.api.get_profile_info()
        self.initial_balance = profile['balance']  # Adjust based on your API's response format
        self.current_balance = self.initial_balance

        # Main trading loop
        try:
            while True:
                logger.info(f"Scanning market at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Scanning market at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}")
                # Get market data
                df = self.get_market_data()
                # Identify market structure
                self.identify_structure()
                # Identify Fair Value Gaps
                self.identify_fvg()
                # Execute trades based on signals
                self.execute_trades()
                # Calculate performance metrics
                performance = self.calculate_performance()
                

        except KeyboardInterrupt:
            logger.info("BitMEXLiveTrader stopped by user")
            print("BitMEXLiveTrader stopped by user")

            # Close any open positions
            if self.in_trade:
                logger.info("=== CLOSING OPEN POSITIONS ===")
                print("=== CLOSING OPEN POSITIONS ===")
                self.api.close_all_positions()
                self.in_trade = False

            # Show final account info
            logger.info("=== FINAL PROFILE ===")
            print("=== FINAL PROFILE ===")
            self.api.get_profile_info()

        except Exception as e:
            logger.error(f"BitMEXLiveTrader error: {str(e)}")
            print(f"BitMEXLiveTrader error: {str(e)}")

            # Try to close positions on error
            if self.in_trade:
                try:
                    logger.info("=== CLOSING POSITIONS ON ERROR ===")
                    print("=== CLOSING POSITIONS ON ERROR ===")
                    self.api.close_all_positions()
                except:
                    pass

        finally:
            # Display summary
            logger.info("=== TRADING SUMMARY ===")
            print("=== TRADING SUMMARY ===")
            logger.info(f"Total trades: {len(self.trades)}")
            print(f"Total trades: {len(self.trades)}")

            winning_trades = sum(1 for trade in self.trades if trade.get('pnl', 0) > 0)
            logger.info(f"Winning trades: {winning_trades}")
            print(f"Winning trades: {winning_trades}")

            if len(self.trades) > 0:
                win_rate = winning_trades / len(self.trades)
                logger.info(f"Win rate: {win_rate:.2%}")
                print(f"Win rate: {win_rate:.2%}")

            total_pnl = sum(trade.get('pnl', 0) for trade in self.trades)
            logger.info(f"Total P&L: {total_pnl}")
            print(f"Total P&L: {total_pnl}")

            if self.initial_balance > 0:
                return_pct = total_pnl / self.initial_balance * 100
                logger.info(f"Return: {return_pct:.2f}%")
                print(f"Return: {return_pct:.2f}%")
        




class BitMEXLiveTrader:
    def __init__(self, api_key="", api_secret="", test=True, symbol="XBTUSD",
                 timeframe="1h", risk_per_trade=0.02, lookback_periods=100):
        """
        Initialize the BitMEXLiveTrader class

        Parameters:
        api_key (str): BitMEX API key
        api_secret (str): BitMEX API secret
        test (bool): Whether to use testnet (True) or live (False)
        symbol (str): Trading symbol (default XBTUSD)
        timeframe (str): Candle timeframe (1m, 5m, 1h, etc.)
        risk_per_trade (float): Risk percentage per trade (0.02 = 2%)
        lookback_periods (int): Number of candles to fetch for analysis
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.test = test
        self.symbol = symbol
        self.timeframe = timeframe
        self.risk_per_trade = risk_per_trade
        self.lookback_periods = lookback_periods

        # Initialize BitMEX API client
        self.api = BitMEXTestAPI(
            api_key=self.api_key,
            api_secret=self.api_secret,
            test=self.test
        )

        # Trading state
        self.in_trade = False
        self.trade_type = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.position_size = 0
        self.trades = []
        self.equity_curve = []
        self.initial_balance = 0
        self.current_balance = 0

        logger.info(f"BitMEXLiveTrader initialized for {symbol} on {timeframe} timeframe")
        print(f"BitMEXLiveTrader initialized for {symbol} on {timeframe} timeframe")

    def get_market_data(self):
        """
        Fetch market data from BitMEX API or fallback to yfinance
        """
        try:
            # Try to get data from BitMEX API
            logger.info(f"Fetching {self.symbol} market data from BitMEX")
            print(f"Fetching {self.symbol} market data from BitMEX")

            # This is a placeholder - you'll need to implement BitMEX data fetching
            # based on your actual API implementation
            data = self.api.get_candles(
                symbol=self.symbol,
                timeframe=self.timeframe,
                count=self.lookback_periods
            )

            # Convert to DataFrame format expected by your SMC logic
            df = pd.DataFrame(data)

            #logger.info(f"Retrieved {len(df)} candles from BitMEX")
            print(f"Retrieved {len(df)} candles from BitMEX")

            return df

        except Exception as e:
            # Fallback to yfinance for data (mostly for testing purposes)
            #logger.warning(f"Failed to get data from BitMEX API: {str(e)}. Falling back to yfinance.")
            print(f"Failed to get data from BitMEX API: {str(e)}. Falling back to yfinance.")
            
            # For crypto we'll use a different ticker format
            crypto_ticker = self.symbol.replace('USD', '-USD')
            end_date = sast_now.strftime('%Y-%m-%d %H:%M:%S')
            start_date = end_date - timedelta(days=30)  # Adjust as needed

            data = yf.download(
                crypto_ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=self.timeframe
            )

            logger.info(f"Retrieved {len(data)} candles from yfinance")
            print(f"Retrieved {len(data)} candles from yfinance")

            return data

    def analyze_market_structure(self, df):
        """
        Apply SMC analysis to the market data
        """
        logger.info("Analyzing market structure using SMC logic")
        print("Analyzing market structure using SMC logic")

        # Create temporary SMCStrategy instance for analysis
        smc = SMCStrategy(df, initial_balance=10000, risk_per_trade=self.risk_per_trade)

        # Identify market structure
        smc.identify_structure()

        # Identify Fair Value Gaps
        smc.identify_fvg()
        logger.info("Done Analyzing market structure using SMC logic")

        # Return the processed dataframe with SMC signals
        return smc.df

    def check_for_signals(self, df):
        """
        Check for trading signals based on the SMC analysis
        """
        logger.info("Checking for trading signals")
        print("Checking for trading signals")

        # Get the latest candle
        current_candle = df.iloc[-1]
        current_price = current_candle['close']

        signal = None

        # If we're already in a trade, check for exit signals
        if self.in_trade:
            if self.trade_type == 'long':
                # Check if stop loss hit
                if current_candle['low'] <= self.stop_loss:
                    signal = {'action': 'exit', 'reason': 'stop_loss', 'price': self.stop_loss}

                # Check if take profit hit
                elif current_candle['high'] >= self.take_profit:
                    signal = {'action': 'exit', 'reason': 'take_profit', 'price': self.take_profit}

            elif self.trade_type == 'short':
                # Check if stop loss hit
                if current_candle['high'] >= self.stop_loss:
                    signal = {'action': 'exit', 'reason': 'stop_loss', 'price': self.stop_loss}

                # Check if take profit hit
                elif current_candle['low'] <= self.take_profit:
                    signal = {'action': 'exit', 'reason': 'take_profit', 'price': self.take_profit}

        else:
            # Check for entry signals, similar to execute_trades in SMCStrategy
            i = len(df) - 1  # Index of current candle

            # Check for bullish setup
            if df.iloc[i-1].get('bos_up', False) and current_candle.get('choch_up', False):
                # Look back for unmitigated bullish FVGs
                for j in range(i-10, i):
                    if j >= 0 and df.iloc[j].get('bullish_fvg', False) and not df.iloc[j].get('bullish_fvg_mitigated', False):
                        fvg_low = df.iloc[j]['bullish_fvg_low']
                        fvg_high = df.iloc[j]['bullish_fvg_high']
                        sl_idx = int(df.iloc[j]['bullish_fvg_sl_index'])

                        # Check if price is near the FVG
                        if fvg_low <= current_price <= fvg_high:
                            # Setup stop loss at the low of the FVG-forming candle
                            stop_loss = df.iloc[sl_idx]['low']

                            # Find recent structure low for take profit
                            recent_lows = df.iloc[i-20:i]['low'].tolist()
                            min_idx = recent_lows.index(min(recent_lows))
                            take_profit = df.iloc[i-20+min_idx]['low']

                            signal = {
                                'action': 'entry',
                                'side': 'long',
                                'price': current_price,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit
                            }
                            break

            # Check for bearish setup
            if df.iloc[i-1].get('bos_down', False) and current_candle.get('choch_down', False):
                # Look back for unmitigated bearish FVGs
                for j in range(i-10, i):
                    if j >= 0 and df.iloc[j].get('bearish_fvg', False) and not df.iloc[j].get('bearish_fvg_mitigated', False):
                        fvg_low = df.iloc[j]['bearish_fvg_low']
                        fvg_high = df.iloc[j]['bearish_fvg_high']
                        sl_idx = int(df.iloc[j]['bearish_fvg_sl_index'])

                        # Check if price is near the FVG
                        if fvg_low <= current_price <= fvg_high:
                            # Setup stop loss at the high of the FVG-forming candle
                            stop_loss = df.iloc[sl_idx]['high']

                            # Find recent structure high for take profit
                            recent_highs = df.iloc[i-20:i]['high'].tolist()
                            max_idx = recent_highs.index(max(recent_highs))
                            take_profit = df.iloc[i-20+max_idx]['high']

                            signal = {
                                'action': 'entry',
                                'side': 'short',
                                'price': current_price,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit
                            }
                            break
        logger.info("Done Checking for trading signals")
        return signal

    def execute_signal(self, signal):
        """
        Execute the trading signal via BitMEX API
        """
        if signal is None:
            logger.info("No trading signal detected")
            print("No trading signal detected")
            return

        if signal['action'] == 'entry':
            self.execute_entry(signal)
        elif signal['action'] == 'exit':
            self.execute_exit(signal)

    def execute_entry(self, signal):
        """
        Execute an entry order based on the signal
        """
        side = signal['side']
        price = signal['price']
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']

        logger.info(f"=== OPENING {side.upper()} POSITION ===")
        print(f"=== OPENING {side.upper()} POSITION ===")

        # Calculate position size based on risk
        account_info = self.api.get_profile_info()
        self.current_balance = account_info['balance']  # Adjust based on your API's response format

        risk_amount = self.current_balance * self.risk_per_trade
        risk_per_unit = abs(price - stop_loss)
        position_size = risk_amount / risk_per_unit

        # Round position size to appropriate precision
        position_size = max(1, round(position_size))  # Minimum size of 1

        logger.info(f"Entry Price: {price}, Stop Loss: {stop_loss}, Take Profit: {take_profit}")
        logger.info(f"Position Size: {position_size} contracts")
        print(f"Entry Price: {price}, Stop Loss: {stop_loss}, Take Profit: {take_profit}")
        print(f"Position Size: {position_size} contracts")

        try:
            # Execute order via BitMEX API
            order_side = "Buy" if side == "long" else "Sell"
            order_result = self.api.open_test_position(side=order_side, quantity=position_size)

            # Update trading state
            self.in_trade = True
            self.trade_type = side
            self.entry_price = price
            self.stop_loss = stop_loss
            self.take_profit = take_profit
            self.position_size = position_size

            logger.info(f"{side.capitalize()} position opened successfully")
            print(f"{side.capitalize()} position opened successfully")

            # Record trade
            trade = {
                'entry_time': sast_now.strftime('%Y-%m-%d %H:%M:%S'),
                'entry_price': price,
                'side': side,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_amount': risk_amount
            }
            self.trades.append(trade)

        except Exception as e:
            logger.error(f"Error opening {side} position: {str(e)}")
            print(f"Error opening {side} position: {str(e)}")

    def execute_exit(self, signal):
        """
        Execute an exit order based on the signal
        """
        reason = signal['reason']
        price = signal['price']

        logger.info(f"=== CLOSING {self.trade_type.upper()} POSITION ({reason}) ===")
        print(f"=== CLOSING {self.trade_type.upper()} POSITION ({reason}) ===")

        try:
            # Close position via BitMEX API
            self.api.close_all_positions()

            # Calculate profit/loss
            if self.trade_type == 'long':
                pnl = (price - self.entry_price) * self.position_size
            else:  # short
                pnl = (self.entry_price - price) * self.position_size

            # Update the latest trade record
            current_trade = self.trades[-1]
            current_trade['exit_time'] = sast_now.strftime('%Y-%m-%d %H:%M:%S')
            current_trade['exit_price'] = price
            current_trade['exit_reason'] = reason
            current_trade['pnl'] = pnl

            # Update account balance
            self.api.get_profile_info()  # Refresh account info

            logger.info(f"Position closed with P&L: {pnl}")
            print(f"Position closed with P&L: {pnl}")

            # Reset trading state
            self.in_trade = False
            self.trade_type = None
            self.entry_price = None
            self.stop_loss = None
            self.take_profit = None
            self.position_size = 0

        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            print(f"Error closing position: {str(e)}")

    def run(self, scan_interval=60):
        """
        Main loop for live trading

        Parameters:
        scan_interval (int): Seconds between market scans
        """
        logger.info("Starting BitMEXLiveTrader")
        print("Starting BitMEXLiveTrader")

        # Display initial profile info
        logger.info("=== INITIAL PROFILE ===")
        print("=== INITIAL PROFILE ===")
        profile = self.api.get_profile_info()
        self.initial_balance = profile['balance']  # Adjust based on your API's response format
        self.current_balance = self.initial_balance

        # Main trading loop
        try:
            while True:
                logger.info(f"Scanning market at {sast_now.strftime('%Y-%m-%d %H:%M:%S') }")
                print(f"Scanning market at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}")

                # Get market data
                df = self.get_market_data()
                logger.info(f"Got market data :\n {df.tail(1) }")
                # Analyze market structure
                df_analyzed = self.analyze_market_structure(df)
                logger.info(f"Analysis of market structure :\n {df_analyzed.tail(1)}")
                # Check for signals
                signal = self.check_for_signals(df_analyzed)
                
                # Show current balance after trade execution
                if signal is not None:
                    logger.info(f"Found signal {signal}") 
                    logger.info(f"Results of Checking for signals :\n {signal.tail(1)}")
                    #  signal if any
                    self.execute_signal(signal)
                    #logger.info("=== CURRENT PROFILE ===")
                    print("=== CURRENT PROFILE ===")
                    profile = self.api.get_profile_info()
                    self.current_balance = profile['balance']  # Adjust based on your API's response format
                elif signal == None:
                    logger.info(f"No signal")
                    #continue
                    #  signal if any
                # Wait for next scan
                logger.info(f"Waiting for {scan_interval} seconds until next scan...")
                print(f"Waiting for {scan_interval} seconds until next scan...")
                time.sleep(scan_interval)

        except KeyboardInterrupt:
            logger.info("BitMEXLiveTrader stopped by user")
            print("BitMEXLiveTrader stopped by user")

            # Close any open positions
            if self.in_trade:
                logger.info("=== CLOSING OPEN POSITIONS ===")
                print("=== CLOSING OPEN POSITIONS ===")
                self.api.close_all_positions()
                self.in_trade = False

            # Show final account info
            logger.info("=== FINAL PROFILE ===")
            print("=== FINAL PROFILE ===")
            self.api.get_profile_info()

        except Exception as e:
            logger.error(f"BitMEXLiveTrader error: {str(e)}")
            print(f"BitMEXLiveTrader error: {str(e)}")

            # Try to close positions on error
            if self.in_trade:
                try:
                    logger.info("=== CLOSING POSITIONS ON ERROR ===")
                    print("=== CLOSING POSITIONS ON ERROR ===")
                    self.api.close_all_positions()
                except:
                    pass

        finally:
            # Display summary
            logger.info("=== TRADING SUMMARY ===")
            print("=== TRADING SUMMARY ===")
            logger.info(f"Total trades: {len(self.trades)}")
            print(f"Total trades: {len(self.trades)}")

            winning_trades = sum(1 for trade in self.trades if trade.get('pnl', 0) > 0)
            logger.info(f"Winning trades: {winning_trades}")
            print(f"Winning trades: {winning_trades}")

            if len(self.trades) > 0:
                win_rate = winning_trades / len(self.trades)
                logger.info(f"Win rate: {win_rate:.2%}")
                print(f"Win rate: {win_rate:.2%}")

            total_pnl = sum(trade.get('pnl', 0) for trade in self.trades)
            logger.info(f"Total P&L: {total_pnl}")
            print(f"Total P&L: {total_pnl}")

            if self.initial_balance > 0:
                return_pct = total_pnl / self.initial_balance * 100
                logger.info(f"Return: {return_pct:.2f}%")
                print(f"Return: {return_pct:.2f}%")
                
                
                

