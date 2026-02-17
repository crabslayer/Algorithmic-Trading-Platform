
"""
Execution models and paper trading for the trading platform.
Includes realistic slippage/cost estimation and paper trading simulation.
"""

import datetime
from datetime import timezone, timedelta
import logging

import numpy as np

_BEIJING_TZ = timezone(timedelta(hours=8))
import pandas as pd
from colorama import Fore, Style
from typing import Dict, List, Tuple

class RealisticExecutionModel:
    """Realistic execution model with slippage and market impact for A-shares"""
    
    def __init__(self, system):
        self.system = system
        self.logger = logging.getLogger(__name__)
        
        # A-share specific parameters
        self.price_tick = 0.01  # Minimum price movement
        self.lot_size = 100     # Minimum trading unit
        
        # Commission structure for A-shares
        self.commission_rate = 0.0008  # 0.08% (negotiable, using higher retail rate)
        self.stamp_duty = 0.001        # 0.1% on sells only
        self.min_commission = 5.0      # Minimum 5 CNY per trade
        
    def estimate_slippage(self, symbol: str, action: str, quantity: int) -> float:
        """
        Estimate slippage based on order size, liquidity, and market conditions
        
        Factors considered:
        - Order size relative to average volume
        - Current volatility
        - Time of day
        - Market regime
        """
        try:
            # Get recent market data
            df = self.system.get_market_data_cached(symbol, days=20)
            if df is None:
                return 0.002  # Default 0.2% if no data
            
            # Calculate average daily volume
            avg_volume = df['volume'].tail(10).mean()
            if avg_volume == 0:
                return 0.005  # High slippage for illiquid stocks
            
            # Order size impact
            order_impact = quantity / avg_volume
            
            # Volatility impact
            returns = df['close'].pct_change()
            volatility = returns.tail(10).std()
            
            # Time of day impact (use Beijing time for A-share market)
            now = datetime.datetime.now(_BEIJING_TZ)
            time_factor = self._get_time_impact(now)
            
            # Market regime impact
            regime_factor = 1.0
            if self.system.market_regime:
                regime = self.system.market_regime.get('regime', 'unknown')
                regime_factors = {
                    'crisis': 2.0,
                    'volatile_bull': 1.5,
                    'bear': 1.3,
                    'range_bound': 1.0,
                    'steady_bull': 0.8
                }
                regime_factor = regime_factors.get(regime, 1.0)
            
            # Calculate base slippage
            base_slippage = 0.001  # 0.1% base
            
            # Order size component (linear + square root for large orders)
            size_impact = order_impact * 0.1 + np.sqrt(order_impact) * 0.02
            
            # Volatility component
            vol_impact = volatility * 10  # Roughly converts daily vol to slippage
            
            # Direction component (buying usually has more slippage in A-shares)
            direction_factor = 1.2 if action == 'BUY' else 1.0
            
            # Combine all factors
            total_slippage = (base_slippage + size_impact + vol_impact) * time_factor * regime_factor * direction_factor
            
            # Cap slippage at reasonable levels
            max_slippage = 0.01  # 1% maximum
            total_slippage = min(total_slippage, max_slippage)
            
            self.logger.info(f"Slippage estimate for {symbol}: {total_slippage:.3%} "
                           f"(impact: {order_impact:.1%}, vol: {volatility:.1%})")
            
            return total_slippage
            
        except Exception as e:
            self.logger.error(f"Error estimating slippage: {e}")
            return 0.003  # Default 0.3% on error
    
    def _get_time_impact(self, current_time: datetime.datetime) -> float:
        """Get time-based impact factor for A-share market"""
        hour = current_time.hour
        minute = current_time.minute
        
        # Opening auction (9:15-9:25) and first 30 min (9:00-9:29): high impact
        if hour == 9 and minute < 30:
            return 1.5
        
        # Closing 30 minutes: high impact
        elif hour == 14 and minute >= 30:
            return 1.5
        
        # Lunch reopening: medium impact
        elif hour == 13 and minute < 30:
            return 1.2
        
        # Normal trading hours: normal impact
        else:
            return 1.0
    
    def calculate_transaction_costs(self, symbol: str, action: str, quantity: int, price: float) -> Dict:
        """Calculate all transaction costs for A-share trading"""
        order_value = quantity * price
        
        # Commission (both buy and sell)
        commission = max(order_value * self.commission_rate, self.min_commission)
        
        # Stamp duty (sell only)
        stamp_duty = order_value * self.stamp_duty if action == 'SELL' else 0
        
        # Slippage estimate
        slippage_rate = self.estimate_slippage(symbol, action, quantity)
        slippage_cost = order_value * slippage_rate
        
        # Total costs
        total_costs = commission + stamp_duty + slippage_cost
        
        return {
            'commission': commission,
            'stamp_duty': stamp_duty,
            'slippage': slippage_cost,
            'slippage_rate': slippage_rate,
            'total_costs': total_costs,
            'cost_rate': total_costs / order_value,
            'estimated_fill_price': price * (1 + slippage_rate) if action == 'BUY' else price * (1 - slippage_rate)
        }
    
    def validate_order(self, symbol: str, action: str, quantity: int, price: float) -> Tuple[bool, str]:
        """Validate order against A-share trading rules"""
        # Check lot size (must be multiple of 100)
        if quantity % self.lot_size != 0:
            return False, f"Quantity must be multiple of {self.lot_size}"
        
        # Check minimum order size
        if quantity < self.lot_size:
            return False, f"Minimum order size is {self.lot_size} shares"
        
        # Check price tick compliance
        if price % self.price_tick > 0.0001:  # Small tolerance for float precision
            return False, f"Price must be multiple of {self.price_tick}"
        
        # Check daily price limits (±10% for main board, ±20% for STAR/ChiNext with registration)
        df = self.system.get_market_data_cached(symbol, days=2)
        if df is not None and len(df) >= 2:
            prev_close = df['close'].iloc[-2]
            
            # Determine board type (simplified - you may need more sophisticated logic)
            if symbol.startswith('688') or symbol.startswith('300'):
                # STAR Market or ChiNext with registration system
                price_limit = 0.20
            else:
                # Main board
                price_limit = 0.10
            
            price_change = (price - prev_close) / prev_close
            if abs(price_change) > price_limit:
                return False, f"Price exceeds daily limit ({price_change:.1%} vs ±{price_limit:.0%})"
        
        return True, "Order validated"

# ============================================================================
# PAPER TRADING MODE (Original)
# ============================================================================

class PaperTradingMode:
    """Paper trading mode for testing strategies without real money"""
    
    def __init__(self, real_system, initial_capital: float = 1000000):
        self.real_system = real_system
        self.initial_capital = initial_capital
        self.paper_cash = initial_capital
        self.paper_positions = {}
        self.paper_trades = []
        self.paper_orders = []
        self.start_date = datetime.datetime.now(_BEIJING_TZ).date()
        self.execution_model = RealisticExecutionModel(real_system)
        
        # Performance tracking
        self.daily_values = []
        self.peak_value = initial_capital
        self.max_drawdown = 0
        
        # Initialize paper trading database tables
        self._init_paper_db()
    
    def _init_paper_db(self):
        """Initialize paper trading tables in database"""
        # Tables are already created in DatabaseManager
        pass
    
    def execute_trade(self, symbol: str, action: str, quantity: int, order_price: float, 
                     signal_strength: float = 0, signal_reason: str = "") -> Dict:
        """Execute paper trade with realistic simulation"""
        
        # Validate order
        valid, reason = self.execution_model.validate_order(symbol, action, quantity, order_price)
        if not valid:
            return {'success': False, 'reason': reason}
        
        # Calculate transaction costs
        costs = self.execution_model.calculate_transaction_costs(symbol, action, quantity, order_price)
        fill_price = costs['estimated_fill_price']
        
        # Check if we have enough cash for buy orders
        if action == 'BUY':
            required_cash = quantity * fill_price + costs['commission']
            if required_cash > self.paper_cash:
                return {'success': False, 'reason': f'Insufficient cash: need {required_cash:.2f}, have {self.paper_cash:.2f}'}
        
        # Check if we have position for sell orders
        elif action == 'SELL':
            if symbol not in self.paper_positions or self.paper_positions[symbol]['quantity'] < quantity:
                return {'success': False, 'reason': 'Insufficient position to sell'}
        
        # Execute trade
        timestamp = datetime.datetime.now(_BEIJING_TZ)
        trade_pnl = 0
        return_pct = 0
        
        if action == 'BUY':
            # Update cash
            total_cost = quantity * fill_price + costs['commission']
            self.paper_cash -= total_cost
            
            # Update or create position
            if symbol in self.paper_positions:
                pos = self.paper_positions[symbol]
                new_quantity = pos['quantity'] + quantity
                new_cost_basis = pos['cost_basis'] + total_cost
                new_avg_price = new_cost_basis / new_quantity
                
                self.paper_positions[symbol] = {
                    'quantity': new_quantity,
                    'avg_price': new_avg_price,
                    'cost_basis': new_cost_basis,
                    'last_price': fill_price,
                    'last_update': timestamp
                }
            else:
                self.paper_positions[symbol] = {
                    'quantity': quantity,
                    'avg_price': fill_price,
                    'cost_basis': total_cost,
                    'last_price': fill_price,
                    'last_update': timestamp
                }
        
        else:  # SELL
            pos = self.paper_positions[symbol]
            
            # Calculate proceeds and P&L
            gross_proceeds = quantity * fill_price
            net_proceeds = gross_proceeds - costs['commission'] - costs['stamp_duty']
            self.paper_cash += net_proceeds
            
            # Calculate P&L
            cost_basis = pos['avg_price'] * quantity
            trade_pnl = net_proceeds - cost_basis
            return_pct = (trade_pnl / cost_basis) * 100
            
            # Update position
            remaining_quantity = pos['quantity'] - quantity
            if remaining_quantity == 0:
                del self.paper_positions[symbol]
            else:
                pos['quantity'] = remaining_quantity
                pos['cost_basis'] = pos['avg_price'] * remaining_quantity
        
        # Record trade
        portfolio_value = self._calculate_portfolio_value()
        
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'order_price': order_price,
            'fill_price': fill_price,
            'commission': costs['commission'],
            'stamp_duty': costs['stamp_duty'],
            'slippage': costs['slippage'],
            'slippage_rate': costs['slippage_rate'],
            'pnl': trade_pnl,
            'return_pct': return_pct,
            'cash_after': self.paper_cash,
            'portfolio_value': portfolio_value,
            'signal_strength': signal_strength,
            'signal_reason': signal_reason
        }
        
        self.paper_trades.append(trade_record)
        self._save_trade_to_db(trade_record)
        
        # Log trade
        action_color = Fore.GREEN if action == 'BUY' else Fore.RED
        print(f"\n{Fore.CYAN}[PAPER TRADE]{Style.RESET_ALL} {action_color}{action}{Style.RESET_ALL} {symbol}")
        print(f"  Quantity: {quantity:,}")
        print(f"  Order price: {order_price:.2f}")
        print(f"  Fill price: {fill_price:.2f} (slippage: {costs['slippage_rate']:.2%})")
        print(f"  Commission: {costs['commission']:.2f}")
        if action == 'SELL':
            print(f"  Stamp duty: {costs['stamp_duty']:.2f}")
            pnl_color = Fore.GREEN if trade_pnl > 0 else Fore.RED
            print(f"  P&L: {pnl_color}{trade_pnl:+.2f} ({return_pct:+.1f}%){Style.RESET_ALL}")
        print(f"  Cash: {self.paper_cash:,.2f}")
        print(f"  Portfolio value: {portfolio_value:,.2f}")
        
        return {
            'success': True,
            'trade': trade_record
        }
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = 0
        
        for symbol, pos in self.paper_positions.items():
            # Get current price
            df = self.real_system.get_market_data_cached(symbol)
            if df is not None and len(df) > 0:
                current_price = df['close'].iloc[-1]
                pos['last_price'] = current_price
                positions_value += pos['quantity'] * current_price
            else:
                # Use last known price
                positions_value += pos['quantity'] * pos.get('last_price', pos['avg_price'])
        
        return self.paper_cash + positions_value
    
    def update_daily_performance(self):
        """Update daily performance metrics"""
        today = datetime.datetime.now(_BEIJING_TZ).date()
        portfolio_value = self._calculate_portfolio_value()
        
        # Calculate positions value
        positions_value = portfolio_value - self.paper_cash
        
        # Calculate returns
        total_return = (portfolio_value / self.initial_capital - 1) * 100
        
        # Calculate daily return
        daily_return = 0
        if self.daily_values:
            last_value = self.daily_values[-1]['portfolio_value']
            daily_return = (portfolio_value / last_value - 1) * 100
        
        # Update max drawdown
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        drawdown = (portfolio_value - self.peak_value) / self.peak_value * 100
        self.max_drawdown = min(self.max_drawdown, drawdown)
        
        # Count today's trades
        today_trades = len([t for t in self.paper_trades 
                          if t['timestamp'].date() == today])
        
        # Save to tracking
        performance = {
            'date': today,
            'portfolio_value': portfolio_value,
            'cash': self.paper_cash,
            'positions_value': positions_value,
            'daily_return': daily_return,
            'total_return': total_return,
            'max_drawdown': self.max_drawdown,
            'n_trades': today_trades,
            'n_positions': len(self.paper_positions)
        }
        
        self.daily_values.append(performance)
        self._save_performance_to_db(performance)
        
        return performance
    
    def generate_report(self) -> Dict:
        """Generate comprehensive paper trading report"""
        if not self.paper_trades:
            return {'error': 'No trades to analyze'}
        
        # Update current performance
        current_performance = self.update_daily_performance()
        
        # Analyze trades
        trades_df = pd.DataFrame(self.paper_trades)
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        
        # Calculate metrics
        total_trades = len(self.paper_trades)
        buy_trades_count = len(trades_df[trades_df['action'] == 'BUY'])
        sell_trades_count = len(sell_trades)
        
        # Win rate and P&L stats
        if len(sell_trades) > 0:
            winning_trades = sell_trades[sell_trades['pnl'] > 0]
            win_rate = len(winning_trades) / len(sell_trades) * 100
            
            avg_win = winning_trades['return_pct'].mean() if len(winning_trades) > 0 else 0
            avg_loss = sell_trades[sell_trades['pnl'] < 0]['return_pct'].mean() if len(sell_trades[sell_trades['pnl'] < 0]) > 0 else 0
            
            total_pnl = sell_trades['pnl'].sum()
            best_trade = sell_trades.loc[sell_trades['return_pct'].idxmax()]
            worst_trade = sell_trades.loc[sell_trades['return_pct'].idxmin()]
        else:
            win_rate = avg_win = avg_loss = total_pnl = 0
            best_trade = worst_trade = None
        
        # Calculate Sharpe ratio
        if len(self.daily_values) > 1:
            returns = pd.Series([d['daily_return'] for d in self.daily_values[1:]])
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe = 0
        
        # Transaction cost analysis
        total_commission = trades_df['commission'].sum()
        total_stamp_duty = trades_df['stamp_duty'].sum()
        total_slippage = trades_df['slippage'].sum()
        total_costs = total_commission + total_stamp_duty + total_slippage
        
        report = {
            'summary': {
                'start_date': self.start_date,
                'end_date': datetime.datetime.now(_BEIJING_TZ).date(),
                'days_active': (datetime.datetime.now(_BEIJING_TZ).date() - self.start_date).days,
                'initial_capital': self.initial_capital,
                'final_value': current_performance['portfolio_value'],
                'total_return': current_performance['total_return'],
                'max_drawdown': self.max_drawdown,
                'sharpe_ratio': sharpe
            },
            'trading_stats': {
                'total_trades': total_trades,
                'buy_trades': buy_trades_count,
                'sell_trades': sell_trades_count,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_pnl': total_pnl,
                'best_trade': {
                    'symbol': best_trade['symbol'],
                    'return': best_trade['return_pct'],
                    'pnl': best_trade['pnl']
                } if best_trade is not None else None,
                'worst_trade': {
                    'symbol': worst_trade['symbol'],
                    'return': worst_trade['return_pct'],
                    'pnl': worst_trade['pnl']
                } if worst_trade is not None else None
            },
            'cost_analysis': {
                'total_commission': total_commission,
                'total_stamp_duty': total_stamp_duty,
                'total_slippage': total_slippage,
                'total_costs': total_costs,
                'costs_pct_of_volume': total_costs / trades_df['quantity'].dot(trades_df['fill_price']) * 100 if len(trades_df) > 0 else 0
            },
            'current_state': {
                'cash': self.paper_cash,
                'positions': len(self.paper_positions),
                'positions_list': [
                    {
                        'symbol': symbol,
                        'quantity': pos['quantity'],
                        'avg_price': pos['avg_price'],
                        'last_price': pos.get('last_price', pos['avg_price']),
                        'unrealized_pnl': pos['quantity'] * (pos.get('last_price', pos['avg_price']) - pos['avg_price']),
                        'unrealized_pct': (pos.get('last_price', pos['avg_price']) / pos['avg_price'] - 1) * 100
                    }
                    for symbol, pos in self.paper_positions.items()
                ]
            }
        }
        
        return report
    
    def display_report(self):
        """Display formatted paper trading report"""
        report = self.generate_report()
        
        if 'error' in report:
            print(f"{Fore.RED}{report['error']}{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}PAPER TRADING PERFORMANCE REPORT{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
        # Summary
        summary = report['summary']
        print(f"\n{Fore.YELLOW}Summary:{Style.RESET_ALL}")
        print(f"  Period: {summary['start_date']} to {summary['end_date']} ({summary['days_active']} days)")
        print(f"  Initial capital: {summary['initial_capital']:,.0f}")
        print(f"  Final value: {summary['final_value']:,.0f}")
        
        return_color = Fore.GREEN if summary['total_return'] > 0 else Fore.RED
        print(f"  Total return: {return_color}{summary['total_return']:+.2f}%{Style.RESET_ALL}")
        
        dd_color = Fore.YELLOW if summary['max_drawdown'] > -10 else Fore.RED
        print(f"  Max drawdown: {dd_color}{summary['max_drawdown']:.2f}%{Style.RESET_ALL}")
        print(f"  Sharpe ratio: {summary['sharpe_ratio']:.2f}")
        
        # Trading stats
        stats = report['trading_stats']
        print(f"\n{Fore.YELLOW}Trading Statistics:{Style.RESET_ALL}")
        print(f"  Total trades: {stats['total_trades']} ({stats['buy_trades']} buys, {stats['sell_trades']} sells)")
        
        if stats['sell_trades'] > 0:
            wr_color = Fore.GREEN if stats['win_rate'] > 50 else Fore.YELLOW if stats['win_rate'] > 40 else Fore.RED
            print(f"  Win rate: {wr_color}{stats['win_rate']:.1f}%{Style.RESET_ALL}")
            print(f"  Avg win: +{stats['avg_win']:.2f}%")
            print(f"  Avg loss: {stats['avg_loss']:.2f}%")
            
            pnl_color = Fore.GREEN if stats['total_pnl'] > 0 else Fore.RED
            print(f"  Total P&L: {pnl_color}{stats['total_pnl']:+,.2f}{Style.RESET_ALL}")
            
            if stats['best_trade']:
                print(f"  Best trade: {stats['best_trade']['symbol']} (+{stats['best_trade']['return']:.1f}%)")
            if stats['worst_trade']:
                print(f"  Worst trade: {stats['worst_trade']['symbol']} ({stats['worst_trade']['return']:.1f}%)")
        
        # Cost analysis
        costs = report['cost_analysis']
        print(f"\n{Fore.YELLOW}Transaction Costs:{Style.RESET_ALL}")
        print(f"  Commission: {costs['total_commission']:,.2f}")
        print(f"  Stamp duty: {costs['total_stamp_duty']:,.2f}")
        print(f"  Slippage: {costs['total_slippage']:,.2f}")
        print(f"  Total costs: {costs['total_costs']:,.2f} ({costs['costs_pct_of_volume']:.2%} of volume)")
        
        # Current positions
        current = report['current_state']
        print(f"\n{Fore.YELLOW}Current State:{Style.RESET_ALL}")
        print(f"  Cash: {current['cash']:,.2f}")
        print(f"  Positions: {current['positions']}")
        
        if current['positions_list']:
            print(f"\n  {Fore.CYAN}Open Positions:{Style.RESET_ALL}")
            for pos in current['positions_list']:
                pnl_color = Fore.GREEN if pos['unrealized_pnl'] > 0 else Fore.RED
                print(f"    {pos['symbol']}: {pos['quantity']} @ {pos['avg_price']:.2f}")
                print(f"      Current: {pos['last_price']:.2f} ({pnl_color}{pos['unrealized_pct']:+.1f}%{Style.RESET_ALL})")
    
    def _save_trade_to_db(self, trade: Dict):
        """Save paper trade to database"""
        with self.real_system.db.get_connection() as conn:
            conn.execute("""
                INSERT INTO paper_trades 
                (timestamp, symbol, action, quantity, order_price, fill_price,
                 commission, stamp_duty, slippage, pnl, return_pct,
                 cash_after, portfolio_value, signal_strength, signal_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade['timestamp'], trade['symbol'], trade['action'],
                trade['quantity'], trade['order_price'], trade['fill_price'],
                trade['commission'], trade['stamp_duty'], trade['slippage'],
                trade['pnl'], trade['return_pct'], trade['cash_after'],
                trade['portfolio_value'], trade['signal_strength'], trade['signal_reason']
            ))
            conn.commit()
    
    def _save_performance_to_db(self, performance: Dict):
        """Save daily performance to database"""
        with self.real_system.db.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO paper_performance
                (date, portfolio_value, cash, positions_value, daily_return,
                 total_return, max_drawdown, n_trades, n_positions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                performance['date'], performance['portfolio_value'],
                performance['cash'], performance['positions_value'],
                performance['daily_return'], performance['total_return'],
                performance['max_drawdown'], performance['n_trades'],
                performance['n_positions']
            ))
            conn.commit()
