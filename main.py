
"""
Main entry point and interactive menu for the Enhanced Manual Trading System.
All user inputs are wrapped in safe helpers to prevent crashes on bad input.
"""

from colorama import init, Fore, Style

from .execution import PaperTradingMode
from .trading_system import EnhancedManualTradingSystem

# Initialize colorama for colored output
init(autoreset=True)


# ---------------------------------------------------------------------------
# Safe input helpers
# ---------------------------------------------------------------------------

def safe_int(prompt: str, default: int | None = None) -> int | None:
    """Prompt the user for an integer.  Returns *default* on empty/bad input."""
    raw = input(prompt).strip()
    if not raw and default is not None:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"{Fore.RED}Invalid number, please enter an integer.{Style.RESET_ALL}")
        return None


def safe_float(prompt: str, default: float | None = None) -> float | None:
    """Prompt the user for a float.  Returns *default* on empty/bad input."""
    raw = input(prompt).strip()
    if not raw and default is not None:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"{Fore.RED}Invalid number, please enter a decimal number.{Style.RESET_ALL}")
        return None


def require_int(prompt: str, default: int | None = None) -> int:
    """Keep asking until we get a valid integer."""
    while True:
        v = safe_int(prompt, default)
        if v is not None:
            return v


def require_float(prompt: str, default: float | None = None) -> float:
    """Keep asking until we get a valid float."""
    while True:
        v = safe_float(prompt, default)
        if v is not None:
            return v


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

def main():
    """Main function with interactive menu"""
    system = EnhancedManualTradingSystem()
    
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}ENHANCED MANUAL TRADING SYSTEM{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print("Professional-grade A-share trading with all enhancements")
    print(f"{Fore.GREEN}✓ Pre-market checks{Style.RESET_ALL}")
    print(f"{Fore.GREEN}✓ Realistic execution model{Style.RESET_ALL}")
    print(f"{Fore.GREEN}✓ Paper trading mode{Style.RESET_ALL}")
    print(f"{Fore.GREEN}✓ Enhanced risk management{Style.RESET_ALL}")
    print(f"{Fore.GREEN}✓ Improved ML ensemble models{Style.RESET_ALL}")
    print(f"{Fore.GREEN}✓ Adaptive signal thresholds{Style.RESET_ALL}")
    print(f"{Fore.GREEN}✓ Robust error handling{Style.RESET_ALL}")
    
    try:
        _main_loop(system)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Interrupted. Cleaning up...{Style.RESET_ALL}")
    finally:
        if system.monitoring_active:
            system.stop_monitoring()
        system.db.close_connection()
        print(f"\n{Fore.CYAN}Thank you for using Enhanced Manual Trading System{Style.RESET_ALL}")


def _main_loop(system):
    """Main interactive menu loop"""
    while True:
        print(f"\n{Fore.YELLOW}=== MAIN MENU ==={Style.RESET_ALL}")
        print("1. Daily trading report (with pre-market checks)")
        print("2. Check positions")
        print("3. Generate trading signals")
        print("4. Add position")
        print("5. Remove position (sell)")
        print("6. Risk dashboard")
        print("7. Watchlist management")
        print("8. Backtest strategy")
        print("9. Performance analysis")
        print("10. Start/stop monitoring")
        print("11. Market analysis")
        print("12. View alerts")
        print("13. System controls")
        print("14. Database maintenance")
        print("15. Retrain ML model")
        print("16. Analyze signal weakness")
        print("17. Paper trading")
        print("18. Pre-market checklist")
        print("19. Estimate trade costs")
        print("20. ML model health")
        print("21. Signal performance tracking")
        print("22. Error summary")
        print("0. Exit")
        
        choice = input("\nEnter choice: ").strip()
        
        if choice == '1':
            system.daily_report_enhanced()
            
        elif choice == '2':
            system.check_positions_enhanced()
            
        elif choice == '3':
            sub_menu = True
            while sub_menu:
                print(f"\n{Fore.YELLOW}Signal Generation:{Style.RESET_ALL}")
                print("1. Single symbol")
                print("2. All positions")
                print("3. Watchlist")
                print("4. Universe scan")
                print("5. Diagnose signals")
                print("0. Back")
                
                sub_choice = input("Choice: ").strip()
                
                if sub_choice == '1':
                    symbol = input("Enter symbol: ").strip().upper()
                    signals = system.generate_enhanced_signals(symbol)
                    
                    print(f"\n{Fore.CYAN}Signals for {symbol}:{Style.RESET_ALL}")
                    signal_color = Fore.GREEN if 'BUY' in signals['signal'] else Fore.RED if 'SELL' in signals['signal'] else Fore.YELLOW
                    print(f"Signal: {signal_color}{signals['signal']}{Style.RESET_ALL}")
                    print(f"Strength: {signals['strength']:.1f}")
                    print(f"Buy score: {signals['buy_score']:.1f}")
                    print(f"Sell score: {signals['sell_score']:.1f}")
                    print(f"Volatility: {signals['volatility']:.1%}")
                    print(f"Data quality: {signals.get('data_quality', 0):.1%}")
                    print(f"ML confidence: {signals.get('ml_confidence', 0):.1%}")
                    print(f"ML quality: {signals.get('ml_quality', 'unknown')}")
                    print(f"Buy signals: {signals.get('buy_signals', 0)}")
                    print(f"Sell signals: {signals.get('sell_signals', 0)}")
                    print(f"Reasons: {', '.join(signals['reasons'])}")
                    
                    # Show adaptive thresholds used
                    if 'thresholds_used' in signals:
                        print(f"\nThresholds used:")
                        print(f"  Buy: {signals['thresholds_used']['buy_threshold']:.1f}")
                        print(f"  Sell: {signals['thresholds_used']['sell_threshold']:.1f}")
                        print(f"  Min signals: {signals['thresholds_used']['min_signals']}")
                    
                    # Execution timing
                    timing = system.get_optimal_execution_time(symbol, 'BUY' if 'BUY' in signals['signal'] else 'SELL')
                    print(f"\nExecution timing: {timing['suggestions'][0] if timing['suggestions'] else 'Neutral'}")
                    
                elif sub_choice == '2':
                    for symbol in system.positions:
                        signals = system.generate_enhanced_signals(symbol)
                        signal_color = Fore.GREEN if 'BUY' in signals['signal'] else Fore.RED if 'SELL' in signals['signal'] else Fore.YELLOW
                        print(f"\n{symbol}: {signal_color}{signals['signal']}{Style.RESET_ALL} "
                              f"(strength: {signals['strength']:.1f}, ML: {signals.get('ml_confidence', 0):.1%}, "
                              f"quality: {signals.get('data_quality', 0):.1%}, "
                              f"confirmations: {signals.get('buy_signals', 0) + signals.get('sell_signals', 0)})")
                
                elif sub_choice == '3':
                    for symbol in system.watchlist:
                        signals = system.generate_enhanced_signals(symbol)
                        signal_color = Fore.GREEN if 'BUY' in signals['signal'] else Fore.RED if 'SELL' in signals['signal'] else Fore.YELLOW
                        print(f"\n{symbol}: {signal_color}{signals['signal']}{Style.RESET_ALL} "
                              f"(strength: {signals['strength']:.1f}, ML: {signals.get('ml_confidence', 0):.1%}, "
                              f"quality: {signals.get('data_quality', 0):.1%}, "
                              f"confirmations: {signals.get('buy_signals', 0) + signals.get('sell_signals', 0)})")
                
                elif sub_choice == '4':
                    print("Scanning universe (this may take a moment)...")
                    universe = system.get_universe_symbols()[:20]
                    strong_signals = []
                    
                    for symbol in universe:
                        if symbol not in system.positions:
                            signals = system.generate_enhanced_signals(symbol)
                            if signals['signal'] in ['BUY', 'STRONG_BUY'] and signals['strength'] > 2 and signals.get('buy_signals', 0) >= 2:
                                strong_signals.append((symbol, signals))
                    
                    strong_signals.sort(key=lambda x: x[1]['strength'], reverse=True)
                    
                    print(f"\n{Fore.GREEN}Top Buy Signals:{Style.RESET_ALL}")
                    for symbol, signals in strong_signals[:5]:
                        print(f"{symbol}: {signals['signal']} (strength: {signals['strength']:.1f}, "
                              f"confirmations: {signals.get('buy_signals', 0)}, "
                              f"ML: {signals.get('ml_confidence', 0):.1%}, "
                              f"quality: {signals.get('data_quality', 0):.1%})")
                
                elif sub_choice == '5':
                    symbol = input("Symbol to diagnose: ").strip().upper()
                    days = require_int("Days to analyze (default 30): ", default=30)
                    system.diagnose_signals(symbol, days)
                
                elif sub_choice == '0':
                    sub_menu = False
                    
        elif choice == '4':
            symbol = input("Symbol: ").strip().upper()
            quantity = require_int("Quantity: ")
            price = require_float("Buy price: ")
            system.add_position(symbol, quantity, price)
            
        elif choice == '5':
            if system.positions:
                print("\nCurrent positions:")
                for symbol, pos in system.positions.items():
                    print(f"  {symbol}: {pos['quantity']} shares @ {pos['buy_price']:.2f}")
                
                symbol = input("\nSymbol to sell: ").strip().upper()
                if symbol in system.positions:
                    price_val = safe_float("Sell price: ")
                    if price_val is None:
                        continue
                    system.remove_position(symbol, price_val)
                else:
                    print(f"{Fore.RED}Symbol not in positions{Style.RESET_ALL}")
            else:
                print("No positions to sell")
        
        elif choice == '6':
            system.generate_risk_dashboard()
        
        elif choice == '7':
            sub_menu = True
            while sub_menu:
                print(f"\n{Fore.YELLOW}Watchlist:{Style.RESET_ALL}")
                print("1. View watchlist")
                print("2. Add symbol")
                print("3. Remove symbol")
                print("0. Back")
                
                sub_choice = input("Choice: ").strip()
                
                if sub_choice == '1':
                    if system.watchlist:
                        print("\nWatchlist:")
                        for symbol in system.watchlist:
                            print(f"  {symbol}")
                    else:
                        print("Watchlist is empty")
                
                elif sub_choice == '2':
                    symbol = input("Symbol to add: ").strip().upper()
                    priority = require_int("Priority (0-10, default 0): ", default=0)
                    notes = input("Notes (optional): ").strip()
                    system.add_to_watchlist(symbol, priority, notes)
                
                elif sub_choice == '3':
                    symbol = input("Symbol to remove: ").strip().upper()
                    system.remove_from_watchlist(symbol)
                
                elif sub_choice == '0':
                    sub_menu = False
        
        elif choice == '8':
            symbol = input("Symbol for backtest: ").strip().upper()
            start_date = input("Start date (YYYY-MM-DD): ").strip()
            end_date = input("End date (YYYY-MM-DD): ").strip()
            capital = require_float("Initial capital (default 100000): ", default=100000.0)
            aggressive = input("Aggressive mode? (y/n, default n): ").strip().lower() == 'y'
            system.backtest_strategy(symbol, start_date, end_date, initial_capital=capital, aggressive=aggressive)
        
        elif choice == '9':
            system.analyze_performance()
        
        elif choice == '10':
            if system.monitoring_active:
                system.stop_monitoring()
                print(f"{Fore.YELLOW}Monitoring stopped{Style.RESET_ALL}")
            else:
                system.start_monitoring()
                print(f"{Fore.GREEN}Monitoring started{Style.RESET_ALL}")
        
        elif choice == '11':
            sub_menu = True
            while sub_menu:
                print(f"\n{Fore.YELLOW}Market Analysis:{Style.RESET_ALL}")
                print("1. Market regime")
                print("2. Sector analysis")
                print("3. Portfolio correlations")
                print("0. Back")
                
                sub_choice = input("Choice: ").strip()
                
                if sub_choice == '1':
                    regime = system.detect_market_regime()
                    print(f"\nMarket Regime: {regime}")
                
                elif sub_choice == '2':
                    sectors = system.analyze_sector_rotation()
                    if sectors and 'rankings' in sectors and sectors['rankings']:
                        print(f"\n{Fore.CYAN}Sector Rankings:{Style.RESET_ALL}")
                        for sector, data in sectors['rankings']:
                            trend_color = Fore.GREEN if data['trend'] == 'BULLISH' else Fore.RED if data['trend'] == 'BEARISH' else Fore.YELLOW
                            print(f"  {sector}: {trend_color}{data['trend']}{Style.RESET_ALL} "
                                  f"(5d: {data['momentum_5d']:.1f}%, 20d: {data['momentum_20d']:.1f}%, RSI: {data['rsi']:.0f})")
                        if 'rotation_signal' in sectors:
                            print(f"\nRotation signal: {sectors['rotation_signal']}")
                    else:
                        print("Unable to fetch sector data")
                
                elif sub_choice == '3':
                    correlations = system.analyze_correlations()
                    if not correlations.empty:
                        print(f"\n{Fore.CYAN}Position Correlations:{Style.RESET_ALL}")
                        for _, row in correlations.iterrows():
                            print(f"  {row['pair']}: {row['correlation']:.2f} - {row['relationship']}")
                    else:
                        print("Need at least 2 positions for correlation analysis")
                
                elif sub_choice == '0':
                    sub_menu = False
        
        elif choice == '12':
            hours = require_int("View alerts for last N hours (default 24): ", default=24)
            system.view_alerts(hours)
        
        elif choice == '13':
            sub_menu = True
            while sub_menu:
                print(f"\n{Fore.YELLOW}System Controls:{Style.RESET_ALL}")
                print("1. Emergency stop")
                print("2. Release emergency stop")
                print("3. System health check")
                print("0. Back")
                
                sub_choice = input("Choice: ").strip()
                
                if sub_choice == '1':
                    confirm = input("Activate emergency stop? (yes/no): ").strip()
                    if confirm.lower() == 'yes':
                        system.failsafe.activate_emergency_stop("Manual activation")
                        print(f"{Fore.RED}EMERGENCY STOP ACTIVATED{Style.RESET_ALL}")
                
                elif sub_choice == '2':
                    system.failsafe.deactivate_emergency_stop()
                    print(f"{Fore.GREEN}Emergency stop deactivated{Style.RESET_ALL}")
                
                elif sub_choice == '3':
                    ok, issues = system.failsafe.check_all_systems()
                    if ok:
                        print(f"{Fore.GREEN}All systems nominal{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}Issues detected:{Style.RESET_ALL}")
                        for issue in issues:
                            print(f"  - {issue}")
                
                elif sub_choice == '0':
                    sub_menu = False
        
        elif choice == '14':
            sub_menu = True
            while sub_menu:
                print(f"\n{Fore.YELLOW}Database Maintenance:{Style.RESET_ALL}")
                print("1. Backup database")
                print("2. Verify integrity")
                print("3. View statistics")
                print("4. View trade history")
                print("5. Clear old data")
                print("6. Export positions")
                print("0. Back")
                
                sub_choice = input("Choice: ").strip()
                
                if sub_choice == '1':
                    if system.db.backup_database():
                        print(f"{Fore.GREEN}Backup successful{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}Backup failed{Style.RESET_ALL}")
                
                elif sub_choice == '2':
                    if system.db.force_integrity_check():
                        print(f"{Fore.GREEN}Database integrity OK{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}Database integrity check failed!{Style.RESET_ALL}")
                
                elif sub_choice == '3':
                    with system.db.get_connection() as conn:
                        for table in ['positions', 'trade_history', 'watchlist', 'alerts',
                                     'data_quality_log', 'paper_trades']:
                            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                            print(f"  {table}: {count} records")
                
                elif sub_choice == '4':
                    with system.db.get_connection() as conn:
                        trades = conn.execute("""
                            SELECT * FROM trade_history 
                            ORDER BY trade_date DESC
                            LIMIT 10
                        """).fetchall()
                        
                        if trades:
                            print("\nRecent trades:")
                            for trade in trades:
                                action_color = Fore.GREEN if trade['action'] == 'BUY' else Fore.RED
                                pnl_color = Fore.GREEN if trade['pnl'] > 0 else Fore.RED if trade['pnl'] < 0 else Fore.WHITE
                                print(f"{trade['trade_date']} - {action_color}{trade['action']}{Style.RESET_ALL} {trade['symbol']} "
                                      f"{trade['quantity']} @ {trade['price']:.2f}")
                                if trade['pnl'] is not None:
                                    print(f"  P&L: {pnl_color}{trade['pnl']:+.0f} ({trade['return_pct']:+.1f}%){Style.RESET_ALL}")
                        else:
                            print("No trade history")
                elif sub_choice == '5':
                    confirm = input("Clear data older than 30 days? (yes/no): ").strip()
                    if confirm.lower() == 'yes':
                        with system.db.get_connection() as conn:
                            conn.execute("DELETE FROM data_quality_log WHERE check_timestamp < datetime('now', '-30 days')")
                            conn.execute("DELETE FROM alerts WHERE timestamp < datetime('now', '-30 days')")
                            conn.commit()
                        print("Old data cleared")
                elif sub_choice == '6':
                    import csv
                    with open('positions_export.csv', 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Symbol', 'Quantity', 'Buy Price', 'Buy Date', 'Cost Basis'])
                        for symbol, pos in system.positions.items():
                            writer.writerow([symbol, pos['quantity'], pos['buy_price'], 
                                           pos['buy_date'], pos['cost_basis']])
                    print(f"{Fore.GREEN}Positions exported to positions_export.csv{Style.RESET_ALL}")
                elif sub_choice == '0':
                    sub_menu = False
                    
        elif choice == '15':
            print("Retraining ML model with enhanced features...")
            success = system.retrain_ml_model_enhanced()
            if success:
                print(f"{Fore.GREEN}Model retrained successfully!{Style.RESET_ALL}")
                system._analyze_model_performance()
            else:
                print(f"{Fore.RED}Model training failed{Style.RESET_ALL}")
                
        elif choice == '16':
            n_symbols = require_int("Number of symbols to analyze (default 20): ", default=20)
            system.analyze_signal_weakness(n_symbols)
            
        elif choice == '17':
            sub_menu = True
            while sub_menu:
                print(f"\n{Fore.YELLOW}Paper Trading:{Style.RESET_ALL}")
                print("1. Execute paper trade")
                print("2. View paper positions")
                print("3. View paper trading report")
                print("4. Reset paper trading")
                print("0. Back")
                
                sub_choice = input("Choice: ").strip()
                
                if sub_choice == '1':
                    symbol = input("Symbol: ").strip().upper()
                    action = input("Action (BUY/SELL): ").strip().upper()
                    if action not in ('BUY', 'SELL'):
                        print(f"{Fore.RED}Invalid action — must be BUY or SELL{Style.RESET_ALL}")
                        continue
                    quantity = require_int("Quantity: ")
                    price = require_float("Price: ")
                    
                    signals = system.generate_enhanced_signals(symbol)
                    
                    result = system.execute_paper_trade(
                        symbol=symbol,
                        action=action,
                        quantity=quantity,
                        price=price,
                        signal_strength=signals['strength'],
                        signal_reason=signals['signal']
                    )
                    
                    if not result['success']:
                        print(f"{Fore.RED}Paper trade failed: {result['reason']}{Style.RESET_ALL}")
                
                elif sub_choice == '2':
                    if system.paper_trading.paper_positions:
                        print("\nPaper positions:")
                        for symbol, pos in system.paper_trading.paper_positions.items():
                            pnl_pct = (pos.get('last_price', pos['avg_price']) / pos['avg_price'] - 1) * 100
                            pnl_color = Fore.GREEN if pnl_pct > 0 else Fore.RED
                            print(f"  {symbol}: {pos['quantity']} @ {pos['avg_price']:.2f} "
                                  f"({pnl_color}{pnl_pct:+.1f}%{Style.RESET_ALL})")
                    else:
                        print("No paper positions")
                
                elif sub_choice == '3':
                    system.paper_trading.display_report()
                
                elif sub_choice == '4':
                    confirm = input("Reset paper trading? This will clear all paper trades (yes/no): ").strip()
                    if confirm.lower() == 'yes':
                        system.paper_trading = PaperTradingMode(system)
                        print("Paper trading reset")
                
                elif sub_choice == '0':
                    sub_menu = False
        
        elif choice == '18':
            passed, results = system.run_pre_market_checks()
            if passed:
                print(f"\n{Fore.GREEN}All pre-market checks passed!{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.RED}Some pre-market checks failed - review before trading{Style.RESET_ALL}")
        
        elif choice == '19':
            symbol = input("Symbol: ").strip().upper()
            action = input("Action (BUY/SELL): ").strip().upper()
            if action not in ('BUY', 'SELL'):
                print(f"{Fore.RED}Invalid action — must be BUY or SELL{Style.RESET_ALL}")
                continue
            quantity = require_int("Quantity: ")
            price = require_float("Price: ")
            
            costs = system.estimate_trade_costs(symbol, action, quantity, price)
            
            print(f"\n{Fore.CYAN}=== TRANSACTION COST ESTIMATE ==={Style.RESET_ALL}")
            print(f"Order value: {quantity * price:,.2f}")
            print(f"Commission: {costs['commission']:.2f}")
            if action == 'SELL':
                print(f"Stamp duty: {costs['stamp_duty']:.2f}")
            print(f"Estimated slippage: {costs['slippage']:.2f} ({costs['slippage_rate']:.2%})")
            print(f"Total costs: {costs['total_costs']:.2f} ({costs['cost_rate']:.2%})")
            print(f"Estimated fill price: {costs['estimated_fill_price']:.2f}")
        
        elif choice == '20':
            system.display_ml_health()
        
        elif choice == '21':
            system.track_signal_performance()
        
        elif choice == '22':
            system.display_error_summary()
            
        elif choice == '0':
            break
        
        else:
            print(f"{Fore.RED}Invalid choice{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
