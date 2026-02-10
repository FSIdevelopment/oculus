#!/usr/bin/env python3
"""Analyze backtest trades"""
import json

with open('backtest_results.json') as f:
    results = json.load(f)

trades = results.get('trades', [])
print('=== ALL TRADES (sorted by PnL) ===')
sorted_trades = sorted(trades, key=lambda x: x.get('pnl_pct', 0), reverse=True)
for t in sorted_trades:
    symbol = t.get('symbol', 'N/A')
    pnl = t.get('pnl_pct', 0)
    reason = t.get('exit_reason', 'N/A')
    entry = t.get('entry_date', 'N/A')[:10] if t.get('entry_date') else 'N/A'
    exit_d = t.get('exit_date', 'N/A')[:10] if t.get('exit_date') else 'N/A'
    print(f'{symbol}: {pnl:+.2f}% | {entry} -> {exit_d} | {reason}')

print()
winners = [t.get('pnl_pct', 0) for t in trades if t.get('pnl_pct', 0) > 0]
losers = [t.get('pnl_pct', 0) for t in trades if t.get('pnl_pct', 0) < 0]
if winners:
    print(f'Avg Winner: {sum(winners)/len(winners):.2f}% ({len(winners)} trades)')
if losers:
    print(f'Avg Loser: {sum(losers)/len(losers):.2f}% ({len(losers)} trades)')

