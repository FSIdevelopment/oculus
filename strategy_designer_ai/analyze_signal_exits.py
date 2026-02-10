#!/usr/bin/env python3
"""Analyze SIGNAL exits from backtest results."""
import json
from datetime import datetime

with open('backtest_results.json') as f:
    results = json.load(f)
trades = results.get('trades', [])

# Analyze SIGNAL exits specifically
signal_exits = [t for t in trades if t.get('exit_reason') == 'SIGNAL']
print('=== SIGNAL EXITS ANALYSIS ===')
print(f'Total SIGNAL exits: {len(signal_exits)}')

winners = [t for t in signal_exits if t.get('pnl_pct', 0) > 0]
losers = [t for t in signal_exits if t.get('pnl_pct', 0) <= 0]
print(f'Winners: {len(winners)}')
print(f'Losers: {len(losers)}')

if winners:
    print(f'Avg winner: {sum(t["pnl_pct"] for t in winners)/len(winners):.2f}%')
if losers:
    print(f'Avg loser: {sum(t["pnl_pct"] for t in losers)/len(losers):.2f}%')

print('\n=== SIGNAL EXITS DETAILS (sorted by PnL) ===')
for t in sorted(signal_exits, key=lambda x: x.get('pnl_pct', 0), reverse=True):
    entry = t.get('entry_date', '')[:10] if t.get('entry_date') else 'N/A'
    exit_d = t.get('exit_date', '')[:10] if t.get('exit_date') else 'N/A'
    print(f'{t["symbol"]}: {t["pnl_pct"]:+.2f}% | {entry} -> {exit_d}')

# Calculate holding periods
print('\n=== HOLDING PERIODS ===')
total_days = []
for t in signal_exits:
    if t.get('entry_date') and t.get('exit_date'):
        entry = datetime.fromisoformat(t['entry_date'].replace('Z', '+00:00').replace('+00:00', ''))
        exit_d = datetime.fromisoformat(t['exit_date'].replace('Z', '+00:00').replace('+00:00', ''))
        days = (exit_d - entry).days
        total_days.append(days)
        print(f'{t["symbol"]}: {days} days, {t["pnl_pct"]:+.2f}%')

if total_days:
    print(f'\nAverage holding period: {sum(total_days)/len(total_days):.1f} days')

# Overall stats
print('\n=== OVERALL BACKTEST STATS ===')
print(f"Total Return: {results.get('total_return_pct', 'N/A')}%")
print(f"Total Trades: {results.get('total_trades', 'N/A')}")
print(f"Win Rate: {results.get('win_rate_pct', 'N/A')}%")

# Exit reason breakdown
from collections import Counter
reasons = Counter(t.get('exit_reason', 'Unknown') for t in trades)
print('\n=== EXIT REASONS ===')
for reason, count in reasons.most_common():
    print(f'  {reason}: {count}')

