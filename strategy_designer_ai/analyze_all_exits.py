#!/usr/bin/env python3
"""Analyze ALL exits from backtest results."""
import json
from datetime import datetime
from collections import Counter

with open('backtest_results.json') as f:
    results = json.load(f)
trades = results.get('trades', [])

print('=== EXIT REASON BREAKDOWN ===')
reasons = Counter(t.get('exit_reason', 'Unknown') for t in trades)
for reason, count in reasons.most_common():
    subset = [t for t in trades if t.get('exit_reason') == reason]
    avg_pnl = sum(t.get('pnl_pct', 0) for t in subset) / len(subset) if subset else 0
    total_pnl = sum(t.get('pnl_pct', 0) for t in subset)
    print(f'{reason}: {count} trades, avg={avg_pnl:+.2f}%, total={total_pnl:+.2f}%')

print('\n=== STOP LOSS EXITS (what are we losing?) ===')
stop_losses = [t for t in trades if t.get('exit_reason') == 'STOP_LOSS']
for t in sorted(stop_losses, key=lambda x: x.get('pnl_pct', 0)):
    entry = t.get('entry_date', '')[:10] if t.get('entry_date') else 'N/A'
    exit_d = t.get('exit_date', '')[:10] if t.get('exit_date') else 'N/A'
    print(f'{t["symbol"]}: {t["pnl_pct"]:+.2f}% | {entry} -> {exit_d}')

print('\n=== TAKE PROFIT EXITS ===')
take_profits = [t for t in trades if t.get('exit_reason') == 'TAKE_PROFIT']
for t in sorted(take_profits, key=lambda x: x.get('pnl_pct', 0), reverse=True):
    entry = t.get('entry_date', '')[:10] if t.get('entry_date') else 'N/A'
    exit_d = t.get('exit_date', '')[:10] if t.get('exit_date') else 'N/A'
    print(f'{t["symbol"]}: {t["pnl_pct"]:+.2f}% | {entry} -> {exit_d}')

print('\n=== END OF BACKTEST POSITIONS ===')
eob = [t for t in trades if t.get('exit_reason') == 'END_OF_BACKTEST']
for t in sorted(eob, key=lambda x: x.get('pnl_pct', 0), reverse=True):
    entry = t.get('entry_date', '')[:10] if t.get('entry_date') else 'N/A'
    print(f'{t["symbol"]}: {t["pnl_pct"]:+.2f}% | Entry: {entry}')

print('\n=== OVERALL SUMMARY ===')
winners = [t for t in trades if t.get('pnl_pct', 0) > 0]
losers = [t for t in trades if t.get('pnl_pct', 0) < 0]
print(f"Total Trades: {len(trades)}")
print(f"Winners: {len(winners)} ({100*len(winners)/len(trades):.1f}%)")
print(f"Losers: {len(losers)} ({100*len(losers)/len(trades):.1f}%)")
if winners:
    print(f"Avg Winner: +{sum(t['pnl_pct'] for t in winners)/len(winners):.2f}%")
    print(f"Total from Winners: +{sum(t['pnl_pct'] for t in winners):.2f}%")
if losers:
    print(f"Avg Loser: {sum(t['pnl_pct'] for t in losers)/len(losers):.2f}%")
    print(f"Total from Losers: {sum(t['pnl_pct'] for t in losers):.2f}%")

print(f"\nNet Total: {sum(t.get('pnl_pct', 0) for t in trades):.2f}%")

