

# Strategy Creation Guide

## 1. General Principles

- **Capture Rate is the #1 metric to solve.** A strategy with 0% capture rate generates zero live trades regardless of backtest return. Always diagnose capture rate failures before optimizing returns.
- **Entry rules multiply as AND conditions — each additional rule exponentially reduces qualifying signals.** With 8 entry rules all requiring simultaneous satisfaction, the probability of all passing on any given bar becomes vanishingly small. **Aim for 3-5 entry rules maximum**, or use a scoring/threshold system (e.g., "5 of 8 must pass") instead of requiring all to pass.
- **The tension between tight entries and capture rate is the core design challenge.** Iteration 2 of Titan_Gold_V5 had 1204 trades (too many, too small) while Iteration 3 had ~0 trades (too restrictive). The sweet spot requires iterative calibration against historical data.
- **Wider exits help only if you actually enter trades.** Iteration 3's exit redesign was sound in theory (RSI 85, MOM_1M ≤ -15%) but irrelevant because entries never triggered. **Always fix entries before optimizing exits.**
- **Validate entry rule pass rates independently.** Before combining rules, check what percentage of bars each individual rule passes. If any single rule passes < 30% of bars, the combined pass rate will likely be near zero.
- **Total return without capture rate is an illusion.** The 35.84% return in Iteration 3 likely came from forced buy-and-hold or fallback logic, not from the strategy's signal-driven entries. Always confirm returns are signal-driven.

## 2. Optimal ML Settings

### Model Selection
- **MLP_Large is the most reliable baseline model** for daily timeframe strategies. It achieved the best F1 scores across iterations (0.60-0.66).
- **LSTM adds value for trend-following strategies** on assets with sustained directional moves (gold, commodities) but requires more data and longer training. Keep it ON but don't rely on it as primary.
- **F1 score in the 0.60-0.66 range is achievable** for gold trend classification. Scores above 0.70 may indicate overfitting.

### Architecture
- **MLP layers:** Start with `[128, 64, 32]` as baseline. Larger networks `[256, 128, 64]` showed no improvement and risk overfitting. Use the smallest architecture that achieves target F1.
- **MLP epochs:** 200 is reasonable. Monitor for convergence — diminishing returns typically occur after 100-150 epochs.
- **LSTM config:** Hidden=160, Layers=3, Epochs=300 is heavyweight. Consider starting with Hidden=64, Layers=2, Epochs=150 to reduce overfitting risk, then scale up if F1 is insufficient.

### Forward Days & Profit Thresholds
- **Forward days [10, 15, 20] target 2-4 week moves.** This is appropriate for gold's rally characteristics (multi-week sustained trends). For faster-moving assets, use [3, 5, 10].
- **Profit thresholds [3.0, 5.0, 8.0] may be too ambitious** for GLD (non-leveraged gold ETF). GLD's daily ATR is ~1-2%. A 3% move in 10-20 days is reasonable, but 8% in 20 days requires a strong rally. Consider [2.0, 3.5, 5.0] for non-leveraged ETFs, reserve [3.0, 5.0, 8.0] for leveraged products like UGL.
- **Match profit thresholds to the specific symbol's volatility.** UGL (2x leveraged) can target 2x the thresholds of GLD.

## 3. Feature Selection Patterns

### Gold / Precious Metals
| Feature | Role | Effectiveness | Notes |
|---------|------|--------------|-------|
| PRICE_VS_EMA_200 | Trend filter | ★★★★★ | Gold's best trend indicator. 29% above EMA200 in strong bull. Use ≥ 3-5% for entries. |
| EMA_STACK_SCORE | Trend confirmation | ★★★★ | Multi-timeframe alignment. Score ≥ 2 (not 3) for more permissive entries. |
| RSI_14 | Momentum/overbought | ★★★★ | Gold-specific: 42-75 entry band is reasonable. RSI hit 88 before Jan crash. |
| ADX | Trend strength | ★★★ | ≥ 15 is appropriate. Gold consolidation shows ADX ~14. |
| TREND_SLOPE | Trend direction | ★★★ | Useful as exit signal (≤ -0.5) more than entry filter. |
| MOM_1M / MOM_3M | Momentum | ★★★ | MOM_1M ≥ -5% is reasonable for pullback filtering. |
| MACD_HIST | Momentum | ★★ | Adds noise when combined with RSI and MOM. Consider dropping to reduce rule count. |
| BREAKOUT_20D | Breakout detection | ★★★ | Not used in entry rules despite being in priority features. Good candidate for a scoring system. |
| ATR_PCT | Volatility | ★★★ | Better for position sizing than entry/exit signals. Gold ATR tripled to 3.5% during Jan-Feb 2026 crash. |
| BB_PCT_B | Mean reversion | ★★ | Conflicts with trend-following approach. Use cautiously. |
| HIGHER_HIGH / HIGHER_LOW | Price structure | ★★★ | Good trend confirmation but binary signals that may not fire often enough. |

### General Feature Principles
- **Use no more than 4-5 features in entry rules** to maintain capture rate.
- **Separate features into tiers:** Tier 1 (must-have, 2-3 features), Tier 2 (nice-to-have, use in scoring).
- **Redundant momentum indicators kill capture rate.** RSI, MOM_1M, MACD_HIST, and ROC all measure variants of momentum. Pick ONE primary momentum indicator for entries.
- **PRICE_VS_EMA_200 is the single most valuable feature for trend-following** on assets in secular bull markets.

## 4. Risk Management Lessons

### Stop Loss
- **12% stop loss is wide but appropriate for gold** given ATR tripling to 3.5% during volatile periods. A 12% stop accommodates 3-4 days of adverse movement at peak volatility.
- **General rule:** Stop loss should be ≥ 3x the asset's average daily ATR to avoid noise-triggered exits.
- **For leveraged ETFs (UGL = 2x):** The effective stop on the underlying is 6%, which is tighter. Consider 15-18% stops for 2x leveraged products.

### Trailing Stop
- **8% trailing stop is aggressive for gold.** During the Jan 29-30 crash, GLD dropped ~7% in 2 days. An 8% trail would have barely survived. Consider 10-12% for volatile gold regimes.
- **Trail should be wider than stop** is counterintuitive but wrong — trail should be calibrated to the asset's mean reversion characteristics after a peak.

### Position Sizing
- **MaxPos=4 is appropriate** for a 4-symbol strategy (UGL, GLD, BAR, GDX). Allows full deployment across the universe.
- **Consider correlation:** UGL, GLD, and BAR are 95%+ correlated. MaxPos=4 with these symbols is effectively a concentrated bet on gold price. Treat as MaxPos=1 effective position from a risk perspective.

## 5. Common Pitfalls

### CRITICAL: The Capture Rate Death Spiral
- **Pitfall:** Each iteration adds more entry conditions to "improve quality" → capture rate drops to 0% → no trades → strategy is useless.
- **Solution:** After ANY iteration with 0% capture rate, the NEXT iteration must REMOVE entry rules, not modify thresholds. Reduce from N rules to N-3 minimum, or switch to a scoring system.

### The AND-Gate Trap
- **Pitfall:** 8 entry rules all connected by AND logic. If each rule independently passes 70% of bars, combined pass rate = 0.70^8 = 5.7%. If some rules pass only 40% of bars, combined rate approaches 0%.
- **Solution:** Use threshold-based entry (e.g., "score ≥ 4 of 8") or limit hard AND rules to 3-4 maximum. Alternatively, compute expected pass rate: multiply individual pass rates and verify the product exceeds 10%.

### Confusing Backtest Returns with Strategy Returns
- **Pitfall:** Iteration 3 showed 35.84% return with 0% capture rate. This return came from baseline/fallback behavior, not from the designed signal logic.
- **Solution:** Only trust returns where capture rate > 0%. A strategy with 0% capture rate has 0% real return by definition.

### Over-Optimizing Exits Before Fixing Entries
- **Pitfall:** Iteration 3 redesigned exits extensively (wider RSI, wider MOM, wider ROC thresholds) but the problem was entries, not exits.
- **Solution:** Fix entries first. Once capture rate > 20%, then optimize exits. Exits are irrelevant without entries.

### Redundant Feature Stacking
- **Pitfall:** Using RSI_14, MOM_1M, MACD_HIST, and TREND_SLOPE all as entry conditions — these are all momentum derivatives and are highly correlated. Each adds minimal new information but significantly reduces capture rate.
- **Solution:** Choose ONE momentum indicator for entries. Let the ML model learn the relationships between correlated features rather than hardcoding them as rules.

### Threshold Anchoring to Recent Extremes
- **Pitfall:** Setting thresholds based on specific events (e.g., "RSI hit 88 before Jan crash, so exit at 85") creates rules optimized for one historical event.
- **Solution:** Validate thresholds across the FULL backtest period, not just memorable events. Check how many times RSI actually exceeded 85 — if only once, the rule is nearly useless.

### Ignoring the Leverage Mismatch
- **Pitfall:** Applying identical entry/exit thresholds across GLD (1x), UGL (2x), and GDX (gold miners with ~2-3x gold beta). A -5% MOM_1M filter means very different things for each symbol.
- **Solution:** Either normalize features by symbol or create symbol-specific threshold adjustments. Better yet, let the ML model handle this and keep rules minimal.

## 6. Strategy-Specific Notes

### Gold / Precious Metals (GLD, UGL, BAR, GDX)

**Market Characteristics:**
- Secular bull trend with price 29% above EMA200 (as of strategy period)
- Multi-week sustained rallies: Sep-Oct 2025 (+28.5%), Jan 2026 (+28.5%)
- Sharp corrections: Jan 29-30 crash (-13% in 2 days for UGL equivalent)
- ATR can triple during volatile regimes ($5 → $16 for GLD)
- Consolidation periods with ADX < 15 (Nov 2025)

**Recommended Approach:**
- **Entry rules (3-4 max):**
  - PRICE_VS_EMA_200 ≥ 3.0% (primary trend filter — generous enough to maintain entries)
  - RSI_14 between 40-78 (avoid extremes)
  - EMA_STACK_SCORE ≥ 2 (basic trend alignment, not strict)
  - Optional: ADX ≥ 13 or use as ML feature instead
- **Exit rules (2-3 max):**
  - RSI_14 ≥ 85 (extreme overbought)
  - PRICE_VS_EMA_200 ≤ 2.0% (trend collapse)
  - ROC_5 ≤ -10% (crash protection)
- **ML config:**
  - MLP_Large as primary, architecture [128, 64, 32]
  - Forward days [10, 15, 20] for multi-week trend capture
  - Profit thresholds: [2.0, 4.0, 6.0] for GLD; [4.0, 8.0, 12.0] for UGL
- **Risk:** Stop=12%, Trail=10%, MaxPos=4
- **Key lesson from Titan_Gold_V5:** The strategy concept is sound (trend-following with wide exits on gold), but implementation failed due to over-constrained entries. Iteration 2 was closest to working (8.3% capture, 43.3% return). Next iteration should start from Iteration 2's entry logic and SIMPLIFY rather than adding conditions.

**Iteration Roadmap for Gold (if resuming):**
1. Reduce entry rules from 8 to 4 (keep PRICE_VS_EMA_200, RSI band, EMA_STACK_SCORE ≥ 2)
2. Verify each rule's individual pass rate exceeds 60%
3. Target capture rate > 15% before optimizing returns
4. Once capturing trades, tune exit rules to maximize hold duration
5. Only then increase entry selectivity if win rate needs improvement

### Semiconductors
*(No data yet — to be populated after first semiconductor strategy)*

### Broad Market / Index
*(No data yet — to be populated after first index strategy)*

---

## Quick Reference: Pre-Build Checklist

Before building ANY new strategy, verify:

- [ ] **Entry rules ≤ 5** (hard AND conditions)
- [ ] **Each entry rule independently passes > 50% of historical bars**
- [ ] **Combined estimated pass rate > 10%** (multiply individual rates)
- [ ] **No more than 2 momentum-derived features** in entry rules (RSI, MOM, MACD, ROC — pick max 2)
- [ ] **Profit thresholds calibrated to asset volatility** (threshold / ATR_PCT should be achievable in forward_days window)
- [ ] **Stop loss ≥ 3x average daily ATR** of the most volatile symbol in the universe
- [ ] **Trailing stop ≥ 2x average daily ATR** to avoid noise-triggered exits
- [ ] **Leverage differences accounted for** across symbols in the universe
- [ ] **Priority: Capture Rate > 15% first**, then optimize returns