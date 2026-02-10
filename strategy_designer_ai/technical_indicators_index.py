#!/usr/bin/env python3
"""
Comprehensive Technical Indicators Index
Sources: incrediblecharts.com, centralcharts.com, NYU Stern

Contains ALL technical indicators with formulas, signals, and trading rules
for building an AI-powered trading strategy.

Author: SignalSynk AI
"""

# =============================================================================
# TECHNICAL INDICATORS KNOWLEDGE BASE
# Comprehensive index of 70+ indicators organized by category
# =============================================================================

INDICATORS = {
    # =========================================================================
    # MOMENTUM INDICATORS (13 indicators)
    # =========================================================================
    "RSI": {
        "name": "Relative Strength Index",
        "type": "momentum",
        "source": "incrediblecharts.com",
        "default_period": 14,
        "overbought": 70,
        "oversold": 30,
        "formula": "RSI = 100 - (100 / (1 + RS)), RS = Avg Gain / Avg Loss",
        "calculation": """
            1. Calculate price changes: delta = Close - Close.shift(1)
            2. Separate gains and losses: gain = delta.where(delta > 0, 0), loss = -delta.where(delta < 0, 0)
            3. Calculate average gain/loss over period (typically EMA)
            4. RS = avg_gain / avg_loss
            5. RSI = 100 - (100 / (1 + RS))
        """,
        "signals": {
            "ranging_market": {
                "buy": "RSI falls below 30 and rises back above it, or bullish divergence",
                "sell": "RSI rises above 70 and falls back below it, or bearish divergence"
            },
            "trending_market": {
                "uptrend_buy": "RSI falls below 40 and rises back above it",
                "downtrend_sell": "RSI rises above 60 and falls back below it"
            }
        },
        "divergences": {
            "bullish": "Price makes lower low, RSI makes higher low",
            "bearish": "Price makes higher high, RSI makes lower high"
        },
        "key_insight": "Different levels for trending vs ranging markets. For strong uptrends use 40/80, for downtrends use 20/60"
    },
    
    "STOCHASTIC": {
        "name": "Stochastic Oscillator",
        "type": "momentum",
        "source": "incrediblecharts.com, centralcharts.com",
        "default_k_period": 5,
        "default_d_period": 3,
        "overbought": 80,
        "oversold": 20,
        "formula": "%K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100, %D = SMA(%K)",
        "calculation": """
            1. Find lowest low over K period
            2. Find highest high over K period
            3. %K = (Close - Lowest_Low) / (Highest_High - Lowest_Low) * 100
            4. %D = SMA of %K over D period (usually 3)
        """,
        "signals": {
            "buy": [
                "Bullish divergence with first trough below oversold",
                "%K or %D falls below oversold and rises back",
                "%K crosses above %D"
            ],
            "sell": [
                "Bearish divergence with first peak above overbought",
                "%K or %D rises above overbought and falls back",
                "%K crosses below %D"
            ]
        },
        "key_insight": "Narrow, shallow bottoms = weak bears, strong rally; Wide deep bottoms = strong bears"
    },

    "STOCHASTIC_MOMENTUM_INDEX": {
        "name": "Stochastic Momentum Index (SMI)",
        "type": "momentum",
        "source": "centralcharts.com, TrendSpider",
        "description": "Enhanced stochastic that measures close relative to midpoint of high/low range, reducing false signals",
        "formula": "SMI = EMA(EMA(Close - Midpoint, q), r) / EMA(EMA(HH - LL, q), r) * 100",
        "calculation": """
            1. Calculate Highest High (HH) and Lowest Low (LL) over period n (default 13)
            2. Midpoint = (HH + LL) / 2
            3. Distance = Close - Midpoint
            4. Range = HH - LL
            5. Double-smooth the Distance: EMA(EMA(Distance, q), r) where q=default 25, r=default 2
            6. Double-smooth the Range: EMA(EMA(Range / 2, q), r)
            7. SMI = (Smoothed Distance / Smoothed Half-Range) * 100
            8. Signal Line = EMA(SMI, signal_period) default 9
        """,
        "default_periods": {"n": 13, "q": 25, "r": 2, "signal": 9},
        "range": [-100, 100],
        "overbought": 40,
        "oversold": -40,
        "signals": {
            "buy": "SMI crosses above -40 from below, or SMI crosses above signal line",
            "sell": "SMI crosses below 40 from above, or SMI crosses below signal line"
        },
        "key_insight": "Less prone to false signals than standard Stochastic due to double smoothing"
    },

    "MACD": {
        "name": "Moving Average Convergence Divergence",
        "type": "momentum",
        "source": "incrediblecharts.com, centralcharts.com",
        "fast_ema": 12,
        "slow_ema": 26,
        "signal_period": 9,
        "formula": "MACD = 12-EMA - 26-EMA, Signal = 9-EMA of MACD",
        "calculation": """
            1. Calculate 12-period EMA of Close
            2. Calculate 26-period EMA of Close
            3. MACD Line = 12-EMA - 26-EMA
            4. Signal Line = 9-period EMA of MACD Line
            5. Histogram = MACD Line - Signal Line
        """,
        "signals": {
            "standard": {
                "buy": "MACD crosses above signal line",
                "sell": "MACD crosses below signal line"
            },
            "strong_signals": {
                "large_swing_buy": "MACD crosses signal after swing below -2%",
                "large_swing_sell": "MACD reverses after swing above +2%",
                "divergence": "Smaller swing reversing = divergence signal"
            },
            "high_momentum_stocks": {
                "note": "For high momentum stocks like NVDA, don't use crossovers",
                "buy": "MACD crosses above zero line",
                "sell": "MACD crosses below zero line"
            }
        },
        "key_insight": "MACD oscillating above zero = strong uptrend; below zero = strong downtrend"
    },

    "MACD_ZERO_LAG": {
        "name": "MACD Zero Lag",
        "type": "momentum",
        "source": "centralcharts.com, John Ehlers",
        "description": "Reduces lag by using double-smoothed EMA (DEMA) or Zero-Lag EMA (ZLEMA)",
        "formula": "ZLEMA = EMA + (EMA - EMA(EMA)), then apply MACD formula",
        "calculation": """
            1. DEMA(price, period) = 2 * EMA(price, period) - EMA(EMA(price, period), period)
            2. OR ZLEMA(price, period) = EMA(price + (price - price[lag]), period) where lag = (period-1)/2
            3. Fast Line = DEMA(Close, 12) or ZLEMA(Close, 12)
            4. Slow Line = DEMA(Close, 26) or ZLEMA(Close, 26)
            5. MACD Zero Lag = Fast Line - Slow Line
            6. Signal Line = EMA(MACD Zero Lag, 9)
        """,
        "default_periods": {"fast": 12, "slow": 26, "signal": 9},
        "advantage": "Faster signals with 50-70% less lag than standard MACD",
        "signals": {
            "buy": "Zero Lag MACD crosses above signal line OR crosses above zero",
            "sell": "Zero Lag MACD crosses below signal line OR crosses below zero"
        },
        "key_insight": "Best for capturing early trend changes in high-momentum stocks"
    },

    "MACD_RAINBOW": {
        "name": "MACD Rainbow",
        "type": "momentum",
        "source": "centralcharts.com",
        "description": "Multiple MACD lines with different periods for trend confirmation",
        "calculation": """
            1. Calculate multiple MACD lines with different period combinations:
               - MACD_1: EMA(6) - EMA(13), Signal: EMA(5)
               - MACD_2: EMA(12) - EMA(26), Signal: EMA(9)
               - MACD_3: EMA(19) - EMA(39), Signal: EMA(9)
               - MACD_4: EMA(26) - EMA(52), Signal: EMA(9)
            2. Plot all MACD lines with different colors (rainbow effect)
            3. Strongest signals when all lines align
        """,
        "signals": {
            "strong_buy": "All MACD lines positive and rising",
            "strong_sell": "All MACD lines negative and falling",
            "confirmation": "Lines converging = trend continuation"
        },
        "use_case": "Visual confirmation when all MACD lines align - multi-timeframe in one indicator"
    },

    "MOMENTUM": {
        "name": "Momentum",
        "type": "momentum",
        "source": "centralcharts.com, StockCharts",
        "formula": "Momentum = Close - Close[n] or (Close / Close[n]) * 100",
        "calculation": """
            Method 1 (Difference): Momentum = Close - Close[n periods ago]
            Method 2 (Ratio): Momentum = (Close / Close[n periods ago]) * 100

            Example (10-day Momentum):
            1. Get today's closing price: 105
            2. Get closing price 10 days ago: 100
            3. Difference method: 105 - 100 = 5
            4. Ratio method: (105 / 100) * 100 = 105

            Ratio method oscillates around 100 (above = positive momentum)
            Difference method oscillates around 0
        """,
        "default_period": 10,
        "signals": {
            "buy": "Momentum crosses above zero (or 100 if using ratio)",
            "sell": "Momentum crosses below zero (or 100 if using ratio)",
            "divergence_buy": "Price makes lower low, Momentum makes higher low",
            "divergence_sell": "Price makes higher high, Momentum makes lower high"
        },
        "key_insight": "Simple but effective - measures rate of price change"
    },

    "RATE_OF_CHANGE": {
        "name": "Rate of Change (ROC)",
        "type": "momentum",
        "source": "centralcharts.com, StockCharts",
        "formula": "ROC = ((Close - Close[n]) / Close[n]) * 100",
        "calculation": """
            1. Get today's closing price (C)
            2. Get closing price n periods ago (C_n)
            3. ROC = ((C - C_n) / C_n) * 100

            Example (12-day ROC):
            - Today's close: 110
            - 12 days ago close: 100
            - ROC = ((110 - 100) / 100) * 100 = 10%

            ROC oscillates around zero:
            - Positive = price higher than n periods ago
            - Negative = price lower than n periods ago
        """,
        "default_period": 12,
        "signals": {
            "buy": "ROC crosses above zero from below",
            "sell": "ROC crosses below zero from above",
            "extreme_buy": "ROC at extreme negative (historical low) - mean reversion",
            "extreme_sell": "ROC at extreme positive (historical high) - mean reversion"
        },
        "overbought": "Extreme positive values (varies by stock volatility)",
        "oversold": "Extreme negative values (varies by stock volatility)"
    },

    "CHANDE_MOMENTUM": {
        "name": "Chande Momentum Oscillator (CMO)",
        "type": "momentum",
        "source": "centralcharts.com, Tushar Chande",
        "formula": "CMO = ((Sum_Up - Sum_Down) / (Sum_Up + Sum_Down)) * 100",
        "calculation": """
            1. Calculate price changes: delta = Close - Close[1]
            2. Sum_Up = Sum of all positive deltas over n periods
            3. Sum_Down = Absolute sum of all negative deltas over n periods
            4. CMO = ((Sum_Up - Sum_Down) / (Sum_Up + Sum_Down)) * 100

            Example (14-period CMO):
            - Sum of gains over 14 days: 15 points
            - Sum of losses over 14 days: 5 points (absolute value)
            - CMO = ((15 - 5) / (15 + 5)) * 100 = 50

            Unlike RSI, CMO uses raw sums, not averages
        """,
        "default_period": 14,
        "range": [-100, 100],
        "overbought": 50,
        "oversold": -50,
        "signals": {
            "buy": "CMO crosses above -50 from below",
            "sell": "CMO crosses below 50 from above",
            "strong_buy": "CMO below -70 (extreme oversold)",
            "strong_sell": "CMO above 70 (extreme overbought)"
        },
        "key_insight": "Similar to RSI but more sensitive; uses sums not averages"
    },

    "FORCE_INDEX": {
        "name": "Force Index",
        "type": "momentum",
        "source": "centralcharts.com, Alexander Elder",
        "formula": "Force Index = (Close - Close[1]) * Volume",
        "calculation": """
            1. Calculate raw Force Index: FI = (Close - Previous Close) * Volume
            2. Short-term (2-period EMA): FI_2 = EMA(FI, 2) - for entry timing
            3. Long-term (13-period EMA): FI_13 = EMA(FI, 13) - for trend direction

            Example:
            - Today's close: 105, Yesterday's close: 100, Volume: 1,000,000
            - Raw FI = (105 - 100) * 1,000,000 = 5,000,000

            Interpretation:
            - Positive FI = buying pressure (price up on volume)
            - Negative FI = selling pressure (price down on volume)
            - Magnitude indicates strength of move
        """,
        "default_periods": {"short": 2, "long": 13},
        "signals": {
            "buy": "13-period Force Index turns positive in uptrend",
            "sell": "13-period Force Index turns negative in downtrend",
            "entry_buy": "2-period FI dips below zero in uptrend (pullback buy)",
            "entry_sell": "2-period FI rises above zero in downtrend (rally sell)"
        },
        "key_insight": "Combines price and volume to measure buying/selling pressure"
    },

    "ELDERRAY": {
        "name": "Elder Ray Index",
        "type": "momentum",
        "source": "centralcharts.com, Alexander Elder",
        "components": ["Bull Power", "Bear Power"],
        "formula": {
            "bull_power": "High - EMA(Close, 13)",
            "bear_power": "Low - EMA(Close, 13)"
        },
        "calculation": """
            1. Calculate 13-period EMA of closing prices
            2. Bull Power = Daily High - EMA(13)
               (Measures ability of bulls to push price above consensus)
            3. Bear Power = Daily Low - EMA(13)
               (Measures ability of bears to push price below consensus)

            Example (EMA = 100):
            - Day's High: 103, Low: 98
            - Bull Power = 103 - 100 = 3
            - Bear Power = 98 - 100 = -2

            Bull Power is normally positive, Bear Power is normally negative
        """,
        "default_period": 13,
        "signals": {
            "buy": "EMA rising + Bear Power negative but rising toward zero",
            "sell": "EMA falling + Bull Power positive but falling toward zero",
            "strong_buy": "Bear Power crosses above zero (rare but powerful)",
            "strong_sell": "Bull Power crosses below zero (rare but powerful)"
        },
        "use_case": "Best used with trend filter - only buy in uptrends, sell in downtrends"
    },

    "ADVANCE_DECLINE_LINE": {
        "name": "Advance-Decline Line",
        "type": "momentum/breadth",
        "source": "centralcharts.com, StockCharts",
        "formula": "AD Line = Previous AD Line + (Advancing Issues - Declining Issues)",
        "calculation": """
            1. Count advancing issues (stocks closing higher than previous day)
            2. Count declining issues (stocks closing lower than previous day)
            3. Net Advances = Advancing Issues - Declining Issues
            4. AD Line = Previous AD Line + Net Advances

            Example:
            - Yesterday's AD Line: 50,000
            - Today: 1,800 stocks up, 1,200 stocks down
            - Net Advances = 1,800 - 1,200 = 600
            - Today's AD Line = 50,000 + 600 = 50,600

            Cumulative indicator - starting value is arbitrary
        """,
        "use_case": "Market breadth indicator for overall market health",
        "signals": {
            "bullish": "AD Line rising with price - confirms strength",
            "bearish_divergence": "Price rising but AD Line falling = narrow rally, warning",
            "bullish_divergence": "Price falling but AD Line rising = accumulation"
        },
        "key_insight": "New highs in index should be confirmed by new highs in AD Line"
    },

    "ADVANCE_DECLINE_RATIO": {
        "name": "Advance-Decline Ratio",
        "type": "momentum/breadth",
        "source": "centralcharts.com",
        "formula": "AD Ratio = Advancing Issues / Declining Issues",
        "calculation": """
            1. Count advancing issues (up on the day)
            2. Count declining issues (down on the day)
            3. AD Ratio = Advancing / Declining

            Example:
            - 1,800 advancing, 1,200 declining
            - AD Ratio = 1,800 / 1,200 = 1.5

            Often smoothed with moving average (10-day) for signals
        """,
        "interpretation": {
            "above_1": "More stocks advancing than declining - bullish",
            "below_1": "More stocks declining than advancing - bearish",
            "extreme_high": "Ratio > 2 = overbought, possible reversal",
            "extreme_low": "Ratio < 0.5 = oversold, possible reversal"
        }
    },

    "ARMS_INDEX": {
        "name": "ARMS Index (TRIN)",
        "type": "momentum/breadth",
        "source": "centralcharts.com, Richard Arms",
        "formula": "TRIN = (Advancing Issues / Declining Issues) / (Advancing Volume / Declining Volume)",
        "calculation": """
            1. AD Ratio = Advancing Issues / Declining Issues
            2. AD Volume Ratio = Advancing Volume / Declining Volume
            3. TRIN = AD Ratio / AD Volume Ratio

            Example:
            - 1,600 advancing issues, 1,400 declining issues
            - Advancing volume: 800M, Declining volume: 1,200M
            - AD Ratio = 1,600 / 1,400 = 1.14
            - AD Volume Ratio = 800M / 1,200M = 0.67
            - TRIN = 1.14 / 0.67 = 1.70 (bearish - more volume in decliners)

            Often smoothed with 10-day MA for trend
        """,
        "interpretation": {
            "below_1": "Bullish - advancing stocks have disproportionate volume",
            "above_1": "Bearish - declining stocks have disproportionate volume",
            "below_0.5": "Extreme bullish - possibly overbought",
            "above_2.0": "Extreme bearish - possibly oversold (contrarian buy)"
        },
        "key_insight": "Contrarian indicator at extremes; 10-day MA below 1 = bullish market"
    },

    "REPULSE": {
        "name": "Repulse",
        "type": "momentum",
        "source": "centralcharts.com",
        "description": "Measures buying vs selling pressure using high/low relationship to close",
        "formula": "Repulse = ((3 * Close - 2 * Low - Open) / Close) - ((Open + 2 * High - 3 * Close) / Close)",
        "calculation": """
            1. Bull Power = (3 * Close - 2 * Low - Open) / Close * 100
               Measures how close is positioned relative to low (buying pressure)
            2. Bear Power = (Open + 2 * High - 3 * Close) / Close * 100
               Measures how close is positioned relative to high (selling pressure)
            3. Repulse = Bull Power - Bear Power
            4. Smoothed Repulse = EMA(Repulse, period) where period = 5 or 13
        """,
        "default_period": 5,
        "range": "Typically -100 to +100",
        "signals": {
            "buy": "Repulse crosses above zero from below",
            "sell": "Repulse crosses below zero from above",
            "strong_buy": "Repulse rising from extreme negative (< -50)",
            "strong_sell": "Repulse falling from extreme positive (> 50)"
        },
        "smoothed_version": "Smoothed Repulse (EMA applied) for cleaner signals"
    },

    "RELATIVE_STRENGTH": {
        "name": "Relative Strength (vs benchmark)",
        "type": "momentum",
        "source": "centralcharts.com, IBD",
        "formula": "RS = Stock Price / Benchmark Price (scaled to 100)",
        "calculation": """
            1. Calculate RS Ratio = Stock Price / Benchmark Price (e.g., S&P 500)
            2. To track over time: RS Line = RS Ratio normalized
            3. Some versions: Rank all stocks by price performance, scale 1-99

            IBD Style Relative Strength Rating:
            - Compare stock's 12-month price change to all other stocks
            - Score from 1-99 (99 = outperforming 99% of stocks)

            RS Line (Chart):
            - Simply Stock Price / Index Price
            - Rising line = outperforming index
        """,
        "interpretation": {
            "rising": "Stock outperforming benchmark - leader",
            "falling": "Stock underperforming benchmark - laggard",
            "new_high": "RS Line at new high while price consolidates = bullish"
        },
        "key_insight": "Buy stocks with RS > 80 (IBD style) or rising RS Line making new highs"
    },

    "MANSFIELD_RS": {
        "name": "Mansfield's Relative Strength",
        "type": "momentum",
        "source": "centralcharts.com, Stan Weinstein",
        "formula": "Mansfield RS = ((Stock/Index) / SMA(Stock/Index, 52 weeks) - 1) * 100",
        "calculation": """
            1. Calculate RS Ratio = Stock Price / Index Price
            2. Calculate 52-week SMA of RS Ratio
            3. Mansfield RS = ((RS Ratio / 52-week SMA of RS) - 1) * 100

            Example:
            - Stock: $50, Index: $5000, RS Ratio = 0.01
            - 52-week SMA of RS Ratio = 0.0095
            - Mansfield RS = ((0.01 / 0.0095) - 1) * 100 = 5.26

            Oscillates around zero:
            - Positive = outperforming relative to historical norm
            - Negative = underperforming relative to historical norm
        """,
        "default_period": 52,  # weeks
        "signals": {
            "buy": "Mansfield RS above zero and rising (Stage 2 breakout)",
            "sell": "Mansfield RS below zero and falling (Stage 4 decline)",
            "hold": "Mansfield RS positive but falling (Stage 3 top)"
        },
        "key_insight": "Zero-line crossovers identify sector/stock rotation early"
    },
    
    # =========================================================================
    # TREND INDICATORS (23 indicators)
    # =========================================================================
    "DMI": {
        "name": "Directional Movement Index",
        "type": "trend",
        "source": "incrediblecharts.com, centralcharts.com, Welles Wilder",
        "default_period": 14,
        "components": ["+DI (upward trend)", "-DI (downward trend)", "ADX (trend strength)"],
        "formula": {
            "+DM": "Today's High - Yesterday's High (if positive and > -DM)",
            "-DM": "Yesterday's Low - Today's Low (if positive and > +DM)",
            "+DI": "(Smoothed +DM / ATR) * 100",
            "-DI": "(Smoothed -DM / ATR) * 100",
            "DX": "|+DI - -DI| / (+DI + -DI) * 100",
            "ADX": "Smoothed average of DX"
        },
        "calculation": """
            1. Calculate True Range (TR) = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
            2. Calculate +DM = High - Previous High (if > 0 and > -DM, else 0)
            3. Calculate -DM = Previous Low - Low (if > 0 and > +DM, else 0)
            4. Smooth +DM, -DM, and TR using Wilder's smoothing (or 14-period EMA)
            5. +DI = (Smoothed +DM / Smoothed TR) * 100
            6. -DI = (Smoothed -DM / Smoothed TR) * 100
            7. DX = (|+DI - -DI| / (+DI + -DI)) * 100
            8. ADX = Wilder's smoothing of DX over 14 periods

            Wilder's smoothing: Prev + (Current - Prev) / Period
        """,
        "signals": {
            "go_long": "+DI above -DI AND ADX rising above 20",
            "go_short": "-DI above +DI AND ADX rising above 20",
            "exit_long": "+DI crosses below -DI",
            "exit_short": "-DI crosses below +DI",
            "no_trade": "ADX below 20 = ranging market, wait for trend",
            "take_profit": "ADX turns down from above 40 = trend exhaustion"
        },
        "key_insight": "ADX measures trend STRENGTH (not direction); +DI/-DI measure direction"
    },

    "ADX": {
        "name": "Average Directional Index",
        "type": "trend",
        "source": "centralcharts.com, Welles Wilder",
        "default_period": 14,
        "formula": "ADX = Wilder's smoothing of DX over 14 periods",
        "calculation": """
            1. Calculate +DI and -DI (see DMI calculation)
            2. DX = (|+DI - -DI| / (+DI + -DI)) * 100
            3. First ADX = Average of first 14 DX values
            4. Subsequent ADX = ((Prev ADX * 13) + Current DX) / 14

            ADX only measures trend strength, not direction!
            Use +DI/-DI crossovers for direction signals
        """,
        "range": [0, 100],
        "interpretation": {
            "0-20": "Weak or no trend - use oscillator strategies",
            "20-40": "Moderate trend - trend-following may work",
            "40-60": "Strong trend - strong trend-following signals",
            "60-100": "Very strong trend - but watch for exhaustion"
        },
        "signals": {
            "trend_starting": "ADX rises above 20-25 from below",
            "trend_strengthening": "ADX continues rising above 25",
            "trend_weakening": "ADX turns down from above 40",
            "choppy_market": "ADX consistently below 20"
        }
    },

    "BOLLINGER": {
        "name": "Bollinger Bands",
        "type": "trend/volatility",
        "source": "incrediblecharts.com, centralcharts.com",
        "default_period": 20,
        "default_std": 2.0,
        "formula": "Middle = 20-SMA, Upper = Middle + 2*StdDev, Lower = Middle - 2*StdDev",
        "calculation": """
            1. Middle Band = SMA(Close, 20)
            2. StdDev = Standard Deviation of Close over 20 periods
            3. Upper Band = Middle + (2 * StdDev)
            4. Lower Band = Middle - (2 * StdDev)
        """,
        "signals": {
            "squeeze": "Contracting bands warn of upcoming trend, first breakout often false",
            "swing": "In ranging market, move from one band carries to opposite",
            "breakout": "Move outside band = strong trend likely to continue",
            "reversal": "Quick reversal after breakout = expect swing to opposite band",
            "trend_following": {
                "entry": "Price closes above upper band",
                "exit": "Price closes below lower band"
            }
        },
        "key_insight": "100-day Bollinger at 2 std dev for trend following"
    },

    "AROON": {
        "name": "Aroon Indicator",
        "type": "trend",
        "source": "centralcharts.com, Tushar Chande",
        "default_period": 25,
        "components": ["Aroon Up", "Aroon Down", "Aroon Oscillator"],
        "formula": {
            "aroon_up": "((Period - Days since highest high) / Period) * 100",
            "aroon_down": "((Period - Days since lowest low) / Period) * 100",
            "oscillator": "Aroon Up - Aroon Down"
        },
        "calculation": """
            1. Aroon Up = ((N - days since N-period high) / N) * 100
            2. Aroon Down = ((N - days since N-period low) / N) * 100
            3. Aroon Oscillator = Aroon Up - Aroon Down

            Example (25-period):
            - If highest high was 5 days ago: Aroon Up = ((25-5)/25)*100 = 80
            - If lowest low was 20 days ago: Aroon Down = ((25-20)/25)*100 = 20
            - Oscillator = 80 - 20 = 60 (bullish)

            Range: 0 to 100 for Up/Down; -100 to +100 for Oscillator
        """,
        "signals": {
            "strong_uptrend": "Aroon Up > 70, Aroon Down < 30",
            "strong_downtrend": "Aroon Down > 70, Aroon Up < 30",
            "trend_change": "Aroon lines cross (Up crosses above Down = bullish)",
            "consolidation": "Both lines between 30-70",
            "breakout_coming": "Both lines near 100 (recent high AND low)"
        },
        "key_insight": "Aroon identifies how long since high/low - good for catching new trends early"
    },

    "KELTNER_CHANNEL": {
        "name": "Keltner Channel",
        "type": "trend",
        "source": "centralcharts.com, Chester Keltner",
        "formula": {
            "middle": "EMA(Close, 20)",
            "upper": "Middle + (Multiplier * ATR)",
            "lower": "Middle - (Multiplier * ATR)"
        },
        "calculation": """
            1. Middle Line = EMA(Close, 20) - the trend centerline
            2. Calculate ATR over 10 periods (or use same period as EMA)
            3. Upper Band = Middle + (Multiplier * ATR), typically Multiplier = 2
            4. Lower Band = Middle - (Multiplier * ATR)

            Default settings: EMA(20), ATR(10), Multiplier = 2

            Difference from Bollinger: Uses ATR instead of Standard Deviation
            - ATR-based bands are more stable (less reactive to single spikes)
        """,
        "default_periods": {"ema": 20, "atr": 10, "multiplier": 2.0},
        "signals": {
            "buy": "Price closes above upper channel (strong uptrend)",
            "sell": "Price closes below lower channel (strong downtrend)",
            "pullback_buy": "Price touches middle line in uptrend",
            "pullback_sell": "Price touches middle line in downtrend"
        },
        "use_case": "Trend identification and volatility-adjusted support/resistance"
    },

    "DONCHIAN_CHANNEL": {
        "name": "Donchian Channel",
        "type": "trend",
        "source": "centralcharts.com, Richard Donchian",
        "default_period": 20,
        "formula": {
            "upper": "Highest High over N periods",
            "lower": "Lowest Low over N periods",
            "middle": "(Upper + Lower) / 2"
        },
        "calculation": """
            1. Upper Band = Highest High of last N periods (excluding current bar)
            2. Lower Band = Lowest Low of last N periods (excluding current bar)
            3. Middle Band = (Upper + Lower) / 2

            Example (20-period):
            - Look back 20 days
            - Upper = highest high of those 20 days
            - Lower = lowest low of those 20 days

            Turtle Trading: 20-day breakout entry, 10-day breakout exit
        """,
        "signals": {
            "buy": "Price breaks above upper channel (new N-day high)",
            "sell": "Price breaks below lower channel (new N-day low)",
            "exit_long": "Price breaks below 10-day low",
            "exit_short": "Price breaks above 10-day high"
        },
        "key_insight": "Turtle Trading system used 20-day and 55-day Donchian breakouts with ATR-based position sizing"
    },

    "ICHIMOKU": {
        "name": "Ichimoku Cloud (Ichimoku Kinko Hyo)",
        "type": "trend",
        "source": "centralcharts.com, Goichi Hosoda",
        "components": ["Tenkan-sen", "Kijun-sen", "Senkou Span A", "Senkou Span B", "Chikou Span"],
        "formula": {
            "tenkan": "(Highest High 9 + Lowest Low 9) / 2",
            "kijun": "(Highest High 26 + Lowest Low 26) / 2",
            "senkou_a": "(Tenkan + Kijun) / 2, plotted 26 periods ahead",
            "senkou_b": "(Highest High 52 + Lowest Low 52) / 2, plotted 26 periods ahead",
            "chikou": "Close plotted 26 periods back"
        },
        "calculation": """
            1. Tenkan-sen (Conversion Line) = (9-period high + 9-period low) / 2
               - Short-term trend line, similar to 9-period midpoint

            2. Kijun-sen (Base Line) = (26-period high + 26-period low) / 2
               - Medium-term trend line, acts as support/resistance

            3. Senkou Span A (Leading Span A) = (Tenkan + Kijun) / 2
               - Plotted 26 periods AHEAD - first cloud boundary

            4. Senkou Span B (Leading Span B) = (52-period high + 52-period low) / 2
               - Plotted 26 periods AHEAD - second cloud boundary

            5. Chikou Span (Lagging Span) = Current Close
               - Plotted 26 periods BACK - confirms trend

            Cloud (Kumo) = Area between Senkou Span A and B
        """,
        "default_periods": {"tenkan": 9, "kijun": 26, "senkou_b": 52, "displacement": 26},
        "signals": {
            "strong_buy": "Price above cloud + Tenkan above Kijun + Chikou above price 26 periods ago",
            "strong_sell": "Price below cloud + Tenkan below Kijun + Chikou below price 26 periods ago",
            "tk_cross_buy": "Tenkan crosses above Kijun (especially above cloud)",
            "tk_cross_sell": "Tenkan crosses below Kijun (especially below cloud)",
            "kumo_breakout": "Price breaks above/below cloud = major trend change"
        },
        "key_insight": "Complete trading system - trend, momentum, support/resistance all in one indicator"
    },

    "PARABOLIC_SAR": {
        "name": "Parabolic SAR (Stop and Reverse)",
        "type": "trend",
        "source": "incrediblecharts.com, centralcharts.com, Welles Wilder",
        "parameters": {
            "start_af": 0.02,
            "max_af": 0.20,
            "af_increment": 0.02
        },
        "formula": "SAR(tomorrow) = SAR(today) + AF * (EP - SAR(today))",
        "calculation": """
            1. AF (Acceleration Factor) starts at 0.02
            2. EP (Extreme Point) = Highest high (uptrend) or Lowest low (downtrend)
            3. SAR = Prior SAR + AF * (EP - Prior SAR)

            Rules:
            - AF increases by 0.02 each time EP makes new extreme, max 0.20
            - When price crosses SAR, flip direction:
              a. Reset AF to 0.02
              b. EP becomes opposite extreme
              c. SAR becomes previous EP
            - SAR cannot enter current or previous bar's price range

            Example (Uptrend):
            - Prior SAR = 100, EP = 110, AF = 0.04
            - New SAR = 100 + 0.04 * (110 - 100) = 100.40
        """,
        "signals": {
            "buy": "SAR flips from above to below price (dots move under candles)",
            "sell": "SAR flips from below to above price (dots move above candles)",
            "trailing_stop": "Use SAR value as trailing stop level"
        },
        "use_case": "Trailing stop in trending markets",
        "key_insight": "Works best in strong trends; avoid in ranging/choppy markets"
    },

    "SUPERTREND": {
        "name": "SuperTrend",
        "type": "trend",
        "source": "centralcharts.com",
        "formula": {
            "basic_upper": "((High + Low) / 2) + (Multiplier * ATR)",
            "basic_lower": "((High + Low) / 2) - (Multiplier * ATR)"
        },
        "calculation": """
            1. Calculate ATR over N periods (default 10)
            2. Basic Upper Band = ((High + Low) / 2) + (Multiplier * ATR)
            3. Basic Lower Band = ((High + Low) / 2) - (Multiplier * ATR)
            4. Final Upper Band = min(Basic Upper, Previous Final Upper) if Previous Close > Previous Final Upper
            5. Final Lower Band = max(Basic Lower, Previous Final Lower) if Previous Close < Previous Final Lower

            SuperTrend value:
            - If in uptrend (Close > Previous SuperTrend): SuperTrend = Final Lower Band
            - If in downtrend (Close < Previous SuperTrend): SuperTrend = Final Upper Band

            Flip when Close crosses SuperTrend value
        """,
        "default_periods": {"atr": 10, "multiplier": 3.0},
        "signals": {
            "buy": "Close crosses above SuperTrend (line flips to green/below price)",
            "sell": "Close crosses below SuperTrend (line flips to red/above price)",
            "hold_long": "Price stays above green SuperTrend line",
            "hold_short": "Price stays below red SuperTrend line"
        },
        "key_insight": "Simplified trend-following system with built-in ATR-based stops"
    },

    "MOVING_AVERAGES": {
        "name": "Moving Averages",
        "type": "trend",
        "source": "centralcharts.com, NYU Stern",
        "types": {
            "SMA": "Simple Moving Average = Sum(Close, N) / N",
            "EMA": "Exponential Moving Average, more weight on recent prices",
            "WMA": "Weighted Moving Average = Sum(Price * Weight) / Sum(Weights)",
            "DEMA": "Double Exponential MA = 2*EMA - EMA(EMA)",
            "TEMA": "Triple Exponential MA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))",
            "AMA": "Adaptive Moving Average - adjusts to volatility"
        },
        "calculation": """
            SMA = Sum(Close, N) / N

            EMA: Multiplier = 2 / (N + 1)
                 EMA = (Close * Multiplier) + (Previous EMA * (1 - Multiplier))

            WMA: Weights = N, N-1, N-2, ... 1
                 WMA = (Close_0*N + Close_1*(N-1) + ... + Close_N*1) / Sum(1 to N)

            DEMA: EMA1 = EMA(Close, N)
                  EMA2 = EMA(EMA1, N)
                  DEMA = 2*EMA1 - EMA2

            TEMA: EMA1, EMA2, EMA3 = triple nesting
                  TEMA = 3*EMA1 - 3*EMA2 + EMA3
        """,
        "common_periods": [8, 13, 21, 50, 100, 200],
        "signals": {
            "golden_cross": "50-MA crosses above 200-MA = long-term bullish",
            "death_cross": "50-MA crosses below 200-MA = long-term bearish",
            "trend_following": "Price above rising MA = uptrend",
            "pullback_buy": "Price pulls back to rising MA = buy opportunity"
        }
    },

    "ENVELOPES": {
        "name": "Envelopes / Trading Bands",
        "type": "trend",
        "source": "centralcharts.com",
        "formula": {
            "upper": "MA * (1 + percentage)",
            "lower": "MA * (1 - percentage)"
        },
        "calculation": """
            1. Calculate Moving Average (SMA or EMA) over N periods
            2. Upper Envelope = MA * (1 + Percentage/100)
            3. Lower Envelope = MA * (1 - Percentage/100)

            Example (20-SMA with 2.5% envelope):
            - MA = 100
            - Upper = 100 * 1.025 = 102.50
            - Lower = 100 * 0.975 = 97.50

            Common: 2-3% for short-term, 5% for longer-term
        """,
        "default_settings": {"period": 20, "percentage": 2.5},
        "signals": {
            "buy": "Price touches or breaks below lower envelope in uptrend",
            "sell": "Price touches or breaks above upper envelope in downtrend",
            "breakout_buy": "Price closes above upper envelope with volume",
            "breakout_sell": "Price closes below lower envelope with volume"
        }
    },

    "LINEAR_REGRESSION": {
        "name": "Linear Regression",
        "type": "trend",
        "source": "centralcharts.com",
        "description": "Best fit line through price data using least squares method",
        "calculation": """
            Linear Regression Line Formula:
            y = a + bx

            Where:
            b (slope) = (N * Sum(xy) - Sum(x) * Sum(y)) / (N * Sum(x²) - (Sum(x))²)
            a (intercept) = (Sum(y) - b * Sum(x)) / N

            For price data:
            - x = time index (0, 1, 2, ... N-1)
            - y = closing prices

            Linear Regression Value (LRV) = The y value on the regression line for current bar
            Often plotted as moving regression or "end-point" moving average
        """,
        "default_period": 14,
        "signals": {
            "trend_direction": "Positive slope = uptrend, Negative slope = downtrend",
            "reversion": "Price far above/below regression line may revert to it",
            "acceleration": "Slope increasing = trend strengthening"
        },
        "use_case": "Smooth trend identification, regression channels"
    },

    "LINEAR_REGRESSION_SLOPE": {
        "name": "Linear Regression Slope",
        "type": "trend",
        "source": "centralcharts.com",
        "formula": "Slope = (N * Sum(xy) - Sum(x) * Sum(y)) / (N * Sum(x²) - (Sum(x))²)",
        "calculation": """
            1. Take N periods of closing prices (y values)
            2. Use time indices as x values (0, 1, 2, ... N-1)
            3. Calculate slope using least squares formula
            4. Slope value represents rate of price change per period

            Can normalize by dividing by price for percentage slope
        """,
        "default_period": 14,
        "signals": {
            "bullish": "Positive slope and rising (accelerating uptrend)",
            "bearish": "Negative slope and falling (accelerating downtrend)",
            "reversal_warning": "Slope flattening toward zero"
        }
    },

    "PIVOT_POINTS": {
        "name": "Pivot Points",
        "type": "trend/support_resistance",
        "source": "centralcharts.com",
        "formula": {
            "pivot": "(High + Low + Close) / 3",
            "r1": "(2 * Pivot) - Low",
            "s1": "(2 * Pivot) - High",
            "r2": "Pivot + (High - Low)",
            "s2": "Pivot - (High - Low)",
            "r3": "High + 2 * (Pivot - Low)",
            "s3": "Low - 2 * (High - Pivot)"
        },
        "calculation": """
            Using PREVIOUS period's High, Low, Close:

            Standard Pivots:
            1. Pivot Point (PP) = (High + Low + Close) / 3
            2. R1 = (2 * PP) - Low
            3. S1 = (2 * PP) - High
            4. R2 = PP + (High - Low)
            5. S2 = PP - (High - Low)
            6. R3 = High + 2 * (PP - Low)
            7. S3 = Low - 2 * (High - PP)

            Fibonacci Pivots (alternative):
            R1 = PP + 0.382 * (High - Low)
            R2 = PP + 0.618 * (High - Low)
            R3 = PP + 1.0 * (High - Low)
        """,
        "signals": {
            "bullish": "Price opens above Pivot Point",
            "bearish": "Price opens below Pivot Point",
            "buy_target": "R1, R2, R3 as resistance/profit targets",
            "sell_target": "S1, S2, S3 as support/stop levels"
        },
        "use_case": "Intraday support/resistance levels, calculated from daily/weekly data"
    },

    "ZIGZAG": {
        "name": "ZigZag",
        "type": "trend",
        "source": "centralcharts.com, Investopedia",
        "description": "Filters out minor price movements to show significant swings",
        "calculation": """
            1. Set deviation percentage threshold (default 5%)
            2. Start from first bar, mark as potential pivot
            3. For each subsequent bar:
               - If price moves more than deviation% from last pivot:
                 a. If current direction was UP and price falls > deviation% from high: mark new HIGH pivot
                 b. If current direction was DOWN and price rises > deviation% from low: mark new LOW pivot
               - Connect pivots with straight lines
            4. Algorithm is retroactive - pivots are confirmed only after reversal
        """,
        "default_deviation": 5,  # 5% is common
        "parameters": {
            "deviation": "Percentage move required to confirm a new leg (1-10% typical)",
            "depth": "Minimum bars between pivots"
        },
        "signals": {
            "swing_high": "Local maximum confirmed by deviation% reversal",
            "swing_low": "Local minimum confirmed by deviation% reversal"
        },
        "use_case": "Identify swing highs/lows, wave patterns, Elliott Wave analysis",
        "key_insight": "Repainting indicator - last segment may change until confirmed"
    },

    "FRACTAL_DIMENSION_INDEX": {
        "name": "Fractal Dimension Index",
        "type": "trend",
        "source": "centralcharts.com",
        "formula": "FDI = 1 + (log(N-1) + log(N) - log(HL_Diff)) / log(2 * N)",
        "calculation": """
            1. For period N (default 30):
            2. Calculate highest high and lowest low over N periods
            3. HL_Diff = (Highest_High - Lowest_Low) normalized
            4. N = number of periods
            5. FDI = 1 + (log(N-1) + log(2*N)) / log(2*N) adjusted by price range

            Simplified version:
            FDI uses Hurst exponent or box-counting method to measure fractal dimension
        """,
        "default_period": 30,
        "range": [1, 2],
        "interpretation": {
            "near_1": "Trending market - price moving in straight line",
            "near_1.5": "Random walk / no trend - best for mean-reversion",
            "near_2": "Ranging/choppy market - complex patterns"
        },
        "signals": {
            "use_trend_following": "FDI < 1.4",
            "use_mean_reversion": "FDI > 1.6"
        }
    },

    "STANDARD_ERROR": {
        "name": "Standard Error",
        "type": "trend",
        "source": "centralcharts.com",
        "formula": "SE = sqrt(Sum((Close - LinearReg)^2) / (N - 2))",
        "calculation": """
            1. Calculate Linear Regression line over N periods
            2. For each bar, calculate deviation from regression: (Close - Regression_Value)
            3. Square each deviation
            4. Sum of squared deviations
            5. Standard Error = sqrt(Sum / (N - 2))

            Can also create bands: Regression ± (Multiplier * Standard Error)
        """,
        "default_period": 21,
        "interpretation": {
            "low_SE": "Price closely follows trend line - strong trend",
            "high_SE": "Price deviates significantly - choppy/volatile"
        },
        "use_case": "Measure reliability of linear regression trend, create regression channels"
    },
    
    # =========================================================================
    # VOLATILITY INDICATORS (7 indicators)
    # =========================================================================
    "ATR": {
        "name": "Average True Range",
        "type": "volatility",
        "source": "incrediblecharts.com, centralcharts.com",
        "default_period": 14,
        "formula": "TR = max(H-L, |H-PrevC|, |L-PrevC|), ATR = EMA(TR) or SMA(TR)",
        "calculation": """
            1. True Range = max(High - Low, |High - PrevClose|, |Low - PrevClose|)
            2. ATR = EMA or SMA of True Range over N periods
        """,
        "signals": {
            "high_values": "Warn of market tops and bottoms, increased volatility",
            "low_values": "Indicate ranging markets, low volatility"
        },
        "use_cases": ["Position sizing", "Stop loss calculation", "Trailing stops", "Volatility breakout systems"],
        "key_insight": "Peaks before price bottoms; use for ATR-based trailing stops (2-3x ATR)"
    },

    "BOLLINGER_BANDWIDTH": {
        "name": "Bollinger Bandwidth",
        "type": "volatility",
        "source": "centralcharts.com, John Bollinger",
        "formula": "Bandwidth = (Upper Band - Lower Band) / Middle Band * 100",
        "calculation": """
            1. Calculate Bollinger Bands (20-SMA, 2 StdDev)
            2. Bandwidth = ((Upper - Lower) / Middle) * 100

            Example:
            - Upper Band = 105, Lower Band = 95, Middle = 100
            - Bandwidth = ((105 - 95) / 100) * 100 = 10%

            %B (related): (Price - Lower) / (Upper - Lower)
        """,
        "signals": {
            "squeeze": "Bandwidth at 6-month low = major breakout imminent",
            "expansion": "Bandwidth rapidly increasing = strong trend in progress",
            "contraction": "Bandwidth declining = trend weakening, consolidation"
        },
        "key_insight": "Bandwidth squeeze (low volatility) often precedes explosive moves"
    },

    "STANDARD_DEVIATION": {
        "name": "Standard Deviation",
        "type": "volatility",
        "source": "centralcharts.com",
        "formula": "StdDev = sqrt(Sum((x - mean)²) / N)",
        "calculation": """
            1. Calculate mean of prices over N periods
            2. For each price, calculate (Price - Mean)²
            3. Sum all squared deviations
            4. Divide by N (population) or N-1 (sample)
            5. Take square root

            Example (4 prices: 10, 12, 11, 13):
            - Mean = 11.5
            - Deviations: -1.5, 0.5, -0.5, 1.5
            - Squared: 2.25, 0.25, 0.25, 2.25
            - Sum = 5.0, StdDev = sqrt(5/4) = 1.118
        """,
        "default_period": 20,
        "use_case": "Measure of price volatility around the mean, basis for Bollinger Bands"
    },

    "HISTORIC_VOLATILITY": {
        "name": "Historic Volatility (HV)",
        "type": "volatility",
        "source": "centralcharts.com",
        "formula": "HV = StdDev(log returns) * sqrt(252) * 100 (annualized)",
        "calculation": """
            1. Calculate log returns: ln(Close / Previous Close)
            2. Calculate standard deviation of log returns over N periods
            3. Annualize: Multiply by sqrt(trading days per year)
               - Daily data: sqrt(252) ≈ 15.87
               - Weekly data: sqrt(52) ≈ 7.21
               - Monthly data: sqrt(12) ≈ 3.46
            4. Multiply by 100 for percentage

            Example:
            - 20-day StdDev of log returns = 0.02 (2%)
            - Annualized HV = 0.02 * sqrt(252) * 100 = 31.75%
        """,
        "default_period": 20,
        "use_case": "Compare current volatility to historical norms, options pricing"
    },

    "CHAIKIN_VOLATILITY": {
        "name": "Chaikin's Volatility",
        "type": "volatility",
        "source": "centralcharts.com, Marc Chaikin",
        "formula": "CV = ((EMA(H-L, 10) - EMA(H-L, 10)[10 periods ago]) / EMA(H-L, 10)[10]) * 100",
        "calculation": """
            1. Calculate High - Low (daily range) for each bar
            2. Calculate 10-period EMA of the High-Low range
            3. Compare today's EMA to EMA 10 periods ago
            4. CV = ((EMA_today - EMA_10_ago) / EMA_10_ago) * 100

            Measures rate of change in trading range, not absolute range
        """,
        "default_period": 10,
        "signals": {
            "rising": "Increasing volatility - possible trend change or continuation",
            "falling": "Decreasing volatility - trend slowing, possible reversal",
            "spike": "Sharp rise in CV often accompanies market bottoms"
        }
    },

    "CHANDE_KROLL_STOP": {
        "name": "Chande Kroll Stop",
        "type": "volatility",
        "source": "centralcharts.com, FXOpen, Tushar Chande & Stanley Kroll",
        "description": "ATR-based trailing stop indicator that adapts to volatility",
        "formula": {
            "long_stop": "Highest(Highest High - X * ATR, Q)",
            "short_stop": "Lowest(Lowest Low + X * ATR, Q)"
        },
        "calculation": """
            Parameters: P (ATR period, default 10), X (ATR multiplier, default 1), Q (lookback, default 9)

            1. Calculate ATR over P periods
            2. First Stop High = Highest High over P periods - (X * ATR)
            3. First Stop Low = Lowest Low over P periods + (X * ATR)
            4. Long Stop = Highest(First Stop High, Q periods) - smoothed stop for long positions
            5. Short Stop = Lowest(First Stop Low, Q periods) - smoothed stop for short positions

            The Q parameter creates a stepped trailing stop effect
        """,
        "default_periods": {"P": 10, "X": 1.0, "Q": 9},
        "interpretation": {
            "long_position": "Exit when price closes below Long Stop (green/blue line)",
            "short_position": "Exit when price closes above Short Stop (red/orange line)"
        },
        "signals": {
            "buy": "Long Stop crosses above Short Stop (bullish momentum)",
            "sell": "Short Stop crosses below Long Stop (bearish momentum)",
            "stop_loss_long": "Place stop below Long Stop line",
            "stop_loss_short": "Place stop above Short Stop line"
        },
        "use_case": "Dynamic stop-loss placement that adapts to market volatility",
        "key_insight": "Higher X values = wider stops, less whipsaws but more risk"
    },

    "MASS_INDEX": {
        "name": "Mass Index",
        "type": "volatility",
        "source": "centralcharts.com, Donald Dorsey",
        "formula": "Mass Index = Sum(EMA(H-L, 9) / EMA(EMA(H-L, 9), 9)) over 25 periods",
        "calculation": """
            1. Calculate Single EMA: EMA1 = EMA(High - Low, 9)
            2. Calculate Double EMA: EMA2 = EMA(EMA1, 9)
            3. Calculate EMA Ratio: Ratio = EMA1 / EMA2
            4. Mass Index = Sum of Ratio over 25 periods

            The ratio is always close to 1, but:
            - When volatility expands: EMA1 rises faster than EMA2, ratio > 1
            - When volatility contracts: EMA1 falls faster, ratio < 1

            Normal range: approximately 21-27
        """,
        "default_periods": {"ema": 9, "sum": 25},
        "signals": {
            "reversal_bulge": "Mass Index rises above 27, then falls below 26.5 = trend reversal likely",
            "trend_continuation": "Mass Index stays below 27 = current trend continues"
        },
        "key_insight": "Identifies trend reversals by detecting volatility bulges, regardless of direction"
    },

    # =========================================================================
    # VOLUME INDICATORS (10 indicators)
    # =========================================================================
    "OBV": {
        "name": "On Balance Volume",
        "type": "volume",
        "source": "incrediblecharts.com, centralcharts.com",
        "formula": "If Close > PrevClose: OBV = PrevOBV + Volume; If Close < PrevClose: OBV = PrevOBV - Volume",
        "calculation": """
            1. If Close > Previous Close: OBV = Previous OBV + Volume
            2. If Close < Previous Close: OBV = Previous OBV - Volume
            3. If Close = Previous Close: OBV = Previous OBV
        """,
        "signals": {
            "ranging": {
                "rising_obv": "Warns of upward breakout",
                "falling_obv": "Warns of downward breakout"
            },
            "trending": {
                "rising_obv": "Confirms uptrend",
                "falling_obv": "Confirms downtrend"
            },
            "divergences": {
                "bullish": "Price makes lower low, OBV makes higher low = bottom warning",
                "bearish": "Price makes higher high, OBV makes lower high = top warning"
            }
        },
        "key_insight": "Triple divergences are especially powerful signals"
    },

    "ACCUMULATION_DISTRIBUTION": {
        "name": "Accumulation/Distribution Line",
        "type": "volume",
        "source": "centralcharts.com, Marc Chaikin",
        "formula": {
            "CLV": "((Close - Low) - (High - Close)) / (High - Low)",
            "AD": "Previous AD + CLV * Volume"
        },
        "calculation": """
            1. Calculate Close Location Value (CLV):
               CLV = ((Close - Low) - (High - Close)) / (High - Low)
               CLV ranges from -1 (close at low) to +1 (close at high)

            2. Money Flow Volume = CLV * Volume

            3. AD Line = Previous AD + Money Flow Volume

            Example:
            - High=55, Low=50, Close=54
            - CLV = ((54-50) - (55-54)) / (55-50) = (4-1)/5 = 0.6
            - If Volume = 1M, Money Flow = 0.6 * 1M = 600K
            - AD increases by 600K
        """,
        "signals": {
            "rising": "Accumulation - buying pressure (closes near highs with volume)",
            "falling": "Distribution - selling pressure (closes near lows with volume)",
            "divergence": "AD diverging from price = potential reversal"
        },
        "key_insight": "Unlike OBV, considers where close is within daily range"
    },

    "CHAIKIN_MONEY_FLOW": {
        "name": "Chaikin Money Flow",
        "type": "volume",
        "source": "incrediblecharts.com, centralcharts.com, Marc Chaikin",
        "default_period": 21,
        "formula": "CMF = Sum(CLV * Volume, N) / Sum(Volume, N)",
        "calculation": """
            1. Calculate CLV = ((Close - Low) - (High - Close)) / (High - Low)
            2. Calculate Money Flow Volume = CLV * Volume
            3. Sum Money Flow Volume over N periods
            4. Sum Volume over N periods
            5. CMF = Sum(Money Flow Volume) / Sum(Volume)

            Result oscillates between -1 and +1
        """,
        "range": [-1, 1],
        "signals": {
            "buy": "CMF above 0.10 (strong accumulation)",
            "sell": "CMF below -0.10 (strong distribution)",
            "accumulation": "CMF consistently positive for extended period",
            "distribution": "CMF consistently negative for extended period"
        }
    },

    "CHAIKIN_OSCILLATOR": {
        "name": "Chaikin Oscillator",
        "type": "volume",
        "source": "centralcharts.com, Marc Chaikin",
        "formula": "Chaikin Osc = EMA(AD Line, 3) - EMA(AD Line, 10)",
        "calculation": """
            1. Calculate Accumulation/Distribution Line (see above)
            2. Calculate 3-period EMA of AD Line (fast)
            3. Calculate 10-period EMA of AD Line (slow)
            4. Chaikin Oscillator = Fast EMA - Slow EMA

            Measures momentum of the AD Line
        """,
        "default_periods": {"fast": 3, "slow": 10},
        "signals": {
            "buy": "Oscillator crosses above zero (AD momentum turning positive)",
            "sell": "Oscillator crosses below zero (AD momentum turning negative)",
            "divergence_buy": "Price makes lower low, Oscillator makes higher low",
            "divergence_sell": "Price makes higher high, Oscillator makes lower high"
        }
    },

    "MONEY_FLOW_INDEX": {
        "name": "Money Flow Index (MFI)",
        "type": "volume",
        "source": "centralcharts.com",
        "default_period": 14,
        "formula": "100 - (100 / (1 + Money Flow Ratio))",
        "overbought": 80,
        "oversold": 20,
        "calculation": """
            1. Typical Price = (High + Low + Close) / 3
            2. Money Flow = Typical Price * Volume
            3. Positive/Negative Money Flow based on Typical Price direction
            4. MFI = 100 - (100 / (1 + Positive MF / Negative MF))
        """,
        "key_insight": "Volume-weighted RSI - more reliable than standard RSI"
    },

    "EASE_OF_MOVEMENT": {
        "name": "Ease of Movement (EMV)",
        "type": "volume",
        "source": "centralcharts.com, Richard Arms",
        "formula": "EMV = Distance / Box Ratio",
        "calculation": """
            1. Distance = ((High + Low) / 2) - ((PrevHigh + PrevLow) / 2)
               (How far the midpoint moved)

            2. Box Ratio = (Volume / scale) / (High - Low)
               (Volume relative to price range - scale often 1,000,000)

            3. EMV = Distance / Box Ratio

            4. Usually smoothed with 14-period SMA for signals

            Example:
            - Today: H=55, L=50; Yesterday: H=54, L=49
            - Distance = 52.5 - 51.5 = 1
            - Volume = 2M, Range = 5
            - Box Ratio = (2M / 1M) / 5 = 0.4
            - EMV = 1 / 0.4 = 2.5 (easy upward movement)
        """,
        "default_period": 14,  # for smoothing
        "signals": {
            "positive": "Easy upward movement (price rising on light volume)",
            "negative": "Easy downward movement (price falling on light volume)",
            "high_positive": "Strong upward momentum with good ease",
            "high_negative": "Strong downward momentum with good ease"
        },
        "key_insight": "High EMV = price moving easily; Low EMV = price struggling (heavy volume for small moves)"
    },

    "PVT": {
        "name": "Price Volume Trend",
        "type": "volume",
        "source": "centralcharts.com",
        "formula": "PVT = Previous PVT + (Volume * (Close - PrevClose) / PrevClose)",
        "calculation": """
            1. Calculate percentage change: PCT = (Close - Previous Close) / Previous Close
            2. Add portion of volume based on percentage change
            3. PVT = Previous PVT + (Volume * PCT)

            Example:
            - Previous Close = 100, Close = 102, Volume = 1M
            - PCT = (102 - 100) / 100 = 0.02 (2%)
            - PVT adds = 1M * 0.02 = 20,000

            Unlike OBV (all or nothing), PVT adds proportional volume
        """,
        "signals": {
            "rising": "Bullish volume confirmation - accumulation",
            "falling": "Bearish volume confirmation - distribution",
            "divergence_buy": "Price lower low, PVT higher low = reversal up",
            "divergence_sell": "Price higher high, PVT lower high = reversal down"
        },
        "key_insight": "More nuanced than OBV - reflects percentage moves, not just direction"
    },

    "NEGATIVE_VOLUME_INDEX": {
        "name": "Negative Volume Index (NVI)",
        "type": "volume",
        "source": "centralcharts.com, Paul Dysart (1930s)",
        "description": "Tracks price changes on days with lower volume than previous day",
        "formula": "If Volume < Previous Volume: NVI = Previous NVI + ROC",
        "calculation": """
            1. Start with NVI = 1000 (or any base value)
            2. For each day:
               - If Today's Volume < Yesterday's Volume:
                 NVI = Previous NVI + ((Close - Previous Close) / Previous Close) * Previous NVI
               - Else: NVI = Previous NVI (unchanged)
            3. Often plotted with 255-day (1 year) EMA
        """,
        "signals": {
            "bullish": "NVI above its 255-day EMA (96% probability of bull market)",
            "bearish": "NVI below its 255-day EMA (53% probability of bear market)"
        },
        "interpretation": "Smart money trades on low volume days - NVI captures their activity",
        "key_insight": "Developed by Paul Dysart in 1930s; NVI above 1-year MA = strong bull market indicator"
    },

    "POSITIVE_VOLUME_INDEX": {
        "name": "Positive Volume Index (PVI)",
        "type": "volume",
        "source": "centralcharts.com, Paul Dysart",
        "description": "Tracks price changes on days with higher volume than previous day",
        "formula": "If Volume > Previous Volume: PVI = Previous PVI + ROC",
        "calculation": """
            1. Start with PVI = 1000 (or any base value)
            2. For each day:
               - If Today's Volume > Yesterday's Volume:
                 PVI = Previous PVI + ((Close - Previous Close) / Previous Close) * Previous PVI
               - Else: PVI = Previous PVI (unchanged)
            3. Often plotted with 255-day (1 year) EMA
        """,
        "signals": {
            "bullish": "PVI above its 255-day EMA",
            "bearish": "PVI below its 255-day EMA"
        },
        "interpretation": "Retail/emotional trading on high volume days - less reliable than NVI",
        "use_with": "Use with NVI - when both agree, signal is stronger"
    },

    # =========================================================================
    # OSCILLATOR INDICATORS (17 indicators)
    # =========================================================================
    "BOLLINGER_PERCENT_B": {
        "name": "Bollinger %b",
        "type": "oscillator",
        "source": "centralcharts.com, John Bollinger",
        "formula": "%b = (Price - Lower Band) / (Upper Band - Lower Band)",
        "calculation": """
            1. Calculate Bollinger Bands (20-SMA, 2 StdDev default)
            2. %b = (Close - Lower Band) / (Upper Band - Lower Band)

            Example:
            - Upper = 110, Lower = 90, Close = 105
            - %b = (105 - 90) / (110 - 90) = 15 / 20 = 0.75

            Interpretation:
            - %b = 1.0: Price at upper band
            - %b = 0.5: Price at middle band
            - %b = 0.0: Price at lower band
            - %b > 1.0: Price above upper band
            - %b < 0.0: Price below lower band
        """,
        "range": "Unbounded (typically 0-1, but can exceed)",
        "interpretation": {
            "above_1": "Price above upper band - strong/overbought",
            "below_0": "Price below lower band - weak/oversold",
            "at_0.5": "Price at middle band - neutral"
        },
        "signals": {
            "buy": "%b rises from below 0 back above 0",
            "sell": "%b falls from above 1 back below 1"
        }
    },

    "CCI": {
        "name": "Commodity Channel Index",
        "type": "oscillator",
        "source": "centralcharts.com, Donald Lambert",
        "default_period": 20,
        "formula": "CCI = (Typical Price - SMA(TP, N)) / (0.015 * Mean Deviation)",
        "calculation": """
            1. Typical Price (TP) = (High + Low + Close) / 3
            2. Calculate SMA of Typical Price over N periods
            3. Mean Deviation = Average of |TP - SMA(TP)| over N periods
            4. CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)

            The 0.015 constant scales so roughly 70-80% of values fall between -100 and +100

            Example:
            - TP = 102, SMA(TP) = 100, Mean Deviation = 1.5
            - CCI = (102 - 100) / (0.015 * 1.5) = 2 / 0.0225 = 88.9
        """,
        "range": "Unbounded (typically -200 to +200)",
        "overbought": 100,
        "oversold": -100,
        "signals": {
            "buy": "CCI crosses above -100 from below (emerging from oversold)",
            "sell": "CCI crosses below 100 from above (emerging from overbought)",
            "strong_buy": "CCI crosses above +100 (momentum breakout)",
            "strong_sell": "CCI crosses below -100 (momentum breakdown)"
        }
    },

    "WILLIAMS_R": {
        "name": "Williams %R",
        "type": "oscillator",
        "source": "centralcharts.com, Larry Williams",
        "default_period": 14,
        "formula": "%R = ((Highest High - Close) / (Highest High - Lowest Low)) * -100",
        "calculation": """
            1. Find Highest High over N periods
            2. Find Lowest Low over N periods
            3. %R = ((Highest High - Close) / (Highest High - Lowest Low)) * -100

            Example (14-period):
            - Highest High = 110, Lowest Low = 90, Close = 105
            - %R = ((110 - 105) / (110 - 90)) * -100 = (5/20) * -100 = -25

            Note: %R is inverted Stochastic %K
            - %R = %K - 100 or %R = -100 * (1 - %K/100)
        """,
        "range": [-100, 0],
        "overbought": -20,
        "oversold": -80,
        "signals": {
            "buy": "%R rises above -80 from below (leaving oversold)",
            "sell": "%R falls below -20 from above (leaving overbought)",
            "divergence_buy": "Price lower low, %R higher low",
            "divergence_sell": "Price higher high, %R lower high"
        },
        "key_insight": "Inverted scale - -100 is lowest (oversold), 0 is highest (overbought)"
    },

    "DPO": {
        "name": "Detrended Price Oscillator",
        "type": "oscillator",
        "source": "centralcharts.com",
        "formula": "DPO = Close - SMA(Close, N/2 + 1 periods ago)",
        "calculation": """
            1. Calculate N-period SMA
            2. Shift the SMA back by (N/2 + 1) periods
            3. DPO = Close - Shifted SMA

            Example (20-period DPO):
            - Shift = 20/2 + 1 = 11 periods
            - DPO = Today's Close - SMA value from 11 days ago

            This removes the trend, leaving only cycles
            Note: DPO is not plotted at current date but shifted back
        """,
        "default_period": 20,
        "signals": {
            "cycle_high": "DPO peaks above zero",
            "cycle_low": "DPO troughs below zero",
            "zero_cross_up": "Momentum shifting positive",
            "zero_cross_down": "Momentum shifting negative"
        },
        "use_case": "Identify cycles by removing trend - helps spot cycle lengths"
    },

    "TRIX": {
        "name": "TRIX",
        "type": "oscillator",
        "source": "centralcharts.com, StockCharts",
        "description": "Triple Exponential Average - momentum oscillator showing rate of change of triple-smoothed EMA",
        "formula": "TRIX = ((EMA3 - EMA3[1]) / EMA3[1]) * 100",
        "calculation": """
            1. Calculate first EMA: EMA1 = EMA(Close, period)
            2. Calculate second EMA: EMA2 = EMA(EMA1, period)
            3. Calculate third EMA: EMA3 = EMA(EMA2, period)
            4. TRIX = Percent change of EMA3: ((EMA3 - EMA3[1]) / EMA3[1]) * 100
            5. Signal Line = EMA(TRIX, signal_period)

            Default: period = 15, signal = 9
        """,
        "default_periods": {"period": 15, "signal": 9},
        "signals": {
            "buy": "TRIX crosses above zero (bullish) or crosses above signal line",
            "sell": "TRIX crosses below zero (bearish) or crosses below signal line",
            "divergence_buy": "Price makes lower low, TRIX makes higher low",
            "divergence_sell": "Price makes higher high, TRIX makes lower high"
        },
        "interpretation": {
            "positive": "Triple EMA is rising - bullish momentum",
            "negative": "Triple EMA is falling - bearish momentum"
        },
        "key_insight": "Triple smoothing filters out minor price fluctuations; similar to MACD but smoother"
    },

    "AWESOME_OSCILLATOR": {
        "name": "Awesome Oscillator (AO)",
        "type": "oscillator",
        "source": "centralcharts.com, Bill Williams",
        "formula": "AO = SMA((High+Low)/2, 5) - SMA((High+Low)/2, 34)",
        "calculation": """
            1. Calculate Midpoint Price = (High + Low) / 2
            2. Fast SMA = SMA(Midpoint, 5 periods)
            3. Slow SMA = SMA(Midpoint, 34 periods)
            4. AO = Fast SMA - Slow SMA
            5. Plotted as histogram (green bars = AO rising, red = AO falling)
        """,
        "signals": {
            "zero_cross_buy": "AO crosses above zero line",
            "zero_cross_sell": "AO crosses below zero line",
            "saucer_buy": "AO above zero, 2 red bars followed by green bar",
            "saucer_sell": "AO below zero, 2 green bars followed by red bar",
            "twin_peaks_buy": "Two peaks below zero, second higher, followed by green bar",
            "twin_peaks_sell": "Two peaks above zero, second lower, followed by red bar"
        },
        "key_insight": "Simplified MACD using median price - good for trend confirmation"
    },

    "CYCLE": {
        "name": "Cycle Indicator",
        "type": "oscillator",
        "source": "centralcharts.com, John Ehlers",
        "description": "Identifies cyclical patterns in price using Fourier or Hilbert Transform analysis",
        "calculation": """
            1. Detrend the price data (remove linear trend)
            2. Apply bandpass filter or Hilbert Transform
            3. Identify dominant cycle length
            4. Plot oscillator showing position within cycle

            Common approach: Use autocorrelation or spectral analysis
        """,
        "use_case": "Identify market cycles, time entries/exits with cycle phase"
    },

    "R_SQUARED": {
        "name": "R² Correlation (Coefficient of Determination)",
        "type": "oscillator",
        "source": "centralcharts.com",
        "formula": "R² = 1 - (SS_res / SS_tot)",
        "calculation": """
            1. Calculate linear regression over N periods
            2. SS_res = Sum of squared residuals (actual - predicted)²
            3. SS_tot = Sum of squared total (actual - mean)²
            4. R² = 1 - (SS_res / SS_tot)

            Alternatively: R² = Correlation² between price and time
        """,
        "default_period": 14,
        "range": [0, 1],
        "interpretation": {
            "above_0.8": "Strong trend - 80%+ of price movement explained by trend",
            "0.5_to_0.8": "Moderate trend",
            "below_0.2": "No trend / ranging - use mean reversion strategies"
        },
        "key_insight": "High R² = trend-following works; Low R² = oscillator strategies work"
    },

    "ROCNROLL": {
        "name": "ROCnRoll",
        "type": "oscillator",
        "source": "centralcharts.com",
        "description": "Rate of Change variant with rolling window smoothing",
        "formula": "EMA(ROC, period)",
        "calculation": """
            1. Calculate ROC = ((Close - Close[n]) / Close[n]) * 100
            2. Apply EMA smoothing to ROC
            3. Some versions use multiple ROC periods combined
        """,
        "default_period": 12,
        "signals": {
            "buy": "ROCnRoll crosses above zero",
            "sell": "ROCnRoll crosses below zero"
        }
    },

    "DYNAMIC_ZONE_RSI": {
        "name": "Dynamic Zone RSI",
        "type": "oscillator",
        "source": "centralcharts.com",
        "description": "RSI with dynamically adjusted overbought/oversold levels based on volatility",
        "calculation": """
            1. Calculate standard RSI
            2. Calculate standard deviation of RSI over lookback period
            3. Dynamic Overbought = Mean RSI + (Multiplier * StdDev)
            4. Dynamic Oversold = Mean RSI - (Multiplier * StdDev)
            5. Zones adjust based on recent RSI volatility
        """,
        "default_periods": {"rsi": 14, "lookback": 50, "multiplier": 2},
        "signals": {
            "buy": "RSI crosses above Dynamic Oversold zone",
            "sell": "RSI crosses below Dynamic Overbought zone"
        },
        "key_insight": "Adapts to market conditions - tighter zones in low volatility, wider in high volatility"
    },

    "DYNAMIC_ZONE_STOCHASTIC": {
        "name": "Dynamic Zone Stochastic",
        "type": "oscillator",
        "source": "centralcharts.com",
        "description": "Stochastic with dynamically adjusted levels based on recent behavior",
        "calculation": """
            1. Calculate standard Stochastic %K and %D
            2. Calculate mean and standard deviation of Stochastic over lookback
            3. Dynamic Overbought = Mean + (Multiplier * StdDev)
            4. Dynamic Oversold = Mean - (Multiplier * StdDev)
        """,
        "signals": {
            "buy": "Stochastic crosses above Dynamic Oversold",
            "sell": "Stochastic crosses below Dynamic Overbought"
        },
        "advantage": "Reduces false signals in trending markets"
    },

    # =========================================================================
    # DIVERGENCE DETECTION INDICATORS
    # =========================================================================
    "RSI_DIVERGENCE": {
        "name": "RSI Divergences",
        "type": "oscillator/divergence",
        "source": "centralcharts.com",
        "description": "Automatic detection of RSI divergences",
        "calculation": """
            1. Identify swing highs/lows in price (using ZigZag or peak detection)
            2. Identify corresponding swing highs/lows in RSI
            3. Bullish Divergence: Price makes Lower Low, RSI makes Higher Low
            4. Bearish Divergence: Price makes Higher High, RSI makes Lower High
            5. Hidden Bullish: Price makes Higher Low, RSI makes Lower Low (trend continuation)
            6. Hidden Bearish: Price makes Lower High, RSI makes Higher High
        """,
        "signals": {
            "bullish_divergence": "Strong buy signal - momentum shifting up",
            "bearish_divergence": "Strong sell signal - momentum shifting down",
            "hidden_bullish": "Buy on pullback in uptrend",
            "hidden_bearish": "Sell on rally in downtrend"
        }
    },

    "MACD_DIVERGENCE": {
        "name": "MACD Divergences",
        "type": "oscillator/divergence",
        "source": "centralcharts.com",
        "description": "Automatic detection of MACD divergences",
        "calculation": """
            1. Identify swing highs/lows in price
            2. Identify corresponding peaks/troughs in MACD histogram or MACD line
            3. Bullish Divergence: Price Lower Low, MACD Higher Low
            4. Bearish Divergence: Price Higher High, MACD Lower High
        """,
        "signals": {
            "bullish_divergence": "MACD shows accumulation despite falling price",
            "bearish_divergence": "MACD shows distribution despite rising price"
        },
        "key_insight": "MACD divergences often precede major trend reversals by 1-3 bars"
    },

    "CCI_DIVERGENCE": {
        "name": "CCI Divergence",
        "type": "oscillator/divergence",
        "source": "centralcharts.com",
        "description": "Automatic detection of CCI divergences",
        "calculation": """
            1. Calculate CCI with standard formula
            2. Identify price swing points and CCI swing points
            3. Compare to detect divergences (same logic as RSI/MACD)
        """,
        "signals": {
            "bullish_divergence": "Price Lower Low + CCI Higher Low = reversal up",
            "bearish_divergence": "Price Higher High + CCI Lower High = reversal down"
        },
        "key_insight": "CCI divergences particularly effective at extreme levels (beyond ±200)"
    },
}

