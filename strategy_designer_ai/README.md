# AI Strategy Designer

An automated system for creating ML-powered trading strategies from technical indicators. The system trains machine learning models on historical data, extracts trading rules, and generates complete strategy packages ready for backtesting and live trading.

Optionally, the `--LLM` flag enables Claude AI (claude-opus-4-6 with extended thinking) to design the strategy before ML training. Claude researches the symbols, selects indicators, defines entry/exit rules, and sets ML training parameters. The ML pipeline then finds optimal weights for Claude's design. On each re-iteration, Claude receives the full history of previous designs and results so it can pivot to new approaches rather than repeating what didn't work.

## Overview

The AI Strategy Designer uses a multi-step pipeline to create optimized trading strategies:

1. **LLM Strategy Design** *(optional, with `--LLM`)* - Claude API designs the strategy structure, rules, and parameters
2. **Data Collection** - Fetches historical OHLCV data from Polygon.io
3. **Feature Engineering** - Calculates 60+ technical indicators and pattern features
4. **ML Model Training** - Trains multiple models (GradientBoosting, LightGBM, XGBoost, RandomForest)
5. **Rule Extraction** - Extracts interpretable trading rules from the best model (merged with LLM rules if `--LLM`)
6. **Strategy Generation** - Creates complete strategy files with entry/exit logic
7. **Parameter Optimization** - Tunes risk management parameters for optimal performance

## Prerequisites

### Required Environment Variables

```bash
export POLYGON_MASSIVE_API_KEY="your_polygon_api_key"

# Required only when using --LLM flag
export ANTHROPIC_API_KEY="your_anthropic_api_key"
```

### Required Python Packages

```bash
pip install pandas numpy scikit-learn lightgbm xgboost polygon-api-client

# Required only when using --LLM flag
pip install anthropic
```

## Quick Start

### Create a New Strategy (ML only)

```bash
python strategy_designer_ai/strategy_builder.py \
    --name "Semiconductor_Momentum" \
    --symbols "NVDA,AMD,AVGO,TSM,QCOM,MU,MRVL,AMAT,LRCX,KLAC,ASML" \
    --timeframe 1d \
    --years 2
```

### Create a Strategy with LLM-Assisted Design

```bash
python strategy_designer_ai/strategy_builder.py \
    --name "Semiconductor_Momentum" \
    --symbols "NVDA,AMD,AVGO,TSM,QCOM" \
    --LLM \
    --desc "Momentum-based swing trading strategy for semiconductor stocks, buying on pullbacks and selling on strength" \
    --target 50 \
    --max-iter 3
```

When `--LLM` is used, the pipeline adds a Step 0 where Claude:
1. Researches the symbols via AlphaVantage tools
2. Designs entry/exit rules with specific indicator thresholds
3. Selects which features the ML pipeline should prioritize
4. Sets ML training parameters (forward return days, profit thresholds)
5. Recommends risk management parameters (stop loss, trailing stop, position sizing)

The ML pipeline then trains models to find optimal weights for Claude's design. If the capture rate target isn't met, Claude is called again with the full backtest results and complete history of all previous designs, ensuring each iteration tries a meaningfully different approach.

## Detailed Workflow

### Step 1: Create a New Strategy

The `strategy_builder.py` is the main entry point that orchestrates the entire process.

```bash
python strategy_designer_ai/strategy_builder.py \
    --name "My_Strategy" \
    --symbols "AAPL,MSFT,GOOGL" \
    --timeframe 1d \
    --years 2
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--name` | Strategy name (used for folder) | Required |
| `--symbols` | Comma-separated stock symbols | Required |
| `--timeframe` | Data timeframe (1d, 1h, etc.) | 1d |
| `--years` | Years of historical data | 5 |
| `--target` | Target capture rate percentage | 50 |
| `--max-iter` | Maximum training iterations | 5 |
| `--LLM` | Use Claude AI to pre-design strategy | Off |
| `--desc` | Strategy description (required with `--LLM`) | - |

### Step 2: ML Model Training (Automatic)

The strategy builder automatically trains multiple ML models:

- **GradientBoosting** - Robust ensemble method
- **LightGBM** - Fast gradient boosting
- **XGBoost** - Extreme gradient boosting
- **RandomForest** - Bagging ensemble
- **NeuralNetwork** - Multi-layer perceptron
- **LongShortTermMemory** - Recurrent neural network

Each model is trained with:
- Multiple forward return windows (5, 10, 15, 20 days)
- Multiple profit thresholds (2%, 3%, 5%)
- Hyperparameter tuning via RandomizedSearchCV
- Cross-validation for robust evaluation

**Features Used (60+):**
- Momentum: RSI, MACD, Stochastic, ROC
- Trend: EMA (8, 21, 50, 200), ADX, Trend Slope
- Volatility: ATR, Bollinger Bands, Keltner Channels
- Volume: OBV, CMF, Volume Ratio, VWAP
- Macro/Trend: 52-week high/low, EMA stack, momentum acceleration

### Step 3: Rule Extraction (Automatic)

The `rule_extractor.py` analyzes the trained model to extract interpretable rules:

```python
# Example extracted rules:
Entry Conditions:
  - RSI_14 < 35.0 (oversold)
  - MACD_HIST > 0 (bullish momentum)
  - EMA_STACK_SCORE >= 2 (bullish alignment)

Exit Conditions:
  - RSI_14 > 70.0 (overbought)
  - PRICE_VS_EMA_50 < -5.0 (below trend)
```

### Step 4: Strategy Generation (Automatic)

The builder generates a complete strategy package in `strategies/<name>/`:

```
strategies/Semiconductor_Momentum/
├── config.json          # Strategy configuration
├── strategy.py          # Trading logic with entry/exit rules
├── backtest.py          # Portfolio backtester
├── data_provider.py     # Market data fetcher
├── risk_manager.py      # Stop loss, trailing stop logic
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker container config
└── docker-compose.yml   # Docker compose config
```

### Step 5: Optimize the Strategy

After generation, optimize risk management parameters (automatically done by strategy_builder.py):

```bash
python strategy_designer_ai/strategy_optimizer.py \
    --strategy strategies/Semiconductor_Momentum \
    --iterations 50 \
    --apply
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--strategy`, `-s` | Path to strategy directory | Required |
| `--iterations`, `-n` | Max parameter combinations to test | 50 |
| `--stop-loss` | Stop loss percentages to test | 0.03-0.15 |
| `--trailing-stop` | Trailing stop percentages to test | 0.02-0.10 |
| `--max-positions` | Max positions to test | 2-5 |
| `--min-entry-score` | Min entry scores to test | 2-5 |
| `--apply`, `-a` | Apply best params to config | False |

**Parameters Optimized:**
- Stop Loss % (3% - 15%)
- Trailing Stop % (2% - 10%)
- Max Positions (2 - 5)
- Min Entry Score (2 - 5)

### Step 6: Run Backtest

Run the strategy backtest to validate performance:

```bash
cd strategies/Semiconductor_Momentum
python backtest.py
```

**Backtest Output Includes:**
- Total Return %
- Win Rate
- Max Drawdown
- Sharpe Ratio (approximation)
- Buy & Hold Comparison
- Max Potential Profit
- Per-Symbol Performance
- Exit Reason Breakdown

## Configuration

### config.json Structure

```json
{
  "strategy_name": "Semiconductor_Momentum",
  "version": "1.0.0",
  "symbols": ["NVDA", "AMD", "AVGO"],
  "timeframe": "1d",
  "trading": {
    "max_positions": 3,
    "min_entry_score": 3
  },
  "risk_management": {
    "stop_loss_pct": 0.05,
    "trailing_stop_pct": 0.03,
    "max_position_size": 0.1,
    "max_drawdown": 0.20,
    "max_daily_loss": 0.05
  },
  "ml_model": {
    "model_type": "GradientBoosting",
    "f1_score": 0.58,
    "precision": 0.65,
    "recall": 0.52
  }
}
```

## Example Usage

### Semiconductor Strategy (ML only)

```bash
# Create strategy
python strategy_designer_ai/strategy_builder.py \
    --name "Semiconductor_Momentum" \
    --symbols "NVDA,AMD,AVGO,TSM,QCOM,MU,MRVL,AMAT,LRCX,KLAC,ASML" \
    --years 2

# Optimize parameters
python strategy_designer_ai/strategy_optimizer.py \
    -s strategies/Semiconductor_Momentum \
    -n 100 \
    --apply

# Run backtest
cd strategies/Semiconductor_Momentum && python backtest.py
```

### LLM-Assisted Semiconductor Strategy

```bash
# Claude designs the strategy, ML finds the weights
python strategy_designer_ai/strategy_builder.py \
    --name "Semiconductor_Momentum_v2" \
    --symbols "NVDA,AMD,AVGO,TSM,QCOM" \
    --LLM \
    --desc "Momentum-based swing trading for semiconductors, buying oversold dips with strong trend confirmation" \
    --target 50 \
    --max-iter 3
```

### LLM-Assisted Gold/Commodity Strategy

```bash
python strategy_designer_ai/strategy_builder.py \
    --name "Gold_Macro_Swing" \
    --symbols "GLD,GDX,NEM,GOLD,AEM" \
    --LLM \
    --desc "Macro-driven gold strategy using inflation hedging and safe-haven flows during market uncertainty" \
    --target 40 \
    --years 3
```

### Biotech Strategy (ML only)

```bash
# Create strategy
python strategy_designer_ai/strategy_builder.py \
    --name "Biotech_Momentum" \
    --symbols "VRTX,REGN,ARGX,ALNY,BMRN,INSM" \
    --years 2

# Optimize and backtest
python strategy_designer_ai/strategy_optimizer.py \
    -s strategies/Biotech_Momentum --apply
cd strategies/Biotech_Momentum && python backtest.py
```

## LLM-Assisted Design Details

When `--LLM` is enabled, the pipeline works as follows:

### Initial Design (Step 0)
Claude receives a prompt with:
- Strategy name, symbols, description, target return, and timeframe
- The full catalog of 70+ available technical indicators
- The exact feature names the ML pipeline uses (60+ features)
- Access to AlphaVantage tools for real-time market data research

Claude outputs a structured JSON design containing:
- Priority features to focus on
- Entry/exit rules with specific indicator thresholds
- ML training parameters (forward return days, profit thresholds)
- Risk management recommendations (stop loss, trailing stop, position sizing)

### Re-Iteration Refinement
When the capture rate target isn't met, Claude is called again with:
- **Complete iteration history** - every previous design (features, rules, thresholds, ML params) paired with its backtest results
- **ML training metrics** - F1 scores, training data size, hyperparameter iterations per attempt
- **Per-symbol performance** - which symbols performed well or poorly
- **Exit reason breakdown** - stop losses, trailing stops, signal exits
- **Explicit anti-duplication instructions** - Claude must try meaningfully different approaches each time

This creates a feedback loop: `Claude designs -> ML trains -> Backtest evaluates -> Claude refines -> ML re-trains -> ...`

## Files Reference

| File | Purpose |
|------|---------|
| `strategy_builder.py` | Main orchestrator - runs entire pipeline |
| `llm_strategy_designer.py` | Claude API integration for LLM-assisted design |
| `ml_enhanced_trainer.py` | ML model training with hyperparameter tuning |
| `ml_strategy_designer.py` | Feature engineering (60+ indicators) |
| `rule_extractor.py` | Extract trading rules from ML models |
| `strategy_optimizer.py` | Optimize risk management parameters |
| `technical_indicators_index.py` | Technical indicator definitions |

## Troubleshooting

### Missing Polygon API Key
```
⚠️ Polygon client not configured. Set POLYGON_MASSIVE_API_KEY.
```
**Solution:** Export your Polygon API key:
```bash
export POLYGON_MASSIVE_API_KEY="your_api_key"
```

### Insufficient Data
```
❌ Insufficient common dates
```
**Solution:** Reduce `--years` parameter or check symbol availability.

### Module Not Found
```
ModuleNotFoundError: No module named 'polygon'
```
**Solution:** Install required packages:
```bash
pip install polygon-api-client lightgbm xgboost
```

## Performance Tips

1. **Use more training data** - Increase `--years` for better model accuracy
2. **Optimize parameters** - Always run the optimizer after strategy generation
3. **Test with Docker** - Use the generated Dockerfile for consistent environments
4. **Monitor drawdown** - Adjust `max_drawdown` in config for risk tolerance

