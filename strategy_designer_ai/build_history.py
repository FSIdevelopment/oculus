#!/usr/bin/env python3
"""
Build History Manager - Persistent memory system for strategy builds.

Maintains structured records of all past strategy builds so the LLM
can learn from what worked and what didn't across different asset classes,
features, and configurations.

Two data files:
- build_history.json: Array of build records (one per strategy)
- feature_tracker.json: Aggregated feature effectiveness data

Author: SignalSynk AI
"""

import json
import os
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


# =============================================================================
# Asset Class Inference
# =============================================================================

ASSET_CLASS_MAP = {
    "gold": {"GLD", "UGL", "GDX", "BAR", "IAU", "SGOL", "PHYS", "GDXJ", "RING", "AAAU"},
    "semiconductors": {
        "NVDA", "AMD", "AVGO", "TSM", "QCOM", "MU", "MRVL", "AMAT", "LRCX",
        "KLAC", "ASML", "INTC", "TXN", "ADI", "NXPI", "ON", "SWKS", "MCHP",
    },
    "tech": {
        "AAPL", "MSFT", "GOOGL", "GOOG", "META", "AMZN", "NFLX", "CRM",
        "ADBE", "ORCL", "NOW", "SNOW", "PLTR", "UBER",
    },
    "energy": {
        "XLE", "XOP", "OIH", "CVX", "XOM", "COP", "SLB", "EOG", "PXD",
        "USO", "BNO", "UCO",
    },
    "bonds": {"TLT", "IEF", "SHY", "BND", "AGG", "TMF", "TBT", "HYG", "LQD"},
    "crypto": {"BITO", "GBTC", "ETHE", "COIN", "MARA", "RIOT", "CLSK"},
    "index": {"SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "TQQQ", "SQQQ", "SPXL"},
}


def infer_asset_class(symbols: List[str]) -> str:
    """Infer asset class from a list of symbols."""
    symbol_set = set(s.upper() for s in symbols)
    best_match = "mixed"
    best_overlap = 0

    for asset_class, class_symbols in ASSET_CLASS_MAP.items():
        overlap = len(symbol_set & class_symbols)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = asset_class

    # Require at least one symbol match
    return best_match if best_overlap > 0 else "other"


# =============================================================================
# Build History Manager
# =============================================================================

class BuildHistoryManager:
    """Manages build_history.json - structured memory across strategy builds."""

    HISTORY_PATH = Path(__file__).parent / "build_history.json"

    def __init__(self):
        self.history: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        """Load existing build history from JSON file."""
        try:
            if self.HISTORY_PATH.exists():
                return json.loads(self.HISTORY_PATH.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: Could not load build history: {e}")
        return []

    def _save(self):
        """Write current history to JSON file (atomic write)."""
        try:
            content = json.dumps(self.history, indent=2)
            # Atomic write: write to temp file then rename
            fd, tmp_path = tempfile.mkstemp(
                dir=self.HISTORY_PATH.parent, suffix=".tmp"
            )
            try:
                os.write(fd, content.encode())
                os.close(fd)
                os.replace(tmp_path, self.HISTORY_PATH)
            except Exception:
                os.close(fd) if not os.get_inheritable(fd) else None
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
        except Exception as e:
            print(f"  Warning: Could not save build history: {e}")

    def add_build(self, build_record: Dict):
        """Append a completed build record. Deduplicates by strategy_name."""
        name = build_record.get("strategy_name", "")
        # Remove existing entry with same name (replace with updated data)
        self.history = [b for b in self.history if b.get("strategy_name") != name]
        self.history.append(build_record)
        self._save()

    def get_relevant_builds(
        self,
        symbols: Optional[List[str]] = None,
        asset_class: Optional[str] = None,
        max_results: int = 5,
    ) -> List[Dict]:
        """Return most relevant past builds for current design task.

        Relevance scoring:
        - Same asset_class: +10 points
        - Each overlapping symbol: +3 points
        - Successful build (hit target): +5 points
        - Recency tiebreaker: +0-2 points
        """
        if not self.history:
            return []

        symbol_set = set(s.upper() for s in (symbols or []))
        scored = []

        for build in self.history:
            score = 0
            build_symbols = set(s.upper() for s in build.get("symbols", []))

            # Asset class match
            if asset_class and build.get("asset_class") == asset_class:
                score += 10

            # Symbol overlap
            overlap = len(symbol_set & build_symbols)
            score += overlap * 3

            # Success bonus
            if build.get("success"):
                score += 5

            # Recency bonus (max 2 points)
            try:
                build_date = datetime.strptime(build.get("build_date", ""), "%Y-%m-%d")
                days_ago = (datetime.now() - build_date).days
                score += max(0, 2 - days_ago / 30)  # 2 pts if recent, decays over months
            except (ValueError, TypeError):
                pass

            scored.append((score, build))

        # Sort by score descending, then by build_date descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [build for _, build in scored[:max_results]]

    def format_for_prompt(self, builds: List[Dict]) -> str:
        """Format relevant builds as concise text for Claude's prompt."""
        if not builds:
            return ""

        lines = []
        for b in builds:
            status = "HIT TARGET" if b.get("success") else "below target"
            bt = b.get("backtest", {})
            model = b.get("model", {})
            risk = b.get("risk_params", {})
            features = b.get("features", {})

            # Use `or 0` to handle None values (key exists but value is None)
            total_return = bt.get("total_return") or 0
            return_str = f"{total_return:.1f}%" if total_return else "N/A"
            win_rate = bt.get("win_rate") or 0
            max_dd = bt.get("max_drawdown") or 0
            model_f1 = model.get("f1") or 0
            sl = risk.get("stop_loss") or 0
            ts = risk.get("trailing_stop") or 0

            lines.append(f"### {b.get('strategy_name', '?')} ({status})")
            lines.append(
                f"- Symbols: {', '.join(b.get('symbols', []))} | "
                f"Asset: {b.get('asset_class', '?')} | "
                f"Timeframe: {b.get('timeframe', '?')}"
            )
            lines.append(
                f"- Return: {return_str} | Win Rate: {win_rate:.0f}% | "
                f"Max DD: {max_dd:.1f}% | Target: {b.get('target') or '?'}%"
            )
            lines.append(
                f"- Model: {model.get('name') or '?'} (F1: {model_f1:.3f}) | "
                f"Training: {model.get('training_years') or '?'}yrs"
            )
            lines.append(
                f"- Stop Loss: {sl*100:.1f}% | "
                f"Trail: {ts*100:.1f}% | "
                f"MaxPos: {risk.get('max_positions') or '?'}"
            )

            priority = features.get("priority", [])
            ml_features = features.get("top_ml_features", [])
            if priority:
                lines.append(f"- LLM Priority Features: {', '.join(priority[:8])}")
            if ml_features:
                lines.append(f"- Top ML Features: {', '.join(ml_features[:8])}")

            # Top entry rules
            top_entry = features.get("top_entry_rules", [])
            if top_entry:
                rules_str = "; ".join(
                    f"{r['feature']} {r['operator']} {r['threshold']}"
                    for r in top_entry[:3]
                )
                lines.append(f"- Key Entry Rules: {rules_str}")

            # Iteration summary
            iters = b.get("iterations", {})
            if iters.get("results"):
                iter_strs = [
                    f"Iter{r['iteration']}={r.get('return', 0):.1f}%"
                    for r in iters["results"]
                ]
                lines.append(f"- Iterations: {', '.join(iter_strs)}")

            lines.append("")  # blank line between builds

        return "\n".join(lines)


# =============================================================================
# Feature Tracker
# =============================================================================

class FeatureTracker:
    """Aggregates feature effectiveness data from build history."""

    TRACKER_PATH = Path(__file__).parent / "feature_tracker.json"

    def __init__(self):
        self.data: Dict[str, Dict] = self._load()

    def _load(self) -> Dict:
        try:
            if self.TRACKER_PATH.exists():
                return json.loads(self.TRACKER_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
        return {}

    def _save(self):
        try:
            content = json.dumps(self.data, indent=2)
            fd, tmp_path = tempfile.mkstemp(
                dir=self.TRACKER_PATH.parent, suffix=".tmp"
            )
            try:
                os.write(fd, content.encode())
                os.close(fd)
                os.replace(tmp_path, self.TRACKER_PATH)
            except Exception:
                os.close(fd) if not os.get_inheritable(fd) else None
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
        except Exception as e:
            print(f"  Warning: Could not save feature tracker: {e}")

    def rebuild_from_history(self, history: List[Dict]):
        """Regenerate feature tracker from build history data."""
        tracker: Dict[str, Dict] = {}

        for build in history:
            name = build.get("strategy_name", "?")
            asset_class = build.get("asset_class", "other")
            bt = build.get("backtest", {})
            total_return = bt.get("total_return")
            features = build.get("features", {})

            # Collect all feature names used in this build
            all_features = set()
            for f in features.get("priority", []):
                all_features.add(f)
            for f in features.get("top_ml_features", []):
                all_features.add(f)

            # Update feature entries
            for feat in all_features:
                if feat not in tracker:
                    tracker[feat] = {
                        "used_in": [],
                        "times_used": 0,
                        "returns_when_used": [],
                        "asset_classes": [],
                        "as_entry_rule": {"count": 0, "thresholds": [], "operators": []},
                        "as_exit_rule": {"count": 0, "thresholds": [], "operators": []},
                    }
                entry = tracker[feat]
                if name not in entry["used_in"]:
                    entry["used_in"].append(name)
                    entry["times_used"] += 1
                    entry["returns_when_used"].append(total_return)
                    if asset_class not in entry["asset_classes"]:
                        entry["asset_classes"].append(asset_class)

            # Track entry rules
            for rule in features.get("top_entry_rules", []):
                feat = rule.get("feature", "")
                if feat in tracker:
                    tracker[feat]["as_entry_rule"]["count"] += 1
                    tracker[feat]["as_entry_rule"]["thresholds"].append(rule.get("threshold"))
                    tracker[feat]["as_entry_rule"]["operators"].append(rule.get("operator"))

            # Track exit rules
            for rule in features.get("top_exit_rules", []):
                feat = rule.get("feature", "")
                if feat in tracker:
                    tracker[feat]["as_exit_rule"]["count"] += 1
                    tracker[feat]["as_exit_rule"]["thresholds"].append(rule.get("threshold"))
                    tracker[feat]["as_exit_rule"]["operators"].append(rule.get("operator"))

        self.data = tracker
        self._save()

    def format_for_prompt(
        self, asset_class: Optional[str] = None, top_n: int = 15
    ) -> str:
        """Format feature effectiveness data for Claude's prompt."""
        if not self.data:
            return ""

        # Filter by asset class if specified
        features = []
        for feat, info in self.data.items():
            if asset_class and asset_class not in info.get("asset_classes", []):
                continue
            features.append((feat, info))

        if not features:
            # Fall back to all features if nothing matches the asset class
            features = list(self.data.items())

        # Sort by times_used descending
        features.sort(key=lambda x: x[1]["times_used"], reverse=True)
        features = features[:top_n]

        lines = []
        for feat, info in features:
            returns = [r for r in info.get("returns_when_used", []) if r is not None]
            avg_return = sum(returns) / len(returns) if returns else 0
            entry_info = info.get("as_entry_rule", {})
            exit_info = info.get("as_exit_rule", {})
            assets = ", ".join(info.get("asset_classes", []))

            line = f"- **{feat}**: used {info['times_used']}x"
            if returns:
                line += f", avg return {avg_return:.1f}%"
            if entry_info.get("count"):
                thresholds = [t for t in entry_info.get("thresholds", []) if t is not None]
                if thresholds:
                    line += f", entry thresholds: {[round(t, 2) for t in thresholds[:3]]}"
            if exit_info.get("count"):
                thresholds = [t for t in exit_info.get("thresholds", []) if t is not None]
                if thresholds:
                    line += f", exit thresholds: {[round(t, 2) for t in thresholds[:3]]}"
            line += f" ({assets})"
            lines.append(line)

        return "\n".join(lines)


# =============================================================================
# Retroactive Scanner
# =============================================================================

def scan_existing_strategies(strategies_dir: Optional[Path] = None) -> List[Dict]:
    """Scan all existing strategy directories to build history records.

    Handles both old-format configs (titan_gld, ugl_hybrid) and new-format
    configs (titan_gold_v5, semiconductor_momentum).
    """
    if strategies_dir is None:
        strategies_dir = Path(__file__).parent.parent / "strategies"

    if not strategies_dir.exists():
        print(f"  Strategies directory not found: {strategies_dir}")
        return []

    records = []

    for strategy_dir in sorted(strategies_dir.iterdir()):
        if not strategy_dir.is_dir():
            continue
        if strategy_dir.name.startswith("_"):
            continue  # Skip _template etc.

        config_path = strategy_dir / "config.json"
        if not config_path.exists():
            continue

        try:
            record = _parse_strategy_dir(strategy_dir)
            if record:
                records.append(record)
                print(f"  Scanned: {strategy_dir.name} -> {record['asset_class']}")
        except Exception as e:
            print(f"  Warning: Could not parse {strategy_dir.name}: {e}")

    return records


def _parse_strategy_dir(strategy_dir: Path) -> Optional[Dict]:
    """Parse a single strategy directory into a standardized build record."""
    config_path = strategy_dir / "config.json"
    config = json.loads(config_path.read_text())

    # Detect format: new format has 'strategy' top-level key
    is_new_format = "strategy" in config

    # --- Parse config ---
    if is_new_format:
        name = config["strategy"].get("name", strategy_dir.name)
        model_metrics = config["strategy"].get("model_metrics", {})
        llm_assisted = config["strategy"].get("llm_assisted", False)
        generated = config["strategy"].get("generated", "")
        symbols = config.get("trading", {}).get("symbols", [])
        timeframe = config.get("trading", {}).get("timeframe", "1d")
        risk = config.get("risk_management", {})
        entry_threshold = config.get("trading", {}).get("min_entry_score", 3)

        # Risk values are already decimals in new format
        stop_loss = risk.get("stop_loss_pct", 0.05)
        trailing_stop = risk.get("trailing_stop_pct", 0.03)
        max_positions = config.get("trading", {}).get("max_positions", 3)
        position_size = risk.get("max_position_size", 0.10)

        # Top features from config
        top_features_config = config.get("extracted_rules", {}).get("top_features", [])

        # LLM design metadata
        llm_design = config.get("llm_design", {})
        priority_features = llm_design.get("priority_features", [])
        description = llm_design.get("description", "")
        rationale = llm_design.get("rationale", "")
    else:
        # Old format (titan_gld, ugl_hybrid)
        name = config.get("strategy_name", strategy_dir.name)
        model_metrics = {}
        llm_assisted = False
        generated = ""
        symbols = config.get("trading", {}).get("symbols", [])
        if not symbols:
            ticker = config.get("ticker", "")
            symbols = [ticker] if ticker else []
        timeframe = config.get("trading", {}).get("timeframe", "1d")
        risk = config.get("risk_management", {})
        entry_threshold = 3

        # Old format has risk values as percentages (50.0 = 50%)
        # Need to normalize to decimals
        raw_sl = risk.get("stop_loss_pct", 5.0)
        raw_ts = risk.get("trailing_stop_pct", 3.0)
        stop_loss = raw_sl / 100.0 if raw_sl > 1 else raw_sl
        trailing_stop = raw_ts / 100.0 if raw_ts > 1 else raw_ts
        max_positions = config.get("trading", {}).get("pyramiding", 3)
        position_size = 0.10

        top_features_config = []
        priority_features = []
        description = config.get("description", "")
        rationale = ""

    # --- Parse build date ---
    build_date = ""
    if generated:
        try:
            build_date = datetime.fromisoformat(generated).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass
    if not build_date:
        # Fall back to file modification time
        try:
            mtime = config_path.stat().st_mtime
            build_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
        except OSError:
            build_date = ""

    # --- Parse extracted_rules.json ---
    top_entry_rules = []
    top_exit_rules = []
    top_ml_features = top_features_config
    rules_path = strategy_dir / "extracted_rules.json"
    if rules_path.exists():
        try:
            rules = json.loads(rules_path.read_text())
            # Override top features from rules file (has importance data)
            if rules.get("top_features"):
                top_ml_features = [
                    f["feature"] if isinstance(f, dict) else f
                    for f in rules["top_features"][:10]
                ]
            for rule in rules.get("entry_rules", [])[:5]:
                top_entry_rules.append({
                    "feature": rule["feature"],
                    "operator": rule["operator"],
                    "threshold": round(rule["threshold"], 4),
                })
            for rule in rules.get("exit_rules", [])[:5]:
                top_exit_rules.append({
                    "feature": rule["feature"],
                    "operator": rule["operator"],
                    "threshold": round(rule["threshold"], 4),
                })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"    Warning: Could not parse rules for {name}: {e}")

    # --- Parse backtest_results.json ---
    backtest = {
        "total_return": None,
        "win_rate": None,
        "max_drawdown": None,
        "sharpe_ratio": None,
        "total_trades": None,
        "profit_factor": None,
    }
    bt_path = strategy_dir / "backtest_results.json"
    if bt_path.exists():
        try:
            bt_data = json.loads(bt_path.read_text())
            backtest = {
                "total_return": bt_data.get("total_return_pct", bt_data.get("total_return")),
                "win_rate": bt_data.get("win_rate_pct", bt_data.get("win_rate")),
                "max_drawdown": bt_data.get("max_drawdown_pct", bt_data.get("max_drawdown")),
                "sharpe_ratio": bt_data.get("sharpe_ratio"),
                "total_trades": bt_data.get("total_trades"),
                "profit_factor": bt_data.get("profit_factor"),
            }
        except (json.JSONDecodeError, KeyError):
            pass

    # --- Build standardized record ---
    asset_class = infer_asset_class(symbols)

    return {
        "strategy_name": name,
        "strategy_dir": str(strategy_dir.relative_to(strategy_dir.parent.parent)),
        "symbols": symbols,
        "asset_class": asset_class,
        "description": description[:500] if description else "",
        "timeframe": timeframe,
        "target": None,  # Not stored in config; unknown for scanned strategies
        "build_date": build_date,
        "llm_assisted": llm_assisted,
        "model": {
            "name": model_metrics.get("model_name"),
            "f1": model_metrics.get("f1"),
            "precision": model_metrics.get("precision"),
            "recall": model_metrics.get("recall"),
            "auc": model_metrics.get("auc"),
            "training_years": model_metrics.get("training_years"),
            "hp_iterations": model_metrics.get("hp_iterations"),
        },
        "backtest": backtest,
        "risk_params": {
            "stop_loss": round(stop_loss, 4),
            "trailing_stop": round(trailing_stop, 4),
            "max_positions": max_positions,
            "position_size": round(position_size, 4),
            "entry_score_threshold": entry_threshold,
        },
        "features": {
            "priority": priority_features,
            "top_ml_features": top_ml_features,
            "entry_rules_count": config.get("extracted_rules", {}).get("entry_rules_count", len(top_entry_rules)),
            "exit_rules_count": config.get("extracted_rules", {}).get("exit_rules_count", len(top_exit_rules)),
            "top_entry_rules": top_entry_rules,
            "top_exit_rules": top_exit_rules,
        },
        "iterations": {"count": 0, "results": []},
        "success": None,  # Unknown for scanned strategies
    }
