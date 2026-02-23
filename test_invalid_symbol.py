#!/usr/bin/env python3
"""
Test script to verify invalid symbol handling.

This script tests that:
1. Invalid symbols are detected during validation
2. A clear error message is returned
3. The error is properly propagated through the worker
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy_designer_ai.data_provider import DataProvider
from strategy_designer_ai.ml_enhanced_trainer import EnhancedMLTrainer, EnhancedMLConfig


async def test_invalid_symbol_validation():
    """Test that invalid symbols are properly detected."""
    print("\n" + "="*60)
    print("TEST 1: Symbol Validation")
    print("="*60)
    
    provider = DataProvider()
    
    # Test with mix of valid and invalid symbols
    symbols = ["AAPL", "APPL", "MSFT"]  # APPL is invalid
    print(f"\nTesting symbols: {symbols}")
    
    results = await provider.validate_symbols(symbols, days=30, interval='1d')
    
    print("\nValidation Results:")
    for symbol, is_valid in results.items():
        status = "✓ VALID" if is_valid else "✗ INVALID"
        print(f"  {symbol}: {status}")
    
    invalid = [s for s, v in results.items() if not v]
    if invalid:
        print(f"\n❌ Found invalid symbols: {', '.join(invalid)}")
        return False
    else:
        print("\n✓ All symbols are valid")
        return True


async def test_trainer_with_invalid_symbol():
    """Test that the trainer properly handles invalid symbols."""
    print("\n" + "="*60)
    print("TEST 2: Trainer with Invalid Symbol")
    print("="*60)
    
    config = EnhancedMLConfig(
        forward_returns_days_options=[5, 10],
        profit_threshold_options=[2.0, 3.0],
        n_iter_search=5,  # Small for quick test
        cv_folds=2,
        priority_features=["RSI_14", "MACD_HIST"],
        train_neural_nets=False,
        train_lstm=False,
    )
    
    # Use invalid symbol APPL
    trainer = EnhancedMLTrainer(
        symbols=["APPL"],  # Invalid symbol
        period_years=1,
        config=config,
        timeframe="1d"
    )
    
    try:
        print("\nAttempting to fetch data for invalid symbol APPL...")
        await trainer.fetch_all_data()
        print("\n❌ ERROR: Should have raised ValueError for invalid symbol!")
        return False
    except ValueError as e:
        error_msg = str(e)
        if "Invalid symbol" in error_msg and "APPL" in error_msg:
            print(f"\n✓ Correctly caught invalid symbol error: {error_msg}")
            return True
        else:
            print(f"\n❌ Wrong error message: {error_msg}")
            return False
    except Exception as e:
        print(f"\n❌ Unexpected error type: {type(e).__name__}: {e}")
        return False


async def test_trainer_with_valid_symbol():
    """Test that the trainer works with valid symbols."""
    print("\n" + "="*60)
    print("TEST 3: Trainer with Valid Symbol")
    print("="*60)
    
    config = EnhancedMLConfig(
        forward_returns_days_options=[5],
        profit_threshold_options=[2.0],
        n_iter_search=5,
        cv_folds=2,
        priority_features=["RSI_14"],
        train_neural_nets=False,
        train_lstm=False,
    )
    
    trainer = EnhancedMLTrainer(
        symbols=["AAPL"],  # Valid symbol
        period_years=1,
        config=config,
        timeframe="1d"
    )
    
    try:
        print("\nAttempting to fetch data for valid symbol AAPL...")
        data = await trainer.fetch_all_data()
        if data and "AAPL" in data and len(data["AAPL"]) > 0:
            print(f"\n✓ Successfully fetched {len(data['AAPL'])} bars for AAPL")
            return True
        else:
            print("\n❌ No data returned for AAPL")
            return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {type(e).__name__}: {e}")
        return False


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("INVALID SYMBOL HANDLING TESTS")
    print("="*60)
    
    results = []
    
    # Test 1: Symbol validation
    results.append(await test_invalid_symbol_validation())
    
    # Test 2: Trainer with invalid symbol
    results.append(await test_trainer_with_invalid_symbol())
    
    # Test 3: Trainer with valid symbol
    results.append(await test_trainer_with_valid_symbol())
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

