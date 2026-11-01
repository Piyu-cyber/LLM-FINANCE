import pytest
from calculator import (
    calculate,
    CalculationError,
    infer_direction,
    calculate_financial_change,
)


def test_addition():
    result = calculate("add", [100, 50])
    assert result["result"] == 150.0
    assert result["direction"] == "increase"


def test_subtraction():
    result = calculate("subtract", [600, 500])
    assert result["result"] == 100.0
    assert result["direction"] == "increase"


def test_subtraction_negative_operands():
    # Demonstrating the naive signed subtraction on negative expenses:
    # 2009: -600, 2008: -511 -> -600 - (-511) = -89
    result = calculate("subtract", [-600, -511])
    assert result["result"] == -89.0
    assert result["direction"] == "decrease"  # Naive signed direction
    # But magnitude difference is +89 (the expense grew by 89)
    assert result["magnitude_change"] == 89.0
    assert result["economic_direction"] == "increase"


def test_magnitude_change():
    # Comparing expense magnitude change: |-600| - |-511| = +89
    result = calculate("magnitude_change", [-600, -511])
    assert result["result"] == 89.0
    assert result["direction"] == "increase"


def test_economic_change_expense():
    # Expense line items stored as negative
    result = calculate("economic_change", [-600, -511], item_type="expense")
    assert result["result"] == 89.0
    assert result["direction"] == "increase"

    # Revenue line items stored as positive
    result_rev = calculate("economic_change", [600, 500], item_type="revenue")
    assert result_rev["result"] == 100.0
    assert result_rev["direction"] == "increase"


def test_multiplication():
    result = calculate("multiply", [3.8, 2])
    assert result["result"] == 7.6


def test_division():
    result = calculate("divide", [600, 500])
    assert result["result"] == 1.2


def test_percentage_change_positive():
    result = calculate("percentage_change", [500, 600])
    assert result["result"] == 20.0


def test_percentage_change_negative_expense():
    # For interest expense in FinQA: 2008 = -511, 2009 = -600
    # The table shows: % change 2009 v 2008: 17%
    result = calculate("percentage_change", [-511, -600], item_type="expense")
    assert round(result["result"], 1) == 17.4
    assert result["direction"] == "increase"


def test_division_by_zero():
    with pytest.raises(CalculationError, match="Division by zero is not allowed"):
        calculate("divide", [100, 0])


def test_calculate_financial_change():
    res = calculate_financial_change(-600, -511, item_type="interest expense")
    assert res["economic_change"] == 89.0
    assert res["signed_change"] == -89.0
    assert res["economic_direction"] == "increase"
    assert round(res["percentage_change"], 1) == 17.4


def test_infer_direction():
    assert infer_direction(89) == "increase"
    assert infer_direction(-89) == "decrease"
    assert infer_direction(0) == "unchanged"


if __name__ == "__main__":
    pytest.main([__file__])
