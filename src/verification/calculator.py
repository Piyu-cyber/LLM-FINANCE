from decimal import Decimal, InvalidOperation


class CalculationError(Exception):
    """Raised when a calculation cannot be safely performed."""
    pass


def to_decimal(value):
    """
    Convert a numeric value to Decimal.
    """
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise CalculationError(
            f"Invalid numeric value: {value}"
        )


def is_expense_like(item_type: str) -> bool:
    """Check if an item name or type represents an expense, cost, loss, or outflow."""
    if not item_type:
        return False
    item_type = item_type.lower().strip()
    keywords = ("expense", "cost", "outflow", "loss", "tax", "charge", "expenditure", "payment")
    return any(kw in item_type for kw in keywords)


def calculate(operation, operands, **kwargs):
    """
    Deterministically execute a financial calculation.

    Supported operations:

        add
        subtract
        multiply
        divide
        percentage_change
        magnitude_change (abs(curr) - abs(prev))
        economic_change (accounting polarity aware)
        abs

    Parameters
    ----------
    operation : str
        Name of the operation.

    operands : list
        Numeric operands.

    Returns
    -------
    dict
        Calculation result and directional metadata.
    """

    if not isinstance(operands, list):
        raise CalculationError(
            "Operands must be a list."
        )

    if len(operands) == 0:
        raise CalculationError(
            "At least one operand is required."
        )

    values = [
        to_decimal(value)
        for value in operands
    ]

    operation = operation.lower().strip()

    # --------------------------------------------------
    # Addition
    # --------------------------------------------------

    if operation == "add":

        if len(values) < 2:
            raise CalculationError(
                "Addition requires at least two operands."
            )

        result = sum(
            values,
            Decimal("0")
        )

    # --------------------------------------------------
    # Subtraction
    # --------------------------------------------------

    elif operation == "subtract":

        if len(values) < 2:
            raise CalculationError(
                "Subtraction requires at least two operands."
            )

        result = values[0]

        for value in values[1:]:
            result -= value

    # --------------------------------------------------
    # Magnitude change (|curr| - |prev|)
    # --------------------------------------------------

    elif operation in ("magnitude_change", "abs_change"):

        if len(values) != 2:
            raise CalculationError(
                "Magnitude change requires exactly two operands (current, previous)."
            )

        result = abs(values[0]) - abs(values[1])

    # --------------------------------------------------
    # Economic change (accounting-aware)
    # --------------------------------------------------

    elif operation == "economic_change":

        if len(values) != 2:
            raise CalculationError(
                "Economic change requires exactly two operands (current, previous)."
            )

        item_type = kwargs.get("item_type", "expense").lower()

        # For expenses, costs, and cash outflows typically stored as negative numbers
        if (is_expense_like(item_type) or (values[0] < 0 and values[1] < 0)) and values[0] <= 0 and values[1] <= 0:
            result = abs(values[0]) - abs(values[1])
        else:
            result = values[0] - values[1]

    # --------------------------------------------------
    # Absolute value
    # --------------------------------------------------

    elif operation == "abs":

        if len(values) != 1:
            raise CalculationError(
                "Absolute value requires exactly one operand."
            )

        result = abs(values[0])

    # --------------------------------------------------
    # Multiplication
    # --------------------------------------------------

    elif operation == "multiply":

        result = Decimal("1")

        for value in values:
            result *= value

    # --------------------------------------------------
    # Division
    # --------------------------------------------------

    elif operation == "divide":

        if len(values) != 2:
            raise CalculationError(
                "Division requires exactly two operands."
            )

        numerator = values[0]
        denominator = values[1]

        if denominator == 0:
            raise CalculationError(
                "Division by zero is not allowed."
            )

        result = numerator / denominator

    # --------------------------------------------------
    # Percentage change
    # --------------------------------------------------

    elif operation == "percentage_change":

        if len(values) != 2:
            raise CalculationError(
                "Percentage change requires exactly "
                "two operands: old value and new value."
            )

        old_value = values[0]
        new_value = values[1]

        if old_value == 0:
            raise CalculationError(
                "Cannot calculate percentage change "
                "from a zero baseline."
            )

        # For expenses reported as negative numbers (e.g. -511 to -600),
        # standard financial convention measures growth on magnitudes (|600| - |511|) / |511| = +17.4%
        item_type = kwargs.get("item_type", "").lower()
        if is_expense_like(item_type) or (old_value < 0 and new_value < 0):
            result = (
                (abs(new_value) - abs(old_value))
                / abs(old_value)
            ) * Decimal("100")
        else:
            result = (
                (new_value - old_value)
                / abs(old_value)
            ) * Decimal("100")

    else:

        raise CalculationError(
            f"Unsupported operation: {operation}"
        )

    direction = (
        "increase" if result > 0
        else ("decrease" if result < 0 else "unchanged")
    )

    response = {
        "operation": operation,
        "operands": [
            float(value)
            for value in values
        ],
        "result": float(result),
        "direction": direction,
    }

    if len(values) == 2 and operation == "subtract":
        # Include economic magnitude difference as reference for expense/outflow items
        response["magnitude_change"] = float(abs(values[0]) - abs(values[1]))
        response["economic_direction"] = (
            "increase" if response["magnitude_change"] > 0
            else ("decrease" if response["magnitude_change"] < 0 else "unchanged")
        )

    return response


def infer_direction(value):
    """Infer financial direction from a numeric value."""
    if value is None:
        return "unknown"
    try:
        dec = to_decimal(value)
        if dec > 0:
            return "increase"
        elif dec < 0:
            return "decrease"
        return "unchanged"
    except CalculationError:
        return "unknown"


def calculate_financial_change(val_curr, val_prev, item_type="expense"):
    """
    Compute financial change accounting for line item polarity conventions.

    Returns dict with:
        - economic_change: change in magnitude (e.g. +89 for expense increase)
        - signed_change: raw signed difference (-600 - (-511) = -89)
        - economic_direction: "increase" | "decrease" | "unchanged"
        - percentage_change: float (%)
    """
    curr_dec = to_decimal(val_curr)
    prev_dec = to_decimal(val_prev)

    signed_diff = curr_dec - prev_dec

    is_negative_expense = (
        is_expense_like(item_type)
        and curr_dec <= 0 and prev_dec <= 0
    )

    if is_negative_expense:
        econ_diff = abs(curr_dec) - abs(prev_dec)
        pct = ((abs(curr_dec) - abs(prev_dec)) / abs(prev_dec)) * Decimal("100") if prev_dec != 0 else Decimal("0")
    else:
        econ_diff = signed_diff
        pct = (signed_diff / abs(prev_dec)) * Decimal("100") if prev_dec != 0 else Decimal("0")

    direction = (
        "increase" if econ_diff > 0
        else ("decrease" if econ_diff < 0 else "unchanged")
    )

    return {
        "current_value": float(curr_dec),
        "previous_value": float(prev_dec),
        "item_type": item_type,
        "economic_change": float(econ_diff),
        "signed_change": float(signed_diff),
        "economic_direction": direction,
        "percentage_change": float(pct),
    }
