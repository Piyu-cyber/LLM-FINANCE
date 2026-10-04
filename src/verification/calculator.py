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


def calculate(operation, operands):
    """
    Deterministically execute a financial calculation.

    Supported operations:

        add
        subtract
        multiply
        divide
        percentage_change

    Parameters
    ----------
    operation : str
        Name of the operation.

    operands : list
        Numeric operands.

    Returns
    -------
    dict
        Calculation result.
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

        result = (
            (new_value - old_value)
            / old_value
        ) * Decimal("100")

    else:

        raise CalculationError(
            f"Unsupported operation: {operation}"
        )

    return {
        "operation": operation,
        "operands": [
            float(value)
            for value in values
        ],
        "result": float(result)
    }