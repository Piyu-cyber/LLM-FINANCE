from calculator import (
    calculate,
    CalculationError
)


def test_addition():

    result = calculate(
        "add",
        [100, 50]
    )

    print("Addition:")
    print(result)


def test_subtraction():

    result = calculate(
        "subtract",
        [600, 500]
    )

    print("\nSubtraction:")
    print(result)


def test_multiplication():

    result = calculate(
        "multiply",
        [3.8, 2]
    )

    print("\nMultiplication:")
    print(result)


def test_division():

    result = calculate(
        "divide",
        [600, 500]
    )

    print("\nDivision:")
    print(result)


def test_percentage_change():

    result = calculate(
        "percentage_change",
        [500, 600]
    )

    print("\nPercentage change:")
    print(result)


def test_division_by_zero():

    try:

        calculate(
            "divide",
            [100, 0]
        )

    except CalculationError as e:

        print("\nDivision by zero correctly rejected:")
        print(e)


if __name__ == "__main__":

    test_addition()
    test_subtraction()
    test_multiplication()
    test_division()
    test_percentage_change()
    test_division_by_zero()