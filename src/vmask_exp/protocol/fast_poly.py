from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


try:
    import flint
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "python-flint is required for the full Threshold SNIP benchmark"
    ) from exc


def _as_ints(values: Iterable[object]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def batch_inverse(values: Sequence[int], modulus: int) -> tuple[int, ...]:
    if not values:
        return ()
    q = int(modulus)
    prefixes = [1] * len(values)
    product = 1
    for index, value in enumerate(values):
        normalized = int(value) % q
        if normalized == 0:
            raise ZeroDivisionError("batch inversion input contains zero")
        prefixes[index] = product
        product = product * normalized % q
    inverse_product = pow(product, -1, q)
    inverses = [0] * len(values)
    for index in range(len(values) - 1, -1, -1):
        normalized = int(values[index]) % q
        inverses[index] = inverse_product * prefixes[index] % q
        inverse_product = inverse_product * normalized % q
    return tuple(inverses)


@dataclass
class ConsecutiveInterpolator:
    modulus: int
    point_count: int

    def __post_init__(self) -> None:
        self.modulus = int(self.modulus)
        self.point_count = int(self.point_count)
        if self.point_count <= 0:
            raise ValueError("point_count must be positive")
        if self.point_count >= self.modulus:
            raise ValueError("interpolation points must be distinct in the field")
        self.context = flint.fmpz_mod_poly_ctx(self.modulus)
        self.points = tuple(range(self.point_count))
        leaves = [self.context([(-point) % self.modulus, 1]) for point in self.points]
        self.product_levels = [leaves]
        current = leaves
        while len(current) > 1:
            next_level = []
            for index in range(0, len(current), 2):
                if index + 1 < len(current):
                    next_level.append(current[index] * current[index + 1])
                else:
                    next_level.append(current[index])
            self.product_levels.append(next_level)
            current = next_level
        derivative_values = _as_ints(
            self.product_levels[-1][0].derivative().multipoint_evaluate(self.points)
        )
        self.inverse_derivative_values = batch_inverse(
            derivative_values,
            self.modulus,
        )

    def interpolate_poly(self, values: Sequence[int]):
        if len(values) != self.point_count:
            raise ValueError("interpolation value count mismatch")
        current = [
            self.context([int(value) * inverse % self.modulus])
            for value, inverse in zip(values, self.inverse_derivative_values)
        ]
        for level in range(len(self.product_levels) - 1):
            products = self.product_levels[level]
            next_level = []
            for index in range(0, len(current), 2):
                if index + 1 < len(current):
                    combined = (
                        current[index] * products[index + 1]
                        + current[index + 1] * products[index]
                    )
                else:
                    combined = current[index]
                next_level.append(combined)
            current = next_level
        return current[0]


class FlintPolynomialBackend:
    def __init__(self, modulus: int):
        self.modulus = int(modulus)
        self.context = flint.fmpz_mod_poly_ctx(self.modulus)

    def coefficients(self, polynomial, expected_length: int | None = None) -> tuple[int, ...]:
        coefficients = list(_as_ints(polynomial.coeffs()))
        if expected_length is not None:
            if len(coefficients) > expected_length:
                raise ValueError("polynomial exceeds expected coefficient count")
            coefficients.extend([0] * (expected_length - len(coefficients)))
        return tuple(coefficients)

    def polynomial(self, coefficients: Sequence[int]):
        return self.context([int(value) % self.modulus for value in coefficients])

    def multiply(
        self,
        left,
        right,
        expected_length: int | None = None,
    ) -> tuple[int, ...]:
        return self.coefficients(left * right, expected_length)

    def evaluate(self, coefficients: Sequence[int], point: int) -> int:
        return int(self.polynomial(coefficients).evaluate(int(point)))

    def multipoint_evaluate(
        self,
        coefficients: Sequence[int],
        points: Sequence[int],
    ) -> tuple[int, ...]:
        return _as_ints(self.polynomial(coefficients).multipoint_evaluate(points))


def consecutive_lagrange_weights(
    multiplication_count: int,
    target: int,
    modulus: int,
) -> tuple[int, ...]:
    m = int(multiplication_count)
    q = int(modulus)
    r = int(target) % q
    if 0 <= r <= m:
        return tuple(1 if index == r else 0 for index in range(m + 1))

    differences = tuple((r - index) % q for index in range(m + 1))
    inverse_differences = batch_inverse(differences, q)
    product = 1
    for value in differences:
        product = product * value % q

    factorial = 1
    for value in range(2, m + 1):
        factorial = factorial * value % q
    inverse_factorials = [1] * (m + 1)
    inverse_factorials[m] = pow(factorial, -1, q)
    for value in range(m, 0, -1):
        inverse_factorials[value - 1] = inverse_factorials[value] * value % q

    weights = [0] * (m + 1)
    for index in range(m + 1):
        barycentric = (
            inverse_factorials[index]
            * inverse_factorials[m - index]
            % q
        )
        if (m - index) & 1:
            barycentric = (-barycentric) % q
        weights[index] = product * barycentric % q * inverse_differences[index] % q
    return tuple(weights)
