from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Self

import jvm
import jvm.state as jvms


class Poset(ABC):
    @abstractmethod
    def __eq__(self, other: Self) -> bool:
        ...

    @abstractmethod
    def __lt__(self, other: Self) -> bool:
        ...

    def __le__(self, other: Self) -> bool:
        return self == other or self < other

    @abstractmethod
    def __gt__(self, other: Self) -> bool:
        ...

    def __ge__(self, other: Self) -> bool:
        return self == other or self > other


def is_reflexive(a: Poset):
    assert a == a, (
        f"Reflexivity check failed: Element {a} "
        "is not related to itself."
    )


def is_antisymmetric(a: Poset, b: Poset):
    if a != b:
        assert not (a <= b <= a), (
            f"Antisymmetry violation: Found elements where "
            f"{a} <= {b} and {b} <= {a}, but {a} != {b}."
        )


def is_transitive(a: Poset, b: Poset, c: Poset):
    if a <= b <= c:
        assert a <= c, (
            f"Transitivity violation: Found path "
            f"{a} <= {b} and {b} <= {c}, but {a} is not related to {c}."
        )


def is_poset(a: Poset, b: Poset, c: Poset):
    for x in (a, b, c):
        is_reflexive(x)

    is_antisymmetric(a, b)
    is_antisymmetric(a, c)
    is_antisymmetric(b, c)

    is_transitive(a, b, c)


class Lattice(Poset):
    @classmethod
    @abstractmethod
    def top(cls) -> Self:
        raise NotImplementedError

    @classmethod
    def bot(cls) -> Self:
        return cls.abstract([])

    @abstractmethod
    def __or__(self, other: Self) -> Self:
        ...

    @abstractmethod
    def __and__(self, other: Self) -> Self:
        ...


def is_lattice_identities(a: Lattice):
    assert (a | a) == a
    assert (a & a) == a

    assert (a | a.top()) == a.top()
    assert (a.top() | a) == a.top()

    assert (a & a.top()) == a
    assert (a.top() & a) == a

    assert (a & a.bot()) == a.bot()
    assert (a.bot() & a) == a.bot()

    assert (a.bot() | a) == a
    assert (a | a.bot()) == a


def is_lattice(a: Lattice, b: Lattice, c: Lattice):
    for x in (a, b, c):
        is_lattice_identities(x)

    if not (a | b <= c):
        assert not (a <= c and b <= c)

    if not (c <= a & b):
        assert not (c <= a and c <= b)


class Abstraction(ABC):
    @classmethod
    @abstractmethod
    def abstract(
        cls,
        values: Iterable[jvms.StackValue],
    ) -> Self:
        raise NotImplementedError

    @abstractmethod
    def __contains__(
        self,
        value: jvms.StackValue,
    ) -> bool:
        ...


def is_galoi(
    a: set[jvms.StackValue],
    b: Abstraction,
):
    alpha_a = b.abstract(a)

    for value in a:
        assert value in alpha_a

    if alpha_a <= b:
        for value in a:
            assert value in b


type Sign = Literal[-1, 0, 1]


def to_sign(value: jvms.StackValue) -> Sign:
    match value:
        case jvms.StackInt(v):
            return (v > 0) - (v < 0)

        case _:
            raise NotImplementedError(
                f"Unsupported value {value!r}"
            )


@dataclass(frozen=True)
class SignSet(Abstraction, Lattice):
    signs: frozenset[Sign]

    def __str__(self):
        result = ""

        if -1 in self.signs:
            result += "-"

        if 0 in self.signs:
            result += "0"

        if 1 in self.signs:
            result += "+"

        return "{" + result + "}"

    def __sexpr__(self):
        return str(self)

    @classmethod
    def from_sign(cls, code: str) -> Self:
        mapping = {
            "-": -1,
            "0": 0,
            "+": 1,
        }

        return cls(
            frozenset(
                mapping[ch]
                for ch in code
                if ch in mapping
            )
        )

    @classmethod
    def abstract(
        cls,
        values: Iterable[jvms.StackValue],
    ) -> Self:
        signs = set()

        for value in values:
            signs.add(to_sign(value))

        return cls(frozenset(signs))

    @classmethod
    def top(cls) -> Self:
        return cls(frozenset({-1, 0, 1}))

    @classmethod
    def bot(cls) -> Self:
        return cls(frozenset())

    def __lt__(self, other: Self) -> bool:
        if not isinstance(other, SignSet):
            return NotImplemented

        return self.signs < other.signs

    def __gt__(self, other: Self) -> bool:
        if not isinstance(other, SignSet):
            return NotImplemented

        return self.signs > other.signs

    def __or__(self, other: Self) -> Self:
        if not isinstance(other, SignSet):
            return NotImplemented

        return SignSet(self.signs | other.signs)

    def __and__(self, other: Self) -> Self:
        if not isinstance(other, SignSet):
            return NotImplemented

        return SignSet(self.signs & other.signs)

    def __contains__(
        self,
        value: jvms.StackValue,
    ) -> bool:
        return to_sign(value) in self.signs

    @staticmethod
    def _add_sign(x: Sign, y: Sign) -> set[Sign]:
        if x == -1 and y == -1:
            return {-1}

        if x == 0:
            return {y}

        if y == 0:
            return {x}

        if x == 1 and y == 1:
            return {1}

        if {x, y} == {-1, 1}:
            return {-1, 0, 1}

        raise AssertionError(
            f"Unhandled addition: {x}, {y}"
        )

    @staticmethod
    def _sub_sign(x: Sign, y: Sign) -> set[Sign]:
        return SignSet._add_sign(x, -y)

    @staticmethod
    def _mul_sign(x: Sign, y: Sign) -> set[Sign]:
        if x == 0 or y == 0:
            return {0}

        if x == y:
            return {1}

        return {-1}

    @staticmethod
    def _div_sign(
        x: Sign,
        y: Sign,
    ) -> tuple[set[Sign], set[str]]:
        if y == 0:
            return set(), {"divide by zero"}

        if x == 0:
            return {0}, set()

        if x == y:
            return {1}, set()

        return {-1}, set()

    @staticmethod
    def _rem_sign(
        x: Sign,
        y: Sign,
    ) -> tuple[set[Sign], set[str]]:
        if y == 0:
            return set(), {"divide by zero"}

        if x == 0:
            return {0}, set()

        return {x, 0}, set()

    def arithmetic(
        self,
        other: "SignSet",
        opr: jvm.BinaryOpr,
    ) -> tuple["SignSet", set[str]]:
        if not isinstance(other, SignSet):
            return NotImplemented

        result = set()
        errors = set()

        for x in self.signs:
            for y in other.signs:
                match opr:
                    case jvm.BinaryOpr.Add:
                        result.update(
                            self._add_sign(x, y)
                        )

                    case jvm.BinaryOpr.Sub:
                        result.update(
                            self._sub_sign(x, y)
                        )

                    case jvm.BinaryOpr.Mul:
                        result.update(
                            self._mul_sign(x, y)
                        )

                    case jvm.BinaryOpr.Div:
                        signs, errs = self._div_sign(x, y)
                        result.update(signs)
                        errors.update(errs)

                    case jvm.BinaryOpr.Rem:
                        signs, errs = self._rem_sign(x, y)
                        result.update(signs)
                        errors.update(errs)

                    case _:
                        raise NotImplementedError(
                            f"TODO: {opr}"
                        )

        return SignSet(frozenset(result)), errors


    @staticmethod
    def _compare_signs(
        x: Sign,
        y: Sign,
        opr: jvm.CmpOpr,
    ) -> set[bool]:
        match opr:
            case jvm.CmpOpr.Eq:
                if x != y:
                    return {False}

                # Same sign does not imply same value.
                if x == 0:
                    return {True}

                return {False, True}

            case jvm.CmpOpr.Ne:
                if x != y:
                    return {True}

                if x == 0:
                    return {False}

                return {False, True}

            case jvm.CmpOpr.Lt:
                if x < y:
                    return {True}

                if x > y:
                    return {False}

                # Same sign: ordering is unknown except for zero.
                if x == 0:
                    return {False}

                return {False, True}

            case jvm.CmpOpr.Le:
                if x < y:
                    return {True}

                if x > y:
                    return {False}

                # Same sign.
                if x == 0:
                    return {True}

                return {False, True}

            case jvm.CmpOpr.Gt:
                if x > y:
                    return {True}

                if x < y:
                    return {False}

                # Same sign: ordering is unknown except for zero.
                if x == 0:
                    return {False}

                return {False, True}

            case jvm.CmpOpr.Ge:
                if x > y:
                    return {True}

                if x < y:
                    return {False}

                # Same sign.
                if x == 0:
                    return {True}

                return {False, True}

            case _:
                raise NotImplementedError(
                    f"TODO: {opr}"
                )

    def compare(
        self,
        other: "SignSet",
        opr: jvm.CmpOpr,
    ) -> Iterable[bool]:
        if not isinstance(other, SignSet):
            return NotImplemented

        results: set[bool] = set()

        for x in self.signs:
            for y in other.signs:
                results.update(
                    self._compare_signs(
                        x,
                        y,
                        opr,
                    )
                )

        return results


@dataclass(frozen=True)
class Interval(Abstraction, Lattice):
    min: int | None
    max: int | None

    def __str__(self):
        minimum = self.min if self.min is not None else "-inf"
        maximum = self.max if self.max is not None else "inf"

        if self.is_bot():
            return "bot"

        return f"{minimum}:{maximum}"

    def __sexpr__(self):
        return str(self)

    def is_bot(self) -> bool:
        # An interval with min > max is the empty interval.
        return (
            self.min is not None
            and self.max is not None
            and self.min > self.max
        )

    @classmethod
    def abstract(
        cls,
        values: Iterable[jvms.StackValue],
    ) -> Self:
        values = list(values)

        if not values:
            return cls.bot()

        ints = [
            value.value
            for value in values
            if isinstance(value, jvms.StackInt)
        ]

        if not ints:
            return cls.bot()

        return cls(
            min(ints),
            max(ints),
        )

    def __contains__(
        self,
        value: jvms.StackValue,
    ) -> bool:
        if not isinstance(value, jvms.StackInt):
            return False

        if self.is_bot():
            return False

        if self.min is not None and value.value < self.min:
            return False

        if self.max is not None and value.value > self.max:
            return False

        return True

    @classmethod
    def top(cls) -> Self:
        return cls(None, None)

    @classmethod
    def bot(cls) -> Self:
        return cls(1, 0)

    def __or__(self, other: Self) -> Self:
        if not isinstance(other, Interval):
            return NotImplemented

        if self.is_bot():
            return other

        if other.is_bot():
            return self

        if self.min is None or other.min is None:
            minimum = None
        else:
            minimum = min(self.min, other.min)

        if self.max is None or other.max is None:
            maximum = None
        else:
            maximum = max(self.max, other.max)

        return Interval(minimum, maximum)

    def __and__(self, other: Self) -> Self:
        if not isinstance(other, Interval):
            return NotImplemented

        if self.is_bot() or other.is_bot():
            return Interval.bot()

        if self.min is None:
            minimum = other.min
        elif other.min is None:
            minimum = self.min
        else:
            minimum = max(self.min, other.min)

        if self.max is None:
            maximum = other.max
        elif other.max is None:
            maximum = self.max
        else:
            maximum = min(self.max, other.max)

        result = Interval(minimum, maximum)

        if result.min is not None and result.max is not None:
            if result.min > result.max:
                return Interval.bot()

        return result

    def __lt__(self, other: Self) -> bool:
        if not isinstance(other, Interval):
            return NotImplemented

        # self < other in the lattice means:
        # self is a strict subset of other.
        if self == other:
            return False

        if self.is_bot():
            return not other.is_bot()

        if other.is_bot():
            return False

        lower_ok = (
            other.min is None
            or (
                self.min is not None
                and other.min <= self.min
            )
        )

        upper_ok = (
            other.max is None
            or (
                self.max is not None
                and self.max <= other.max
            )
        )

        return lower_ok and upper_ok

    def __gt__(self, other: Self) -> bool:
        if not isinstance(other, Interval):
            return NotImplemented

        return other < self