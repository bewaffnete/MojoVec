from std.collections import List

from mojovec.api.metadata import MetadataValue


comptime WHERE_EQ = 0
comptime WHERE_NE = 1
comptime WHERE_GT = 2
comptime WHERE_GTE = 3
comptime WHERE_LT = 4
comptime WHERE_LTE = 5
comptime WHERE_ALL = 6
comptime WHERE_ANY = 7
comptime WHERE_NOT = 8


struct WhereNode(Movable):
    """One predicate or logical instruction in a postfix Where expression."""

    var _operation: Int
    var _key: String
    var _value: MetadataValue
    var _arity: Int

    def __init__(
        out self,
        operation: Int,
        key: String,
        value: MetadataValue,
        arity: Int = 0,
    ):
        self._operation = operation
        self._key = key.copy()
        self._value = value.copy()
        self._arity = arity

    def __init__(out self, *, deinit take: Self):
        self._operation = take._operation
        self._key = take._key^
        self._value = take._value^
        self._arity = take._arity

    def copy(self) -> Self:
        return Self(
            self._operation,
            self._key,
            self._value,
            self._arity,
        )

    def key(self) -> String:
        return self._key.copy()

    def arity(self) -> Int:
        return self._arity

    def is_predicate(self) -> Bool:
        return self._operation <= WHERE_LTE

    def is_all(self) -> Bool:
        return self._operation == WHERE_ALL

    def is_any(self) -> Bool:
        return self._operation == WHERE_ANY

    def is_not(self) -> Bool:
        return self._operation == WHERE_NOT

    def _numeric_value(self, value: MetadataValue) raises -> Float64:
        if value.is_int():
            return Float64(value.as_int())
        if value.is_float():
            return value.as_float()
        raise Error("Ordered metadata comparisons require Int or Float64.")

    def matches(self, candidate: MetadataValue) raises -> Bool:
        """Evaluates this typed predicate against an existing field value."""
        if self._operation == WHERE_EQ:
            return candidate.equals(self._value)
        if self._operation == WHERE_NE:
            return not candidate.equals(self._value)

        if (
            (not self._value.is_int() and not self._value.is_float())
            or (not candidate.is_int() and not candidate.is_float())
        ):
            return False

        var left = self._numeric_value(candidate)
        var right = self._numeric_value(self._value)
        if self._operation == WHERE_GT:
            return left > right
        if self._operation == WHERE_GTE:
            return left >= right
        if self._operation == WHERE_LT:
            return left < right
        return left <= right


struct Where(Movable):
    """
    A typed metadata filter.

    Predicates are overloaded by scalar type. Ordered predicates accept only
    Int and Float64. Logical constructors combine complete expressions.
    """

    var _nodes: List[WhereNode]

    def __init__(out self):
        self._nodes = List[WhereNode]()

    def __init__(out self, *, deinit take: Self):
        self._nodes = take._nodes^

    def copy(self) -> Self:
        var result = Self()
        result._append(self)
        return result^

    def _append(mut self, other: Self):
        for index in range(len(other._nodes)):
            self._nodes.append(other._nodes[index].copy())

    @staticmethod
    def _predicate(
        operation: Int,
        key: String,
        value: MetadataValue,
    ) -> Self:
        var result = Self()
        result._nodes.append(WhereNode(operation, key, value))
        return result^

    @staticmethod
    def eq(key: String, value: String) -> Self:
        return Self._predicate(WHERE_EQ, key, MetadataValue(value))

    @staticmethod
    def eq(key: String, value: Int) -> Self:
        return Self._predicate(WHERE_EQ, key, MetadataValue(value))

    @staticmethod
    def eq(key: String, value: Float64) -> Self:
        return Self._predicate(WHERE_EQ, key, MetadataValue(value))

    @staticmethod
    def eq(key: String, value: Bool) -> Self:
        return Self._predicate(WHERE_EQ, key, MetadataValue(value))

    @staticmethod
    def ne(key: String, value: String) -> Self:
        return Self._predicate(WHERE_NE, key, MetadataValue(value))

    @staticmethod
    def ne(key: String, value: Int) -> Self:
        return Self._predicate(WHERE_NE, key, MetadataValue(value))

    @staticmethod
    def ne(key: String, value: Float64) -> Self:
        return Self._predicate(WHERE_NE, key, MetadataValue(value))

    @staticmethod
    def ne(key: String, value: Bool) -> Self:
        return Self._predicate(WHERE_NE, key, MetadataValue(value))

    @staticmethod
    def gt(key: String, value: Int) -> Self:
        return Self._predicate(WHERE_GT, key, MetadataValue(value))

    @staticmethod
    def gt(key: String, value: Float64) -> Self:
        return Self._predicate(WHERE_GT, key, MetadataValue(value))

    @staticmethod
    def gte(key: String, value: Int) -> Self:
        return Self._predicate(WHERE_GTE, key, MetadataValue(value))

    @staticmethod
    def gte(key: String, value: Float64) -> Self:
        return Self._predicate(WHERE_GTE, key, MetadataValue(value))

    @staticmethod
    def lt(key: String, value: Int) -> Self:
        return Self._predicate(WHERE_LT, key, MetadataValue(value))

    @staticmethod
    def lt(key: String, value: Float64) -> Self:
        return Self._predicate(WHERE_LT, key, MetadataValue(value))

    @staticmethod
    def lte(key: String, value: Int) -> Self:
        return Self._predicate(WHERE_LTE, key, MetadataValue(value))

    @staticmethod
    def lte(key: String, value: Float64) -> Self:
        return Self._predicate(WHERE_LTE, key, MetadataValue(value))

    @staticmethod
    def in_(key: String, values: List[String]) raises -> Self:
        if len(values) == 0:
            raise Error("Where.in_ requires at least one value.")
        var conditions = List[Self](capacity=len(values))
        for value in values:
            conditions.append(Self.eq(key, value))
        return Self.any(conditions)

    @staticmethod
    def in_(key: String, values: List[Int]) raises -> Self:
        if len(values) == 0:
            raise Error("Where.in_ requires at least one value.")
        var conditions = List[Self](capacity=len(values))
        for value in values:
            conditions.append(Self.eq(key, value))
        return Self.any(conditions)

    @staticmethod
    def in_(key: String, values: List[Float64]) raises -> Self:
        if len(values) == 0:
            raise Error("Where.in_ requires at least one value.")
        var conditions = List[Self](capacity=len(values))
        for value in values:
            conditions.append(Self.eq(key, value))
        return Self.any(conditions)

    @staticmethod
    def in_(key: String, values: List[Bool]) raises -> Self:
        if len(values) == 0:
            raise Error("Where.in_ requires at least one value.")
        var conditions = List[Self](capacity=len(values))
        for value in values:
            conditions.append(Self.eq(key, value))
        return Self.any(conditions)

    @staticmethod
    def not_in(key: String, values: List[String]) raises -> Self:
        if len(values) == 0:
            raise Error("Where.not_in requires at least one value.")
        var conditions = List[Self](capacity=len(values))
        for value in values:
            conditions.append(Self.ne(key, value))
        return Self.all(conditions)

    @staticmethod
    def not_in(key: String, values: List[Int]) raises -> Self:
        if len(values) == 0:
            raise Error("Where.not_in requires at least one value.")
        var conditions = List[Self](capacity=len(values))
        for value in values:
            conditions.append(Self.ne(key, value))
        return Self.all(conditions)

    @staticmethod
    def not_in(key: String, values: List[Float64]) raises -> Self:
        if len(values) == 0:
            raise Error("Where.not_in requires at least one value.")
        var conditions = List[Self](capacity=len(values))
        for value in values:
            conditions.append(Self.ne(key, value))
        return Self.all(conditions)

    @staticmethod
    def not_in(key: String, values: List[Bool]) raises -> Self:
        if len(values) == 0:
            raise Error("Where.not_in requires at least one value.")
        var conditions = List[Self](capacity=len(values))
        for value in values:
            conditions.append(Self.ne(key, value))
        return Self.all(conditions)

    @staticmethod
    def all(conditions: List[Self]) raises -> Self:
        if len(conditions) == 0:
            raise Error("Where.all requires at least one condition.")
        var result = Self()
        for index in range(len(conditions)):
            result._append(conditions[index])
        result._nodes.append(
            WhereNode(
                WHERE_ALL,
                "",
                MetadataValue(False),
                len(conditions),
            )
        )
        return result^

    @staticmethod
    def any(conditions: List[Self]) raises -> Self:
        if len(conditions) == 0:
            raise Error("Where.any requires at least one condition.")
        var result = Self()
        for index in range(len(conditions)):
            result._append(conditions[index])
        result._nodes.append(
            WhereNode(
                WHERE_ANY,
                "",
                MetadataValue(False),
                len(conditions),
            )
        )
        return result^

    @staticmethod
    def and_(conditions: List[Self]) raises -> Self:
        return Self.all(conditions)

    @staticmethod
    def or_(conditions: List[Self]) raises -> Self:
        return Self.any(conditions)

    @staticmethod
    def not_(condition: Self) -> Self:
        var result = condition.copy()
        result._nodes.append(
            WhereNode(WHERE_NOT, "", MetadataValue(False), 1)
        )
        return result^

    def nodes(self) -> List[WhereNode]:
        """Returns an owned instruction list for query evaluation."""
        var result = List[WhereNode](capacity=len(self._nodes))
        for index in range(len(self._nodes)):
            result.append(self._nodes[index].copy())
        return result^
