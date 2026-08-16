from std.collections import List


comptime METADATA_STRING = 0
comptime METADATA_INT = 1
comptime METADATA_FLOAT = 2
comptime METADATA_BOOL = 3


struct MetadataValue(Movable, Copyable, Writable):
    """A scalar metadata value supported by the Collection API."""

    var _kind: Int
    var _string_value: String
    var _int_value: Int
    var _float_value: Float64
    var _bool_value: Bool

    def __init__(out self, value: String):
        self._kind = METADATA_STRING
        self._string_value = value.copy()
        self._int_value = 0
        self._float_value = 0.0
        self._bool_value = False

    def __init__(out self, value: Int):
        self._kind = METADATA_INT
        self._string_value = String("")
        self._int_value = value
        self._float_value = 0.0
        self._bool_value = False

    def __init__(out self, value: Float64):
        self._kind = METADATA_FLOAT
        self._string_value = String("")
        self._int_value = 0
        self._float_value = value
        self._bool_value = False

    def __init__(out self, value: Bool):
        self._kind = METADATA_BOOL
        self._string_value = String("")
        self._int_value = 0
        self._float_value = 0.0
        self._bool_value = value

    def __init__(out self, *, deinit move: Self):
        self._kind = move._kind
        self._string_value = move._string_value^
        self._int_value = move._int_value
        self._float_value = move._float_value
        self._bool_value = move._bool_value

    def kind(self) -> Int:
        return self._kind

    def is_string(self) -> Bool:
        return self._kind == METADATA_STRING

    def is_int(self) -> Bool:
        return self._kind == METADATA_INT

    def is_float(self) -> Bool:
        return self._kind == METADATA_FLOAT

    def is_bool(self) -> Bool:
        return self._kind == METADATA_BOOL

    def equals(self, other: Self) -> Bool:
        """Compares both the scalar type and value."""
        if self._kind != other._kind:
            return False
        if self._kind == METADATA_STRING:
            return self._string_value == other._string_value
        if self._kind == METADATA_INT:
            return self._int_value == other._int_value
        if self._kind == METADATA_FLOAT:
            return self._float_value == other._float_value
        return self._bool_value == other._bool_value

    def as_string(self) raises -> String:
        if not self.is_string():
            raise Error("Metadata value is not a String.")
        return self._string_value.copy()

    def as_int(self) raises -> Int:
        if not self.is_int():
            raise Error("Metadata value is not an Int.")
        return self._int_value

    def as_float(self) raises -> Float64:
        if not self.is_float():
            raise Error("Metadata value is not a Float64.")
        return self._float_value

    def as_bool(self) raises -> Bool:
        if not self.is_bool():
            raise Error("Metadata value is not a Bool.")
        return self._bool_value

    def write_to[W: Writer](self, mut writer: W):
        if self.is_string():
            writer.write(self._string_value)
        elif self.is_int():
            writer.write(self._int_value)
        elif self.is_float():
            writer.write(self._float_value)
        else:
            writer.write(self._bool_value)


struct Metadata(Movable, Copyable, Writable):
    """
    A deterministic string-to-scalar metadata record.

    Metadata objects are expected to contain a small number of fields, so a
    compact ordered list keeps serialization deterministic and avoids exposing
    hash-table implementation details. Setting an existing key replaces it.
    """

    var _keys: List[String]
    var _values: List[MetadataValue]

    def __init__(out self):
        self._keys = List[String]()
        self._values = List[MetadataValue]()

    def __init__(out self, *, deinit move: Self):
        self._keys = move._keys^
        self._values = move._values^

    def copy(self) -> Metadata:
        var result = Metadata()
        for index in range(len(self._keys)):
            result._keys.append(self._keys[index].copy())
            result._values.append(self._values[index].copy())
        return result^

    def count(self) -> Int:
        return len(self._keys)

    def keys(self) -> List[String]:
        """Returns an owned list of field names in insertion order."""
        var result = List[String](capacity=len(self._keys))
        for key in self._keys:
            result.append(key.copy())
        return result^

    def _find(self, key: String) -> Int:
        for index in range(len(self._keys)):
            if self._keys[index] == key:
                return index
        return -1

    def contains(self, key: String) -> Bool:
        return self._find(key) >= 0

    def set(mut self, key: String, value: String):
        var index = self._find(key)
        if index >= 0:
            self._values[index] = MetadataValue(value)
        else:
            self._keys.append(key.copy())
            self._values.append(MetadataValue(value))

    def set(mut self, key: String, value: Int):
        var index = self._find(key)
        if index >= 0:
            self._values[index] = MetadataValue(value)
        else:
            self._keys.append(key.copy())
            self._values.append(MetadataValue(value))

    def set(mut self, key: String, value: Float64):
        var index = self._find(key)
        if index >= 0:
            self._values[index] = MetadataValue(value)
        else:
            self._keys.append(key.copy())
            self._values.append(MetadataValue(value))

    def set(mut self, key: String, value: Bool):
        var index = self._find(key)
        if index >= 0:
            self._values[index] = MetadataValue(value)
        else:
            self._keys.append(key.copy())
            self._values.append(MetadataValue(value))

    def get(self, key: String) raises -> MetadataValue:
        var index = self._find(key)
        if index < 0:
            raise Error("Metadata key does not exist.")
        return self._values[index].copy()

    def get_string(self, key: String) raises -> String:
        return self.get(key).as_string()

    def get_int(self, key: String) raises -> Int:
        return self.get(key).as_int()

    def get_float(self, key: String) raises -> Float64:
        return self.get(key).as_float()

    def get_bool(self, key: String) raises -> Bool:
        return self.get(key).as_bool()

    def _key_at(self, index: Int) -> String:
        return self._keys[index].copy()

    def _value_at(self, index: Int) -> MetadataValue:
        return self._values[index].copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("{")
        for index in range(len(self._keys)):
            if index > 0:
                writer.write(", ")
            writer.write(
                self._keys[index],
                ": ",
                self._values[index],
            )
        writer.write("}")
