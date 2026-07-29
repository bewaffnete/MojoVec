from std.collections import Dict, List

from mojovec.api.metadata import Metadata, MetadataValue
from mojovec.api.where import WhereNode


comptime BITMAP_WORD_BITS = 64
comptime MAX_BITMAP_DISTINCT_VALUES = 1024


struct MetadataBitmapPosting(Movable):
    """A sparse packed bitmap for one typed metadata value."""

    var _value: MetadataValue
    var _word_indexes: List[Int]
    var _words: List[UInt64]

    def __init__(out self, value: MetadataValue):
        self._value = value.copy()
        self._word_indexes = List[Int]()
        self._words = List[UInt64]()

    def __init__(out self, *, deinit take: Self):
        self._value = take._value^
        self._word_indexes = take._word_indexes^
        self._words = take._words^

    def value_matches(self, node: WhereNode) raises -> Bool:
        return node.matches(self._value)

    def same_value(self, value: MetadataValue) -> Bool:
        return self._value.equals(value)

    def add(mut self, internal_id: Int):
        var word_index = internal_id // BITMAP_WORD_BITS
        var mask = UInt64(1) << UInt64(
            internal_id % BITMAP_WORD_BITS
        )
        var size = len(self._word_indexes)
        if size > 0 and self._word_indexes[size - 1] == word_index:
            self._words[size - 1] |= mask
            return
        self._word_indexes.append(word_index)
        self._words.append(mask)

    def union_into(self, mut destination: List[UInt64]):
        for index in range(len(self._word_indexes)):
            destination[self._word_indexes[index]] |= self._words[index]


struct MetadataBitmapField(Movable):
    """Value postings for one metadata key."""

    var _postings: List[MetadataBitmapPosting]
    var _enabled: Bool

    def __init__(out self):
        self._postings = List[MetadataBitmapPosting]()
        self._enabled = True

    def __init__(out self, *, deinit take: Self):
        self._postings = take._postings^
        self._enabled = take._enabled

    def enabled(self) -> Bool:
        return self._enabled

    def add(mut self, internal_id: Int, value: MetadataValue):
        if not self._enabled:
            return
        for index in range(len(self._postings)):
            if self._postings[index].same_value(value):
                self._postings[index].add(internal_id)
                return

        if len(self._postings) >= MAX_BITMAP_DISTINCT_VALUES:
            self._postings.clear()
            self._enabled = False
            return

        var posting = MetadataBitmapPosting(value)
        posting.add(internal_id)
        self._postings.append(posting^)

    def evaluate(
        self,
        node: WhereNode,
        mut destination: List[UInt64],
    ) raises:
        for index in range(len(self._postings)):
            if self._postings[index].value_matches(node):
                self._postings[index].union_into(destination)


struct MetadataBitmapIndex(Movable):
    """
    Per-field typed bitmap postings.

    High-cardinality fields switch to collection-level scan evaluation after
    1024 distinct values, preventing unbounded per-value index overhead.
    """

    var _field_lookup: Dict[String, Int]
    var _fields: List[MetadataBitmapField]

    def __init__(out self):
        self._field_lookup = Dict[String, Int]()
        self._fields = List[MetadataBitmapField]()

    def __init__(out self, *, deinit take: Self):
        self._field_lookup = take._field_lookup^
        self._fields = take._fields^

    def add(
        mut self,
        internal_id: Int,
        metadata: Metadata,
    ) raises:
        for metadata_index in range(metadata.count()):
            var key = metadata._key_at(metadata_index)
            var value = metadata._value_at(metadata_index)
            var field_index: Int
            if key in self._field_lookup:
                field_index = self._field_lookup[key]
            else:
                field_index = len(self._fields)
                self._field_lookup[key] = field_index
                self._fields.append(MetadataBitmapField())
            self._fields[field_index].add(internal_id, value)

    def contains_field(self, key: String) -> Bool:
        return key in self._field_lookup

    def can_evaluate(self, key: String) raises -> Bool:
        if key not in self._field_lookup:
            return True
        return self._fields[self._field_lookup[key]].enabled()

    def evaluate(
        self,
        node: WhereNode,
        total_records: Int,
    ) raises -> List[UInt64]:
        var word_count = (
            total_records + BITMAP_WORD_BITS - 1
        ) // BITMAP_WORD_BITS
        var result = List[UInt64](unsafe_uninit_length=word_count)
        for index in range(word_count):
            result[index] = 0

        var key = node.key()
        if key not in self._field_lookup:
            return result^
        var field_index = self._field_lookup[key]
        self._fields[field_index].evaluate(node, result)
        return result^
