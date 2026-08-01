"""Typed metadata-filter planning and bitmap evaluation for Collection."""

from std.collections import Dict, List

from mojovec.api.metadata import Metadata
from mojovec.api.metadata_bitmap import BITMAP_WORD_BITS, MetadataBitmapIndex
from mojovec.api.where import Where, WhereNode


def _empty_bitmap(record_count: Int) -> List[UInt64]:
    var word_count = (record_count + BITMAP_WORD_BITS - 1) // BITMAP_WORD_BITS
    var result = List[UInt64](unsafe_uninit_length=word_count)
    for index in range(word_count):
        result[index] = 0
    return result^


def _scan_predicate_bitmap(
    node: WhereNode,
    record_count: Int,
    metadata_by_internal: Dict[Int, Int],
    metadatas: List[Metadata],
) raises -> List[UInt64]:
    var result = _empty_bitmap(record_count)
    var key = node.key()
    for internal_id in range(record_count):
        if internal_id not in metadata_by_internal:
            continue
        var metadata_index = metadata_by_internal[internal_id]
        if not metadatas[metadata_index].contains(key):
            continue
        var value = metadatas[metadata_index].get(key)
        if node.matches(value):
            var word_index = internal_id // BITMAP_WORD_BITS
            result[word_index] |= (
                UInt64(1) << UInt64(internal_id % BITMAP_WORD_BITS)
            )
    return result^


def _predicate_bitmap(
    node: WhereNode,
    record_count: Int,
    metadata_by_internal: Dict[Int, Int],
    metadatas: List[Metadata],
    metadata_index: MetadataBitmapIndex,
) raises -> List[UInt64]:
    var key = node.key()
    if not metadata_index.contains_field(key):
        return _empty_bitmap(record_count)
    if metadata_index.can_evaluate(key):
        return metadata_index.evaluate(node, record_count)
    return _scan_predicate_bitmap(
        node, record_count, metadata_by_internal, metadatas
    )


def _build_where_filter(
    where: Where,
    user_ids: List[Int],
    is_deleted: List[UInt8],
    metadata_by_internal: Dict[Int, Int],
    metadatas: List[Metadata],
    metadata_index: MetadataBitmapIndex,
) raises -> List[UInt8]:
    """Builds the HNSW exclusion mask from a typed postfix expression."""
    var nodes = where.nodes()
    if len(nodes) == 0:
        raise Error("Where expression cannot be empty.")

    # Validate the postfix expression and calculate its maximum bitmap stack
    # depth so compound filters allocate only the required slots.
    var depth = 0
    var max_depth = 0
    for node_index in range(len(nodes)):
        if nodes[node_index].is_predicate():
            depth += 1
            max_depth = max(max_depth, depth)
        elif nodes[node_index].is_not():
            if depth < 1:
                raise Error("Invalid Where expression.")
        else:
            var arity = nodes[node_index].arity()
            if arity <= 0 or depth < arity:
                raise Error("Invalid Where expression.")
            depth = depth - arity + 1
    if depth != 1:
        raise Error("Invalid Where expression.")

    var record_count = len(user_ids)
    var word_count = (
        record_count + BITMAP_WORD_BITS - 1
    ) // BITMAP_WORD_BITS
    var stack_words = List[UInt64](
        unsafe_uninit_length=max_depth * word_count
    )
    var stack_size = 0
    for node_index in range(len(nodes)):
        if nodes[node_index].is_predicate():
            var predicate = _predicate_bitmap(
                nodes[node_index],
                record_count,
                metadata_by_internal,
                metadatas,
                metadata_index,
            )
            var destination = stack_size * word_count
            for word_index in range(word_count):
                stack_words[destination + word_index] = predicate[word_index]
            stack_size += 1
        elif nodes[node_index].is_not():
            var destination = (stack_size - 1) * word_count
            for word_index in range(word_count):
                stack_words[destination + word_index] = ~stack_words[
                    destination + word_index
                ]
        else:
            var arity = nodes[node_index].arity()
            var first = stack_size - arity
            var destination = first * word_count
            for word_index in range(word_count):
                var combined = stack_words[destination + word_index]
                for operand in range(1, arity):
                    var operand_word = stack_words[
                        (first + operand) * word_count + word_index
                    ]
                    if nodes[node_index].is_all():
                        combined &= operand_word
                    elif nodes[node_index].is_any():
                        combined |= operand_word
                    else:
                        raise Error("Invalid Where expression.")
                stack_words[destination + word_index] = combined
            stack_size = first + 1

    # NOT may set unused high bits in the final word.
    var remainder = record_count % BITMAP_WORD_BITS
    if word_count > 0 and remainder > 0:
        var valid_bits = (UInt64(1) << UInt64(remainder)) - UInt64(1)
        stack_words[word_count - 1] &= valid_bits

    var exclusion = List[UInt8](unsafe_uninit_length=record_count)
    for internal_id in range(record_count):
        var word = stack_words[internal_id // BITMAP_WORD_BITS]
        var mask = UInt64(1) << UInt64(internal_id % BITMAP_WORD_BITS)
        var matches = (word & mask) != 0
        exclusion[internal_id] = (
            1 if is_deleted[internal_id] > 0 or not matches else 0
        )
    return exclusion^
