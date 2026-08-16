from std.collections import Dict, List
from std.collections.string import Codepoint, StringSlice


def add_standard_stopwords(mut stopwords: Dict[String, Bool]):
    """Adds the bundled English and Russian stop-word lists."""
    var english = String(
        "i me my myself we us our ours ourselves you your yours yourself "
        "yourselves he him his himself she her hers herself it its itself "
        "they them their theirs themselves what which who whom this that "
        "these those am is are was were be been being have has had having "
        "do does did doing will would shall should can could may might must "
        "ought a an the and but if or because as until while of at by for "
        "with about against between into through during before after above "
        "below to from up down in out on off over under again further then "
        "once here there when where why how all any both each few more most "
        "other some such no nor not only own same so than too very cannot "
        "don't dont isn't aren't wasn't weren't hasn't haven't hadn't "
        "doesn't didn't won't wouldn't shouldn't couldn't mustn't"
    )
    for word in english.split():
        stopwords[String(word)] = True

    var russian = String(
        "и в во не что он на я с со как а то все она так его но да ты к у "
        "же вы за бы по только ее мне было вот от меня еще нет о из ему "
        "теперь когда даже ну вдруг ли если уже или ни быть был него до вас "
        "нибудь опять уж вам сказал ведь там потом себя ничего ей может они "
        "тут где есть надо ней для мы тебя их чем была сам чтоб без будто "
        "человек чего раз тоже себе под жизнь будет ж тогда кто этот говорил "
        "того потому этого какой совсем ним здесь этом один почти мой тем "
        "чтобы нее кажется сейчас были куда зачем сказать всех никогда "
        "сегодня можно при наконец два об другой хоть после над больше тот "
        "через эти нас про всего них какая много разве сказала три эту моя "
        "впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой "
        "им более всегда конечно всю между"
    )
    for word in russian.split():
        stopwords[String(word)] = True


def _is_ascii_word(codepoint: Int) -> Bool:
    return (
        (codepoint >= 48 and codepoint <= 57)
        or (codepoint >= 97 and codepoint <= 122)
        or codepoint == 39
        or codepoint == 95
    )


def _is_unicode_boundary(codepoint: Int) -> Bool:
    """Returns true for whitespace, punctuation, symbols, and emoji."""
    if codepoint < 128:
        return not _is_ascii_word(codepoint)
    return (
        (codepoint >= 0x0080 and codepoint <= 0x009F)
        or (codepoint >= 0x00A0 and codepoint <= 0x00BF)
        or codepoint == 0x00D7
        or codepoint == 0x00F7
        or codepoint == 0x037E
        or codepoint == 0x0387
        or (codepoint >= 0x055A and codepoint <= 0x055F)
        or codepoint == 0x0589
        or codepoint == 0x058A
        or codepoint == 0x05BE
        or codepoint == 0x05C0
        or codepoint == 0x05C3
        or codepoint == 0x05C6
        or (codepoint >= 0x0600 and codepoint <= 0x0605)
        or codepoint == 0x0609
        or codepoint == 0x060A
        or codepoint == 0x060C
        or codepoint == 0x060D
        or codepoint == 0x061B
        or codepoint == 0x061D
        or codepoint == 0x061E
        or codepoint == 0x061F
        or (codepoint >= 0x066A and codepoint <= 0x066D)
        or codepoint == 0x06D4
        or (codepoint >= 0x2000 and codepoint <= 0x2BFF)
        or (codepoint >= 0x3000 and codepoint <= 0x303F)
        or (codepoint >= 0xFE10 and codepoint <= 0xFE1F)
        or (codepoint >= 0xFE30 and codepoint <= 0xFE6F)
        or (codepoint >= 0xFF01 and codepoint <= 0xFF0F)
        or (codepoint >= 0xFF1A and codepoint <= 0xFF20)
        or (codepoint >= 0xFF3B and codepoint <= 0xFF40)
        or (codepoint >= 0xFF5B and codepoint <= 0xFF65)
        or (codepoint >= 0x1F000 and codepoint <= 0x1FAFF)
    )


def _trim_connectors(token: String) -> String:
    var start = 0
    var end = token.byte_length()
    var bytes = token.as_bytes()
    while start < end and (bytes[start] == 39 or bytes[start] == 95):
        start += 1
    while end > start and (bytes[end - 1] == 39 or bytes[end - 1] == 95):
        end -= 1
    return String(StringSlice(token)[byte=start:end])


def standard_word_tokens(text: String) raises -> List[String]:
    """
    Lowercases Unicode text and splits it at Unicode word boundaries.

    Letters, combining marks, and numbers remain in tokens. An ASCII
    apostrophe or underscore is retained only within a token. Whitespace,
    punctuation, symbols, and emoji become boundaries.
    """
    var lowered = text.lower()
    var source = lowered.as_bytes()
    var normalized = List[UInt8](unsafe_uninit_length=len(source))
    var offset = 0
    for codepoint_slice in lowered.codepoint_slices():
        var codepoint = Int(Codepoint.ord(codepoint_slice))
        var boundary = _is_unicode_boundary(codepoint)
        var width = codepoint_slice.byte_length()
        for byte_index in range(width):
            normalized[offset + byte_index] = (
                32 if boundary else source[offset + byte_index]
            )
        offset += width

    var normalized_text = String(from_utf8=normalized)
    var result = List[String]()
    for token_slice in normalized_text.split():
        var token = _trim_connectors(String(token_slice))
        if token.byte_length() > 0:
            result.append(token^)
    return result^


struct StandardBM25Analyzer(Movable):
    """
    BM25 text analyzer with Unicode lowercase, word boundaries, and stopwords.

    It intentionally does not stem words, so indexing and querying preserve
    their normalized surface forms.
    """

    var _stopwords: Dict[String, Bool]

    def __init__(out self):
        self._stopwords = Dict[String, Bool]()
        add_standard_stopwords(self._stopwords)

    def __init__(out self, *, deinit move: Self):
        self._stopwords = move._stopwords^

    def analyze(self, text: String) raises -> List[String]:
        var tokens = standard_word_tokens(text)
        var result = List[String](capacity=len(tokens))
        for token in tokens:
            if token not in self._stopwords:
                result.append(token.copy())
        return result^
