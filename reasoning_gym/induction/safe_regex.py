"""A small regular-expression engine that is safe to run on untrusted patterns.

Python's ``re`` module uses a backtracking matcher, so a pattern such as ``(a+)+b`` can take
exponential time on an input like ``"aaaa...a"`` (so-called ReDoS). ``score_answer`` in
``regex_synthesis`` has to run regexes written by a model, so it uses this engine instead.

The engine parses a well-defined subset of Python's regex syntax, compiles it to a Thompson NFA and
simulates the NFA, so ``fullmatch`` runs in ``O(len(text) * num_states)`` time. Pattern length,
repetition counts and the number of NFA states are capped, which bounds the cost of compiling as well.

Supported syntax (same meaning as in Python's ``re``):

- literal characters and escaped punctuation such as ``\\.``
- ``.`` (any character except newline)
- character classes ``[abc]``, ``[a-c]``, ``[^ab]``
- escapes ``\\d \\D \\w \\W \\s \\S`` (ASCII semantics), ``\\n`` and ``\\t``
- groups ``(...)`` and non-capturing groups ``(?:...)``
- alternation ``|``
- quantifiers ``* + ? {m} {m,} {,n} {m,n}``, optionally lazy (``*?`` etc.; laziness does not change
  which strings fully match)
- ``^`` at the very start and ``$`` at the very end of the pattern (redundant for full matches)

Everything else (back-references, lookarounds, inline flags, possessive quantifiers, word boundaries,
anchors in the middle of the pattern, ...) raises :class:`UnsupportedRegexError`.
"""

import re
import string
from dataclasses import dataclass
from typing import Optional

MAX_PATTERN_LENGTH = 300
MAX_REPEAT = 20
MAX_STATES = 1000
MAX_CLASS_RANGE = 2000

_QUANTIFIER_RE = re.compile(r"\{(\d{0,3})(,?)(\d{0,3})\}")


class UnsupportedRegexError(ValueError):
    """Raised when a pattern is malformed, uses unsupported syntax or is too large."""


@dataclass(frozen=True)
class CharSet:
    """A set of characters, optionally negated."""

    chars: frozenset
    negated: bool = False

    def matches(self, ch: str) -> bool:
        return (ch in self.chars) != self.negated


@dataclass(frozen=True)
class CharMatcher:
    """Matches one character: the union of ``sets``, optionally negated as a whole (``[^...]``)."""

    sets: tuple
    negated: bool = False

    def matches(self, ch: str) -> bool:
        return any(s.matches(ch) for s in self.sets) != self.negated


_DIGITS = frozenset(string.digits)
_WORD = frozenset(string.ascii_letters + string.digits + "_")
_SPACE = frozenset(" \t\n\r\f\v")

_ESCAPE_SETS = {
    "d": CharSet(_DIGITS),
    "D": CharSet(_DIGITS, negated=True),
    "w": CharSet(_WORD),
    "W": CharSet(_WORD, negated=True),
    "s": CharSet(_SPACE),
    "S": CharSet(_SPACE, negated=True),
}
_ESCAPE_LITERALS = {"n": "\n", "t": "\t"}
_DOT = CharMatcher((CharSet(frozenset("\n"), negated=True),))


def _literal(ch: str) -> CharMatcher:
    return CharMatcher((CharSet(frozenset(ch)),))


# AST nodes are plain tuples:
#   ("char", CharMatcher) | ("cat", [nodes]) | ("alt", [nodes]) | ("rep", node, lo, hi_or_None)


class _Parser:
    def __init__(self, pattern: str):
        self.p = pattern
        self.i = 0

    def peek(self) -> Optional[str]:
        return self.p[self.i] if self.i < len(self.p) else None

    def error(self, msg: str) -> UnsupportedRegexError:
        return UnsupportedRegexError(f"{msg} at position {self.i}")

    def parse(self):
        node = self.parse_alt()
        if self.i != len(self.p):
            raise self.error("unbalanced parenthesis")
        return node

    def parse_alt(self):
        branches = [self.parse_concat()]
        while self.peek() == "|":
            self.i += 1
            branches.append(self.parse_concat())
        return branches[0] if len(branches) == 1 else ("alt", branches)

    def parse_concat(self):
        items = []
        while self.peek() is not None and self.peek() not in "|)":
            items.append(self.parse_repeat())
        return ("cat", items)

    def parse_repeat(self):
        atom = self.parse_atom()
        quantifier = self.parse_quantifier()
        if quantifier is None:
            return atom
        if self.peek() == "?":  # lazy quantifier: same set of full matches
            self.i += 1
        if self.peek() is not None and self.peek() in "*+?{":
            raise self.error("multiple repeat or possessive quantifier")
        lo, hi = quantifier
        return ("rep", atom, lo, hi)

    def parse_quantifier(self) -> Optional[tuple[int, Optional[int]]]:
        c = self.peek()
        if c == "*":
            self.i += 1
            return 0, None
        if c == "+":
            self.i += 1
            return 1, None
        if c == "?":
            self.i += 1
            return 0, 1
        if c == "{":
            m = _QUANTIFIER_RE.match(self.p, self.i)
            if m is None:
                raise self.error("unsupported or malformed '{' quantifier")
            lo_s, comma, hi_s = m.groups()
            if not comma:
                if not lo_s:
                    raise self.error("empty quantifier")
                lo = hi = int(lo_s)
            else:
                lo = int(lo_s) if lo_s else 0
                hi = int(hi_s) if hi_s else None
            if lo > MAX_REPEAT or (hi is not None and hi > MAX_REPEAT):
                raise self.error(f"repetition count above {MAX_REPEAT}")
            if hi is not None and hi < lo:
                raise self.error("min repeat greater than max repeat")
            self.i = m.end()
            return lo, hi
        return None

    def parse_atom(self):
        c = self.peek()
        if c == "(":
            self.i += 1
            if self.p.startswith("?:", self.i):
                self.i += 2
            elif self.peek() == "?":
                raise self.error("unsupported group extension")
            node = self.parse_alt()
            if self.peek() != ")":
                raise self.error("missing ')'")
            self.i += 1
            return node
        if c == "[":
            return ("char", self.parse_class())
        if c == ".":
            self.i += 1
            return ("char", _DOT)
        if c == "\\":
            charset, _ = self.parse_escape()
            return ("char", CharMatcher((charset,)))
        if c in "*+?{":
            raise self.error("nothing to repeat")
        if c in "^$":
            raise self.error("anchors are only supported at the start and end of the pattern")
        self.i += 1
        return ("char", _literal(c))

    def parse_escape(self) -> tuple[CharSet, Optional[str]]:
        """Parse an escape sequence. Returns the character set and, for single characters, the character."""
        self.i += 1  # backslash
        c = self.peek()
        if c is None:
            raise self.error("trailing backslash")
        self.i += 1
        if c in _ESCAPE_SETS:
            return _ESCAPE_SETS[c], None
        if c in _ESCAPE_LITERALS:
            ch = _ESCAPE_LITERALS[c]
            return CharSet(frozenset(ch)), ch
        if c.isalnum() or c == "_":
            raise self.error(f"unsupported escape '\\{c}'")
        return CharSet(frozenset(c)), c

    def parse_class_char(self) -> tuple[CharSet, Optional[str]]:
        if self.peek() == "\\":
            return self.parse_escape()
        c = self.peek()
        self.i += 1
        return CharSet(frozenset(c)), c

    def parse_class(self) -> CharMatcher:
        self.i += 1  # '['
        negated = False
        if self.peek() == "^":
            negated = True
            self.i += 1
        sets = []
        first = True
        while True:
            c = self.peek()
            if c is None:
                raise self.error("unterminated character class")
            if c == "]" and not first:
                self.i += 1
                break
            first = False
            charset, lo = self.parse_class_char()
            is_range = lo is not None and self.peek() == "-" and self.i + 1 < len(self.p) and self.p[self.i + 1] != "]"
            if not is_range:
                sets.append(charset)
                continue
            self.i += 1  # '-'
            _, hi = self.parse_class_char()
            if hi is None:
                raise self.error("bad character range")
            if ord(hi) < ord(lo):
                raise self.error("reversed character range")
            if ord(hi) - ord(lo) > MAX_CLASS_RANGE:
                raise self.error("character range too large")
            sets.append(CharSet(frozenset(chr(x) for x in range(ord(lo), ord(hi) + 1))))
        return CharMatcher(tuple(sets), negated)


class _NFABuilder:
    def __init__(self):
        self.eps: list[list[int]] = []
        self.edges: list[list[tuple[CharMatcher, int]]] = []

    def new_state(self) -> int:
        if len(self.eps) >= MAX_STATES:
            raise UnsupportedRegexError(f"pattern too complex (more than {MAX_STATES} NFA states)")
        self.eps.append([])
        self.edges.append([])
        return len(self.eps) - 1

    def chain(self, fragments: list[tuple[int, int]]) -> tuple[int, int]:
        if not fragments:
            s = self.new_state()
            return s, s
        start, end = fragments[0]
        for s, e in fragments[1:]:
            self.eps[end].append(s)
            end = e
        return start, end

    def build(self, node) -> tuple[int, int]:
        kind = node[0]
        if kind == "char":
            s, e = self.new_state(), self.new_state()
            self.edges[s].append((node[1], e))
            return s, e
        if kind == "cat":
            return self.chain([self.build(item) for item in node[1]])
        if kind == "alt":
            s, e = self.new_state(), self.new_state()
            for branch in node[1]:
                bs, be = self.build(branch)
                self.eps[s].append(bs)
                self.eps[be].append(e)
            return s, e
        if kind == "rep":
            _, inner, lo, hi = node
            fragments = [self.build(inner) for _ in range(lo)]
            if hi is None:
                loop = self.new_state()
                bs, be = self.build(inner)
                self.eps[loop].append(bs)
                self.eps[be].append(loop)
                fragments.append((loop, loop))
            else:
                for _ in range(hi - lo):
                    s, e = self.new_state(), self.new_state()
                    bs, be = self.build(inner)
                    self.eps[s].extend([bs, e])
                    self.eps[be].append(e)
                    fragments.append((s, e))
            return self.chain(fragments)
        raise AssertionError(f"unknown node {kind}")


def _strip_anchors(pattern: str) -> str:
    if pattern.startswith("^"):
        pattern = pattern[1:]
    if pattern.endswith("$"):
        backslashes = len(pattern) - 1 - len(pattern[:-1].rstrip("\\"))
        if backslashes % 2 == 0:
            pattern = pattern[:-1]
    return pattern


class SafeRegex:
    """A compiled pattern. Use :func:`compile_safe_regex` to create one."""

    def __init__(self, pattern: str):
        self.pattern = pattern
        ast = _Parser(_strip_anchors(pattern)).parse()
        builder = _NFABuilder()
        self.start, self.accept = builder.build(ast)
        self.eps = builder.eps
        self.edges = builder.edges

    @property
    def num_states(self) -> int:
        return len(self.eps)

    def _closure(self, states: set[int]) -> set[int]:
        stack = list(states)
        seen = set(states)
        while stack:
            for t in self.eps[stack.pop()]:
                if t not in seen:
                    seen.add(t)
                    stack.append(t)
        return seen

    def fullmatch(self, text: str) -> bool:
        """Return True if the whole of ``text`` matches the pattern."""
        current = self._closure({self.start})
        for ch in text:
            nxt = {t for s in current for matcher, t in self.edges[s] if matcher.matches(ch)}
            if not nxt:
                return False
            current = self._closure(nxt)
        return self.accept in current


def compile_safe_regex(pattern: str) -> SafeRegex:
    """Compile ``pattern`` or raise :class:`UnsupportedRegexError`."""
    if not isinstance(pattern, str):
        raise UnsupportedRegexError("pattern must be a string")
    if len(pattern) > MAX_PATTERN_LENGTH:
        raise UnsupportedRegexError(f"pattern longer than {MAX_PATTERN_LENGTH} characters")
    return SafeRegex(pattern)
