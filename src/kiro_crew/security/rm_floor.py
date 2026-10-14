"""Recursive-force ``rm`` floor behind the two ``local-destructive-rm-rf-*`` rows.

The catalog patterns ``rm -rf /.*`` and ``rm -rf ~.*`` match one spelling; a shell
runs the same wipe for ``rm -fr /``, ``rm -r -f /``, ``rm --recursive --force /`` and
``rm -rf $HOME``. This floor reads each ``rm`` argv for recursive and force in any
packing, order or long form, against the root or the home directory ITSELF.

It writes no shell reader of its own. Commands are cut on the shared quote state
machine (``_iter_shell_chars``) and read with the shared ``_self_tokens`` the
self-protection floor uses, which removes quotes and expands ``~`` / ``$HOME``.
Quoted shell payloads (``bash -c '...'``), descendants, exec wrappers (``sudo``),
parameter-expansion defaults and interpreter bodies stay with the regex rows and
the OS sandbox.
"""

from __future__ import annotations

import os
import posixpath
import re

from ..trust_patterns import ENV_ASSIGNMENT_RE
from . import shell_normalizer as _shell
from .denied_rules import BUILTIN_DENIED_RULES

#: The catalog row each wipe target maps to. The floor is gated on that row's
#: pattern, so an operator who disables the row disables the floor with it.
_RM_FLOOR_RULE_IDS: dict[str, str] = {
    "root": "local-destructive-rm-rf-root",
    "home": "local-destructive-rm-rf-home",
}
_RM_FLOOR_BY_KIND: dict[str, tuple[str, str]] = {
    kind: (rule.id, rule.pattern)
    for kind, rule_id in _RM_FLOOR_RULE_IDS.items()
    for rule in BUILTIN_DENIED_RULES
    if rule.id == rule_id
}
_RM_FLOOR_NOTE = (
    "Matched structurally on the command's argv, not by the pattern text above: "
    "rm with recursive and force flags, in any spelling, targets this directory itself."
)
_RM_FLOOR_UNREAD_NOTE = (
    "Refused because the rm floor could not read this command, so it cannot rule "
    "out a wipe of this directory. Simplify it."
)

#: Cheap necessary condition: the program is a literal ``rm`` (quote splices
#: allowed) or is built by ``$`` / ``\\`` / a backtick.
_CAN_FIRE_RE = re.compile(r"r[^\w\s]*m|[$\\`]")
#: A redirection glued to its target at the program position (``2>/dev/null rm``).
_LEADING_REDIRECT_RE = re.compile(r"^\d*(?:[<>]|>>|&>)&?\S+$")
#: Unquoted characters that end a simple command.
_CUT_CHARS = frozenset(";&|\n()`")
#: Reserved words that keep the next word at command position.
_KEYWORDS = frozenset({"if", "then", "else", "elif", "while", "until", "do", "!", "{"})


def _can_fire(command: str) -> bool:
    """False when *command* cannot hold a recursive-force ``rm`` at all."""
    return bool(_CAN_FIRE_RE.search(command))


def _target_kind(operand: str, home: str) -> "str | None":
    """``"root"`` / ``"home"`` when *operand* is that directory itself, else None."""
    if operand.endswith("/*"):
        operand = operand[:-2] or "/"
    if operand.startswith("/") and posixpath.normpath(operand).strip("/") == "":
        return "root"
    if home and os.path.normpath(operand) == home:
        return "home"
    return None


def _wipe_kinds(args: "list[str]", home: str) -> "set[str]":
    """Targets an ``rm`` with *args* wipes recursively and forcibly, else empty."""
    recursive = force = options_done = False
    operands: list[str] = []
    for arg in args:
        if options_done or arg == "-" or not arg.startswith("-"):
            operands.append(arg)
        elif arg == "--":
            options_done = True
        elif arg.startswith("--"):
            name = arg.split("=", 1)[0]
            recursive = recursive or "--recursive".startswith(name)
            force = force or "--force".startswith(name)
        else:
            recursive = recursive or "r" in arg or "R" in arg
            force = force or "f" in arg
    if not (recursive and force):
        return set()
    return {kind for op in operands if (kind := _target_kind(op, home))}


def _raw_commands(command: str) -> "list[str]":
    """*command* cut into simple commands BEFORE quotes come off.

    Cuts at every unquoted operator, paren, backtick and newline the shared
    quote state machine (``_iter_shell_chars``) marks active, so a quoted
    ``'a;b'`` or ``'#x'`` stays one word of its command. ``2>&1`` and ``&>`` are
    redirections, not cuts. An unquoted ``#`` at a word start runs to the newline.
    """
    text = _shell._fold_line_continuations(command)
    out: list[str] = []
    piece: list[str] = []
    comment = False
    for step in _shell._iter_shell_chars(text):
        if comment:
            comment = not (step.active and step.char == "\n")
            continue
        if step.active and step.char == "#" and (not piece or piece[-1].isspace()):
            comment = True
        elif not step.active or step.char not in _CUT_CHARS or _is_redirect_amp(text, step):
            piece.append(step.text)
            continue
        out.append("".join(piece))
        piece = []
    out.append("".join(piece))
    return out


def _is_redirect_amp(text: str, step: "_shell._ShellChar") -> bool:
    """True for the ``&`` of ``>&`` / ``<&`` / ``&>``, which redirects, not cuts."""
    at = step.offset
    return step.char == "&" and (text[at - 1 : at] in ("<", ">") or text[at + 1 : at + 2] == ">")


def _rm_wipe_targets(command: str) -> "set[str]":
    """The wipe targets (``root`` / ``home``) any ``rm`` in *command* hits."""
    if not _can_fire(command):
        return set()
    home = os.path.normpath(os.path.expanduser("~"))
    home = "" if home in ("~", "/") else home
    kinds: set[str] = set()
    for piece in _raw_commands(command):
        argv = _shell._self_tokens(piece)
        start = 0
        while start < len(argv) and (
            argv[start] in _KEYWORDS
            or ENV_ASSIGNMENT_RE.match(argv[start])
            or _LEADING_REDIRECT_RE.match(argv[start])
        ):
            start += 1
        if start < len(argv) and _shell._program_basename(argv[start]) == "rm":
            kinds |= _wipe_kinds(argv[start + 1 :], home)
    return kinds
