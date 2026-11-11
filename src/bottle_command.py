"""Which coloured bottle to pick: the three-bottle table and the typed command that selects one.

A command is plain text ("bring me the green bottle").  ``parse_command`` accepts it only when it
names exactly one colour; anything else is refused *before the chair moves* with a reason, so a
mistyped or ambiguous command can never start a task.  Perception then has to find that colour:
a colour that is not on the table inhibits driving, as a missing bottle already does.
"""
import re
from dataclasses import dataclass
from typing import Optional

BOTTLE_COLOURS = {"red": (.85, .12, .12), "green": (.12, .70, .20), "blue": (.10, .60, .90)}
# World positions on the table: one row at the single-bottle task's validated depth (x 0.62), 0.36 m apart,
# centred on the chair's start line (green straight ahead, red and blue 0.36 m to either side). The chair
# always docks with the commanded bottle at the same place relative to it, so:
#  * every bottle stands at the same depth: a bottle 85 mm deeper brought the table's near edge 85 mm closer
#    to the chair and the auxiliary arm's hand-off pose dipped 4.7 mm into it;
#  * 0.36 m sideways: the main arm (mounted on the left) and the cap bin (0.185 m to the target's left) work
#    in the area to the target's left; at 0.26 m the palm hit the left neighbour carrying the cap (3.7 N).
DEFAULT_LAYOUT = {"red": (.62, .575), "green": (.62, .215), "blue": (.62, -.145)}
NEGATIONS = re.compile(r"\b(not|don't|dont|never|except|instead of|rather than|no)\b")


@dataclass(frozen=True)
class Command:
    text: str
    colour: Optional[str]
    accepted: bool
    reason: str


def parse_command(text, known=tuple(BOTTLE_COLOURS)):
    """Turn a typed command into a colour, or refuse it with the reason."""
    words = re.sub(r"[^a-z' ]", " ", str(text).lower())
    if not words.split():
        return Command(str(text), None, False, "Empty command: say which colour to pick (red, green or blue)")
    if NEGATIONS.search(words):
        return Command(text, None, False, "Command contains a negation; say the colour to pick, e.g. 'pick the red bottle'")
    named = [c for c in known if re.search(rf"\b{c}\b", words)]
    if not named:
        return Command(text, None, False, f"No known colour in the command (choose from {', '.join(known)})")
    if len(named) > 1:
        return Command(text, None, False, f"Ambiguous command: it names {' and '.join(named)}; pick one colour")
    return Command(text, named[0], True, f"Pick the {named[0]} bottle")


def table_layout(present=None, layout=DEFAULT_LAYOUT):
    """Bottles on the table: all three by default, or only the listed colours."""
    return {c: xy for c, xy in layout.items() if present is None or c in present}


DISTRACTOR_GEOM = re.compile(r"^bottle_(red|green|blue)_collision$")


def resting_distractor(names):
    """True for a contact that is only a distractor bottle standing on the table (legal everywhere)."""
    names = tuple(n or "" for n in names)
    return len(names) == 2 and "grasp_table" in names and any(DISTRACTOR_GEOM.match(n) for n in names)
