"""Parse the single supported drawer retrieval goal without an LLM."""

import re

DEFAULT_INSTRUCTION = "Open the drawer and place the object on the table."


def parse_instruction(instruction: str) -> dict[str, str]:
    """Normalize supported complete commands; reject all other instructions.

    Args:
        instruction: Natural-language command for the existing task.

    Returns:
        The supported structured goal.

    Raises:
        ValueError: The command is unsupported.
    """
    if not isinstance(instruction, str):
        raise ValueError("Instruction must be text")
    text = " ".join(instruction.lower().strip().rstrip(".!?").split())
    item = r"(?:the )?(?:object|utensil)"
    patterns = (
        rf"open the drawer and (?:place|put) {item} on the table",
        rf"retrieve {item} from the drawer",
        rf"(?:take|retrieve) {item} from the drawer and (?:put|place) it on the table",
    )
    if any(re.fullmatch(pattern, text) for pattern in patterns):
        return {"task": "drawer_to_table"}
    raise ValueError(f"Unsupported instruction: {instruction!r}")
