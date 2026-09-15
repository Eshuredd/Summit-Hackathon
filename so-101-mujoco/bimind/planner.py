"""Pure, deterministic one-step planning from measured drawer-scene state."""

from math import hypot, isfinite
from typing import Any, TypedDict

XY_TOLERANCE = 0.03
Z_TOLERANCE = 0.005


class Action(TypedDict):
    """Describe one skill invocation or a terminal STOP without executing anything."""

    skill: str
    args: list[str]
    reason: str


def _action(skill: str, args: list[str], reason: str) -> Action:
    """Build a single serializable decision."""
    return {"skill": skill, "args": args, "reason": reason}


def _xyz(position: Any) -> tuple[float, float, float]:
    """Validate a measured finite XYZ position before using it in a decision."""
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise ValueError("position must contain three coordinates")
    if any(isinstance(value, bool) for value in position):
        raise ValueError("position coordinates must be numbers")
    values = tuple(float(value) for value in position)
    if not all(isfinite(value) for value in values):
        raise ValueError("position coordinates must be finite")
    return values


def object_at_target(state: dict[str, Any]) -> bool:
    """Compare measured object and target XYZ, including table height.

    Args:
        state: Scene observation containing drawer_object and table_target positions.

    Returns:
        Whether XY error is below 30 mm and Z error is below 5 mm.

    Raises:
        KeyError: A required position is missing.
        ValueError: Coordinates are malformed or nonfinite.
    """
    obj = _xyz(state["objects"]["drawer_object"]["position"])
    target = _xyz(state["targets"]["table_target"]["position"])
    return (
        hypot(obj[0] - target[0], obj[1] - target[1]) < XY_TOLERANCE
        and abs(obj[2] - target[2]) < Z_TOLERANCE
    )


def next_action(state: dict[str, Any], confidence_threshold: float = 0.65) -> Action:
    """Choose exactly one next skill from observations, without motion or history.

    Args:
        state: Allowlisted RGB and robot observation, including acknowledged placement.
        confidence_threshold: Minimum visual confidence required before any action.

    Returns:
        A dictionary with skill, positional args, and a human-readable reason.
        STOP means failure or an unsupported observation; it is never a motion call.
    """
    if not isinstance(state, dict):
        return _action("STOP", [], "invalid scene observation: expected a dictionary")
    if state.get("failure") is not None:
        return _action("STOP", [], f"active failure: {state['failure']}")
    try:
        if state["instruction"] != {"task": "drawer_to_table"}:
            raise ValueError("unsupported instruction goal")
        drawer = state["drawer"]
        visual = drawer["visual_state"]
        confidence = drawer["confidence"]
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not isfinite(confidence)
            or not 0 <= confidence <= 1
            or not 0 <= confidence_threshold <= 1
        ):
            raise ValueError("invalid confidence or threshold")
        if visual not in ("closed", "open") or confidence < confidence_threshold:
            return _action("STOP", [], "drawer partial, unknown, or below confidence threshold")
        closed, opened = visual == "closed", visual == "open"
        monitor = state["holding"]["drawer_monitor_active"]
        at_target = state["holding"]["placement_confirmed"]
        if not isinstance(monitor, bool) or not isinstance(at_target, bool):
            raise ValueError("holding flags must be booleans")
        left = state["arms"]["left"]["holding_objects"]
        right = state["arms"]["right"]["holding_objects"]
        if not all(
            isinstance(names, list) and all(isinstance(n, str) for n in names)
            for names in (left, right)
        ):
            raise ValueError("holding_objects must be lists of names")
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return _action("STOP", [], f"invalid scene observation: {error}")
    if any(name != "drawer_handle" for name in left) or any(
        name != "drawer_object" for name in right
    ):
        return _action("STOP", [], "unsupported object ownership in drawer task")
    left_holds = "drawer_handle" in left
    right_holds = "drawer_object" in right
    # Completion takes precedence: the released drawer intentionally remains open.
    # A held object at the target still needs the place skill to release it.
    if at_target and not right_holds:
        if left_holds:
            return _action(
                "release_drawer", ["left"], "object at table target; left still holds drawer"
            )
        if monitor:
            return _action("STOP", [], "drawer hold monitor active but handle support is lost")
        return _action("finish", [], "object at table target and drawer released")
    if closed:
        if right_holds or left_holds:
            return _action(
                "STOP", [], "closed drawer with an active grasp is not a supported restart"
            )
        return _action("open_drawer", ["left"], "drawer closed")
    if not opened:
        return _action("STOP", [], "drawer partially open; no validated recovery skill")
    if not monitor or not left_holds:
        return _action("hold_drawer", ["left"], "drawer open but left hold is not maintained")
    if right_holds:
        reason = (
            "right holds object; release at table target"
            if at_target
            else "right holds object away from table target"
        )
        return _action("place_object", ["right", "table_target"], reason)
    return _action(
        "pick_object", ["right", "drawer_object"], "drawer held; right is not holding object"
    )
