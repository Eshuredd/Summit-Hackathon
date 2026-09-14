"""Viewer shutdown is user cancellation, separate from physical task failures."""


class ViewerClosed(Exception):
    """Signal deliberate viewer termination without requesting reset or recovery."""


def check_viewer(viewer):
    """Stop an interactive call when its viewer closes; headless calls are unchanged.

    Args:
        viewer: Optional MuJoCo viewer handle.

    Raises:
        ViewerClosed: The interactive viewer is no longer running.
    """
    if viewer is not None and not viewer.is_running():
        raise ViewerClosed("Viewer closed by user")
