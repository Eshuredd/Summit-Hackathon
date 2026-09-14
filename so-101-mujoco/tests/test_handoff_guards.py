"""Check that missing support and excessive drop cannot pass the handoff guards."""

from bimanual_handoff import Handoff


def test_missing_support():
    """A cube resting on the table must not count as support by either robot."""
    experiment = Handoff()
    assert not experiment.supports("left")
    assert not experiment.supports("right")
    try:
        experiment.tick(required=("right",))
    except RuntimeError as error:
        assert "Lost right opposing support" in str(error)
    else:
        raise AssertionError("Missing right support was accepted")


def test_excessive_drop():
    """A transfer reference 20 mm above the live cube must trigger the drop guard."""
    experiment = Handoff()
    experiment.transfer_z = float(experiment.position()[2] + 0.02)
    try:
        experiment.tick()
    except RuntimeError as error:
        assert "dropped more than 10 mm" in str(error)
        assert experiment.max_drop > 0.01
    else:
        raise AssertionError("Excessive transfer drop was accepted")


if __name__ == "__main__":
    test_missing_support()
    test_excessive_drop()
    print("Handoff negative checks passed.")
