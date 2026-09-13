"""Generate two namespaced robot instances while sharing immutable CAD assets."""

import copy
import xml.etree.ElementTree as ET
from pathlib import Path


def build():
    """Write dual_scene.xml without modifying either single-arm source file."""
    assets = Path(__file__).parent / "assets"
    robot = ET.parse(assets / "so101.xml").getroot()
    scene = ET.parse(assets / "scene.xml").getroot()
    root = ET.Element("mujoco", model="dual_so101")
    for node in robot:
        if node.tag not in {
            "worldbody",
            "actuator",
            "sensor",
            "equality",
            "contact",
            "tendon",
            "keyframe",
        }:
            root.append(copy.deepcopy(node))
    for node in scene:
        if node.tag not in {"include", "worldbody"}:
            root.append(copy.deepcopy(node))
    world = ET.SubElement(root, "worldbody")
    sections = {
        tag: ET.SubElement(root, tag)
        for tag in ("actuator", "sensor", "equality", "contact", "tendon")
    }
    references = {
        "body",
        "body1",
        "body2",
        "joint",
        "joint1",
        "joint2",
        "site",
        "site1",
        "site2",
        "geom",
        "geom1",
        "geom2",
        "tendon",
        "actuator",
        "objname",
        "refname",
    }
    for side, position, quat in (
        ("left", "0 -0.04 0", "0.7071067812 0 0 0.7071067812"),
        ("right", "0.36 0.08 0", "0.7071067812 0 0 -0.7071067812"),
    ):
        for tag in ("worldbody", *sections):
            source = robot.find(tag)
            if source is None:
                continue
            for original in source:
                node = copy.deepcopy(original)
                for element in node.iter():
                    for key, value in list(element.attrib.items()):
                        if key == "name" or key in references:
                            element.set(key, f"{side}_{value}")
                if tag == "worldbody":
                    node.set("pos", position)
                    node.set("quat", quat)
                    world.append(node)
                else:
                    sections[tag].append(node)
    for node in scene.find("worldbody"):
        world.append(copy.deepcopy(node))
    for node in list(root):
        if node.tag in sections and not len(node):
            root.remove(node)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(assets / "dual_scene.xml", encoding="unicode")


if __name__ == "__main__":
    build()
