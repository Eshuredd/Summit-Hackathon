"""Build convex jaw sections by clipping the original CAD triangles into bands."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull


def build() -> None:
    """Write fitted convex sections, preserving each original mesh transform."""
    assets = Path(__file__).parent / "assets"
    path = assets / "so101.xml"
    tree = ET.parse(path, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    root = tree.getroot()
    specs = (
        (
            "wrist_roll_follower_so101_v1",
            "fixed",
            2,
            [-0.001, 0.02, 0.04, 0.06, 0.075, 0.09, 0.106],
        ),
        ("moving_jaw_so101_v1", "moving", 1, [-0.083, -0.07, -0.055, -0.035, -0.015, 0.011]),
    )
    for source, side, axis, bounds in specs:
        raw = (assets / "meshes" / f"{source}.stl").read_bytes()
        dtype = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
        triangles = np.frombuffer(raw[84:], dtype=dtype)["v"]
        body = next(
            b
            for b in root.iter("body")
            if any(
                g.get("mesh") == source and g.get("class") == "visual" for g in b.findall("geom")
            )
        )
        old = next(
            g
            for g in body.findall("geom")
            if g.get("mesh") == source and g.get("class") == "visual"
        )
        for geom in list(body.findall("geom")):
            if geom.get("class") == "collision" and (
                geom.get("mesh") == source or geom.get("name", "").startswith(f"{side}_")
            ):
                body.remove(geom)
        for mesh in list(root.find("asset").findall("mesh")):
            if mesh.get("name", "").startswith(f"{side}_"):
                root.find("asset").remove(mesh)
        for i, (low, high) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
            points = []
            for triangle in triangles:
                polygon = list(triangle.astype(float))
                for bound, sign in ((low, 1), (high, -1)):
                    clipped = []
                    for a, b in zip(polygon, polygon[1:] + polygon[:1], strict=True):
                        inside_a = sign * (a[axis] - bound) >= 0
                        inside_b = sign * (b[axis] - bound) >= 0
                        if inside_a:
                            clipped.append(a)
                        if inside_a != inside_b:
                            clipped.append(a + (b - a) * ((bound - a[axis]) / (b[axis] - a[axis])))
                    polygon = clipped
                points.extend(polygon)
            vertices = np.unique(np.round(points, 9), axis=0)
            hull = ConvexHull(vertices)
            pad = low >= 0.075 if side == "fixed" else high <= -0.055
            name = f"{side}_{'pad' if pad else 'shell'}_{i}"
            lines = ["v " + " ".join(map(str, p)) for p in vertices]
            lines += ["f " + " ".join(str(j + 1) for j in face) for face in hull.simplices]
            (assets / "meshes" / f"{name}.obj").write_text("\n".join(lines) + "\n")
            ET.SubElement(root.find("asset"), "mesh", name=name, file=f"{name}.obj")
            attributes = dict(old.attrib, mesh=name, name=name)
            attributes["class"] = "collision"
            ET.SubElement(body, "geom", attributes)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="unicode", xml_declaration=True)


if __name__ == "__main__":
    build()
