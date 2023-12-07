from math import pi, radians, sqrt

import pytest

from compas.geometry import Polyhedron
from compas.geometry import Polygon
from compas.geometry import Rotation
from compas.geometry import allclose
from compas.geometry import angle_vectors
from compas.geometry import angle_planes
from compas.geometry import angles_vectors
from compas.geometry import centroid_points
from compas.geometry import centroid_polyhedron
from compas.geometry import close
from compas.geometry import length_vector
from compas.geometry import subtract_vectors
from compas.geometry import volume_polyhedron
from compas.geometry import area_polygon
from compas.geometry import area_polygon_xy
from compas.geometry import area_triangle
from compas.geometry import area_triangle_xy
from compas.geometry import normal_polygon


@pytest.fixture
def R():
    return Rotation.from_axis_and_angle([0, 1, 0], radians(-90))


@pytest.fixture
def square():
    return Polygon.from_sides_and_radius_xy(4, sqrt(0.5**2 + 0.5**2))


@pytest.fixture
def triangle():
    return Polygon([[0, 0, 0], [1, 0, 0], [0, 1, 0]])


# ==============================================================================
# angles
# ==============================================================================


@pytest.mark.parametrize(
    ("u", "v", "angle"),
    [
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0),  # X - X
        ([0.0, 1.0, 0.0], [0.0, 1.0, 0.0], 0.0),  # Y - Y
        ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 0.0),  # Z - Z
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], pi / 2),  # X - Y
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], pi / 2),  # Y - X
        ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], pi / 2),  # X - Z
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], pi / 2),  # Z - X
        ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], pi / 2),  # Y - Z
        ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0], pi / 2),  # Z - Y
    ],
)
def test_angle_vectors(u, v, angle):
    assert close(angle_vectors(u, v), angle)


@pytest.mark.parametrize(
    ("u", "v"),
    [
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ],
)
def test_angle_vectors_fails_when_input_is_zero(u, v):
    with pytest.raises(ZeroDivisionError):
        angle_vectors(u, v)


@pytest.mark.parametrize(
    ("u", "v", "angles"),
    [
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], (0.0, 2 * pi)),  # X - X
        ([0.0, 1.0, 0.0], [0.0, 1.0, 0.0], (0.0, 2 * pi)),  # Y - Y
        ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], (0.0, 2 * pi)),  # Z - Z
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], (pi / 2, 3 * pi / 2)),  # X - Y
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], (pi / 2, 3 * pi / 2)),  # Y - X
        ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], (pi / 2, 3 * pi / 2)),  # X - Z
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], (pi / 2, 3 * pi / 2)),  # Z - X
        ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], (pi / 2, 3 * pi / 2)),  # Y - Z
        ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0], (pi / 2, 3 * pi / 2)),  # Z - Y
    ],
)
def test_angles_vectors(u, v, angles):
    a, b = angles
    assert allclose(angles_vectors(u, v), (a, b))


@pytest.mark.parametrize(
    ("u", "v"),
    [
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ],
)
def test_angles_vectors_fails_when_input_is_zero(u, v):
    with pytest.raises(ZeroDivisionError):
        angles_vectors(u, v)


@pytest.mark.parametrize(
    ("a", "b", "angle"),
    [
        (
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            0.5 * pi,
        ),
        (
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            0.5 * pi,
        ),
        ([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, +1.0]], 0.0),
        ([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0]], pi),
    ],
)
def test_angle_planes(a, b, angle):
    assert close(angle_planes(a, b), angle)


# ==============================================================================
# average
# ==============================================================================


@pytest.mark.parametrize(
    ("points", "centroid"),
    [
        ([[0.0, 0.0, 0.0]], [0.0, 0.0, 0.0]),
        ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [0.0, 0.0, 0.0]),
        ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [0.0, 0.0, 0.0]),
        ([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 1.0, 0.0]], [0.1666, 0.6666, 0.0]),
        (
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [0.5, 0.5, 0.0],
        ),
        (
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.1, 0.0],
            ],
            [0.35, 0.35, 0.0],
        ),
    ],
)
def test_centroid_points(points, centroid):
    if centroid is None:
        x, y, z = 0.0, 0.0, 0.0
    else:
        x, y, z = centroid
    assert allclose(centroid_points(points), (x, y, z), tol=1e-03)


@pytest.mark.parametrize(("points"), [[0.0, 0.0, 0.0]])
def test_centroid_points_fails_when_input_is_not_list_of_lists(points):
    with pytest.raises(TypeError):
        centroid_points(points)


@pytest.mark.parametrize(
    ("points"),
    [
        [[0.0, 0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0]],
    ],
)
def test_centroid_points_fails_when_input_is_not_complete_points(points):
    with pytest.raises(ValueError):
        centroid_points(points)


@pytest.mark.parametrize(
    ("polyhedron", "centroid"),
    [
        (Polyhedron.from_platonicsolid(6), [0.0, 0.0, 0.0]),
    ],
)
def test_centroid_polyhedron(polyhedron, centroid):
    x, y, z = centroid
    assert allclose(centroid_polyhedron(polyhedron), (x, y, z))


# ==============================================================================
# size
# ==============================================================================


@pytest.mark.parametrize(("polyhedron", "volume"), [(Polyhedron.from_platonicsolid(6), None)])
def test_volume_polyhedron(polyhedron, volume):
    if volume is None:
        L = length_vector(subtract_vectors(polyhedron.vertices[0], polyhedron.vertices[1]))
        volume = L * L * L
    V = volume_polyhedron(polyhedron)
    assert close(V, volume)


def test_area_square(square, R):
    assert close(area_polygon(square.points), 1)
    assert close(area_polygon_xy(square.points), 1)
    assert close(square.area, 1)
    square.transform(R)
    assert close(area_polygon(square.points), 1)
    assert close(area_polygon_xy(square.points), 0)
    assert close(square.area, 1)


def test_area_triangle(triangle, R):
    assert close(area_polygon(triangle.points), 0.5)
    assert close(area_polygon_xy(triangle.points), 0.5)
    assert close(area_triangle(triangle.points), 0.5)
    assert close(area_triangle_xy(triangle.points), 0.5)
    assert close(triangle.area, 0.5)
    triangle.transform(R)
    assert close(area_polygon(triangle.points), 0.5)
    assert close(area_polygon_xy(triangle.points), 0.0)
    assert close(area_triangle(triangle.points), 0.5)
    assert close(area_triangle_xy(triangle.points), 0.0)
    assert close(triangle.area, 0.5)


# ==============================================================================
# normals
# ==============================================================================\


def test_normal_polygon():
    polygon = [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)]
    normal = normal_polygon(polygon, unitized=False)
    area = area_polygon(polygon)
    assert close(area, 100.0)
    assert close(area, length_vector(normal))
