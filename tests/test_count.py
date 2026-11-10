import numpy as np
import pytest

from linecount.count import Line, LineCounter, anchor_points


def feed(counter, track_id, xs, ys, name="car", start=0):
    for k, (x, y) in enumerate(zip(xs, ys)):
        counter.update(start + k, np.array([track_id]), np.array([[x, y]]), [name])


def test_parse_and_geometry():
    line = Line.parse("0,100,200,100")
    assert line.length == 200
    # below a left-to-right horizontal line is the right-hand side (positive)
    assert line.signed_distance(np.array([[50, 120]]))[0] == pytest.approx(20)
    assert line.signed_distance(np.array([[50, 80]]))[0] == pytest.approx(-20)
    assert line.position(np.array([[50, 0]]))[0] == pytest.approx(50)
    with pytest.raises(ValueError):
        Line.parse("1,2,3")
    with pytest.raises(ValueError):
        Line.parse("5,5,5,5")


def test_downward_crossing_is_in_upward_is_out():
    c = LineCounter(Line(0, 100, 200, 100), margin=2)
    feed(c, 1, [50] * 5, [80, 90, 99, 110, 120])  # moves down
    feed(c, 2, [150] * 5, [130, 115, 100, 90, 70])  # moves up
    assert [(e.track_id, e.direction) for e in c.events] == [(1, "in"), (2, "out")]
    assert c.counts() == {"car": {"in": 1, "out": 1}}
    assert c.events[0].frame == 3  # first frame confirmed beyond the dead zone


def test_jitter_on_the_line_is_not_counted():
    c = LineCounter(Line(0, 100, 200, 100), margin=4)
    feed(c, 1, [50] * 8, [90, 98, 101, 99, 102, 97, 103, 99])  # never leaves the dead zone on the far side
    assert c.total == 0


def test_object_counted_once_even_if_it_comes_back():
    c = LineCounter(Line(0, 100, 200, 100), margin=2)
    feed(c, 1, [50] * 6, [80, 120, 80, 120, 80, 120])
    assert c.total == 1
    c2 = LineCounter(Line(0, 100, 200, 100), margin=2, once_per_track=False)
    feed(c2, 1, [50] * 6, [80, 120, 80, 120, 80, 120])
    assert c2.total == 5


def test_crossing_beside_the_segment_is_ignored():
    c = LineCounter(Line(0, 100, 200, 100), margin=2)
    feed(c, 1, [300] * 4, [80, 90, 110, 120])  # passes to the right of the segment's end
    assert c.total == 0


def test_track_that_starts_beyond_the_line_is_not_counted():
    # a new ID that appears already past the line (e.g. after an ID switch) has no "before" side
    c = LineCounter(Line(0, 100, 200, 100), margin=2)
    feed(c, 7, [50] * 3, [120, 130, 140])
    assert c.total == 0


def test_id_switch_on_the_line_gives_a_double_count():
    # documents a real failure mode: the old ID crosses, a new ID starts behind the line and crosses again
    c = LineCounter(Line(0, 100, 200, 100), margin=2)
    feed(c, 1, [50] * 3, [90, 95, 105])
    feed(c, 2, [50] * 3, [96, 98, 106], start=3)
    assert c.total == 2


def test_class_is_the_weighted_vote_over_the_track():
    c = LineCounter(Line(0, 100, 200, 100), margin=2)
    names = ["truck", "car", "car", "car"]
    for k, (y, n) in enumerate(zip([80, 90, 95, 110], names)):
        c.update(k, np.array([1]), np.array([[50, y]]), [n], np.array([0.9]))
    assert c.events[0].class_name == "car"


def test_vertical_line_direction():
    # drawn top to bottom: the right-hand side on screen is the left of the image, so moving left is "in"
    c = LineCounter(Line(100, 0, 100, 200), margin=2)
    feed(c, 1, [130, 110, 90, 70], [100] * 4)
    feed(c, 2, [70, 90, 110, 130], [50] * 4)
    assert [e.direction for e in c.events] == ["in", "out"]


def test_anchor_points():
    boxes = np.array([[10, 20, 30, 60]])
    assert anchor_points(boxes, "bottom_center").tolist() == [[20, 60]]
    assert anchor_points(boxes, "center").tolist() == [[20, 40]]
