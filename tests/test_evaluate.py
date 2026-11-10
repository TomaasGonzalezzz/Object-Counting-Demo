from linecount.evaluate import match_events, score


def gt(event, frame, direction="in", position=100, cls="car"):
    return {"event": str(event), "frame": frame, "direction": direction, "position": position, "class": cls}


def pred(frame, direction="in", position=100, cls="car", track_id=1):
    return {"frame": frame, "direction": direction, "position": position, "class": cls, "track_id": track_id}


def test_matching_respects_direction_and_tolerance():
    g = [gt(1, 100), gt(2, 200, "out")]
    p = [pred(105), pred(200, "in"), pred(205, "out")]
    assert sorted(match_events(g, p, tol_frames=10, tol_px=50)) == [(0, 0), (1, 2)]


def test_score_breakdown():
    g = [gt(1, 100), gt(2, 150), gt(3, 300, position=400)]
    p = [
        pred(102, track_id=1),           # correct
        pred(108, track_id=9),           # second count of event 1 -> double count
        pred(152, cls="truck", track_id=2),  # correct but wrong class
        pred(500, track_id=3),           # nothing there -> false crossing
    ]
    s = score(g, p, (0, 1000), tol_frames=20, tol_px=50)
    assert s["true_count"] == 3 and s["predicted_count"] == 4
    assert (s["correct"], s["missed"], s["double_counts"], s["false_crossings"]) == (2, 1, 1, 1)
    assert s["class_accuracy"] == 0.5
    assert s["pct_error"] == 33.3


def test_ignore_rows_and_window():
    g = [gt(1, 5), gt(2, 100), gt(3, 120, cls="ignore")]
    p = [pred(6), pred(101), pred(121, track_id=5)]
    s = score(g, p, (15, 1000), tol_frames=10, tol_px=50)
    # event 1 is before the window, event 3 is "ignore": only event 2 counts on both sides
    assert s["true_count"] == 1 and s["predicted_count"] == 1
    assert s["correct"] == 1 and s["false_crossings"] == 0
