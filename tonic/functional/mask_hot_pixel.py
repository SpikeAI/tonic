from .utils import guess_event_ordering_numpy


def mask_hot_pixel(events, coordinates, sensor_size, ordering=None):
    """Drops events for certain pixel locations, to suppress pixels that constantly fire (e.g. due to faulty hardware).

    Args:
        events: ndarray of shape [num_events, num_event_channels]
        coordinates: list of (x,y) coordinates for which all events will be deleted.
        sensor_size: size of the sensor that was used [W,H]
        ordering: ordering of the event tuple inside of events, if None
                  the system will take a guess through
                  guess_event_ordering_numpy. This function requires 'x' and
                  'y' to be in the ordering

    Returns:
        masked set of events.
    """

    if ordering is None:
        ordering = guess_event_ordering_numpy(events)
    assert "x" and "y" in ordering

    x_index = ordering.find("x")
    y_index = ordering.find("y")

    for (x, y) in coordinates:
        xs = events[:, x_index] == x
        ys = events[:, y_index] == y
        events = events[~(xs & ys)]

    return events
