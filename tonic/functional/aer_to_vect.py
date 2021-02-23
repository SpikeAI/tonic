import numpy as np

from .utils import guess_event_ordering_numpy


def aer_to_vect(
    events,
    cumulate,
    tau,
    sample_events,
    sample_space,
    sensor_size,
    ordering=None,
    use_ravel=True
):
    """

    Args:
        -

    Returns:
        -
    """

    if ordering is None:
        ordering = guess_event_ordering_numpy(events)
    assert "x" and "y" in ordering

    x_index = ordering.find("x")
    y_index = ordering.find("y")
    t_index = ordering.find("t")
    p_index = ordering.find("p")

    N_p = len(np.unique(events[:, p_index]))
    n_events = len(events[:, t_index])

    c_int = lambda n, d : ((n - 1) // d) + 1

    # presynaptic potential
    Vm = np.zeros((c_int(sensor_size[0],sample_space),
                     c_int(sensor_size[1],sample_space),
                     N_p))

    # what is recorded
    if use_ravel:
        X = np.zeros((c_int(n_events, sample_events),
                      len(Vm.ravel())
                    ))
    else:
        X = np.zeros((c_int(n_events, sample_events),
                      c_int(sensor_size[0],sample_space),
                      c_int(sensor_size[1],sample_space),
                      N_p
                    ))

    for i_event in range(n_events):
        if i_event>0:
            dt = events[i_event, t_index]-events[i_event-1, t_index]
            Vm *= np.exp(-dt/tau)

        x_pos = events[i_event,x_index]//sample_space
        y_pos = events[i_event,y_index]//sample_space
        p = events[i_event,p_index]

        Vm[i_event, x_pos, y_pos, p] = 1.

        if i_event % sample_events == sample_events//2:
            if use_ravel:
                X[i_event//sample_events, :] = Vm.ravel()
            else:
                X[i_event//sample_events, :] = Vm

    return X
