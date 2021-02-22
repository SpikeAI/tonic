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
    
    N_p = len(np.unique(events[:,p_index]))
    n_events = len(events[:,t_index])

    c_int = lambda n, d : ((n - 1) // d) + 1
    
    data = np.zeros((c_int(sensor_size[0],sample_space),
                     c_int(sensor_size[1],sample_space),
                     N_p))

    X = np.zeros((c_int(n_events, sample_events), len(data.ravel())))
    #y = np.zeros((c_int(n_events, sample_events), ))
    for i_event in range(1, n_events):
        if np.exp(-(events[i_event,t_index]-events[i_event-1,t_index])/tau)==0:
            print((events[i_event,t_index]-events[i_event-1,t_index]))
        data *= np.exp(-(events[i_event,t_index]-events[i_event-1,t_index])/tau)

        x_pos = events[i_event,x_index]//sample_space
        y_pos = events[i_event,y_index]//sample_space
        p = events[i_event,p_index]
        data[int(x_pos), int(y_pos), int(p)] = 1.

        if i_event % sample_events == sample_events//2:
            X[i_event//sample_events, :] = data.ravel()
                #y[i_event//sample_events] = events_in[-3][i_event]
    
    return X
