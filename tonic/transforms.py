from . import functional

class Compose:
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transform(s) to compose. Even when using a single transform, the Compose wrapper is necessary.

    Example:
        >>> transforms.Compose([
        >>>     transforms.Denoise(),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        for t in self.transforms:
            events, images = t(
                events=events,
                images=images,
                sensor_size=sensor_size,
                ordering=ordering,
                multi_image=multi_image,
            )
        if multi_image and images.all() != None:
            return events, images
        else:
            return events

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Crop:
    """Crops the sensor size to a smaller sensor and removes events outsize of the target sensor and maps."""

    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        return functional.crop_numpy(
            events, images, sensor_size, ordering, self.target_size, multi_image
        )


class DropEvents:
    """Drops events with  a certain probability."""

    def __init__(self, drop_probability=0.5, random_drop_probability=False):
        self.drop_probability = drop_probability
        self.random_drop_probability = random_drop_probability

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.drop_events_numpy(
            events, self.drop_probability, self.random_drop_probability
        )
        return events, images


class FlipLR:
    """Mirrors x coordinates of events and images (if present)"""

    def __init__(self, flip_probability=0.5):
        self.flip_probability_lr = flip_probability

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        return functional.flip_lr_numpy(
            events, images, sensor_size, ordering, self.flip_probability_lr, multi_image
        )


class FlipPolarity:
    """Changes polarities 1 to -1 and polarities [-1, 0] to 1"""

    def __init__(self, flip_probability=0.5):
        self.flip_probability_pol = flip_probability

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.flip_polarity_numpy(
            events, self.flip_probability_pol, ordering
        )
        return events, images


class FlipUD:
    """Mirrors y coordinates of events and images (if present)"""

    def __init__(self, flip_probability=0.5):
        self.flip_probability_ud = flip_probability

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        return functional.flip_ud_numpy(
            events, images, sensor_size, ordering, self.flip_probability_ud, multi_image
        )


class Denoise:
    """Cycles through all events and drops it if there is no other event within
    a time of time_filter and a spatial neighbourhood of 1."""

    def __init__(self, time_filter=10000):
        self.time_filter = time_filter

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.denoise_numpy(
            events, sensor_size, ordering, self.time_filter
        )
        return events, images


class MaskHotPixel:
    """Drops events for certain pixel locations, to suppress pixels that constantly fire
    (e.g. due to faulty hardware)."""

    def __init__(self, coordinates):
        self.coordinates = coordinates

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.mask_hot_pixel(
            events, self.coordinates, sensor_size, ordering
        )
        return events, images


class RefractoryPeriod:
    """Cycles through all events and drops event if within refractory period for
    that pixel."""

    def __init__(self, refractory_period=0.5):
        self.refractory_period = refractory_period

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.refractory_period_numpy(
            events, sensor_size, ordering, self.refractory_period
        )
        return events, images


class SpatialJitter:
    """Blurs x and y coordinates of events. Integer or subpixel precision possible."""

    def __init__(
        self,
        variance_x=1,
        variance_y=1,
        sigma_x_y=0,
        integer_coordinates=True,
        clip_outliers=True,
    ):
        self.variance_x = variance_x
        self.variance_y = variance_y
        self.sigma_x_y = sigma_x_y
        self.integer_coordinates = integer_coordinates
        self.clip_outliers = clip_outliers

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.spatial_jitter_numpy(
            events,
            sensor_size,
            ordering,
            self.variance_x,
            self.variance_y,
            self.sigma_x_y,
            self.integer_coordinates,
            self.clip_outliers,
        )
        return events, images


class SpatioTemporalTransform:
    """Choose arbitrary spatial and temporal transformation."""

    def __init__(self, spatial_transform, temporal_transform, roll=False):
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.roll = roll

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.st_transform(
            events,
            sensor_size,
            ordering,
            self.spatial_transform,
            self.temporal_transform,
            self.roll,
        )
        return events, images


class TimeJitter:
    """Blurs timestamps of events. Will clip negative timestamps by default."""

    def __init__(
        self, std=1, integer_jitter=False, clip_negative=False, shift_first_as_zero=False, sort_timestamps=False,
    ):
        self.std = std
        self.integer_jitter = integer_jitter
        self.clip_negative = clip_negative
        self.shift_first_as_zero = shift_first_as_zero
        self.sort_timestamps = sort_timestamps

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.time_jitter_numpy(
            events,
            ordering,
            self.std,
            self.integer_jitter,
            self.clip_negative,
            self.shift_first_as_zero,
            self.sort_timestamps,
        )
        return events, images


class TimeReversal:
    """Will reverse the timestamps of events with a certain probability."""

    def __init__(self, flip_probability=0.5):
        self.flip_probability_t = flip_probability

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        return functional.time_reversal_numpy(
            events, images, sensor_size, ordering, self.flip_probability_t, multi_image
        )


class TimeSkew:
    """Scale and/or offset all timestamps."""

    def __init__(self, coefficient=0.9, offset=0):
        self.coefficient = coefficient
        self.offset = offset

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.time_skew_numpy(
            events, ordering, self.coefficient, self.offset
        )
        return events, images


class UniformNoise:
    """Inject noise events."""

    def __init__(self, noise_density=1e-8):
        self.noise_density = noise_density

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        surfaces = functional.uniform_noise_numpy(
            events, sensor_size, ordering, self.noise_density
        )
        return events, images


class ToAveragedTimesurface:
    """Creates Averaged Time Surfaces."""

    def __init__(
        self,
        cell_size=10,
        surface_size=7,
        temporal_window=5e5,
        tau=5e3,
        decay="lin",
        merge_polarities=False,
    ):
        assert surface_size % 2 == 1
        self.cell_size = cell_size
        self.surface_size = surface_size
        self.temporal_window = temporal_window
        self.tau = tau
        self.decay = decay
        self.merge_polarities = merge_polarities

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        surfaces = functional.to_averaged_timesurface(
            events,
            sensor_size,
            ordering,
            self.cell_size,
            self.surface_size,
            self.temporal_window,
            self.tau,
            self.decay,
            self.merge_polarities,
        )
        return surfaces, images


class ToRatecodedFrame:
    """Bin events to frames."""

    def __init__(self, frame_time=5000, merge_polarities=True):
        self.frame_time = frame_time
        self.merge_polarities = merge_polarities

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        frames = functional.to_ratecoded_frame_numpy(
            events,
            sensor_size,
            ordering,
            frame_time=self.frame_time,
            merge_polarities=self.merge_polarities,
        )
        return frames, images


class ToSparseTensor:
    """Turn event array (N,E) into sparse Tensor (B,T,W,H)."""

    def __init__(self, merge_polarities=False):
        self.merge_polarities = merge_polarities

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        tensor = functional.to_sparse_tensor_pytorch(
            events, sensor_size, ordering, merge_polarities=self.merge_polarities
        )
        return tensor, images


class ToTimesurface:
    """Create Time surfaces for each event."""

    def __init__(
        self, surface_dimensions=(7, 7), tau=5e3, decay="lin", merge_polarities=False
    ):
        assert len(surface_dimensions) == 2
        assert surface_dimensions[0] % 2 == 1 and surface_dimensions[1] % 2 == 1
        self.surface_dimensions = surface_dimensions
        self.tau = tau
        self.decay = decay
        self.merge_polarities = merge_polarities

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        surfaces = functional.to_timesurface_numpy(
            events,
            sensor_size,
            ordering,
            self.surface_dimensions,
            self.tau,
            self.decay,
            self.merge_polarities,
        )
        return surfaces, images


class Repeat:
    """Copies target n times. Useful to transform sample labels into sequences."""

    def __init__(self, repetitions):
        self.repetitions = repetitions

    def __call__(self, target):
        return np.tile(np.expand_dims(target, 0), [self.repetitions, 1])


class ToOneHotEncoding:
    """Transforms one or more targets into a one hot encoding scheme."""

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, target):
        return np.eye(self.n_classes)[target]


class NumpyAsType(object):
    def __init__(self, cast_to):
        self.cast_to = cast_to

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = events.astype(self.cast_to)

        if images is not None:
            images = images.astype(self.cast_to)

        return events, images

class AERtoVector:
    """Transforms one or more targets into ((N_xgrid * N_ygrid * N_polarities) * N_events) matrices where all events are mapped as a function of their address (column) and indice (line). An exponential decay is applied on their timing. """

    def __init__(self, nb_pola = 2, tau=150000, sample_event=1, sample_space=1, cumulate=False):
        self.tau = tau
        self.cumulate = cumulate
        self.sample_event = sample_event
        self.sample_space = sample_space
        self.nb_pola = nb_pola

    def __call__(self, events, sensor_size, ordering, images=None, multi_image=None):
        events = functional.aer_to_vect(
            events, self.cumulate, self.tau, self.sample_event, self.sample_space, sensor_size, self.nb_pola, ordering, 
        )
        return events, images
