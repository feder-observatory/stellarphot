from astropy import units as u

__all__ = ['Camera']


class Camera:
    """
    A class to represent a CCD-based camera.

    Parameters
    ----------
    gain : `astropy.quantity.Quantity`
        The gain of the camera in units such that the unit of the product `gain` times the image data matches the unit of the `read_noise`.
    read_noise : `astropy.quantity.Quantity`
        The read noise of the camera in units of electrons.
    dark_current : `astropy.quantity.Quantity`
        The dark current of the camera in units such that, when multiplied by exposure time, the unit matches the unit of the `read_noise`.

    Attributes
    ----------
    gain : float times u.electron / u.adu,
        The gain of the camera in units of electrons per ADU.
    read_noise : float times u.electron
        The read noise of the camera in units of electrons.
    dark_current : float times u.electron / u.second
        The dark current of the camera in units of electrons per second.

    Notes
    -----
    The gain, read noise, and dark current are all assumed to be constant
    across the entire CCD.

    Examples
    --------
    >>> from astropy import units as u
    >>> from stellarphot import Camera
    >>> camera = Camera(gain=1.0 * u.electron / u.adu,
    ...                 read_noise=1.0 * u.electron,
    ...                 dark_current=0.01 * u.electron / u.second)
    >>> camera.gain
    <Quantity 1. electron / adu>
    >>> camera.read_noise
    <Quantity 1. electron>
    >>> camera.dark_current
    <Quantity 0.01 electron / s>

    """
    def __init__(self, gain=1.0 * u.electron / u.adu,
                 read_noise=1.0 * u.electron,
                 dark_current=0.01 * u.electron / u.second):
        super().__init__()
        self.gain = gain
        self.read_noise = read_noise
        self.dark_current = dark_current
