from astropy import units as u

__all__ = ['Camera']


class Camera:
    """docstring for Camera"""
    def __init__(self, gain=1.0 * u.electron / u.adu,
                 read_noise=1.0 * u.electron,
                 dark_current=0.01 * u.electron / u.second):
        super().__init__()
        self.gain = gain
        self.read_noise = read_noise
        self.dark_current = dark_current
