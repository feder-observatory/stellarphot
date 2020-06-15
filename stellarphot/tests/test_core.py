from ..core import Camera


def test_camera_attributes():
    gain = 2.0
    read_noise = 10
    dark_current = 0.01
    c = Camera(gain=gain, read_noise=read_noise, dark_current=dark_current)
    assert c.gain == gain
    assert c.dark_current == dark_current
    assert c.read_noise == read_noise
