import numpy as np


def make_colour_visualizer(colour_func, height=200, width_scale=1):
    """
    Given a colour function return a numpy array with the grayscale colour and its corresponding mapped colour
    beneath each other.

    :param colour_func:     function colour function to be visualized.
    :return:                numpy.ndarray of shape (height, width_scale*255, 3) representing the colour map.
    """
    _image = np.zeros((height, width_scale*256, 3), dtype=int)

    for col in range(0, _image.shape[1]):
        for row in range(0, _image.shape[0] / 2):
            _image[row][col] = colour_func(col)

        for row in range(_image.shape[0] / 2, _image.shape[0]):
            _image[row][col] = np.asarray([col, col, col])

    return _image


def recolour_image(image, colour_func):
    """

    :param image:
    :param colour_func:
    :return:
    """
    _image = np.zeros((image.shape[0], image.shape[1], 3), dtype=int)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            _image[row][col] = colour_func(image[row][col])
    return _image.astype(np.uint8)


class ColourFuncBuilder:

    def __init__(self):
        # Set default colours to all black
        self.colours = np.asarray([[x, x, x] for x in range(256)])

    def set_interval(self, start_colour, end_colour, start_index=0, end_index=255):
        """

        :param start_colour:
        :param end_colour:
        :param start_index:
        :param end_index:
        :return:
        """

        def interp_colour(shade):
            return start_colour + ((end_colour - start_colour) / 255.0) * shade

        for i in range(start_index, end_index):
            # Calculate the index scaled to the size of the interval
            index = int((1.0 * i - start_index) / (end_index - start_index) * 255)

            # Set the colours in this interval to the correct values
            self.colours[i] = interp_colour(index)

        return self

    def build(self):
        """

        :return:
        """

        colours = np.copy(self.colours)

        def colour_func(shade):
            return colours[shade]

        return colour_func


def test():
    colour_factory = ColourFuncBuilder()
    colour_factory.set_interval(np.asarray([0, 0, 0]), np.asarray([255, 255, 255]))
    cfunc1 = colour_factory.build()

    colour_factory.set_interval(np.asarray([0, 0, 0]), np.asarray([0, 255, 0]))
    cfunc2 = colour_factory.build()

    colour_factory.set_interval(np.asarray([0, 0, 0]), np.asarray([255, 0, 0]), 0, 20)
    colour_factory.set_interval(np.asarray([0, 0, 0]), np.asarray([0, 0, 255]), 30, 40)
    cfunc3 = colour_factory.build()

    for i in range(60):
        print i
        print cfunc1(i)
        print cfunc2(i)
        print cfunc3(i)
        print

if __name__ == '__main__':
    test()
