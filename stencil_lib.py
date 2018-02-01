import cv2
import numpy as np
import os
from sklearn.cluster import KMeans


class StencilBuilder:

    def _set_image(self, image):
        # Set up all intermediate images needed to make the stencil
        self._image = np.copy(image)
        self._gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self._flattened_image = np.reshape(self._gray_image, newshape=(-1, 1))

    def __init__(self, image, num_colours=2):
        self._set_image(image)
        self._num_colours = num_colours
        self._stencil = None

    def set_image(self, image):
        self._set_image(image)
        return self

    def set_num_colours(self, num_colours):
        self._num_colours = num_colours
        return self

    def make_stencil(self):
        # Cluster the image into num_colours distinct colours
        kmeans = KMeans(
            n_clusters=self._num_colours,
            random_state=0).fit(self._flattened_image)

        # Recolour the image using the most different shades possible
        vfunc = np.vectorize(lambda index: int(kmeans.cluster_centers_[index]))

        # Map the recolouring function onto the image
        self._stencil = vfunc(np.reshape(kmeans.labels_, newshape=self._gray_image.shape))

        return self

    def get_stencil(self):
        return self._stencil
