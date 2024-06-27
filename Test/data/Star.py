import deeptrack as dt
import numpy as np
from deeptrack.scatterers import Scatterer
from deeptrack.backend.units import ConversionTable
from deeptrack.types import PropertyLike
from deeptrack import units as u

class Star(Scatterer):
    """Generates a square scatterer with extended corners like a 4-corner star.

    Parameters
    ----------
    side_length : float
        Length of the sides of the square in meters.
    corner_extension : float
        Length of the extensions from the corners.
    rotation : float
        Orientation angle of the star-square in the camera plane in radians.
    position : array_like[float, float (, float)]
        The position of the particle. Third index is optional,
        and represents the position in the direction normal to the
        camera plane.
    z : float
        The position in the direction normal to the
        camera plane. Used if `position` is of length 2.
    value : float
        A default value of the characteristic of the particle. Used by
        optics unless a more direct property is set: (eg. `refractive_index`
        for `Brightfield` and `intensity` for `Fluorescence`).
    upsample : int
        Upsamples the calculations of the pixel occupancy fraction.
    """

    __conversion_table__ = ConversionTable(
        side_length=(u.meter, u.meter),
        corner_extension=(u.meter, u.meter),
        rotation=(u.radian, u.radian),
    )

    def __init__(
        self,
        side_length: PropertyLike[float] = 1e-6,
        corner_extension: PropertyLike[float] = 1e-6,
        rotation: PropertyLike[float] = 0,
        **kwargs,
    ):
        super().__init__(
            side_length=side_length, corner_extension=corner_extension, rotation=rotation, **kwargs
        )

    def _process_properties(self, properties: dict) -> dict:
        """Preprocess the input to the method .get()

        Ensures that the side length and corner extension are correctly processed.
        """
        properties = super()._process_properties(properties)

        # Ensure side_length and corner_extension are single values
        side_length = np.array(properties["side_length"])
        if side_length.ndim == 0:
            side_length = np.array([properties["side_length"], properties["side_length"]])
        properties["side_length"] = side_length

        corner_extension = np.array(properties["corner_extension"])
        if corner_extension.ndim == 0:
            corner_extension = np.array([properties["corner_extension"], properties["corner_extension"]])
        properties["corner_extension"] = corner_extension

        return properties

    def get(self, *ignore, side_length, corner_extension, rotation, voxel_size, **kwargs):

        # Create a grid to calculate on
        side = side_length[0]
        half_side = side / 2
        ext = corner_extension[0]
        ceil = int(np.ceil((half_side + ext) / np.min(voxel_size[:2])))
        Y, X = np.meshgrid(
            np.arange(-ceil, ceil) * voxel_size[1],
            np.arange(-ceil, ceil) * voxel_size[0],
        )

        # Rotate the grid if needed
        if rotation != 0:
            Xt = X * np.cos(-rotation) + Y * np.sin(-rotation)
            Yt = -X * np.sin(-rotation) + Y * np.cos(-rotation)
            X = Xt
            Y = Yt

        # Evaluate star-square
        mask_square = (np.abs(X) <= half_side) & (np.abs(Y) <= half_side)
        mask_corners = (np.abs(X) >= half_side) & (np.abs(Y) >= half_side) & (np.abs(X) <= (half_side + ext)) & (np.abs(Y) <= (half_side + ext))
        mask = mask_square | mask_corners
        mask = mask.astype(float)
        mask = np.expand_dims(mask, axis=-1)
        return mask