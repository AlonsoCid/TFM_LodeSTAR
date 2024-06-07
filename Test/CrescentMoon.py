import deeptrack as dt
import numpy as np
from deeptrack.scatterers import Scatterer
from deeptrack.backend.units import ConversionTable
from deeptrack.types import PropertyLike
from deeptrack import units as u

class CrescentMoon(Scatterer):
    __conversion_table__ = ConversionTable(
        size=(u.meter, u.meter),
        rotation=(u.radian, u.radian),
    )

    def __init__(
        self,
        size: PropertyLike[float] = 1e-6,
        rotation: PropertyLike[float] = 0,
        **kwargs,
    ):
        super().__init__(
            size=size, rotation=rotation, **kwargs
        )

    def _process_properties(self, properties: dict) -> dict:
        properties = super()._process_properties(properties)

        size = np.array(properties["size"])
        if size.ndim == 0:
            size = np.array([properties["size"], properties["size"]])
        properties["size"] = size

        return properties

    def get(self, *ignore, size, rotation, voxel_size, **kwargs):
        # Create a grid to calculate on
        radius = size[0] / 2
        ceil = int(np.ceil(radius / np.min(voxel_size[:2])))
        Y, X = np.meshgrid(
            np.arange(-ceil, ceil) * voxel_size[1],
            np.arange(-ceil, ceil) * voxel_size[0],
        )

        # Grid rotation
        if rotation != 0:
            Xt = X * np.cos(-rotation) + Y * np.sin(-rotation)
            Yt = -X * np.sin(-rotation) + Y * np.cos(-rotation)
            X = Xt
            Y = Yt

        # Create crescent moon
        mask_large_circle = X**2 + Y**2 <= radius**2
        mask_small_circle = (X - radius / 2.5)**2 + Y**2 <= (radius / 1.2)**2
        mask = mask_large_circle & ~mask_small_circle
        mask = mask.astype(float)
        mask = np.expand_dims(mask, axis=-1)
        return mask