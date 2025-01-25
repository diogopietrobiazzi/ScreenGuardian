from io import BytesIO

import cv2
import numpy
from PIL import Image


class ImageType:
    """
    Represents an image type that can be stored as:
    - A file path to the image (`str`).
    - A PIL image (Pillow `Image.Image`).
    - An OpenCV matrix (`cv2.typing.MatLike`).

    This class provides methods to convert the image into different formats and manage its internal representation.
    """

    def __init__(self, image: str | Image.Image | cv2.typing.MatLike):
        """
        Represents an image type that can be stored as:
        - A file path to the image (`str`).
        - A PIL image (Pillow `Image.Image`).
        - An OpenCV matrix (`cv2.typing.MatLike`).

        Args:
            image (str | Image.Image | cv2.typing.MatLike):
                The image can be provided as a file path (str),
                a PIL.Image.Image instance, or an OpenCV matrix.
        """
        if isinstance(image, str) and not image.startswith("img"):
            image = f"img/{image}"

        self.image = image

    def __call__(self) -> str | Image.Image | cv2.typing.MatLike:
        """
        Returns the internal representation of the image.

        Returns:
            str | Image.Image | cv2.typing.MatLike: The image stored in the instance.
        """
        return self.image

    def __repr__(self) -> str:
        """
        Returns a textual representation of the instance for debugging.

        Returns:
            str: Detailed representation of the instance.
        """
        return f"ImageType({repr(self.image)})"

    def __str__(self) -> str:
        """
        Returns the string representation of the stored image.

        Returns:
            str: String representation of the image.
        """
        return str(self.image)

    def as_bytes(self, format: str = "png") -> bytes:
        """
        Converts the image to a byte format of `format` parameter.

        Args:
            format (str): The image's format that will be used when converting it into bytes. Default is "png".

        Returns:
            bytes: The image in PNG format.

        Raises:
            ValueError: If the image type is not supported.
        """
        if isinstance(self.image, numpy.ndarray):
            _, image = cv2.imencode(f".{format.lower()}", self.image)
            return image.tobytes()
        elif isinstance(self.image, str):
            image = Image.open(self.image)
        elif isinstance(self.image, Image.Image):
            image = self.image
        else:
            raise ValueError("Unsupported image type for byte conversion.")

        img_bytes = BytesIO()
        image.save(img_bytes, format=format.upper())
        return img_bytes.getvalue()

    def as_cv2_image(self) -> cv2.typing.MatLike:
        """
        Converts the image to an OpenCV-compatible format (numpy.ndarray).

        - If the image is a file path (str), it is read using OpenCV.
        - If the image is a PIL instance (Image.Image), it is converted to BGR (OpenCV format).
        - If the image is already an OpenCV matrix, it is returned directly.

        Returns:
            cv2.typing.MatLike: The image in OpenCV format.

        Raises:
            ValueError: If the image type is not supported for conversion.
        """
        if isinstance(self.image, str):
            return cv2.imread(self.image)
        elif isinstance(self.image, Image.Image):
            return cv2.cvtColor(numpy.array(self.image), cv2.COLOR_RGB2BGR)
        elif isinstance(self.image, numpy.ndarray):
            return self.image

        raise ValueError("Unsupported image type for OpenCV conversion.")

    def as_pil_image(self) -> Image.Image:
        """
        Converts the image to a PIL-compatible format (Image.Image).

        - If the image is a file path (str), it is opened using PIL.
        - If the image is an OpenCV matrix (numpy.ndarray), it is converted to RGB and then to a PIL object.
        - If the image is already a PIL instance, it is returned directly.

        Returns:
            Image.Image: The image in PIL format.

        Raises:
            ValueError: If the image type is not supported for conversion.
        """
        if isinstance(self.image, str):
            return Image.open(self.image)
        elif isinstance(self.image, numpy.ndarray):
            return Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        elif isinstance(self.image, Image.Image):
            return self.image

        raise ValueError("Unsupported image type for PIL conversion.")


class MSSRegion:
    """
    A region compatible with `MSS`, represented as a dictionary with the keys
    `left` (x-axis), `top` (y-axis), `width` (width), and `height` (height).

    >>> mss_region = MSSRegion(100, 200, 300, 400)
    """

    def __init__(self, left: int, top: int, width: int, height: int):
        """
        A region compatible with `MSS`, represented as a dictionary with the keys
        `left` (x-axis), `top` (y-axis), `width` (width), and `height` (height).

        >>> mss_region = MSSRegion(100, 200, 300, 400)

        Args:
            left (int): The x-coordinate (left) of the region.
            top (int): The y-coordinate (top) of the region.
            width (int): The width of the region.
            height (int): The height of the region.

        Raises:
            ValueError: If width or height are less than or equal to zero.
        """
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be greater than zero.")
        self.left = left
        self.top = top
        self.width = width
        self.height = height

    def __call__(self) -> dict[str, int]:
        """
        Returns the `MSSRegion` as a dictionary compatible with the `mss` library when it's instanciated.

        Returns:
            dict[str, int]: Dictionary with the keys `left`, `top`, `width`, `height`.
        """
        return self.as_dict()

    def __repr__(self) -> str:
        """
        Returns the string representation of the MSSRegion instance.

        Returns:
            str: Representation of the region in the format:
                 MSSRegion(left=..., top=..., width=..., height=...)
        """
        return (
            f"MSSRegion(left={self.left}, top={self.top}, "
            f"width={self.width}, height={self.height})"
        )

    def as_dict(self) -> dict[str, int]:
        """
        Returns the region as a dictionary with the keys `left`, `top`, `width`, and `height`.

        Returns:
            dict[str, int]: Dictionary representing the MSS region.
        """
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


class Region:
    """
    Represents a screen region with x, y, width, and height coordinates.

    >>> region = Region(100, 200, 300, 400)
    """

    def __init__(self, x: int, y: int, width: int, height: int):
        """
        Represents a screen region with x, y, width, and height coordinates.

        >>> region = Region(100, 200, 300, 400)

        Args:
            x (int): The x-coordinate of the top-left corner of the region.
            y (int): The y-coordinate of the top-left corner of the region.
            width (int): The width of the region.
            height (int): The height of the region.

        Raises:
            ValueError: If width or height are less than or equal to zero.
        """
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be greater than zero.")
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __call__(self) -> list[int]:
        """
        Returns the region as a list of integers when instanciated.

        Returns:
            list[int]: List in the format [x, y, width, height].
        """
        return self.as_list()

    def __repr__(self) -> str:
        """
        Returns the representation of the Region instance.

        Returns:
            str: Representation in the format:
                 Region(x=..., y=..., width=..., height=...)
        """
        return (
            f"Region(x={self.x}, y={self.y}, width={self.width}, height={self.height})"
        )

    def as_list(self) -> list[int]:
        """
        Converts the region into a list of integers.

        Returns:
            list[int]: List containing the coordinates and dimensions [x, y, width, height].
        """
        return [self.x, self.y, self.width, self.height]

    def as_mss_region(self) -> MSSRegion:
        """
        Converts the current region into an `MSSRegion` object compatible with the `mss` library.

        Returns:
            MSSRegion: An object containing the region dimensions (x, y, width, height)
            in the format accepted by the `mss` library.
        """
        return MSSRegion(
            left=self.x,
            top=self.y,
            width=self.width,
            height=self.height,
        )


def singleton(cls):
    """
    A decorator to ensure a class follows the Singleton pattern.

    The Singleton pattern ensures that a class has only one instance,
    even if instantiated multiple times, and provides a global access point
    to that instance.

    Args:
        cls (type): The class to be transformed into a Singleton.

    Returns:
        function: The `get_instance` function that manages the single instance of the class.

    Example:
        >>> @singleton
        >>> class MyClass:
        >>>     pass
    """
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance
