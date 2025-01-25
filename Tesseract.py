# -*- coding: utf-8 -*-
import os
from types import TracebackType

from PIL import Image
from tesserocr import OEM, PSM, PyTessBaseAPI

from Typing import ImageType


class Tesseract:
    def __init__(
        self,
        psm: PSM = PSM.SINGLE_LINE,
        oem: OEM = OEM.DEFAULT,
    ) -> None:
        """
        Initializes the Tesseract object for text reading.

        Args:
            path (str): Path where the **.traineddata** file for the specified `lang` is located.
            lang (str): Language for text recognition.
            psm (int): Page segmentation mode.
            oem (int): OCR engine mode.
        """
        tessdata = os.getenv("TESSDATA_PREFIX")
        language = os.getenv("TESSDATA_LANG")

        if not tessdata:
            raise EnvironmentError(
                'The Environment Variable "TESSDATA_PREFIX" is not defined or is empty.'
            )
        elif not tessdata.endswith("/"):
            tessdata += "/"

        if not language:
            raise EnvironmentError(
                'The Environment Variable "TESSDATA_PREFIX" is not defined or is empty.'
            )
        elif not language.endswith("/"):
            language += "/"

        self.tesseract_api = PyTessBaseAPI(
            path=tessdata,  # type: ignore
            lang=language,  # type: ignore
            oem=oem,  # type: ignore
            psm=psm,  # type: ignore
        )

    def __enter__(self):
        """
        Starts the Tesseract context and returns the instance.

        This method prepares Tesseract to be used within a `with` block.

        Returns:
            Tesseract: The Tesseract instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_value: Exception | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Finalizes the Tesseract context and releases resources.

        This method is called at the end of the `with` block to ensure that
        Tesseract is properly terminated, releasing any allocated resources.

        Args:
            exc_type (type | None): The type of the exception that occurred (if any).
            exc_value (Exception | None): The instance of the exception (if any).
            traceback (TracebackType | None): The traceback object of the exception (if any).
        """
        self.tesseract_api.End()

    def __del__(self):
        """
        Class destructor. Called as soon as an instance of the class is no longer used/has completed its lifecycle.
        """
        self.tesseract_api.End()

    def GetTextFromImage(self, image: ImageType) -> str:
        """
        Extracts text from the provided image.

        Args:
            image (ImageType): The image from which the text will be extracted.

        Returns:
            str: The text extracted from the image.
        """
        self.SetImage(image)
        text = self.tesseract_api.GetUTF8Text().strip()

        self.tesseract_api.Clear()

        return text

    def SetImage(self, image: ImageType, resize_small_images: bool = True) -> None:
        """
        Sets the image to be processed by Tesseract.

        Args:
            image (ImageType): The image to be configured for text recognition.
            resize_small_images (bool): Controls if the image will be resized if it's too small (height with less than 30 pixels)
        """
        image_pil = image.as_pil_image()

        if resize_small_images:
            width, height = image_pil.size

            if height < 30:
                image_pil = image_pil.resize(
                    (width * 2, height * 2),
                    Image.Resampling.LANCZOS,
                )

        self.tesseract_api.SetImage(image_pil)

    def SetVariable(self, name: str, value: str) -> None:
        """
        Sets a configuration variable for Tesseract.

        Args:
            name (str): The name of the variable to be configured.
            value (str): The value to be assigned to the variable.
        """
        self.tesseract_api.SetVariable(name, value)

    def SetBlacklist(self, blacklist: str) -> None:
        """
        Specifies the `blacklist` string in Tesseract's character blacklist.

        Args:
            blacklist (str): Characters that will not be recognized by Tesseract.
        """
        self.SetVariable("tessedit_char_blacklist", blacklist)
