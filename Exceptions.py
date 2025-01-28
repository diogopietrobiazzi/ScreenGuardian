from Typing import ImageType, Region


class ImageComparisonError(Exception):
    def __init__(
        self,
        error_message: str,
        expected_images: ImageType | list[ImageType],
        images_to_search_into: ImageType | Region | list[ImageType] | list[Region],
    ):
        """Error indicating failure to locate the expected image within the searched image.

        Returns the traceback of previously called methods and adds the images to the test report (*Allure*).

        Args:
            error_message (str): Message explaining the error that occurred.
            expected_images (ImageType | list[ImageType]): Expected image or list of images.
            images_to_search_into (ImageType | Region | list[ImageType] | list[Region]): Image or list of images in which the `expected_images` were searched.
        """
        super().__init__(error_message)

        # TODO: Create logic to send images and traceback to test reports


class TextComparisonError(Exception):
    def __init__(
        self,
        expected_text: str,
        captured_texts: str | list[str],
        captured_text_images: ImageType | list[ImageType] | None = None,
        error_message: str | None = None,
    ):
        """Error indicating failures in text comparisons.

        Adds information about the compared texts to the error, as well as attaching the traceback
        and the text image to the test report.

        Args:
            expected_text (str): Text that is expected to be found.
            captured_texts (str | list[str]): Text or list of texts obtained.
            captured_text_image (ImageType | None): Image of the captured text.
            error_message (str | None): Additional error message.
        """
        # Formatting the error message depending on the type of captured_texts
        if isinstance(captured_texts, list):
            captured_texts = ",\n".join([f"- '{text}';" for text in captured_texts])
            message = f"Text not found! The text '{expected_text}' was not found among the following texts: \n\n{captured_texts}"
        else:
            message = f"Different texts! The expected text was '{expected_text}' but the captured text was '{captured_texts}'."

        if error_message:
            message = f"{error_message}\n\n{message}"

        super().__init__(message)

        # TODO: Create logic to send images and traceback to test reports
