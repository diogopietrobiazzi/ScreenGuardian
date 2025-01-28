import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from operator import attrgetter
from queue import Queue

import cv2
import cv2.typing
import numpy
import numpy.typing

from Exceptions import ImageComparisonError, TextComparisonError
from ImageCapture import ImageCapture
from Tesseract import Tesseract
from Typing import ImageType, MSSRegion, Region, singleton


@singleton
class Screen:
    def __init__(self) -> None:
        self.number_threads = os.cpu_count() or 1
        self.queue_tesseract: Queue[Tesseract] = Queue(self.number_threads)

        for _ in range(self.number_threads):
            self.queue_tesseract.put(Tesseract())

    def any_of_images_exists_in_region(
        self,
        images: list[ImageType],
        region: Region,
        similarity: float = 0.7,
    ) -> bool:
        """
        Returns whether any of the `images` is present within the specified `region`.

        Args:
            images (ImageType): The images to search for.
            region (Region): The region to search in.
            similarity (float): Minimum level (*between 0.0 and 1.0, where 0.0 is not similar and 1.0 is identical*) of similarity required to consider the image found. Default is 0.7.

        Returns:
            bool: `True` if any of the `images` is found within the region; otherwise, `False`.
        """
        region_image = ImageCapture().capture_region(region)

        num_threads = min(len(images), self.number_threads)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(
                    self.image_exists_in_image, image, region_image, similarity
                )
                for image in images
            ]

            for future in as_completed(futures):
                if future.result():
                    return True

            return False

    def get_coordinates_from_any_of_images(
        self,
        images: list[ImageType],
        region: Region,
        similarity: float = 0.7,
    ) -> Region:
        """
        Gets the coordinates of one of the `images` that is displayed within the specified `region`.

        The first image detected will have its coordinates returned. This method is used
        in cases where multiple images might appear at different times during the test.

        Args:
            images (list[str] | list[ImageType]): List of images to check.
            region (Region): Coordinates of the region to search for the images.
            similarity (float): Minimum level (*between 0.0 and 1.0, where 0.0 is not similar and 1.0 is identical*) of similarity required to consider the image found. Default is 0.7.

        Returns:
            Region: The coordinates of the image found within the region.

        Raises:
            RuntimeError: If none of the `images` is found within the `region`.
        """
        self.wait_for_any_of_images_in_region(images, region, similarity)

        region_image = ImageCapture().capture_region(region)

        for image in images:
            if self.image_exists_in_image(image, region_image, similarity):
                return self.get_image_coordinates(image, region, similarity)
        else:
            raise ImageComparisonError(
                "None of the images exists within the specified coordinates.",
                images,
                region_image,
            )

    def get_all_coordinates_from_any_of_images(
        self,
        images: list[ImageType],
        region: Region,
        similarity: float = 0.7,
    ) -> list[Region]:
        """
        Gets the coordinates of all occurrences of any of the `images` displayed within the `region`.

        Args:
            images (list[str] | list[ImageType]): List of images to check.
            region (Region): Coordinates of the region to search for the images.
            similarity (float): Minimum level (*between 0.0 and 1.0*) of similarity required to consider the
                image found. Default is 0.7.

        Returns:
            list[Region]: A list with the coordinates
                of the image found within the region.

        Raises:
            RuntimeError: If none of the `images` is found within the `region`.
        """
        # Input validation
        if not (0.0 <= similarity <= 1.0):
            raise ValueError("The value of `similarity` must be between 0.0 and 1.0.")

        elif not images:
            raise ValueError("The `images` list cannot be empty.")

        coordinates_found: list[Region] = []

        timeout = time.time() + 5

        while time.time() < timeout:
            region_image = ImageCapture().capture_region(region)
            for image in images:
                opencv_image = image.as_cv2_image()

                matches = self.get_images_matches(image, region_image)

                # Gets the X and Y axes that are similar to the image
                locations = numpy.where(matches >= similarity)
                axis = zip(locations[1], locations[0])

                if not axis:
                    continue

                height, width = opencv_image.shape[:2]

                for x, y in axis:
                    coordinates_found.append(
                        Region(
                            int(x) + region.x,
                            int(y) + region.y,
                            width,
                            height,
                        )
                    )

            if coordinates_found:
                break
        else:
            raise ImageComparisonError(
                "None of the images were found in the specified region.",
                images,
                region_image,
            )

        # Adjusts the sorting to match items with the smallest x and y axis values first.
        # This ensures they are ordered as if they appear from left to right and top to bottom on the screen.
        coordinates_found.sort(key=attrgetter("x", "y"))

        return coordinates_found

    def find_image_axis(
        self,
        image_to_search: ImageType | cv2.typing.MatLike,
        image_to_search_into: ImageType | cv2.typing.MatLike,
        similarity: float = 0.7,
    ) -> tuple[int, int]:
        """
        Returns the X and Y axes of the first `image_to_search` found within the `image_to_search_into`.

        Args:
            image_to_search (ImageType): Image to search for.
            image_to_search_into (ImageType): Image where the search will be performed.

        Returns:
            tuple(int, int): The X and Y axes where the `image_to_search` was found.
        """
        matches = self.get_images_matches(image_to_search, image_to_search_into)

        # Checks if there is a match greater than or equal to the specified similarity
        image_exists = numpy.any(matches >= similarity)

        if not image_exists:
            raise RuntimeError(
                "The image does not exist within the specified coordinates."
            )

        # Gets the X and Y axes of the best match found
        axis_y, axis_x = numpy.unravel_index(numpy.argmax(matches), matches.shape)

        return int(axis_x), int(axis_y)

    def get_image_coordinates(
        self,
        image: ImageType,
        region: Region,
        similarity: float = 0.7,
    ) -> Region:
        """
        Gets the coordinates of an `image` located within a `region`.

        Args:
            image (str | ImageType): The image to locate.
            region (Region): The region to search for the image.
            similarity (float): Minimum level (*between 0.0 and 1.0, where 0.0 is not similar and 1.0 is identical*) of similarity required to consider the image found. Default is 0.7.

        Returns:
            Region: The coordinates of the image within the region.

        Raises:
            RuntimeError: If the image is not found within the specified region.
        """
        search_region = ImageCapture().capture_region(region)
        image_opencv = image.as_cv2_image()

        # Gets the X and Y axes of the best match found
        axis_x, axis_y = self.find_image_axis(image_opencv, search_region, similarity)
        height, width = image_opencv.shape[:2]

        # Adjusts the coordinates to correspond to those on the entire screen
        return Region(
            axis_x + region.x,
            axis_y + region.y,
            width,
            height,
        )

    def get_images_matches(
        self,
        image_to_search: ImageType | cv2.typing.MatLike,
        image_to_search_into: ImageType | cv2.typing.MatLike,
    ) -> cv2.typing.MatLike:
        """Returns the regions where `image_to_search` is present within `image_to_search_into`.

        Args:
            image_to_search (ImageType): Image to be searched.
            image_to_search_into (ImageType): Image where the search will be performed.

        Returns:
            cv2.typing.MatLike: Matrix of matching coefficients, where each value represents the similarity
            between a region in `image_to_search_into` and `image_to_search`. Higher values indicate greater similarity.
        """
        if isinstance(image_to_search, ImageType):
            image_to_search = image_to_search.as_cv2_image()

        if isinstance(image_to_search_into, ImageType):
            image_to_search_into = image_to_search_into.as_cv2_image()

        try:
            matches = cv2.matchTemplate(
                image_to_search_into,
                image_to_search,
                cv2.TM_CCOEFF_NORMED,
            )

            return matches

        except Exception as error:
            raise ImageComparisonError(
                f"Ocorreu um erro ao comparar as imagens: {error}",
                ImageType(image_to_search),
                ImageType(image_to_search_into),
            )

    def get_position_of_image_in_regions(
        self,
        image: ImageType,
        regions: list[Region],
        similarity: float = 0.7,
    ) -> int:
        """Returns the position (index) of the region within `regions` where the `image` is found.

        Args:
            image (ImageType): Image to be searched for.
            regions (list[Region]): List of regions where the image will be searched.
            similarity (float): Minimum similarity value (between 0.0 and 1.0) required to consider
                the image as found. Defaults to 0.7.

        Returns:
            int: The index of the region where the image was found.

        Raises:
            ImageComparisonError: If the image is not found in any of the provided regions.
        """
        num_threads = min(len(regions), self.number_threads)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(self.image_exists_in_region, image, region, similarity)
                for region in regions
            ]

            for index, future in enumerate(as_completed(futures)):
                if future.result():
                    return index
            else:
                error_images = ImageCapture().capture_multiple_regions(regions)
                raise ImageComparisonError(
                    "A imagem não foi encontrada dentro das regiões.",
                    image,
                    error_images,
                )

    def get_position_of_text_in_regions(
        self,
        text: str,
        regions: list[Region],
        blacklist: str | None = None,
        error_message: str | None = None,
    ) -> int:
        """Returns the position (index) of the region within `regions` where the `text` is found.

        Args:
            text (str): Text to be searched for.
            regions (list[Region]): List of regions where the text will be searched.
            blacklist (str | None): Characters to be ignored during text reading. Defaults to None.
            error_message (str | None): Custom error message to be used if the text is not found. Defaults to None.

        Returns:
            int: The index of the region where the text was found.

        Raises:
            TextComparisonError: If the text is not found in any region within the timeout period.
        """
        timeout = time.time() + 5

        while timeout > time.time():
            # I used this instead of `get_texts_from_multiple_regions` because we can reuse the images if the timeout is reached
            region_images = ImageCapture().capture_multiple_regions(regions)
            regions_texts = self.get_texts_from_multiple_images(
                region_images,
                blacklist,
            )

            if text in regions_texts:
                return regions_texts.index(text)
        else:
            raise TextComparisonError(
                text,
                regions_texts,
                region_images,
                error_message=error_message,
            )

    def get_text_from_image(
        self,
        image: ImageType,
        blacklist: str | None = None,
    ) -> str:
        """Extracts text from an image using the Tesseract API.

        Args:
            image (ImageType): Image to be read.
            blacklist (str | None): Characters to be ignored while reading text from the image. Defaults to None.

        Returns:
            str: The text extracted from the image, stripped of leading and trailing whitespace.
        """
        tesseract = self.queue_tesseract.get()
        try:
            if blacklist:
                tesseract.SetBlacklist(blacklist)
            return tesseract.GetTextFromImage(image)
        finally:
            self.queue_tesseract.put(tesseract)

    def get_text_from_region(
        self,
        region: Region,
        blacklist: str | None = None,
    ) -> str:
        """Extracts text from a region using the Tesseract API.

        Args:
            region (Region): Screen region from which the text will be extracted.
            blacklist (str | None): Text to be ignored during reading. Defaults to None.

        Returns:
            str: The extracted text from the region, stripped of leading and trailing whitespace.
        """
        region_image = ImageCapture().capture_region(region)
        return self.get_text_from_image(region_image, blacklist)

    def get_texts_from_multiple_images(
        self,
        images: list[ImageType],
        blacklist: str | None = None,
    ) -> list[str]:
        """
        Simultaneously extracts text from a list of images.

        Args:
            images (list[ImageType]): List of images from which text will be extracted.
            blacklist (str | None): Text to be ignored during extraction. Defaults to None.

        Returns:
            list[str]: A list of strings containing the extracted text from each image, in the same order as the input images.
        """
        with ThreadPoolExecutor(max_workers=self.number_threads) as executor:
            # Associate each future with its original position
            futures = {
                executor.submit(self.get_text_from_image, image, blacklist): index
                for index, image in enumerate(images)
            }
            # Initialize a list to store texts in the correct order
            texts = [""] * len(images)

            # Populate the texts as results are completed
            for future in as_completed(futures):
                index = futures[future]
                try:
                    texts[index] = future.result()
                except Exception as error:
                    raise Exception(
                        f"Error processing the image at position {index}: {error}"
                    )

            return texts

    def get_texts_from_multiple_regions(
        self,
        regions: list[Region],
        blacklist: str | None = None,
    ) -> list[str]:
        """
        Simultaneously extracts text from a list of regions.

        Args:
            regions (list[Region]): List of regions from which text will be extracted.
            blacklist (str | None): Text to be ignored during extraction. Defaults to None.

        Returns:
            list[str]: A list containing the extracted text from each region, in the same order as the input regions.
        """
        images = ImageCapture().capture_multiple_regions(regions)
        return self.get_texts_from_multiple_images(images, blacklist)

    def image_exists_in_image(
        self,
        image: ImageType,
        image_to_search: ImageType,
        similarity: float = 0.7,
    ) -> bool:
        """
        Checks if an image is present within another specified image.

        Args:
            image (ImageType): The image to be located.
            image_to_search (ImageType): The image where `image` will be searched.
            similarity (float): The minimum similarity level (*between 0.0 and 1.0,
                where 0.0 means no similarity and 1.0 means identical*) required to
                consider the image found. Defaults to 0.7.

        Returns:
            bool: `True` if the image is found within the region, otherwise `False`.
        """
        match = self.get_images_matches(image, image_to_search)
        return numpy.any(match >= similarity).item()

    def image_exists_in_region(
        self,
        image: ImageType,
        region: Region | MSSRegion,
        similarity: float = 0.7,
    ) -> bool:
        """
        Checks if an image is present within a specified region.

        Args:
            image (ImageType): The image to be located.
            region (Region | MSSRegion): The region where the image will be searched.
            similarity (float): The minimum similarity level (*between 0.0 and 1.0,
                where 0.0 means no similarity and 1.0 means identical*) required to
                consider the image found. Defaults to 0.7.

        Returns:
            bool: `True` if the image is found within the region, otherwise `False`.
        """
        region_image = ImageCapture().capture_region(region)
        return self.image_exists_in_image(image, region_image, similarity)

    def wait_for_image_in_region(
        self,
        image: ImageType,
        region: Region,
        similarity: float = 0.7,
        timeout: float = 5.0,
        error_message: str | None = None,
    ) -> None:
        """
        Waits until an image appears within a region.

        Args:
            image (ImageType): The image to be located.
            region (Region): The region where the image will be searched.
            similarity (float): The minimum similarity level (*between 0.0 and 1.0, where 0.0 means no similarity and 1.0 means identical*) required to consider the image found. Defaults to 0.7.
            timeout (float): The maximum time, in seconds, to wait for the image to appear. Defaults to 5 seconds.
            error_message (str | None): Custom error message to be raised if the `image` is not found within the `timeout`. Defaults to None.

        Raises:
            ImageComparisonError: If the image is not found within the specified `timeout`.
        """
        tested_images: list[ImageType] = []

        stop_time = time.time() + timeout

        while time.time() < stop_time:
            image_captured = ImageCapture().capture_region(region)
            exists_image = self.image_exists_in_image(image, image_captured, similarity)

            if exists_image:
                break
            # Capture only the first 5 interactions.
            elif len(tested_images) < 5:
                tested_images.append(image_captured)
        else:
            if not error_message:
                error_message = f"Unable to find the image within {timeout}s."

            # Add a final state image of the region.
            tested_images.append(image_captured)

            raise ImageComparisonError(error_message, image, tested_images)

    def wait_for_any_of_images_in_region(
        self,
        images: list[ImageType],
        region: Region,
        similarity: float = 0.7,
        timeout: float = 5.0,
    ) -> None:
        """
        Waits until any of the `images` appears within the `region`.

        Args:
            images (list[ImageType]): The images to be validated.
            region (Region): The region where the `images` will be searched.
            similarity (float): The minimum similarity level (*between 0.0 and 1.0, where 0.0 means no similarity and 1.0 means identical*) required to consider an image found. Defaults to 0.7.
            timeout (float): The maximum time, in seconds, for any image to appear in the `region`. Defaults to 5 seconds.

        Raises:
            ImageComparisonError: If none of the images is found within the specified `timeout`.
        """
        stop_time = time.time() + timeout

        while time.time() < stop_time:
            exists_image = self.any_of_images_exists_in_region(
                images,
                region,
                similarity,
            )

            if exists_image:
                break
        else:
            region_image = ImageCapture().capture_region(region)

            raise ImageComparisonError(
                "Unable to find any image within 5 seconds.",
                images,
                region_image,
            )

    def wait_for_text_in_region(
        self,
        text: str,
        region: Region,
        blacklist: str | None = None,
        timeout: float = 1,
        error_message: str | None = None,
    ) -> None:
        """
        Waits for the specified `text` to appear within the `region` for up to the given `timeout`.

        Args:
            text (str): The text to wait for in the specified region.
            region (Region): The screen region to monitor.
            blacklist (str | None): Text to ignore during verification. Defaults to None.
            timeout (float): Maximum time, in seconds, to wait for the text to appear. Defaults to 1 second.
            error_message (str | None): Custom error message to be raised if the text is not found. Defaults to None.

        Raises:
            ImageComparisonError: If the text is not found within the timeout period.
        """
        tesseract = self.queue_tesseract.get()
        try:
            if blacklist:
                tesseract.SetBlacklist(blacklist)

            stop_time = time.time() + timeout

            while stop_time > time.time():
                image = ImageCapture().capture_region(region)
                text_in_image = tesseract.GetTextFromImage(image)

                if text == text_in_image:
                    break
            else:
                message_comparison_error = (
                    f"The text '{text_in_image}' is displayed instead of '{text}'!"
                )

                if error_message:
                    error_message = f"{error_message}\n\n{message_comparison_error}"
                else:
                    error_message = message_comparison_error

                raise TextComparisonError(text, text_in_image, image)
        finally:
            self.queue_tesseract.put(tesseract)

    def wait_for_texts_in_regions(
        self,
        texts: list[str],
        regions: list[Region],
        blacklist: str | None = None,
        timeout: float = 1,
        error_messages: list[str] | None = None,
    ) -> None:
        """
        Simultaneously waits, for up to a specified `timeout`, for the `texts` to appear respectively within the `regions`.

        Each text in `texts` will be validated with the region in the same position of `regions`.

        - **Example:** The first position in `texts` will be compared to the first position in `regions`.

        Args:
            texts (list[str]): Texts to search for in each region.
            regions (list[Region]): Regions where the texts will be searched.
            blacklist (str | None): Characters to ignore during the search.
            timeout (float): Maximum wait time for each text.
            error_messages (list[str] | None): Custom error messages for each text.

        Raises:
            ValueError: If `texts`, `regions`, and `error_messages` (if provided) do not have the same length.
            Exception: Propagates any exceptions raised by `wait_for_text_in_region`.
        """
        if error_messages and len(texts) != len(error_messages):
            raise ValueError(
                f"`texts` (length {len(texts)}), `regions` (length {len(regions)}) and `error_messages` (length {len(error_messages)}) must have the same length."
            )
        elif len(texts) != len(regions):
            raise ValueError(
                f"`texts` (length {len(texts)}) and `regions` (length {len(regions)}) must have the same length."
            )

        num_threads = min(len(regions), self.number_threads)

        # Create default values for `error_messages` if custom messages are not provided.
        error_messages_default = error_messages or [None] * len(texts)

        tasks = zip(texts, regions, error_messages_default)

        exceptions = []

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(
                    self.wait_for_text_in_region,
                    text,
                    region,
                    blacklist,
                    timeout,
                    error_message,
                )
                for text, region, error_message in tasks
            ]

            for future in as_completed(futures):
                try:
                    future.result()  # Wait for the result or capture exceptions
                except Exception as error:
                    exceptions.append(error)

        if exceptions:
            validation_errors = "\n".join(
                f"- {str(exception)}" for exception in exceptions
            )

            # Reraise all captured exceptions as a single error.
            raise Exception(
                f"Errors occurred during execution: \n\n{validation_errors}"
            )

    def wait_for_text_in_regions(
        self,
        text: str,
        regions: list[Region],
        blacklist: str | None = None,
        timeout: float = 1,
        error_message: str | None = None,
    ) -> None:
        """
        Waits, for up to a specified `timeout`, for the `text` to appear within any of the `regions`.

        Args:
            text (str): Text to search for in each region.
            regions (list[Region]): Regions where the text will be awaited.
            blacklist (str | None): Characters to ignore during the search.
            timeout (float): Maximum time, in seconds, to find the text.
            error_message (str | None): Custom error message in case the text is not found.

        Raises:
            ImageComparisonError: If the `text` is not found in any of the `regions` within the specified `timeout`.
        """
        stop_time = time.time() + timeout

        while stop_time > time.time():
            texts = self.get_texts_from_multiple_regions(regions, blacklist)
            if text in texts:
                break
        else:
            if error_message:
                raise TextComparisonError(
                    text,
                    texts,
                    error_message=error_message,
                )
            else:
                raise TextComparisonError(text, texts)

    def wait_while_any_of_images_in_region(
        self,
        images: list[ImageType],
        region: Region,
        similarity: float = 0.7,
        timeout: float = 5.0,
    ) -> None:
        """
        Waits while any of the `images` is present within the `region`.

        Args:
            images (ImageType): The images to be validated.
            region (Region): The region where the `images` will be searched.
            similarity (float): The minimum similarity level (*between 0.0 and 1.0, with 0.0 being not similar and 1.0 being identical*) required to consider the image found. Default is 0.7.
            timeout (float): The maximum time, in seconds, for any of the images to appear in the `region`. Default is 5 seconds.

        Raises:
            ImageComparisonError: If no image is found within the specified `timeout`.
        """
        stop_time = time.time() + timeout

        while stop_time > time.time():
            exists_image = self.any_of_images_exists_in_region(
                images,
                region,
                similarity,
            )

            if not exists_image:
                break
        else:
            region_image = ImageCapture().capture_region(region)

            raise ImageComparisonError(
                f"One of the images remained on screen after {timeout}s.",
                images,
                region_image,
            )

    def wait_while_image_in_region(
        self,
        image: ImageType,
        region: Region,
        similarity: float = 0.7,
        timeout: float = 5.0,
        error_message: str | None = None,
    ) -> None:
        """
        Waits until the `image` is no longer displayed within the `region`.

        Args:
            image (ImageType): The image to be monitored.
            region (Region): The region where the image will be checked.
            similarity (float): The minimum similarity level (*between 0.0 and 1.0, with 0.0 being not similar and 1.0 being identical*) required to consider the image found. Default is 0.7.
            timeout (float): The maximum time, in seconds, to wait for the image to appear. Default is 5 seconds.
            error_message (str | None): Custom error message if the `image` remains on screen after the `timeout`.

        Raises:
            ImageComparisonError: If the image is not removed within the specified `timeout`.
        """
        stop_time = time.time() + timeout

        while stop_time > time.time():
            exists_image = self.image_exists_in_region(image, region, similarity)

            if not exists_image:
                break
        else:
            if not error_message:
                error_message = f"The image remained on screen after the maximum time of {timeout}s."

            raise ImageComparisonError(
                error_message,
                image,
                region,
            )
