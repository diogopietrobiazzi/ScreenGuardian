import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import cast

import mss
from PIL import Image

from Typing import ImageType, MSSRegion, Region


class ImageCapture:
    def capture_region(
        self,
        region: Region | MSSRegion | None,
    ) -> ImageType:
        """
        Captures a region on the screen.

        Args:
            region (Region | MSSRegion | None): Screen region to capture. If `None`, captures the full screen.

        Returns:
            ImageType: The captured region as an image.
        """
        if isinstance(region, Region):
            region = region.as_mss_region()

        with mss.mss() as screen_capturer:
            if region is None:
                monitor = screen_capturer.monitors[1]
                screenshot = screen_capturer.grab(monitor)
            else:
                screenshot = screen_capturer.grab(region.as_dict())

        image = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

        return ImageType(image)

    def capture_multiple_regions(
        self,
        regions: list[Region] | list[MSSRegion],
    ) -> list[ImageType]:
        """
        Simultaneously captures multiple regions on the screen.

        Args:
            regions (list[Region] | list[MSSRegion]): List of regions to capture.

        Returns:
            list[ImageType]: List of captured region images.
        """
        num_threads = min(len(regions), os.cpu_count() or 1)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Associates each future with its original position
            futures = {
                executor.submit(self.capture_region, region): index
                for index, region in enumerate(regions)
            }
            # Initializes a list to store the images in the correct order
            images: list[ImageType] = [cast(ImageType, None)] * len(regions)

            # Fills the images as results are completed
            for future in as_completed(futures):
                index = futures[future]
                try:
                    images[index] = future.result()
                except Exception as error:
                    raise Exception(
                        f"Error while processing the image at position {index}: {error}"
                    )

            return images
