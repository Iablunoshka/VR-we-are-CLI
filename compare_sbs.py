import argparse
from pathlib import Path

import cv2
import numpy as np


def calculate_stats(
    reference: np.ndarray,
    test: np.ndarray,
    name: str,
) -> None:
    """
    Compare two RGB/BGR image regions.

    Difference values are calculated per color channel in the 0-255 range.
    """

    diff = cv2.absdiff(reference, test)

    # Максимальная разница среди всех каналов и пикселей.
    max_diff = int(diff.max())

    # Средняя абсолютная разница по всем каналам.
    avg_diff = float(diff.mean())

    # Пиксель считается отличающимся, если отличается хотя бы один канал.
    different_pixel_mask = np.any(diff != 0, axis=2)
    different_pixels = int(np.count_nonzero(different_pixel_mask))
    total_pixels = int(reference.shape[0] * reference.shape[1])

    different_percent = (
        different_pixels / total_pixels * 100.0
        if total_pixels
        else 0.0
    )

    # Максимальная и средняя разница отдельно по BGR-каналам.
    channel_max = diff.reshape(-1, 3).max(axis=0)
    channel_avg = diff.reshape(-1, 3).mean(axis=0)

    print(f"\n{name}:")
    print(f"  max diff: {max_diff}")
    print(f"  avg diff: {avg_diff:.6f}")
    print(
        f"  different pixels: "
        f"{different_pixels}/{total_pixels} "
        f"({different_percent:.6f}%)"
    )
    print(
        f"  channel max B/G/R: "
        f"{int(channel_max[0])}/"
        f"{int(channel_max[1])}/"
        f"{int(channel_max[2])}"
    )
    print(
        f"  channel avg B/G/R: "
        f"{channel_avg[0]:.6f}/"
        f"{channel_avg[1]:.6f}/"
        f"{channel_avg[2]:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare CPU and GPU SBS PNG outputs and save an "
            "unamplified absolute-difference PNG."
        )
    )
    parser.add_argument(
        "reference",
        type=Path,
        help="Reference SBS PNG, for example CPU output",
    )
    parser.add_argument(
        "test",
        type=Path,
        help="Test SBS PNG, for example GPU output",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("sbs_diff.png"),
        help="Output PNG with raw absolute difference",
    )

    args = parser.parse_args()

    reference = cv2.imread(
        str(args.reference),
        cv2.IMREAD_UNCHANGED,
    )
    test = cv2.imread(
        str(args.test),
        cv2.IMREAD_UNCHANGED,
    )

    if reference is None:
        raise FileNotFoundError(
            f"Cannot read reference image: {args.reference}"
        )

    if test is None:
        raise FileNotFoundError(
            f"Cannot read test image: {args.test}"
        )

    if reference.shape != test.shape:
        raise ValueError(
            "Image shapes do not match:\n"
            f"reference: {reference.shape}\n"
            f"test:      {test.shape}"
        )

    if reference.dtype != test.dtype:
        raise TypeError(
            "Image dtypes do not match:\n"
            f"reference: {reference.dtype}\n"
            f"test:      {test.dtype}"
        )

    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError(
            f"Expected 3-channel image, got {reference.shape}"
        )

    height, full_width, channels = reference.shape

    if full_width % 2 != 0:
        raise ValueError(
            f"SBS width must be even, got {full_width}"
        )

    half_width = full_width // 2

    reference_left = reference[:, :half_width]
    reference_right = reference[:, half_width:]

    test_left = test[:, :half_width]
    test_right = test[:, half_width:]

    print("SBS comparison")
    print(f"resolution: {full_width}x{height}")
    print(f"half resolution: {half_width}x{height}")
    print(f"dtype: {reference.dtype}")

    calculate_stats(
        reference_left,
        test_left,
        "LEFT",
    )
    calculate_stats(
        reference_right,
        test_right,
        "RIGHT",
    )
    calculate_stats(
        reference,
        test,
        "FULL SBS",
    )

    # Обычная абсолютная разница 0-255.
    # Никакого умножения, нормализации или усиления.
    full_diff = cv2.absdiff(reference, test)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(str(args.output), full_diff):
        raise RuntimeError(
            f"Failed to save diff image: {args.output}"
        )

    print(f"\nRaw diff saved: {args.output}")

    if not np.any(full_diff):
        print("Result: images are pixel-identical.")
    else:
        print("Result: images contain pixel differences.")


if __name__ == "__main__":
    main()