"""Fallback chessboard corner detector using classical computer vision.

This module provides a lightweight implementation inspired by the
`chesscog` project. When the neural network corner detector fails to
return all four corners, this detector uses edge detection and Hough
lines to approximate the board boundaries and compute the corners.
"""

from typing import Dict, Tuple, Optional

import cv2
import numpy as np


def detect_corners_fallback(img: np.ndarray) -> Optional[Dict[str, Tuple[int, int]]]:
    """Detect chessboard corners using edge and line analysis.

    The approach follows the classical pipeline employed in the
    `chesscog` repository: detect edges, extract dominant horizontal and
    vertical lines with the Hough transform and intersect the outermost
    lines to obtain the four corners of the board.

    Args:
        img: Input BGR image containing a chessboard.

    Returns:
        Dictionary mapping corner names to ``(x, y)`` coordinates or
        ``None`` if the chessboard could not be located.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    if lines is None:
        return None

    lines = lines[:, 0, :]  # (rho, theta)
    vertical = []
    horizontal = []
    for rho, theta in lines:
        if abs(theta) < np.pi / 4 or abs(theta - np.pi) < np.pi / 4:
            vertical.append((rho, theta))
        elif abs(theta - np.pi / 2) < np.pi / 4:
            horizontal.append((rho, theta))

    if len(vertical) < 2 or len(horizontal) < 2:
        return None

    left = min(vertical, key=lambda l: l[0])
    right = max(vertical, key=lambda l: l[0])
    top = min(horizontal, key=lambda l: l[0])
    bottom = max(horizontal, key=lambda l: l[0])

    def intersection(l1: Tuple[float, float], l2: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        rho1, theta1 = l1
        rho2, theta2 = l2
        A = np.array(
            [[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]]
        )
        b = np.array([[rho1], [rho2]])
        try:
            x0, y0 = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
        return int(np.round(x0)), int(np.round(y0))

    tl = intersection(top, left)
    tr = intersection(top, right)
    br = intersection(bottom, right)
    bl = intersection(bottom, left)

    if None in (tl, tr, br, bl):
        return None

    return {
        "top_left": tl,
        "top_right": tr,
        "bottom_right": br,
        "bottom_left": bl,
    }
