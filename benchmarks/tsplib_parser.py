"""
TSPLIB file parser for reading TSP and ATSP instances.

Supports various edge weight types including EUC_2D, GEO, ATT, EXPLICIT, etc.
"""

import contextlib
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np


class TSPLIBParser:
    def __init__(self):
        self.name = ""
        self.comment = ""
        self.dimension = 0
        self.edge_weight_type = ""
        self.edge_weight_format = ""
        self.node_coord_type = ""
        self.display_data_type = ""
        self.node_coords = []
        self.edge_weights = None
        self.display_coords = []

    def parse_file(self, filepath: Union[str, Path]) -> Dict:
        """Parse a TSPLIB format file."""
        filepath = Path(filepath)

        with open(filepath) as f:
            lines = f.readlines()

        # Parse header
        i = 0
        while i < len(lines) and not lines[i].strip().endswith("SECTION"):
            line = lines[i].strip()
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                if key == "NAME":
                    self.name = value
                elif key == "COMMENT":
                    self.comment = value
                elif key == "TYPE":
                    # Extract just TSP or ATSP from the value
                    if not (value.startswith("TSP") or value.startswith("ATSP")):
                        raise ValueError(f"Unsupported problem type: {value}")
                elif key == "DIMENSION":
                    self.dimension = int(value)
                elif key == "EDGE_WEIGHT_TYPE":
                    self.edge_weight_type = value
                elif key == "EDGE_WEIGHT_FORMAT":
                    self.edge_weight_format = value
                elif key == "NODE_COORD_TYPE":
                    self.node_coord_type = value
                elif key == "DISPLAY_DATA_TYPE":
                    self.display_data_type = value
            i += 1

        # Parse sections
        while i < len(lines):
            line = lines[i].strip()

            if line == "NODE_COORD_SECTION":
                i = self._parse_node_coords(lines, i + 1)
            elif line == "EDGE_WEIGHT_SECTION":
                i = self._parse_edge_weights(lines, i + 1)
            elif line == "DISPLAY_DATA_SECTION":
                i = self._parse_display_data(lines, i + 1)
            elif line == "EOF":
                break
            else:
                i += 1

        # Create distance matrix
        distance_matrix = self._create_distance_matrix()

        return {
            "name": self.name,
            "dimension": self.dimension,
            "edge_weight_type": self.edge_weight_type,
            "distance_matrix": distance_matrix,
            "node_coords": self.node_coords if self.node_coords else None,
        }

    def _parse_node_coords(self, lines: List[str], start_idx: int) -> int:
        """Parse NODE_COORD_SECTION."""
        i = start_idx
        self.node_coords = []

        while i < len(lines):
            line = lines[i].strip()
            if line.endswith("SECTION") or line == "EOF":
                break

            parts = line.split()
            if len(parts) >= 3:
                # node_id, x, y
                x = float(parts[1])
                y = float(parts[2])
                self.node_coords.append((x, y))
            i += 1

        return i

    def _parse_edge_weights(self, lines: List[str], start_idx: int) -> int:
        """Parse EDGE_WEIGHT_SECTION."""
        i = start_idx
        weights = []

        while i < len(lines):
            line = lines[i].strip()
            if line.endswith("SECTION") or line == "EOF":
                break

            # Parse numbers from line
            values = line.split()
            for val in values:
                with contextlib.suppress(ValueError):
                    weights.append(float(val))
            i += 1

        # Convert to matrix based on format
        self.edge_weights = self._weights_to_matrix(weights)
        return i

    def _parse_display_data(self, lines: List[str], start_idx: int) -> int:
        """Parse DISPLAY_DATA_SECTION."""
        i = start_idx
        self.display_coords = []

        while i < len(lines):
            line = lines[i].strip()
            if line.endswith("SECTION") or line == "EOF":
                break

            parts = line.split()
            if len(parts) >= 3:
                x = float(parts[1])
                y = float(parts[2])
                self.display_coords.append((x, y))
            i += 1

        return i

    def _weights_to_matrix(self, weights: List[float]) -> np.ndarray:
        """Convert weight list to matrix based on format."""
        n = self.dimension
        matrix = np.zeros((n, n))

        if self.edge_weight_format == "FULL_MATRIX":
            k = 0
            for i in range(n):
                for j in range(n):
                    matrix[i][j] = weights[k]
                    k += 1

        elif self.edge_weight_format == "UPPER_ROW":
            k = 0
            for i in range(n):
                for j in range(i + 1, n):
                    matrix[i][j] = weights[k]
                    matrix[j][i] = weights[k]
                    k += 1

        elif self.edge_weight_format == "LOWER_ROW":
            k = 0
            for i in range(1, n):
                for j in range(i):
                    matrix[i][j] = weights[k]
                    matrix[j][i] = weights[k]
                    k += 1

        elif self.edge_weight_format == "UPPER_DIAG_ROW":
            k = 0
            for i in range(n):
                for j in range(i, n):
                    matrix[i][j] = weights[k]
                    if i != j:
                        matrix[j][i] = weights[k]
                    k += 1

        elif self.edge_weight_format == "LOWER_DIAG_ROW":
            k = 0
            for i in range(n):
                for j in range(i + 1):
                    matrix[i][j] = weights[k]
                    if i != j:
                        matrix[j][i] = weights[k]
                    k += 1

        elif self.edge_weight_format == "UPPER_COL":
            k = 0
            for j in range(n):
                for i in range(j):
                    matrix[i][j] = weights[k]
                    matrix[j][i] = weights[k]
                    k += 1

        elif self.edge_weight_format == "LOWER_COL":
            k = 0
            for j in range(n):
                for i in range(j + 1, n):
                    matrix[i][j] = weights[k]
                    matrix[j][i] = weights[k]
                    k += 1

        elif self.edge_weight_format == "UPPER_DIAG_COL":
            k = 0
            for j in range(n):
                for i in range(j + 1):
                    matrix[i][j] = weights[k]
                    if i != j:
                        matrix[j][i] = weights[k]
                    k += 1

        elif self.edge_weight_format == "LOWER_DIAG_COL":
            k = 0
            for j in range(n):
                for i in range(j, n):
                    matrix[i][j] = weights[k]
                    if i != j:
                        matrix[j][i] = weights[k]
                    k += 1

        return matrix

    def _create_distance_matrix(self) -> np.ndarray:
        """Create distance matrix based on edge weight type."""
        n = self.dimension

        if self.edge_weights is not None:
            matrix = self.edge_weights.astype(int)
            # Ensure diagonal is zero
            np.fill_diagonal(matrix, 0)
            return matrix

        if not self.node_coords:
            raise ValueError("No node coordinates or edge weights provided")

        matrix = np.zeros((n, n), dtype=int)

        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = self._calculate_distance(i, j)
                    matrix[i][j] = int(dist)

        return matrix

    def _calculate_distance(self, i: int, j: int) -> float:
        """Calculate distance between two nodes based on edge weight type."""
        if self.edge_weight_type == "EUC_2D":
            return self._euclidean_2d(i, j)
        elif self.edge_weight_type == "GEO":
            return self._geo_distance(i, j)
        elif self.edge_weight_type == "ATT":
            return self._att_distance(i, j)
        elif self.edge_weight_type == "CEIL_2D":
            return math.ceil(self._euclidean_2d_raw(i, j))
        else:
            raise ValueError(f"Unsupported edge weight type: {self.edge_weight_type}")

    def _euclidean_2d_raw(self, i: int, j: int) -> float:
        """Raw Euclidean distance."""
        x1, y1 = self.node_coords[i]
        x2, y2 = self.node_coords[j]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def _euclidean_2d(self, i: int, j: int) -> float:
        """Euclidean distance rounded to nearest integer."""
        return round(self._euclidean_2d_raw(i, j))

    def _geo_distance(self, i: int, j: int) -> float:
        """Geographical distance."""
        pi = 3.141592
        rrr = 6378.388

        x1, y1 = self.node_coords[i]
        x2, y2 = self.node_coords[j]

        # Convert to radians
        deg_x1 = int(x1)
        min_x1 = x1 - deg_x1
        lat1 = pi * (deg_x1 + 5.0 * min_x1 / 3.0) / 180.0

        deg_y1 = int(y1)
        min_y1 = y1 - deg_y1
        lon1 = pi * (deg_y1 + 5.0 * min_y1 / 3.0) / 180.0

        deg_x2 = int(x2)
        min_x2 = x2 - deg_x2
        lat2 = pi * (deg_x2 + 5.0 * min_x2 / 3.0) / 180.0

        deg_y2 = int(y2)
        min_y2 = y2 - deg_y2
        lon2 = pi * (deg_y2 + 5.0 * min_y2 / 3.0) / 180.0

        q1 = math.cos(lon1 - lon2)
        q2 = math.cos(lat1 - lat2)
        q3 = math.cos(lat1 + lat2)

        return int(rrr * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)

    def _att_distance(self, i: int, j: int) -> float:
        """ATT (pseudo-Euclidean) distance."""
        x1, y1 = self.node_coords[i]
        x2, y2 = self.node_coords[j]

        xd = x1 - x2
        yd = y1 - y2
        rij = math.sqrt((xd * xd + yd * yd) / 10.0)
        tij = int(rij)

        if tij < rij:
            return tij + 1
        else:
            return tij


def parse_tour_file(filepath: Union[str, Path]) -> Tuple[List[int], int]:
    """Parse a .tour file and return the tour and optimal distance."""
    filepath = Path(filepath)

    with open(filepath) as f:
        lines = f.readlines()

    tour = []
    optimal_distance = None
    in_tour_section = False

    for line in lines:
        line = line.strip()

        if line.startswith("COMMENT"):
            # Extract distance from comment if present
            # Try different patterns
            match = re.search(r"Length\s*[=:]\s*(\d+)", line, re.IGNORECASE)
            if not match:
                match = re.search(r"\(Length\s*(\d+)\)", line, re.IGNORECASE)
            if match:
                optimal_distance = int(match.group(1))

        elif line == "TOUR_SECTION":
            in_tour_section = True

        elif line == "-1" or line == "EOF":
            in_tour_section = False

        elif in_tour_section and line.isdigit():
            # Convert from 1-indexed to 0-indexed
            tour.append(int(line) - 1)

    return tour, optimal_distance
