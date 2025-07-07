"""
Real-world example: Delivery route optimization.

This example demonstrates how to use tsbee for a realistic delivery
route optimization scenario with actual city coordinates.
"""

from typing import List, Tuple

import numpy as np
import tsbee

# Major US cities with approximate coordinates (latitude, longitude)
CITIES = {
    "New York": (40.7128, -74.0060),
    "Los Angeles": (34.0522, -118.2437),
    "Chicago": (41.8781, -87.6298),
    "Houston": (29.7604, -95.3698),
    "Phoenix": (33.4484, -112.0740),
    "Philadelphia": (39.9526, -75.1652),
    "San Antonio": (29.4241, -98.4936),
    "San Diego": (32.7157, -117.1611),
    "Dallas": (32.7767, -96.7970),
    "San Jose": (37.3382, -121.8863),
    "Austin": (30.2672, -97.7431),
    "Jacksonville": (30.3322, -81.6557),
    "Fort Worth": (32.7555, -97.3308),
    "Columbus": (39.9612, -82.9988),
    "San Francisco": (37.7749, -122.4194),
    "Charlotte": (35.2271, -80.8431),
    "Indianapolis": (39.7684, -86.1581),
    "Seattle": (47.6062, -122.3321),
    "Denver": (39.7392, -104.9903),
    "Boston": (42.3601, -71.0589),
}


def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    Returns distance in kilometers.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Earth's radius in kilometers
    r = 6371

    return c * r


def create_distance_matrix(cities: List[str]) -> np.ndarray:
    """Create distance matrix for selected cities."""
    n = len(cities)
    distances = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            if i != j:
                coord1 = CITIES[cities[i]]
                coord2 = CITIES[cities[j]]
                # Convert to integer (multiply by 10 to preserve one decimal)
                distances[i][j] = int(haversine_distance(coord1, coord2) * 10)

    return distances


def format_route(cities: List[str], tour: List[int]) -> List[str]:
    """Format the tour as a list of city names."""
    return [cities[i] for i in tour]


def calculate_total_distance(cities: List[str], tour: List[int]) -> float:
    """Calculate total distance of the tour in kilometers."""
    total = 0
    for i in range(len(tour)):
        j = (i + 1) % len(tour)
        city1 = cities[tour[i]]
        city2 = cities[tour[j]]
        total += haversine_distance(CITIES[city1], CITIES[city2])
    return total


def main():
    print("Delivery Route Optimization Example")
    print("=" * 60)

    # Scenario 1: East Coast delivery route
    print("\nScenario 1: East Coast Delivery Route")
    print("-" * 40)

    east_coast_cities = [
        "Boston",
        "New York",
        "Philadelphia",
        "Charlotte",
        "Jacksonville",
        "Columbus",
        "Indianapolis",
    ]

    # Create distance matrix
    distances = create_distance_matrix(east_coast_cities)

    # Solve TSP
    print(f"Optimizing route for {len(east_coast_cities)} cities...")
    solution = tsbee.solve(distances.tolist())

    # Format results
    route = format_route(east_coast_cities, solution.tour)
    total_km = calculate_total_distance(east_coast_cities, solution.tour)

    print("\nOptimal route:")
    for i, city in enumerate(route):
        print(f"  {i+1}. {city}")
    print(f"  {len(route)+1}. {route[0]} (return)")

    print(f"\nTotal distance: {total_km:.1f} km")
    print(f"Time to compute: {solution.time:.3f} seconds")

    # Scenario 2: Cross-country route
    print("\n\nScenario 2: Cross-Country Distribution")
    print("-" * 40)

    distribution_centers = [
        "Los Angeles",
        "Phoenix",
        "Denver",
        "Dallas",
        "Chicago",
        "Atlanta" if "Atlanta" in CITIES else "Charlotte",  # Use Charlotte as proxy
        "New York",
        "Seattle",
    ]

    distances = create_distance_matrix(distribution_centers)

    print(f"Optimizing route for {len(distribution_centers)} distribution centers...")
    solution = tsbee.solve(distances.tolist(), time_limit=5.0)

    route = format_route(distribution_centers, solution.tour)
    total_km = calculate_total_distance(distribution_centers, solution.tour)

    print("\nOptimal distribution route:")
    for i, city in enumerate(route):
        if i > 0:
            prev_city = route[i - 1]
            segment_distance = haversine_distance(CITIES[prev_city], CITIES[city])
            print(f"  {i}. {prev_city} -> {city}: {segment_distance:.1f} km")

    # Close the loop
    segment_distance = haversine_distance(CITIES[route[-1]], CITIES[route[0]])
    print(f"  {len(route)}. {route[-1]} -> {route[0]}: {segment_distance:.1f} km")

    print(f"\nTotal distance: {total_km:.1f} km")
    print(f"Average distance per segment: {total_km/len(route):.1f} km")

    # Cost estimation
    fuel_cost_per_km = 0.15  # $0.15 per km
    driver_cost_per_hour = 25  # $25 per hour
    avg_speed_kmh = 80  # 80 km/h average

    fuel_cost = total_km * fuel_cost_per_km
    time_hours = total_km / avg_speed_kmh
    driver_cost = time_hours * driver_cost_per_hour
    total_cost = fuel_cost + driver_cost

    print("\nCost estimation:")
    print(f"  Fuel cost: ${fuel_cost:.2f}")
    print(f"  Driver cost ({time_hours:.1f} hours): ${driver_cost:.2f}")
    print(f"  Total cost: ${total_cost:.2f}")


if __name__ == "__main__":
    main()
