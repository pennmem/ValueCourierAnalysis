import random
import argparse
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import math
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import pandas as pd
from handleTSPRoutes import *
# LC: manually selected store lists that are visible from given store location
#     need to update this dictionary manually when the town layout changes

# this is in relation to the NamedStores Container
store_pos_dict: Dict[str, Tuple[float, float, float]] = {
    "post_office": (29.18, -45, 14.06),
    "barber_shop": (4.45, -45, -93.88),
    "bakery": (-46.89, -45, 10.2),
    "bike_shop": (-78.41, -45, -30.49),
    "cafe": (127.2, -45, -7.76),
    "clothing_store": (127.23, -45, -63.88),
    "dentist": (132.31, -45, 51.82),
    "craft_shop": (90.36, -45, 77.95),
    "grocery_store": (146.02, -45, -42.77),
    "jewelry_store": (-1.15, -45, -35.74),
    "florist": (46.02, -45, 140.51),
    "hardware_store": (19.61, -45, -19.5),
    "gym": (36.49, -45, 57.81),
    "pizzeria": (76.23, -45, 59.53),
    "pet_store": (-43.09, -45, -79.9),
    "music_store": (-47.05, -45, 110.15),
    "pharmacy": (-16.94, -45, 88.54),
    "toy_store": (52.3, -45, -51.73),
    "bank": (61.47, -45, 111.55),
    "bookstore": (112.54, -45, 38.16),
    "noodle_house": (68.38, -45, -39.3),
    "party_store": (-0.21, -45, 44.03),
    "burger_joint": (90, -45, 13.03),
    "gelateria": (-26.5, -45, 132.71),
    "salon": (-23.69, -45, -13.81),
    "tech_shop": (95.32, -45, -23.08),
    
}

node_adj_dict = {
    # -------- Island --------
    "island_0": {"island_1", "island_4"},
    "island_1": {"island_0", "island_2", "darkcity_6"},
    "island_2": {"island_1", "island_3", "island_4", "suburbs_11"},
    "island_3": {"island_2", "island_5"},
    "island_4": {"island_0", "island_2", "island_5"},
    "island_5": {"island_3", "island_4"},

    # -------- Suburbs --------
    "suburbs_0": {"suburbs_1", "suburbs_4", "suburbs_10"},
    "suburbs_1": {"suburbs_0", "suburbs_2"},
    "suburbs_2": {"suburbs_1", "suburbs_3", "suburbs_8"},
    "suburbs_3": {"suburbs_2", "suburbs_9"},
    "suburbs_4": {"suburbs_5", "suburbs_6"},
    "suburbs_5": {"suburbs_4", "suburbs_7"},
    "suburbs_6": {"suburbs_4", "suburbs_7","suburbs_11"},
    "suburbs_7": {"suburbs_8", "suburbs_5","suburbs_6"},
    "suburbs_8": {"suburbs_7", "suburbs_2", "suburbs_9"},
    "suburbs_9": {"suburbs_8", "suburbs_3", "suburbs_17"},
    "suburbs_10": {"suburbs_0","suburbs_11","skyscraper_19","darkcity_1","darkcity_3"},
    "suburbs_11": {"suburbs_10", "suburbs_12", "island_2","suburbs_10"},
    "suburbs_12": {"suburbs_11", "suburbs_16","suburbs_13"},
    "suburbs_13": {"suburbs_2", "suburbs_14"},
    "suburbs_14": {"suburbs_13", "suburbs_15"},
    "suburbs_15": {"suburbs_14", "suburbs_16"},
    "suburbs_16": {"suburbs_12", "suburbs_15", "suburbs_17"},
    "suburbs_17": {"suburbs_16", "suburbs_9"},

    # -------- Skyscrapers --------
    "skyscraper_0": {"skyscraper_1", "skyscraper_5"},
    "skyscraper_1": {"skyscraper_0", "skyscraper_2", "skyscraper_6"},
    "skyscraper_2": {"skyscraper_1", "skyscraper_3", "skyscraper_7"},
    "skyscraper_3": {"skyscraper_2", "skyscraper_4", "skyscraper_8"},
    "skyscraper_4": {"skyscraper_3", "skyscraper_9", "suburbs_0"},
    "skyscraper_5": {"skyscraper_0", "skyscraper_6", "skyscraper_10"},
    "skyscraper_6": {"skyscraper_1", "skyscraper_5", "skyscraper_7", "skyscraper_11"},
    "skyscraper_7": {"skyscraper_2", "skyscraper_6", "skyscraper_8", "skyscraper_12"},
    "skyscraper_8": {"skyscraper_3", "skyscraper_7", "skyscraper_9", "skyscraper_13"},
    "skyscraper_9": {"skyscraper_4", "skyscraper_8", "skyscraper_14"},
    "skyscraper_10": {"skyscraper_5", "skyscraper_11", "skyscraper_15"},
    "skyscraper_11": {"skyscraper_6", "skyscraper_10", "skyscraper_12", "skyscraper_16"},
    "skyscraper_12": {"skyscraper_7", "skyscraper_11", "skyscraper_13", "skyscraper_17"},
    "skyscraper_13": {"skyscraper_8", "skyscraper_12", "skyscraper_14", "skyscraper_18"},
    "skyscraper_14": {"skyscraper_9", "skyscraper_13", "skyscraper_19"},
    "skyscraper_15": {"skyscraper_10", "skyscraper_16", "darkcity_4"},
    "skyscraper_16": {"skyscraper_11", "skyscraper_15", "skyscraper_17", "darkcity_5"},
    "skyscraper_17": {"skyscraper_12", "skyscraper_16", "skyscraper_18", "darkcity_0"},
    "skyscraper_18": {"skyscraper_13", "skyscraper_17", "skyscraper_19","darkcity_0"},
    "skyscraper_19": {"skyscraper_14", "skyscraper_18", "suburbs_10", "darkcity_1"},

    # -------- Dark City --------
    "darkcity_0": {"darkcity_6", "skyscraper_17","skyscraper_18"},
    "darkcity_1": {"skyscraper_10", "darkcity_2", "suburbs_17"},
    "darkcity_2": {"skyscraper_10", "darkcity_1", "darkcity_3"},
    "darkcity_3": {"darkcity_2", "darkcity_6", "darkcity_1"},
    "darkcity_4": {"darkcity_5", "skyscraper_15"},
    "darkcity_5": {"darkcity_4", "darkcity_6", "skyscraper_16"},
    "darkcity_6": {"darkcity_3", "darkcity_5", "darkcity_0", "island_1"},

    # -------- Stores --------
    "pet_store": {"suburbs_7","suburbs_8"},
    "florist": {"island_1", "island_2"},
    "gelateria": {"island_2", "island_4"},
    "music_store": {"island_2", "island_3"},
    "bank": {"island_1", "darkcity_6"},
    "pharmacy": {"suburbs_11", "island_2", "party_store"},
    "salon": {"suburbs_12", "suburbs_13", "bakery"},
    "hardware_store": {"suburbs_10", "suburbs_0"},
    "barber_shop": {"suburbs_4", "suburbs_5"},
    "jewelry_store": {"suburbs_11", "suburbs_6"},
    "party_store": {"suburbs_11", "suburbs_2", "pharmacy"},
    "bike_shop": {"suburbs_16", "suburbs_15"},
    "bakery": {"suburbs_16", "suburbs_12", "salon"},
    "post_office": {"suburbs_10"},
    "gym": {"darkcity_6", "darkcity_3"},
    "craft_shop": {"darkcity_5", "darkcity_6"},
    "pizzeria": {"darkcity_6", "bookstore"},
    "dentist": {"darkcity_5", "skyscraper_16"},
    "tech_shop": {"skyscraper_12", "skyscraper_13"},
    "bookstore": {"darkcity_0", "darkcity_6", "pizzeria"},
    "grocery_store": {"skyscraper_11", "skyscraper_6",},
    "clothing_store": {"skyscraper_6", "skyscraper_7",},
    "noodle_house": {"skyscraper_13", "skyscraper_8",},
    "toy_store": {"skyscraper_9", "skyscraper_14",},
    "cafe": {"skyscraper_17", "skyscraper_16",},
    "burger_joint": {"skyscraper_17", "skyscraper_18",},
}



# collected by goes west to east, north to south
# this is in relation to the NamedStores Container
road_pos_dict: Dict[str, Tuple[float, float, float]] = {
    "island_0": (86.8, -45, 164),
    "island_1": (67.1, -45, 140.5),
    "island_2": (-12, -45, 119.8),
    "island_3": (-95.3, -45, 119.6),
    "island_4": (-12, -45, 164),
    "island_5": (-95.3, -45, 164),
    "suburbs_0": (28.22, -45, -108.76),
    "suburbs_1": (14.7, -45,-139.3),
    "suburbs_2": (-66.7, -45, -139),
    "suburbs_3": (-96.1, -45, -139),
    "suburbs_4": (3.4, -45, -70.6),
    "suburbs_5": (-11.8, -45, -83.8),
    "suburbs_6": (-11.8, -45, -54.8),
    "suburbs_7": (-26.2, -45, 70.6),
    "suburbs_8": (-65.6, -45, -69.1),
    "suburbs_9": (-96.8, -45, -39.8),
    "suburbs_10": (28.2, -45, -0.8),
    "suburbs_11": (-12.2, -45, -0.3),
    "suburbs_12": (-32.4, -45, 0),
    "suburbs_13": (-32.4, -45, -35.5),
    "suburbs_14": (-51, -45, -49.1),
    "suburbs_15": (-70.7, -45, -35.5),
    "suburbs_16": (-72, -45, 0),
    "suburbs_17": (-96.7, -45, 0),
    "skyscraper_0": (167.9, -45, -78.8),
    "skyscraper_1": (141.9, -45, -78.8),
    "skyscraper_2": (110.6, -45, -78.8),
    "skyscraper_3": (82.3, -45, -78.8),
    "skyscraper_4": (59.9, -45, -90.4),
    "skyscraper_5": (167.9, -45, -55.4),
    "skyscraper_6": (141.9, -45, -55.4),
    "skyscraper_7": (110.6, -45, -55.4),
    "skyscraper_8": (82.3, -45, -55.4),
    "skyscraper_9": (59.9, -45, -55.4),
    "skyscraper_10": (167.9, -45, -27.1),
    "skyscraper_11": (141.9, -45, -27.1),
    "skyscraper_12": (110.6, -45, -27.1),
    "skyscraper_13": (82.3, -45, -27.1),
    "skyscraper_14": (59.9, -45, -27.1),
    "skyscraper_15": (167.9, -45, 0),
    "skyscraper_16": (141.9, -45, 0),
    "skyscraper_17": (110.6, -45, 0),
    "skyscraper_18": (82.3, -45, 0),
    "skyscraper_19": (59.9, -45, 0),
    "darkcity_0": (108.26, -45, 19.03),
    "darkcity_1": (39.6, -45, 30),
    "darkcity_2": (17.1, -45, 30),
    "darkcity_3": (28.1, -45, 39.6),
    "darkcity_4": (167.8, -45, 86),
    "darkcity_5": (140.5, -45, 86),
    "darkcity_6": (74, -45, 80.9),
    
}

post_close_set = {"pharmacy", "bakery", "gym", "hardware_store", "burger_joint", "bookstore", "toy_store", "bike_shop"}


close_store_dict: Dict[str, List[str]] = {
    "barber_shop": ["jewelry_store", "bike_shop", "bakery", "hardware_store", "toy_store"],
    "jewelry_store": ["barber_shop", "bike_shop", "bakery"],
    "bike_shop": ["barber_shop", "jewelry_store", "bakery", "pharmacy"],
    "bakery": ["barber_shop", "jewelry_store", "bike_shop", "pharmacy", "hardware_store"],
    "pharmacy": ["bakery", "gym", "bike_shop", "music_store", "hardware_store", "florist"],
    "music_store": ["florist", "pharmacy", "pet_store"],
    "florist": ["music_store", "pharmacy", "pet_store", "craft_store", "gym"],
    "pet_store": ["music_store", "pet_store", "craft_store", "florist"],
    "craft_store": ["music_store", "pet_store", "florist", "gym", "dentist", "pizzeria", "cafe"],
    "gym": ["pharmacy", "craft_store", "hardware_store", "florist"],
    "hardware_store": ["pharmacy", "gym", "barber_shop", "bakery", "toy_store"],
    "toy_store": ["hardware_store", "clothing_store", "grocery_store", "cafe"],
    "clothing_store": ["hardware_store", "toy_store", "grocery_store", "cafe"],
    "grocery_store": ["hardware_store", "toy_store", "clothing_store", "cafe"],
    "cafe": ["hardware_store", "toy_store", "clothing_store", "grocery_store", "dentist", "pizzeria"],
    "pizzeria": ["cafe", "dentist", "craft_shop"],
    "dentist": ["cafe", "pizzeria", "craft_shop"],
}

store_quadrant_dict: Dict[str, List[str]] = {
    "suburbs": ["barber_shop", "jewelry_store", "bike_shop", "bakery", "hardware_store", "party_store","salon", "pet_store"],
    "skyscraper": ["toy_store", "clothing_store", "grocery_store", "cafe", "noodle_house", "tech_shop"],
    "darkcity": ["pizzeria", "dentist", "craft_shop", "gym", "bookstore", "burger_joint"],
    "island": ["florist", "music_store", "pharmacy", "bank","gelateria"],
}

quadrant_transition_dict: Dict[str, List[str]] = {
    "suburbs": ["skyscraper", "island"],
    "skyscraper": ["suburbs", "darkcity"],
    "darkcity": ["skyscraper", "island"],
    "island": ["darkcity", "suburbs"],
}

DEBUG = False


def store_distance(store1: Tuple[float, float, float], store2: Tuple[float, float, float]) -> float:
    """Return Euclidean distance between two stores."""
    x1, y1, z1 = store1
    x2, y2, z2 = store2
    return (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2


def route_distances(route: List[str]) -> Dict[Tuple[str, str], float]:
    """
    Given a route (list of store names), return a dictionary:
    {
        (store_i, store_i+1): distance,
        ...
    }
    """
    dist_dict: Dict[Tuple[str, str], float] = {}

    for i in range(len(route) - 1):
        a = route[i]
        b = route[i + 1]
        storea = store_pos_dict[a]
        storeb = store_pos_dict[b]
        
        dist_dict[(a, b)] = store_distance(storea, storeb)

    return dist_dict

def get_trial_stores_radius(
    all_stores: List[str],
    num_deliveries: int,
    rng: random.Random,
    dist_df: pd.DataFrame,
    *,
    radius: Optional[float] = None,
    k_nearest: Optional[int] = 5,
) -> List[str]:
    """
    Build a trial route by preferring nearby next-stores using dist_df.

    Parameters
    ----------
    all_stores : list of store names (may include "post_office")
    num_deliveries : number of deliveries (stores visited between post_office endpoints)
    rng : random.Random for reproducible randomness
    dist_df : pandas DataFrame with index/columns = store labels, values = distances
    radius : if provided, prefer candidates with distance <= radius
    k_nearest : among qualifying candidates, restrict to the k nearest then sample uniformly;
                set to None to sample from all qualifying candidates.

    Returns
    -------
    trial_stores : list like ["post_office", ..., "post_office"]
    """
    # Start with all non–post office stores
    unvisited = [s for s in all_stores if s != "post_office"]
    trial_stores: List[str] = ["post_office"]

    if num_deliveries <= 0 or not unvisited:
        return trial_stores

    # Pick first store randomly
    current_store = unvisited.pop(rng.randrange(len(unvisited)))
    trial_stores.append(current_store)

    # Fill remaining deliveries
    for _ in range(1, num_deliveries):
        if not unvisited:
            break

        next_store: Optional[str] = None

        # --- Prefer a nearby unvisited store based on dist_df ---
        if current_store in dist_df.index:
            # distances from current_store to each unvisited candidate
            dists = dist_df.loc[current_store, unvisited]

            # drop missing / inf
            dists = dists.replace([float("inf")], pd.NA).dropna()

            if len(dists) > 0:
                # optional radius filter
                if radius is not None:
                    dists = dists[dists <= radius]

                if len(dists) > 0:
                    # take k nearest then sample randomly among them
                    dists_sorted = dists.sort_values()
                    if k_nearest is not None:
                        dists_sorted = dists_sorted.iloc[: max(1, min(k_nearest, len(dists_sorted)))]

                    candidates = list(dists_sorted.index)
                    next_store = candidates[rng.randrange(len(candidates))]

        # Fallback: pick any unvisited store at random
        if next_store is None:
            next_store = unvisited[rng.randrange(len(unvisited))]

        unvisited.remove(next_store)
        trial_stores.append(next_store)
        current_store = next_store

    trial_stores.append("post_office")
    return trial_stores

def get_total_list_radius(all_stores: List[str], num_trials: int, num_deliveries: int, rng: random.Random, dist_df, radius, k_nearest) -> List[List[str]]:
    return [get_trial_stores_radius(all_stores, num_deliveries, rng, dist_df=dist_df, radius=radius, k_nearest=k_nearest) for _ in range(num_trials)]

from collections import defaultdict

def generate_quadrant_path(rng: random.Random) -> List[str]:
    """Generate a random path that visits each quadrant exactly once,
    respecting transitions in quadrant_transition_dict."""
    quadrants = list(quadrant_transition_dict.keys())
    target_length = len(quadrants)

    if target_length == 0:
        return []

    while True:
        visited = set()
        path: List[str] = []

        current = quadrants[rng.randrange(len(quadrants))]
        visited.add(current)
        path.append(current)

        while len(path) < target_length:
            neighbors = quadrant_transition_dict.get(current)
            if not neighbors:
                break

            candidates = [q for q in neighbors if q not in visited]
            if not candidates:
                break

            nxt = candidates[rng.randrange(len(candidates))]
            visited.add(nxt)
            path.append(nxt)
            current = nxt

        if len(path) == target_length:
            return path
        # else: retry


def compute_quadrant_delivery_counts(
    num_deliveries: int,
    quadrant_path: List[str],
    rng: random.Random,
) -> Dict[str, int]:
    """Compute deliveries per quadrant:
    - baseCount = numDeliveries / numQuadrants
    - remainder = numDeliveries % numQuadrants
    - randomly distribute +1 to 'remainder' quadrants
    """
    counts: Dict[str, int] = {}

    unique_quads = list(dict.fromkeys(quadrant_path))  # preserve order, unique
    num_quadrants = len(unique_quads)

    if num_quadrants == 0 or num_deliveries <= 0:
        for q in unique_quads:
            counts[q] = 0
        return counts

    base_count = num_deliveries // num_quadrants
    remainder = num_deliveries % num_quadrants

    for q in unique_quads:
        counts[q] = base_count

    if remainder > 0:
        shuffled = unique_quads[:]  # copy
        rng.shuffle(shuffled)
        for i in range(remainder):
            q = shuffled[i]
            counts[q] += 1

    return counts


def get_trial_stores_quadrant(all_stores: List[str], num_deliveries: int, rng: random.Random) -> List[str]:
    """Generate a trial path of stores:
    - Visits quadrants in a random valid path (each quadrant exactly once)
    - Visits (numDeliveries / numQuadrants) stores per quadrant, with remainder distributed randomly
    - Uses close_store_dict to keep each next store close to the previous one when possible
    """
    trial_stores: List[str] = []
    trial_stores.append("post_office")

    # Filter out post office from pool of candidate stores
    all_store_set = {s for s in all_stores if s != "post_office"}

    if num_deliveries <= 0 or not all_store_set:
        trial_stores.append("post_office")
        return trial_stores

    # Build inverse mapping: store -> quadrant
    store_to_quadrant: Dict[str, str] = {}
    for quad, stores in store_quadrant_dict.items():
        for store in stores:
            store_to_quadrant[store] = quad

    # Unvisited stores that we can actually use (must have quadrant mapping)
    unvisited = {s for s in all_store_set if s in store_to_quadrant}

    if not unvisited:
        trial_stores.append("post_office")
        return trial_stores

    # 1) Quadrant path
    quadrant_path = generate_quadrant_path(rng)
    num_quadrants = len(quadrant_path)
    if num_quadrants == 0:
        # Fallback: just sample stores randomly
        pool = list(unvisited)
        while len(trial_stores) < num_deliveries and pool:
            idx = rng.randrange(len(pool))
            trial_stores.append(pool.pop(idx))
        trial_stores.append("post_office")
        return trial_stores

    # 2) Deliveries per quadrant
    deliveries_per_quad = compute_quadrant_delivery_counts(num_deliveries, quadrant_path, rng)

    current_store: Optional[str] = None

    # 3) Walk quadrants in path order and pick stores
    for quad in quadrant_path:
        deliveries_in_quad = deliveries_per_quad.get(quad, 0)
        if deliveries_in_quad <= 0:
            continue

        for _ in range(deliveries_in_quad):
            if not unvisited:
                break

            next_store: Optional[str] = None

            stores_in_quad = [
                s for s in store_quadrant_dict.get(quad, []) if s in unvisited
            ]

            # First store overall: pick random in this quadrant
            if current_store is None:
                if not stores_in_quad:
                    break
                idx = rng.randrange(len(stores_in_quad))
                next_store = stores_in_quad[idx]
            else:
                # Prefer an unvisited neighbor that is also in this quadrant
                neighbors = close_store_dict.get(current_store)
                if neighbors:
                    neighbor_candidates = [
                        s
                        for s in neighbors
                        if s in unvisited and store_to_quadrant.get(s) == quad
                    ]
                    if neighbor_candidates:
                        idx = rng.randrange(len(neighbor_candidates))
                        next_store = neighbor_candidates[idx]

                # If no close neighbor in this quadrant, pick any unvisited store in this quadrant
                if next_store is None and stores_in_quad:
                    idx = rng.randrange(len(stores_in_quad))
                    next_store = stores_in_quad[idx]

            # Final fallback: any unvisited store at all
            if next_store is None:
                pool = list(unvisited)
                if not pool:
                    break
                idx = rng.randrange(len(pool))
                next_store = pool[idx]

            if next_store not in unvisited:
                continue

            unvisited.remove(next_store)
            trial_stores.append(next_store)
            current_store = next_store

            if len(trial_stores) >= num_deliveries:
                break

        if len(trial_stores) >= num_deliveries:
            break

    trial_stores.append("post_office")
    return trial_stores

def get_total_list_quad(all_stores: List[str], num_trials: int, num_deliveries: int, rng: random.Random) -> List[List[str]]:
    return [get_trial_stores_quadrant(all_stores, num_deliveries, rng) for _ in range(num_trials)]


def get_trial_stores_pure(all_stores: List[str], num_deliveries: int, rng: random.Random) -> Tuple[bool, List[str]]:
    unvisited = [s for s in all_stores if s != "post_office"]
    trial_stores: List[str] = []
    trial_stores.append("post_office")
    success = True

    if not unvisited or num_deliveries <= 0:
        return success, trial_stores

    # Pick first store randomly
    random_index = rng.randrange(len(unvisited))
    next_store = unvisited.pop(random_index)
    trial_stores.append(next_store)

    for _ in range(num_deliveries - 1):
        if not unvisited:
            break
        # Relaxed: pick any unvisited store, regardless of visibility
        next_store = unvisited.pop(rng.randrange(len(unvisited)))
        trial_stores.append(next_store)
    trial_stores.append("post_office")
    return success, trial_stores


def list_generator_pure(all_stores: List[str], num_deliveries: int, rng: random.Random) -> List[List[str]]:
    total: List[List[str]] = []
    while len(total) < len(all_stores):
        success, trial = get_trial_stores_pure(all_stores, num_deliveries, rng)
        if success:
            total.append(trial)
    return total


def list_check_pure(list1: List[str], list2: List[str]) -> bool:
    transitions: Dict[str, str] = {}
    for i in range(len(list1) - 1):
        transitions[list1[i]] = list1[i + 1]

    for i in range(len(list2) - 1):
        prev_next = transitions.get(list2[i])
        if prev_next is not None and prev_next == list2[i + 1]:
            return False
    return True


def get_total_list_simple(
    all_stores: List[str],
    num_trials: int,
    num_deliveries: int,
    rng: random.Random,
    pool: Optional[List[List[str]]] = None,
) -> List[List[str]]:
    # Fast greedy fallback generator: try to build numTrials lists quickly from a candidate pool.
    # If pool is null or insufficient, it will generate a fresh pool via list_generator_pure.
    if not pool:
        pool = list_generator_pure(all_stores, num_deliveries, rng)

    pool_size = len(pool)
    if pool_size == 0:
        return []

    store_count = len(all_stores)
    name_to_index = {name: i for i, name in enumerate(all_stores)}

    # pairSets in C# is computed but not actually used in the later logic;
    # we keep the computation for parity but don't use it.
    pair_sets: List[set[int]] = []
    for lst in pool:
        s = set()
        for k in range(len(lst) - 1):
            a = name_to_index[lst[k]]
            b = name_to_index[lst[k + 1]]
            key = (a << 16) | b
            s.add(key)
        pair_sets.append(s)

    # Greedy fallback: choose as many lists as possible that are pairwise-compatible under list_check_pure
    fallback: List[List[str]] = []
    used2 = [False] * pool_size

    for i in range(pool_size):
        if len(fallback) >= num_trials:
            break
        if used2[i]:
            continue

        candidate = pool[i]
        compatible = True
        for existing in fallback:
            if not list_check_pure(existing, candidate):
                compatible = False
                break

        if compatible:
            fallback.append(candidate)
            used2[i] = True

    # If not enough lists, fill remainder with random ones from pool
    if len(fallback) < num_trials:
        remaining_indices = [i for i in range(pool_size) if not used2[i]]
        while len(fallback) < num_trials and remaining_indices:
            idx = rng.randrange(len(remaining_indices))
            chosen_index = remaining_indices.pop(idx)
            fallback.append(pool[chosen_index])

    if DEBUG:
        print("Finished creating store lists (fallback)")

    return fallback


# def get_adjacency_matrix(
#     dist_df: pd.DataFrame,
#     include_list=None,
#     exclude_list=None,
# ):
#     include_list = include_list or []
#     exclude_list = exclude_list or []

#     include_set = set(include_list)
#     exclude_set = set(exclude_list)

#     # ---- Mutual exclusivity rule ----
#     if include_set and exclude_set:
#         raise ValueError(
#             "include_list and exclude_list cannot both be non-empty. "
#             "Specify only one filtering mode."
#         )

#     # ---- Available stores from dist_df ----
#     # (Prefer index; assume columns match index.)
#     all_stores = list(dist_df.index)

#     # ---- Filtering logic ----
#     if include_set:
#         stores = [s for s in all_stores if s in include_set]
#     elif exclude_set:
#         stores = [s for s in all_stores if s not in exclude_set]
#     else:
#         stores = all_stores

#     if len(stores) == 0:
#         return np.empty((0, 0), dtype=float), np.array([], dtype=object)

#     # ---- Slice labeled distance matrix and return numpy ----
#     sub_df = dist_df.loc[stores, stores]

#     # If you want to guarantee float dtype:
#     D = sub_df.to_numpy(dtype=float)

#     return D, np.array(stores, dtype=object)

    
import time
import random
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

class NoImproveStopper(pywrapcp.SearchMonitor):
    def __init__(self, solver, routing, patience_s: float):
        super().__init__(solver)
        self._routing = routing
        self._patience = float(patience_s)
        self._best = None
        self._last_improve = time.time()

    def AtSolution(self):
        obj = self._routing.CostVar().Value()
        now = time.time()

        if self._best is None or obj < self._best:
            self._best = obj
            self._last_improve = now
        elif now - self._last_improve >= self._patience:
            self.solver().FinishCurrentSearch()


def solve_k_tsp_path_ortools(
    D,
    start=None,
    end=None,
    K=8,
    time_limit=10,              # keep a hard cap (recommended)
    no_improve_limit=None,      # e.g., 30 (seconds). None disables.
    rng=None,                   # pass random.Random(seed) if you want reproducible
):
    import random
    from typing import List, Tuple, Dict, Optional
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.cm as cm
    import math
    N = D.shape[0]
    if not (2 <= K <= N):
        raise ValueError(f"K must be in [2, N]. Got K={K}, N={N}.")

    if rng is None:
        rng = random.Random()

    # Decide mode + pick missing endpoints if needed
    if start is None and end is None:
        mode = "tour"
        start = rng.randrange(N)
        end = start
    else:
        mode = "path"
        if start is None:
            start = rng.randrange(N)
        if end is None:
            end = rng.randrange(N - 1)
            if end >= start:
                end += 1
        if start == end:
            raise ValueError("start and end must be different for an open path.")

    manager = pywrapcp.RoutingIndexManager(N, 1, [start], [end])
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(i, j):
        return int(D[manager.IndexToNode(i), manager.IndexToNode(j)])

    dist_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_idx)

    SKIP_PENALTY = 10**9
    for node in range(N):
        if node not in (start, end):
            routing.AddDisjunction([manager.NodeToIndex(node)], SKIP_PENALTY)

    def one_cb(i, j):
        return 1

    one_idx = routing.RegisterTransitCallback(one_cb)
    target_arcs = (K if mode == "tour" else K - 1)

    routing.AddDimension(
        one_idx,
        slack_max=0,
        capacity=target_arcs,
        fix_start_cumul_to_zero=True,
        name="count"
    )
    count_dim = routing.GetDimensionOrDie("count")
    count_dim.CumulVar(routing.End(0)).SetValue(target_arcs)

    # ---- search params ----
    params = pywrapcp.DefaultRoutingSearchParameters()
    if time_limit is not None:
        params.time_limit.seconds = int(time_limit)

    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

    # ---- stop-if-no-improvement monitor ----
    if no_improve_limit is not None:
        monitor = NoImproveStopper(routing.solver(), routing, no_improve_limit)
        routing.solver().AddSearchMonitor(monitor)

    sol = routing.SolveWithParameters(params)
    if sol is None:
        return None, None

    # Extract path + cost
    path = []
    idx = routing.Start(0)
    total_cost = 0
    visited = set()

    while not routing.IsEnd(idx):
        node = manager.IndexToNode(idx)
        path.append(node)
        visited.add(node)
        nxt = sol.Value(routing.NextVar(idx))
        total_cost += routing.GetArcCostForVehicle(idx, nxt, 0)
        idx = nxt

    # end_node = manager.IndexToNode(idx)
    # path.append(end_node)
    # visited.add(end_node)

    if len(visited) != K:
        raise RuntimeError(f"Expected K={K} unique nodes, got {len(visited)}. Mode={mode}. Path={path}")

    return path, total_cost
    
def run_tsp_scoped(D, store_sample, K):
    import random
    from typing import List, Tuple, Dict, Optional
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.cm as cm
    import math
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    path, total_cost = solve_k_tsp_path_ortools(D=D, K=K, time_limit=100)
    # save path to text file
    store_sequence = [store_sample[i] for i in path]
    line = ", ".join(store_sequence) + f" : {total_cost}"
    print(f"Saved path: {line}")
    return line
 
import heapq
import math

def dijkstra_paths_to_stores(adj, src, stores):
    """
    Single-source Dijkstra: shortest paths from src to each store in `stores`.

    Parameters
    ----------
    adj : dict[str, list[tuple[str, float]]]
        Weighted adjacency list (only legal moves included).
    src : str
        Source node name (typically a store).
    stores : Iterable[str]
        Store node names you want outputs for.

    Returns
    -------
    dist_to_store : dict[str, float]
        Shortest distance from src to each store (math.inf if unreachable).
    path_to_store : dict[str, list[str] | None]
        Shortest path (node-name sequence) from src to each store, or None if unreachable.
    """
    stores = list(stores)

    # initialize distances / predecessors for all nodes we might touch
    dist = {node: math.inf for node in adj}
    prev = {node: None for node in adj}

    if src not in adj:
        raise KeyError(f"src '{src}' not in adj")

    dist[src] = 0.0
    pq = [(0.0, src)]

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue

        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    def reconstruct(tgt):
        if tgt not in dist or dist[tgt] == math.inf:
            return None
        path = []
        cur = tgt
        while cur is not None:
            path.append(cur)
            if cur == src:
                break
            cur = prev[cur]
        path.reverse()
        return path if path and path[0] == src else None

    dist_to_store = {s: dist.get(s, math.inf) for s in stores}
    path_to_store = {s: reconstruct(s) for s in stores}

    return dist_to_store, path_to_store



def make_bidirectional(node_adj_dict):
    """
    Ensure for every edge u->v we also have v->u.
    Returns a dict[str, set[str]].
    """
    bidir = defaultdict(set)
    for u, nbrs in node_adj_dict.items():
        for v in nbrs:
            bidir[u].add(v)
            bidir[v].add(u)
    return dict(bidir)

def build_weighted_adj_from_topology(node_adj_dict, node_pos_dict, dist_fn):
    """
    Build weighted adjacency list using only edges in node_adj_dict.
    Missing edges are treated as infinite because they won't appear in adj[u].
    """
    adj_w = defaultdict(list)

    for u, nbrs in node_adj_dict.items():
        if u not in node_pos_dict:
            raise KeyError(f"Missing position for node '{u}'")
        pu = node_pos_dict[u]

        for v in nbrs:
            if v not in node_pos_dict:
                raise KeyError(f"Missing position for node '{v}'")
            pv = node_pos_dict[v]

            w = float(dist_fn(pu, pv))
            adj_w[u].append((v, w))

    return dict(adj_w)


def subset_dist_matrix(dist_mat, store_names, include_list):
    """
    Returns (sub_mat, stores) where stores == include_list in that order.
    """
    idx = {name: i for i, name in enumerate(store_names)}
    inds = [idx[s] for s in include_list]
    sub_mat = dist_mat[np.ix_(inds, inds)]
    return sub_mat, include_list


# if __name__ == "__main__":
#     # run tsp
#     parser = argparse.ArgumentParser(description="Generate K-node TSP routes")

#     parser.add_argument("--seed", type=int, default=123,
#                         help="Random seed for reproducibility")
#     parser.add_argument("--K", type=int, default=15,
#                         help="Number of stores per route")
#     parser.add_argument("--num-routes", type=int, default=5000,
#                         help="Number of TSP routes to generate")
#     parser.add_argument("--out", type=str, default="tsp_path_all.txt",
#                         help="Output file name")

#     args = parser.parse_args()
#     K = args.K
#     num_routes = args.num_routes
#     rng = random.Random(args.seed)
#     out_file = args.out
#     seen = set()
#     stores_og = [s for s in store_pos_dict.keys() if s != "post_office"]
#     stores_sample_list = []
#     if K > len(stores_og):
#         raise ValueError(f"K={K} exceeds available stores ({len(stores_og)})")
#     adj_mats = []
#     for i in range(num_routes):
#         print(f"Route {i+1}/{num_routes}")
#         tries = 0
#         while True:
#             include_list = tuple(sorted(rng.sample(stores_og, k=K)))
#             if include_list not in seen:
#                 seen.add(include_list)
#                 break
#             tries += 1
#             if tries > 1000:
#                 raise RuntimeError("Too many duplicate subsets")
#         adj_mat, stores = get_adjacency_matrix(store_pos_dict, list(include_list))
#         line = run_tsp(adj_mat, stores, K)
#         with open(out_file, "a") as f:
#             f.write(line + "\n")


def worker_generate_and_solve(
    *,
    job_i: int,
    num_routes_job: int,
    K: int,
    seed: int,
    base_out_path: str,
    store_names: list,
    dist_mat,
):
    """
    Each worker:
      - samples num_routes_job subsets of size K
      - builds KxK submatrix from precomputed dist_mat
      - runs TSP
      - writes lines to base_out_path_{job_i}.txt
    """
    import random
    import numpy as np

    # local RNG per worker (distinct but reproducible)
    rng = random.Random(seed + job_i * 10_000)

    out_path = f"{base_out_path}_{job_i}.txt"

    # fast name->index mapping
    idx = {name: i for i, name in enumerate(store_names)}

    def subset_dist_matrix_local(include_list):
        inds = [idx[s] for s in include_list]
        return dist_mat[np.ix_(inds, inds)]

    seen = set()

    
    for _ in range(num_routes_job):
        # sample a unique subset within this worker (local uniqueness)
        tries = 0
        while True:
            include_list = tuple(sorted(rng.sample(store_names, k=K)))
            has_post_close_stores = include_list[0] in post_close_set and include_list[K - 1] in post_close_set
            if include_list not in seen and has_post_close_stores:
                seen.add(include_list)
                break
            tries += 1
            if tries > 1000:
                # extremely unlikely unless K small and num_routes_job huge
                raise RuntimeError(f"[job {job_i}] Too many duplicate subsets")

        stores_k = list(include_list)  # deterministic order for indexing
        Dk = subset_dist_matrix_local(stores_k)

        # optional: skip disconnected subsets
        if not np.isfinite(Dk).all():
            continue
        
        line = run_tsp_scoped(Dk, stores_k, K)
        with open(out_path, "a") as f:
            f.write(line + "\n")

    return out_path


import os
import cmldask.CMLDask as da
from dask.distributed import wait, as_completed, progress

if __name__ == "__main__":
    # run tsp
    parser = argparse.ArgumentParser(description="Generate K-node TSP routes")

    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed for reproducibility")
    parser.add_argument("--K", type=int, default=15,
                        help="Number of stores per route")
    parser.add_argument("--num-routes", type=int, default=5000,
                        help="Number of TSP routes to generate")
    parser.add_argument("--base_out", type=str, default="tsp_path_25_15",
                    help="Base path for per-job outputs (suffix _{i}.txt added)")
    parser.add_argument("--n_jobs", type=int, default=24,
                        help="Number of cluster jobs")
    parser.add_argument("--ram", type=str, default="3GB",
                        help="Output file name")
      

    args = parser.parse_args()
    
    
    logdir = os.path.join(os.path.abspath(os.curdir), 'dask_appendix_logs')
    # first_run = False
    dask_args = {'job_name': "deliverystore", 'memory_per_job': args.ram, 'max_n_jobs': args.n_jobs,
                 'log_directory': '~/ValueCourierAnalysis'}
    os.makedirs(dask_args['log_directory'], exist_ok=True)
    client = da.new_dask_client(**dask_args)
    client.upload_file("deliveryRoutes.py")
    
    
    K = args.K
    num_routes = args.num_routes
    rng = random.Random(args.seed)
    seen = set()
    node_pos_dict = {**road_pos_dict, **store_pos_dict}

    node_adj_dict_bidir = make_bidirectional(node_adj_dict)
    adj = build_weighted_adj_from_topology(node_adj_dict_bidir, node_pos_dict, store_distance)

    store_names = [s for s in store_pos_dict.keys() if s != "post_office"]
    S = len(store_names)
    dist_mat = np.full((S, S), math.inf, dtype=float)
    path_mat = [[None]*S for _ in range(S)]

    for i, src in enumerate(store_names):
        dist_to, path_to = dijkstra_paths_to_stores(adj, src, store_names)
        for j, tgt in enumerate(store_names):
            dist_mat[i, j] = dist_to[tgt]
            path_mat[i][j] = path_to[tgt]
    stores_sample_list = []
    adj_mats = []
    if K > len(store_names):
        raise ValueError(f"K={K} exceeds available stores ({len(store_names)})")

    # split num_routes across jobs
    n_jobs = args.n_jobs
    num_routes = args.num_routes
    base_out_path = args.base_out  # add this arg (see below)

    routes_per_job = [num_routes // n_jobs] * n_jobs
    for i in range(num_routes % n_jobs):
        routes_per_job[i] += 1

    # upload code once
    client.upload_file("deliveryRoutes.py")

    from deliveryRoutes import worker_generate_and_solve

    futures = []
    for job_i in range(n_jobs):
        if routes_per_job[job_i] == 0:
            continue
        fut = client.submit(
            worker_generate_and_solve,
            job_i=job_i,
            num_routes_job=routes_per_job[job_i],
            K=args.K,
            seed=args.seed,
            base_out_path=base_out_path,
            store_names=store_names,
            dist_mat=dist_mat,
            pure=False,
        )
        futures.append(fut)

    out_paths = client.gather(futures)
    print("Wrote:", out_paths)
    
    combine_path_files(base_out_path, n_jobs)

    # optional progress
    from dask.distributed import progress
    progress(futures)

    out_paths = client.gather(futures)
    
