import pandas as pd
import numpy as np
import math
from typing import Dict, List, Set, Tuple, Any
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from google.auth.transport.requests import Request
from google.auth.exceptions import GoogleAuthError
import traceback
# Import OpenRouteService for real routes
import openrouteservice
import json
import os

# Helper function for safe numeric conversion
def safe_float(value, default=0.0):
    """Safely convert a value to float, returning default if conversion fails."""
    if value is None or value == '':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

# Load OpenRouteService API key from config file
ORS_CONFIG_FILE = "ors_config.json"

def load_ors_config():
    """Load OpenRouteService configuration from the config file."""
    try:
        if os.path.exists(ORS_CONFIG_FILE):
            with open(ORS_CONFIG_FILE, 'r') as f:
                config = json.load(f)
                return config.get("api_key", "YOUR_ORS_API_KEY")
        else:
            print(f"Warning: ORS config file '{ORS_CONFIG_FILE}' not found.")
            return "YOUR_ORS_API_KEY"
    except Exception as e:
        print(f"Error loading ORS config: {e}")
        return "YOUR_ORS_API_KEY"

# Get route from OpenRouteService - similar to the function in KK.py
def get_route(start_coords, end_coords, via_points=None):
    """
    Get route from OpenRouteService between two points, optionally with via points.
    
    Args:
        start_coords: Tuple of (longitude, latitude) for start point
        end_coords: Tuple of (longitude, latitude) for end point
        via_points: Optional list of (longitude, latitude) tuples for waypoints
        
    Returns:
        GeoJSON of the route or None if there was an error
    """
    try:
        # Try to get OpenRouteService client
        ors_client = None
        
        # First check if we can access the client from KK.py
        try:
            # First try to import from KK module
            import KK
            ors_client = KK.ors_client
        except (ImportError, AttributeError):
            # If that fails, load API key from config file
            api_key = load_ors_config()
            if api_key and api_key != "YOUR_ORS_API_KEY":
                try:
                    ors_client = openrouteservice.Client(key=api_key)
                except Exception as e:
                    st.warning(f"Could not connect to OpenRouteService. Using straight lines instead. Error: {e}")
                    return None
            
        if ors_client is None:
            return None
        
        # Create coordinates list
        if via_points:
            coordinates = [start_coords] + via_points + [end_coords]
        else:
            coordinates = [start_coords, end_coords]
        
        # Get directions from OpenRouteService
        route = ors_client.directions(
            coordinates=coordinates,
            profile='driving-car',
            format='geojson'
        )
        return route
    except Exception as e:
        print(f"Error getting route: {e}")
        return None

# Constants - same as in the original script
DIMENSION_MAX = 4.4
MERGED_DIMENSION_MAX = 4.5  # Updated to 4.5 for Phase 3
WEIGHT_MAX = 1800  # 1.5 tons in KG
MAX_TRIP_DISTANCE = 60
MAX_CUSTOMER_DISTANCE = 20
SECOND_MERGE_THRESHOLD = 2.0
FAR_AWAY_THRESHOLD = 30
DEFAULT_DIMENSION = 0.1
DEFAULT_WEIGHT = 0.1
MAX_SUPPLIER_DISTANCE = 5  # Max distance between suppliers for merging (in km)
MAX_DIRECTION_DIFF = 0.5  # Max difference in direction (in radians) for merging

# Google Sheet Constants
SERVICE_ACCOUNT_FILE = "Credentials_clean.json"
SPREADSHEET_KEY = "138EPgPhd0Ddp1-z9xPCexsm5s--t5EHjPpK7NSTT1LI"
SPREADSHEET_URL = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_KEY}/edit"

# Initialize gspread client
def init_gspread_client():
    try:
        # Method 1: Try to load directly but with proper handling for BOM
        try:
            # Open file in text mode and let Python handle decoding
            with open(SERVICE_ACCOUNT_FILE) as f:
                import json
                try:
                    service_account_info = json.loads(f.read())
                except json.decoder.JSONDecodeError:
                    # If regular loading fails, try again with utf-8-sig encoding
                    f.seek(0)
                    text = f.read()
                    service_account_info = json.loads(text.encode().decode('utf-8-sig'))
                    
            # Create credentials from parsed JSON
            creds = Credentials.from_service_account_info(
                service_account_info,
                scopes=["https://www.googleapis.com/auth/spreadsheets"]
            )
        except (UnicodeDecodeError, json.decoder.JSONDecodeError):
            # Method 2: If that fails, try binary mode approach
            with open(SERVICE_ACCOUNT_FILE, 'rb') as f:
                content = f.read()
                try:
                    json_content = content.decode('utf-8-sig')
                except UnicodeDecodeError:
                    json_content = content.decode('utf-8', errors='ignore')
                
                service_account_info = json.loads(json_content)
                
            creds = Credentials.from_service_account_info(
                service_account_info,
                scopes=["https://www.googleapis.com/auth/spreadsheets"]
            )
            
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"Authentication error: {e}")
        # Show specific error details to make debugging easier
        st.code(traceback.format_exc())
        return None
# Load data from Google Sheets
@st.cache_data(ttl=600)
def load_sheet_data(sheet_name):
    client = init_gspread_client()
    if client is None:
        return None
    try:
        ss = client.open_by_url(SPREADSHEET_URL)
        ws = ss.worksheet(sheet_name)
        data = ws.get_all_values()
        if not data:
            st.error(f"{sheet_name} sheet is empty.")
            return None
        
        # Use the first row as headers
        headers = data[0]
        
        # Handle empty and duplicate headers
        clean_headers = []
        for i, header in enumerate(headers):
            if header == '':
                clean_headers.append(f"column_{i}")
            else:
                # Count occurrences of this header so far
                count = clean_headers.count(header)
                if count > 0:
                    clean_headers.append(f"{header}_{count+1}")
                else:
                    clean_headers.append(header)
        
        # Create DataFrame with cleaned headers
        return pd.DataFrame(data[1:], columns=clean_headers)
    except gspread.WorksheetNotFound:
        st.error(f"The '{sheet_name}' sheet is not found.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Helper functions
def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula."""
    R = 6371  # Earth's radius in km
    
    # Convert string inputs to float using safe_float to avoid errors
    lat1 = safe_float(lat1)
    lon1 = safe_float(lon1)
    lat2 = safe_float(lat2)
    lon2 = safe_float(lon2)
    
    # Check if any values are invalid (0 is often a placeholder for missing data)
    if lat1 == 0 and lon1 == 0 or lat2 == 0 and lon2 == 0:
        return 0.0
    
    # Now perform the calculation with properly converted values
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) * math.sin(d_lat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(d_lon / 2) * math.sin(d_lon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing (direction) between two points."""
    # Convert string inputs to float
    lat1 = safe_float(lat1)
    lon1 = safe_float(lon1)
    lat2 = safe_float(lat2)
    lon2 = safe_float(lon2)
    
    d_lon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    y = math.sin(d_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    bearing = math.atan2(y, x)
    # Normalize to 0-2π
    bearing = (bearing + 2 * math.pi) % (2 * math.pi)
    return bearing

def calculate_centroid(orders, cust_lat_idx, cust_lon_idx):
    """Calculate average coordinates for a cluster or trip."""
    if len(orders) == 0:
        return {"lat": 0, "lon": 0}
    
    # Convert string values to float before summing
    avg_lat = sum(safe_float(order[cust_lat_idx]) for order in orders) / len(orders)
    avg_lon = sum(safe_float(order[cust_lon_idx]) for order in orders) / len(orders)
    return {"lat": avg_lat, "lon": avg_lon}

def check_customer_distance(trip, new_order, distance_cache, cust_lat_idx, cust_lon_idx, id_idx, max_distance=MAX_CUSTOMER_DISTANCE):
    """Check if a new order meets customer distance constraints with existing trip orders."""
    for order in trip["orders"]:
        key = f"{order[id_idx]}-{new_order[id_idx]}"
        if key in distance_cache:
            distance = distance_cache[key]
        else:
            # Use safe_float to avoid conversion errors
            order_lat = safe_float(order[cust_lat_idx])
            order_lon = safe_float(order[cust_lon_idx])
            new_order_lat = safe_float(new_order[cust_lat_idx])
            new_order_lon = safe_float(new_order[cust_lon_idx])
            
            # Skip distance check if coordinates are invalid (0, 0)
            if (order_lat == 0 and order_lon == 0) or (new_order_lat == 0 and new_order_lon == 0):
                continue
                
            distance = haversine(
                order_lat, order_lon,
                new_order_lat, new_order_lon
            )
            distance_cache[key] = distance
        
        if distance > max_distance:
            return False
    return True

def optimize_route(trip, supplier_locations, cust_lat_idx, cust_lon_idx):
    """Optimize route using nearest-neighbor algorithm."""
    if len(trip["orders"]) <= 1:
        return trip["orders"]
    
    sorted_orders = []
    remaining_orders = list(trip["orders"])
    
    # Use the first supplier's location as starting point
    supplier_id = trip["suppliers"][0]
    if supplier_id not in supplier_locations:
        # Handle missing supplier location
        return trip["orders"]
        
    current_lat = safe_float(supplier_locations[supplier_id]["lat"])
    current_lon = safe_float(supplier_locations[supplier_id]["lon"])
    
    while remaining_orders:
        nearest_index = -1
        min_distance = float('inf')
        
        for i, order in enumerate(remaining_orders):
            # Convert coordinates using safe_float
            order_lat = safe_float(order[cust_lat_idx])
            order_lon = safe_float(order[cust_lon_idx])
            
            # Skip invalid coordinates
            if order_lat == 0 and order_lon == 0:
                continue
                
            distance = haversine(
                current_lat, current_lon,
                order_lat, order_lon
            )
            if distance < min_distance:
                min_distance = distance
                nearest_index = i
        
        # If no valid order found, break
        if nearest_index == -1:
            break
            
        next_order = remaining_orders.pop(nearest_index)
        sorted_orders.append(next_order)
        current_lat = safe_float(next_order[cust_lat_idx])
        current_lon = safe_float(next_order[cust_lon_idx])
    
    # Add any remaining orders that couldn't be processed
    sorted_orders.extend(remaining_orders)
    
    return sorted_orders

def phase_1(df, column_indices):
    """
    Phase 1: Initial trip formation
    
    Args:
        df: pandas DataFrame with order data
        column_indices: dictionary mapping column names to their indices
        
    Returns:
        dict with updated dataframe, supplier locations, merged trips, etc.
    """
    print("Starting Phase 1...")
    
    # Extract column indices
    id_idx = column_indices.get('id')
    cluster_idx = column_indices.get('cluster')
    sup_place_idx = column_indices.get('sup_place_id')
    gmv_idx = column_indices.get('order_gmv')
    dimension_idx = column_indices.get('order_dimension')
    sup_lon_idx = column_indices.get('supplier_longitude')
    sup_lat_idx = column_indices.get('supplier_latitude')
    cust_lon_idx = column_indices.get('customer_longitude')
    cust_lat_idx = column_indices.get('customer_latitude')
    warehouse_area_idx = column_indices.get('warehouse_location_area')
    weight_idx = column_indices.get('Total Order Weight / KG')
    customer_area_idx = column_indices.get('customer_area')
    trip_id_idx = column_indices.get('Trip_ID')
    
    # Convert DataFrame to list of lists for easier manipulation
    headers = df.columns.tolist()
    data = [headers] + df.values.tolist()
    orders = data[1:]  # Skip header
    
    # Initialize tracking variables
    supplier_locations = {}
    supplier_warehouse_areas = {}
    distance_cache = {}
    order_trip_map = {}
    
    # Map supplier locations and warehouse areas
    for order in orders:
        sup_place = order[sup_place_idx]
        if sup_place not in supplier_locations:
            supplier_locations[sup_place] = {"lat": order[sup_lat_idx], "lon": order[sup_lon_idx]}
            supplier_warehouse_areas[sup_place] = order[warehouse_area_idx]
        
        # Handle empty values
        if not order[dimension_idx] or pd.isna(order[dimension_idx]):
            order[dimension_idx] = DEFAULT_DIMENSION
        if not order[weight_idx] or pd.isna(order[weight_idx]):
            order[weight_idx] = DEFAULT_WEIGHT
    
    # Group orders by supplier and cluster
    supplier_cluster_orders = {}
    for order in orders:
        sup_place = order[sup_place_idx]
        cluster = order[cluster_idx]
        key = f"{sup_place}_{cluster}"
        if key not in supplier_cluster_orders:
            supplier_cluster_orders[key] = []
        supplier_cluster_orders[key].append(order)
    
    # Calculate cluster centroids
    cluster_centroids = {}
    for order in orders:
        cluster = order[cluster_idx]
        if cluster not in cluster_centroids:
            cluster_centroids[cluster] = []
        cluster_centroids[cluster].append(order)
    
    for cluster in cluster_centroids:
        cluster_centroids[cluster] = calculate_centroid(
            cluster_centroids[cluster], 
            cust_lat_idx, 
            cust_lon_idx
        )
    
    trips = []
    supplier_trip_counters = {}
    
    # Process each supplier-cluster group
    for key in supplier_cluster_orders:
        sup_place, cluster = key.split("_")
        if sup_place not in supplier_trip_counters:
            supplier_trip_counters[sup_place] = 0
        
        sup_cluster_orders = supplier_cluster_orders[key]
        far_away_orders = []
        close_orders = []
        
        # Split orders into far-away and close
        for order in sup_cluster_orders:
            distance = haversine(
                supplier_locations[sup_place]["lat"],
                supplier_locations[sup_place]["lon"],
                order[cust_lat_idx],
                order[cust_lon_idx]
            )
            if distance > FAR_AWAY_THRESHOLD:
                far_away_orders.append(order)
            else:
                close_orders.append(order)
        
        # Process far-away orders
        far_away_groups = {}
        for order in far_away_orders:
            key = (order[customer_area_idx] if customer_area_idx != -1 and order[customer_area_idx]
                   else f"{order[cust_lat_idx]},{order[cust_lon_idx]}")
            if key not in far_away_groups:
                far_away_groups[key] = []
            far_away_groups[key].append(order)
        
        for key in far_away_groups:
            region_orders = sorted(far_away_groups[key], key=lambda x: x[id_idx])
            
            while region_orders:
                supplier_trip_counters[sup_place] += 1
                trip = {
                    "orders": [],
                    "supplier": sup_place,
                    "suppliers": [sup_place],  # For Phase 3
                    "cluster": cluster,
                    "total_gmv": 0,
                    "total_dimension": 0,
                    "total_weight": 0,
                    "total_distance": 0,
                    "trip_id": f"{sup_place}_Trip_{supplier_trip_counters[sup_place]}",
                    "avg_cust_lat": 0,
                    "avg_cust_lon": 0,
                    "customer_areas": set(),
                    "warehouse_area": supplier_warehouse_areas[sup_place]
                }
                
                # Take first order
                order = region_orders.pop(0)
                trip["orders"].append(order)
                trip["total_gmv"] += safe_float(order[gmv_idx])
                trip["total_dimension"] += safe_float(order[dimension_idx], DEFAULT_DIMENSION)
                trip["total_weight"] += safe_float(order[weight_idx], DEFAULT_WEIGHT)
                trip["total_distance"] = haversine(
                    supplier_locations[sup_place]["lat"],
                    supplier_locations[sup_place]["lon"],
                    order[cust_lat_idx],
                    order[cust_lon_idx]
                )
                trip["avg_cust_lat"] = safe_float(order[cust_lat_idx])
                trip["avg_cust_lon"] = safe_float(order[cust_lon_idx])
                
                area_key = (order[customer_area_idx] if customer_area_idx != -1 and order[customer_area_idx]
                           else f"{order[cust_lat_idx]},{order[cust_lon_idx]}")
                trip["customer_areas"].add(area_key)
                
                order_trip_map[order[id_idx]] = trip["trip_id"]
                
                # Try to add more orders
                i = 0
                while i < len(region_orders):
                    next_order = region_orders[i]
                    new_dimension = trip["total_dimension"] + safe_float(next_order[dimension_idx], DEFAULT_DIMENSION)
                    new_weight = trip["total_weight"] + safe_float(next_order[weight_idx], DEFAULT_WEIGHT)
                    new_distance = haversine(
                        supplier_locations[sup_place]["lat"],
                        supplier_locations[sup_place]["lon"],
                        next_order[cust_lat_idx],
                        next_order[cust_lon_idx]
                    )
                    
                    # Check constraints
                    if (new_dimension <= DIMENSION_MAX and
                        new_weight <= WEIGHT_MAX and
                        new_distance < MAX_TRIP_DISTANCE and
                        check_customer_distance(
                            trip, next_order, distance_cache,
                            cust_lat_idx, cust_lon_idx, id_idx
                        )):
                        
                        trip["orders"].append(next_order)
                        trip["total_gmv"] += safe_float(next_order[gmv_idx])
                        trip["total_dimension"] = new_dimension
                        trip["total_weight"] = new_weight
                        trip["total_distance"] = max(trip["total_distance"], new_distance)
                        
                        # Update average customer location
                        trip["avg_cust_lat"] = ((trip["avg_cust_lat"] * (len(trip["orders"]) - 1) + 
                                              safe_float(next_order[cust_lat_idx])) / len(trip["orders"]))
                        trip["avg_cust_lon"] = ((trip["avg_cust_lon"] * (len(trip["orders"]) - 1) + 
                                              safe_float(next_order[cust_lon_idx])) / len(trip["orders"]))
                        
                        # Add customer area
                        area_key = (next_order[customer_area_idx] if customer_area_idx != -1 and next_order[customer_area_idx]
                                   else f"{next_order[cust_lat_idx]},{next_order[cust_lon_idx]}")
                        trip["customer_areas"].add(area_key)
                        
                        order_trip_map[next_order[id_idx]] = trip["trip_id"]
                        region_orders.pop(i)
                    else:
                        i += 1
                
                trips.append(trip)
        
        # Process close orders (similar logic to far_away_orders)
        close_groups = {}
        for order in close_orders:
            key = (order[customer_area_idx] if customer_area_idx != -1 and order[customer_area_idx]
                   else f"{order[cust_lat_idx]},{order[cust_lon_idx]}")
            if key not in close_groups:
                close_groups[key] = []
            close_groups[key].append(order)
        
        for key in close_groups:
            region_orders = sorted(close_groups[key], key=lambda x: x[id_idx])
            
            while region_orders:
                supplier_trip_counters[sup_place] += 1
                trip = {
                    "orders": [],
                    "supplier": sup_place,
                    "suppliers": [sup_place],  # For Phase 3
                    "cluster": cluster,
                    "total_gmv": 0,
                    "total_dimension": 0,
                    "total_weight": 0,
                    "total_distance": 0,
                    "trip_id": f"{sup_place}_Trip_{supplier_trip_counters[sup_place]}",
                    "avg_cust_lat": 0,
                    "avg_cust_lon": 0,
                    "customer_areas": set(),
                    "warehouse_area": supplier_warehouse_areas[sup_place]
                }
                
                # Take first order
                order = region_orders.pop(0)
                trip["orders"].append(order)
                trip["total_gmv"] += safe_float(order[gmv_idx])
                trip["total_dimension"] += safe_float(order[dimension_idx], DEFAULT_DIMENSION)
                trip["total_weight"] += safe_float(order[weight_idx], DEFAULT_WEIGHT)
                trip["total_distance"] = haversine(
                    supplier_locations[sup_place]["lat"],
                    supplier_locations[sup_place]["lon"],
                    order[cust_lat_idx],
                    order[cust_lon_idx]
                )
                trip["avg_cust_lat"] = safe_float(order[cust_lat_idx])
                trip["avg_cust_lon"] = safe_float(order[cust_lon_idx])
                
                area_key = (order[customer_area_idx] if customer_area_idx != -1 and order[customer_area_idx]
                           else f"{order[cust_lat_idx]},{order[cust_lon_idx]}")
                trip["customer_areas"].add(area_key)
                
                order_trip_map[order[id_idx]] = trip["trip_id"]
                
                # Try to add more orders (same as for far_away_orders)
                i = 0
                while i < len(region_orders):
                    next_order = region_orders[i]
                    new_dimension = trip["total_dimension"] + safe_float(next_order[dimension_idx], DEFAULT_DIMENSION)
                    new_weight = trip["total_weight"] + safe_float(next_order[weight_idx], DEFAULT_WEIGHT)
                    new_distance = haversine(
                        supplier_locations[sup_place]["lat"],
                        supplier_locations[sup_place]["lon"],
                        next_order[cust_lat_idx],
                        next_order[cust_lon_idx]
                    )
                    
                    # Check constraints
                    if (new_dimension <= DIMENSION_MAX and
                        new_weight <= WEIGHT_MAX and
                        new_distance < MAX_TRIP_DISTANCE and
                        check_customer_distance(
                            trip, next_order, distance_cache,
                            cust_lat_idx, cust_lon_idx, id_idx
                        )):
                        
                        trip["orders"].append(next_order)
                        trip["total_gmv"] += safe_float(next_order[gmv_idx])
                        trip["total_dimension"] = new_dimension
                        trip["total_weight"] = new_weight
                        trip["total_distance"] = max(trip["total_distance"], new_distance)
                        
                        # Update average customer location
                        trip["avg_cust_lat"] = ((trip["avg_cust_lat"] * (len(trip["orders"]) - 1) + 
                                              safe_float(next_order[cust_lat_idx])) / len(trip["orders"]))
                        trip["avg_cust_lon"] = ((trip["avg_cust_lon"] * (len(trip["orders"]) - 1) + 
                                              safe_float(next_order[cust_lon_idx])) / len(trip["orders"]))
                        
                        # Add customer area
                        area_key = (next_order[customer_area_idx] if customer_area_idx != -1 and next_order[customer_area_idx]
                                   else f"{next_order[cust_lat_idx]},{next_order[cust_lon_idx]}")
                        trip["customer_areas"].add(area_key)
                        
                        order_trip_map[next_order[id_idx]] = trip["trip_id"]
                        region_orders.pop(i)
                    else:
                        i += 1
                
                trips.append(trip)
    
    # Merge trips in Phase 1 (same cluster only)
    merged_trips = []
    trips_to_merge = [trip for trip in trips if trip["total_dimension"] < DIMENSION_MAX]
    iteration_limit = 1000
    iteration_count = 0
    
    # Add trips that are already full
    for trip in trips:
        if trip["total_dimension"] >= DIMENSION_MAX:
            merged_trips.append(trip)
    
    print(f"Trips to merge in Phase 1: {len(trips_to_merge)}")
    
    while trips_to_merge and iteration_count < iteration_limit:
        iteration_count += 1
        current_trip = trips_to_merge.pop(0)
        best_match = None
        min_distance = float('inf')
        
        for i, other_trip in enumerate(trips_to_merge):
            # Only merge trips from same supplier and cluster
            if (current_trip["supplier"] != other_trip["supplier"] or 
                current_trip["cluster"] != other_trip["cluster"]):
                continue
            
            distance = haversine(
                current_trip["avg_cust_lat"], current_trip["avg_cust_lon"],
                other_trip["avg_cust_lat"], other_trip["avg_cust_lon"]
            )
            
            if ((current_trip["total_dimension"] + other_trip["total_dimension"]) <= MERGED_DIMENSION_MAX and
                (current_trip["total_weight"] + other_trip["total_weight"]) <= WEIGHT_MAX and
                distance < MAX_CUSTOMER_DISTANCE):
                min_distance = distance
                best_match = i
        
        if best_match is not None:
            other_trip = trips_to_merge[best_match]
            
            # Create a new merged trip
            new_trip = {
                "orders": list(current_trip["orders"]),
                "supplier": current_trip["supplier"],
                "suppliers": list(current_trip["suppliers"]),
                "cluster": current_trip["cluster"],
                "total_gmv": current_trip["total_gmv"],
                "total_dimension": current_trip["total_dimension"],
                "total_weight": current_trip["total_weight"],
                "total_distance": current_trip["total_distance"],
                "trip_id": current_trip["trip_id"],
                "avg_cust_lat": current_trip["avg_cust_lat"],
                "avg_cust_lon": current_trip["avg_cust_lon"],
                "customer_areas": set(current_trip["customer_areas"]),
                "warehouse_area": current_trip["warehouse_area"]
            }
            
            remaining_orders = []
            orders_to_merge = list(other_trip["orders"])
            
            for order in orders_to_merge:
                if (new_trip["total_dimension"] + safe_float(order[dimension_idx], DEFAULT_DIMENSION) <= MERGED_DIMENSION_MAX and
                    new_trip["total_weight"] + safe_float(order[weight_idx], DEFAULT_WEIGHT) <= WEIGHT_MAX and
                    check_customer_distance(
                        new_trip, order, distance_cache,
                        cust_lat_idx, cust_lon_idx, id_idx
                    )):
                    
                    new_trip["orders"].append(order)
                    new_trip["total_gmv"] += safe_float(order[gmv_idx])
                    new_trip["total_dimension"] += safe_float(order[dimension_idx], DEFAULT_DIMENSION)
                    new_trip["total_weight"] += safe_float(order[weight_idx], DEFAULT_WEIGHT)
                    
                    area_key = (order[customer_area_idx] if customer_area_idx != -1 and order[customer_area_idx]
                               else f"{order[cust_lat_idx]},{order[cust_lon_idx]}")
                    new_trip["customer_areas"].add(area_key)
                else:
                    remaining_orders.append(order)
            
            # Recalculate average customer location
            new_trip["avg_cust_lat"] = sum(safe_float(order[cust_lat_idx]) for order in new_trip["orders"]) / len(new_trip["orders"])
            new_trip["avg_cust_lon"] = sum(safe_float(order[cust_lon_idx]) for order in new_trip["orders"]) / len(new_trip["orders"])
            
            # Recalculate total distance
            new_trip["total_distance"] = max(
                haversine(
                    supplier_locations[new_trip["supplier"]]["lat"],
                    supplier_locations[new_trip["supplier"]]["lon"],
                    safe_float(order[cust_lat_idx]),
                    safe_float(order[cust_lon_idx])
                )
                for order in new_trip["orders"]
            )
            
            # Update order-trip mapping
            for order in new_trip["orders"]:
                order_trip_map[order[id_idx]] = new_trip["trip_id"]
            
            # Handle remaining orders from the other trip
            if remaining_orders:
                other_trip["orders"] = remaining_orders
                other_trip["total_gmv"] = sum(safe_float(order[gmv_idx]) for order in remaining_orders)
                other_trip["total_dimension"] = sum(safe_float(order[dimension_idx], DEFAULT_DIMENSION) for order in remaining_orders)
                other_trip["total_weight"] = sum(safe_float(order[weight_idx], DEFAULT_WEIGHT) for order in remaining_orders)
                other_trip["avg_cust_lat"] = sum(safe_float(order[cust_lat_idx]) for order in remaining_orders) / len(remaining_orders)
                other_trip["avg_cust_lon"] = sum(safe_float(order[cust_lon_idx]) for order in remaining_orders) / len(remaining_orders)
                other_trip["total_distance"] = max(
                    haversine(
                        supplier_locations[other_trip["supplier"]]["lat"],
                        supplier_locations[other_trip["supplier"]]["lon"],
                        safe_float(order[cust_lat_idx]),
                        safe_float(order[cust_lon_idx])
                    )
                    for order in remaining_orders
                )
                other_trip["customer_areas"] = set(
                    order[customer_area_idx] if customer_area_idx != -1 and order[customer_area_idx]
                    else f"{order[cust_lat_idx]},{order[cust_lon_idx]}"
                    for order in remaining_orders
                )
            else:
                # Remove the other trip as it's completely merged
                trips_to_merge.pop(best_match)
            
            # Decide whether to continue merging or finalize this trip
            if new_trip["total_dimension"] < DIMENSION_MAX:
                trips_to_merge.append(new_trip)
            else:
                merged_trips.append(new_trip)
        else:
            # No suitable match found, finalize this trip
            merged_trips.append(current_trip)
    
    # Add any remaining trips to merge with nearest cluster if needed
    if trips_to_merge:
        print(f"Merging with nearest clusters for {len(trips_to_merge)} trips...")
        
        while trips_to_merge:
            current_trip = trips_to_merge.pop(0)
            best_match = None
            min_distance = float('inf')
            
            for i, other_trip in enumerate(merged_trips):
                if current_trip["supplier"] != other_trip["supplier"]:
                    continue
                
                current_centroid = cluster_centroids[current_trip["cluster"]]
                other_centroid = cluster_centroids[other_trip["cluster"]]
                
                distance = haversine(
                    current_centroid["lat"], current_centroid["lon"],
                    other_centroid["lat"], other_centroid["lon"]
                )
                
                if ((current_trip["total_dimension"] + other_trip["total_dimension"]) <= MERGED_DIMENSION_MAX and
                    (current_trip["total_weight"] + other_trip["total_weight"]) <= WEIGHT_MAX and
                    distance < MAX_CUSTOMER_DISTANCE):
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_match = i
            
            if best_match is not None:
                other_trip = merged_trips[best_match]
                
                # Create a new merged trip (similar logic as before)
                new_trip = {
                    "orders": list(current_trip["orders"]),
                    "supplier": current_trip["supplier"],
                    "suppliers": list(current_trip["suppliers"]),
                    "cluster": current_trip["cluster"],
                    "total_gmv": current_trip["total_gmv"],
                    "total_dimension": current_trip["total_dimension"],
                    "total_weight": current_trip["total_weight"],
                    "total_distance": current_trip["total_distance"],
                    "trip_id": current_trip["trip_id"],
                    "avg_cust_lat": current_trip["avg_cust_lat"],
                    "avg_cust_lon": current_trip["avg_cust_lon"],
                    "customer_areas": set(current_trip["customer_areas"]),
                    "warehouse_area": current_trip["warehouse_area"]
                }
                
                remaining_orders = []
                orders_to_merge = list(other_trip["orders"])
                
                # Try to merge orders (same logic as before)
                for order in orders_to_merge:
                    if (new_trip["total_dimension"] + safe_float(order[dimension_idx], DEFAULT_DIMENSION) <= MERGED_DIMENSION_MAX and
                        new_trip["total_weight"] + safe_float(order[weight_idx], DEFAULT_WEIGHT) <= WEIGHT_MAX and
                        check_customer_distance(
                            new_trip, order, distance_cache,
                            cust_lat_idx, cust_lon_idx, id_idx
                        )):
                        
                        new_trip["orders"].append(order)
                        new_trip["total_gmv"] += safe_float(order[gmv_idx])
                        new_trip["total_dimension"] += safe_float(order[dimension_idx], DEFAULT_DIMENSION)
                        new_trip["total_weight"] += safe_float(order[weight_idx], DEFAULT_WEIGHT)
                        
                        area_key = (order[customer_area_idx] if customer_area_idx != -1 and order[customer_area_idx]
                                   else f"{order[cust_lat_idx]},{order[cust_lon_idx]}")
                        new_trip["customer_areas"].add(area_key)
                    else:
                        remaining_orders.append(order)
                
                # Recalculate values (same as before)
                new_trip["avg_cust_lat"] = sum(safe_float(order[cust_lat_idx]) for order in new_trip["orders"]) / len(new_trip["orders"])
                new_trip["avg_cust_lon"] = sum(safe_float(order[cust_lon_idx]) for order in new_trip["orders"]) / len(new_trip["orders"])
                
                new_trip["total_distance"] = max(
                    haversine(
                        supplier_locations[new_trip["supplier"]]["lat"],
                        supplier_locations[new_trip["supplier"]]["lon"],
                        safe_float(order[cust_lat_idx]),
                        safe_float(order[cust_lon_idx])
                    )
                    for order in new_trip["orders"]
                )
                
                # Update order-trip mapping
                for order in new_trip["orders"]:
                    order_trip_map[order[id_idx]] = new_trip["trip_id"]
                
                # Handle remaining orders (same as before)
                if remaining_orders:
                    other_trip["orders"] = remaining_orders
                    other_trip["total_gmv"] = sum(safe_float(order[gmv_idx]) for order in remaining_orders)
                    other_trip["total_dimension"] = sum(safe_float(order[dimension_idx], DEFAULT_DIMENSION) for order in remaining_orders)
                    other_trip["total_weight"] = sum(safe_float(order[weight_idx], DEFAULT_WEIGHT) for order in remaining_orders)
                    
                    # Recalculate averages
                    other_trip["avg_cust_lat"] = sum(safe_float(order[cust_lat_idx]) for order in remaining_orders) / len(remaining_orders)
                    other_trip["avg_cust_lon"] = sum(safe_float(order[cust_lon_idx]) for order in remaining_orders) / len(remaining_orders)
                    other_trip["total_distance"] = max(
                        haversine(
                            supplier_locations[other_trip["supplier"]]["lat"],
                            supplier_locations[other_trip["supplier"]]["lon"],
                            safe_float(order[cust_lat_idx]),
                            safe_float(order[cust_lon_idx])
                        )
                        for order in remaining_orders
                    )
                    
                    # Recalculate customer areas
                    other_trip["customer_areas"] = set(
                        order[customer_area_idx] if customer_area_idx != -1 and order[customer_area_idx]
                        else f"{order[cust_lat_idx]},{order[cust_lon_idx]}"
                        for order in remaining_orders
                    )
                else:
                    merged_trips.pop(best_match)
                
                merged_trips.append(new_trip)
            else:
                merged_trips.append(current_trip)
    
    if iteration_count >= iteration_limit:
        print("Phase 1 merge stopped due to iteration limit.")
    
    # Optimize routes
    for trip in merged_trips:
        trip["orders"] = optimize_route(trip, supplier_locations, cust_lat_idx, cust_lon_idx)
        
        # Recalculate total distance with optimized route
        current_lat = supplier_locations[trip["supplier"]]["lat"]
        current_lon = supplier_locations[trip["supplier"]]["lon"]
        trip["total_distance"] = 0
        
        for order in trip["orders"]:
            distance = haversine(current_lat, current_lon, safe_float(order[cust_lat_idx]), safe_float(order[cust_lon_idx]))
            trip["total_distance"] += distance
            current_lat = safe_float(order[cust_lat_idx])
            current_lon = safe_float(order[cust_lon_idx])
    
    # Update the dataframe with Trip_ID
    result_df = df.copy()
    result_df["Trip_ID"] = result_df.apply(
        lambda row: order_trip_map.get(row[headers[id_idx]], ""),
        axis=1
    )
    
    print(f"Phase 1 complete. Trips created: {len(merged_trips)}")
    
    return {
        "data": result_df,
        "supplier_locations": supplier_locations,
        "merged_trips": merged_trips,
        "supplier_warehouse_areas": supplier_warehouse_areas,
        "cluster_centroids": cluster_centroids
    }

def phase_2(df, column_indices, supplier_locations, previous_trips, supplier_warehouse_areas, cluster_centroids):
    """
    Phase 2: Secondary merging
    
    Args:
        df: pandas DataFrame with order data and Trip_ID from phase 1
        column_indices: dictionary mapping column names to their indices
        supplier_locations: dictionary of supplier locations from phase 1
        previous_trips: list of trips created in phase 1
        supplier_warehouse_areas: dictionary of warehouse areas for each supplier
        cluster_centroids: dictionary of centroids for each cluster
        
    Returns:
        dict with updated dataframe and merged trips
    """
    print("Starting Phase 2...")
    
    # Extract column indices
    id_idx = column_indices.get('id')
    cluster_idx = column_indices.get('cluster')
    sup_place_idx = column_indices.get('sup_place_id')
    gmv_idx = column_indices.get('order_gmv')
    dimension_idx = column_indices.get('order_dimension')
    sup_lon_idx = column_indices.get('supplier_longitude')
    sup_lat_idx = column_indices.get('supplier_latitude')
    cust_lon_idx = column_indices.get('customer_longitude')
    cust_lat_idx = column_indices.get('customer_latitude')
    warehouse_area_idx = column_indices.get('warehouse_location_area')
    weight_idx = column_indices.get('Total Order Weight / KG')
    customer_area_idx = column_indices.get('customer_area')
    trip_id_idx = column_indices.get('Trip_ID')
    
    # Convert DataFrame to list of lists for easier manipulation
    headers = df.columns.tolist()
    data = [headers] + df.values.tolist()
    orders = data[1:]  # Skip header
    
    trips = []
    order_trip_map = {}
    distance_cache = {}
    
    # Group orders by trip_id
    trip_groups = {}
    for order in orders:
        trip_id = order[trip_id_idx]
        if not trip_id:
            continue
        if trip_id not in trip_groups:
            trip_groups[trip_id] = []
        trip_groups[trip_id].append(order)
    
    # Reconstruct trips from orders
    for trip_id, trip_orders in trip_groups.items():
        sup_place = trip_orders[0][sup_place_idx]
        trip = {
            "orders": trip_orders,
            "supplier": sup_place,
            "suppliers": [sup_place],  # For Phase 3
            "cluster": trip_orders[0][cluster_idx],
            "total_gmv": sum(safe_float(order[gmv_idx]) for order in trip_orders),
            "total_dimension": sum(safe_float(order[dimension_idx], DEFAULT_DIMENSION) for order in trip_orders),
            "total_weight": sum(safe_float(order[weight_idx], DEFAULT_WEIGHT) for order in trip_orders),
            "total_distance": 0,
            "trip_id": trip_id,
            "avg_cust_lat": sum(safe_float(order[cust_lat_idx]) for order in trip_orders) / len(trip_orders),
            "avg_cust_lon": sum(safe_float(order[cust_lon_idx]) for order in trip_orders) / len(trip_orders),
            "customer_areas": set(
                order[customer_area_idx] if customer_area_idx != -1 and order[customer_area_idx]
                else f"{order[cust_lat_idx]},{order[cust_lon_idx]}"
                for order in trip_orders
            ),
            "warehouse_area": supplier_warehouse_areas[sup_place]
        }
        
        # Calculate max distance from supplier to any order
        trip["total_distance"] = max(
            haversine(
                supplier_locations[trip["supplier"]]["lat"],
                supplier_locations[trip["supplier"]]["lon"],
                safe_float(order[cust_lat_idx]),
                safe_float(order[cust_lon_idx])
            )
            for order in trip["orders"]
        )
        
        trips.append(trip)
        for order in trip["orders"]:
            order_trip_map[order[id_idx]] = trip["trip_id"]
    
    # Merge trips with dimension < SECOND_MERGE_THRESHOLD
    merged_trips = []
    trips_to_merge = [trip for trip in trips if trip["total_dimension"] < SECOND_MERGE_THRESHOLD]
    iteration_limit = 1000
    iteration_count = 0
    
    # Add trips that are already over threshold
    for trip in trips:
        if trip["total_dimension"] >= SECOND_MERGE_THRESHOLD:
            merged_trips.append(trip)
    
    print(f"Trips to merge in Phase 2: {len(trips_to_merge)}")
    
    while trips_to_merge and iteration_count < iteration_limit:
        iteration_count += 1
        current_trip = trips_to_merge.pop(0)
        best_match = None
        min_distance = float('inf')
        
        # Check all trips (both in trips_to_merge and merged_trips)
        all_trips = trips_to_merge + merged_trips
        for i, other_trip in enumerate(all_trips):
            if (current_trip["supplier"] != other_trip["supplier"] or 
                current_trip["trip_id"] == other_trip["trip_id"] or 
                current_trip["cluster"] != other_trip["cluster"]):
                continue
            
            distance = haversine(
                current_trip["avg_cust_lat"], current_trip["avg_cust_lon"],
                other_trip["avg_cust_lat"], other_trip["avg_cust_lon"]
            )
            
            if ((current_trip["total_dimension"] + other_trip["total_dimension"]) <= MERGED_DIMENSION_MAX and
                (current_trip["total_weight"] + other_trip["total_weight"]) <= WEIGHT_MAX and
                distance < MAX_CUSTOMER_DISTANCE):
                
                min_distance = distance
                is_in_merged = i >= len(trips_to_merge)
                best_match = {"trip": other_trip, "index": i if not is_in_merged else i - len(trips_to_merge), 
                             "in_merged": is_in_merged}
        
        if best_match is not None:
            other_trip = best_match["trip"]
            
            # Create new merged trip
            new_trip = {
                "orders": list(current_trip["orders"]),
                "supplier": current_trip["supplier"],
                "suppliers": list(current_trip["suppliers"]),
                "cluster": current_trip["cluster"],
                "total_gmv": current_trip["total_gmv"],
                "total_dimension": current_trip["total_dimension"],
                "total_weight": current_trip["total_weight"],
                "total_distance": current_trip["total_distance"],
                "trip_id": current_trip["trip_id"],
                "avg_cust_lat": current_trip["avg_cust_lat"],
                "avg_cust_lon": current_trip["avg_cust_lon"],
                "customer_areas": set(current_trip["customer_areas"]),
                "warehouse_area": current_trip["warehouse_area"]
            }
            
            remaining_orders = []
            orders_to_merge = list(other_trip["orders"])
            
            for order in orders_to_merge:
                if (new_trip["total_dimension"] + safe_float(order[dimension_idx], DEFAULT_DIMENSION) <= MERGED_DIMENSION_MAX and
                    new_trip["total_weight"] + safe_float(order[weight_idx], DEFAULT_WEIGHT) <= WEIGHT_MAX and
                    check_customer_distance(
                        new_trip, order, distance_cache,
                        cust_lat_idx, cust_lon_idx, id_idx
                    )):
                    
                    new_trip["orders"].append(order)
                    new_trip["total_gmv"] += safe_float(order[gmv_idx])
                    new_trip["total_dimension"] += safe_float(order[dimension_idx], DEFAULT_DIMENSION)
                    new_trip["total_weight"] += safe_float(order[weight_idx], DEFAULT_WEIGHT)
                    
                    area_key = (order[customer_area_idx] if customer_area_idx != -1 and order[customer_area_idx]
                               else f"{order[cust_lat_idx]},{order[cust_lon_idx]}")
                    new_trip["customer_areas"].add(area_key)
                else:
                    remaining_orders.append(order)
            
            # Recalculate average customer location
            new_trip["avg_cust_lat"] = sum(safe_float(order[cust_lat_idx]) for order in new_trip["orders"]) / len(new_trip["orders"])
            new_trip["avg_cust_lon"] = sum(safe_float(order[cust_lon_idx]) for order in new_trip["orders"]) / len(new_trip["orders"])
            
            # Recalculate total distance
            new_trip["total_distance"] = max(
                haversine(
                    supplier_locations[new_trip["supplier"]]["lat"],
                    supplier_locations[new_trip["supplier"]]["lon"],
                    safe_float(order[cust_lat_idx]),
                    safe_float(order[cust_lon_idx])
                )
                for order in new_trip["orders"]
            )
            
            # Update order trip mapping
            for order in new_trip["orders"]:
                order_trip_map[order[id_idx]] = new_trip["trip_id"]
            
            # Handle remaining orders
            if remaining_orders:
                other_trip["orders"] = remaining_orders
                other_trip["total_gmv"] = sum(safe_float(order[gmv_idx]) for order in remaining_orders)
                other_trip["total_dimension"] = sum(safe_float(order[dimension_idx], DEFAULT_DIMENSION) for order in remaining_orders)
                other_trip["total_weight"] = sum(safe_float(order[weight_idx], DEFAULT_WEIGHT) for order in remaining_orders)
                
                # Recalculate averages
                other_trip["avg_cust_lat"] = sum(safe_float(order[cust_lat_idx]) for order in remaining_orders) / len(remaining_orders)
                other_trip["avg_cust_lon"] = sum(safe_float(order[cust_lon_idx]) for order in remaining_orders) / len(remaining_orders)
                other_trip["total_distance"] = max(
                    haversine(
                        supplier_locations[other_trip["supplier"]]["lat"],
                        supplier_locations[other_trip["supplier"]]["lon"],
                        safe_float(order[cust_lat_idx]),
                        safe_float(order[cust_lon_idx])
                    )
                    for order in remaining_orders
                )
                
                # Recalculate customer areas
                other_trip["customer_areas"] = set(
                    order[customer_area_idx] if customer_area_idx != -1 and order[customer_area_idx]
                    else f"{order[cust_lat_idx]},{order[cust_lon_idx]}"
                    for order in remaining_orders
                )
            else:
                if best_match["in_merged"]:
                    merged_trips.pop(best_match["index"])
                else:
                    trips_to_merge.pop(best_match["index"])
            
            # Decide where to put the new trip
            if new_trip["total_dimension"] < SECOND_MERGE_THRESHOLD:
                trips_to_merge.append(new_trip)
            else:
                merged_trips.append(new_trip)
        else:
            # Try to merge with trips from other clusters if same-cluster merge not possible
            best_match = None
            min_distance = float('inf')
            
            for i, other_trip in enumerate(merged_trips):
                if current_trip["supplier"] != other_trip["supplier"] or current_trip["trip_id"] == other_trip["trip_id"]:
                    continue
                
                current_centroid = cluster_centroids[current_trip["cluster"]]
                other_centroid = cluster_centroids[other_trip["cluster"]]
                
                distance = haversine(
                    current_centroid["lat"], current_centroid["lon"],
                    other_centroid["lat"], other_centroid["lon"]
                )
                
                if ((current_trip["total_dimension"] + other_trip["total_dimension"]) <= MERGED_DIMENSION_MAX and
                    (current_trip["total_weight"] + other_trip["total_weight"]) <= WEIGHT_MAX and
                    distance < MAX_CUSTOMER_DISTANCE):
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_match = {"trip": other_trip, "index": i}
            
            if best_match is not None:
                other_trip = best_match["trip"]
                
                # Create new merged trip (similar to above)
                new_trip = {
                    "orders": list(current_trip["orders"]),
                    "supplier": current_trip["supplier"],
                    "suppliers": list(current_trip["suppliers"]),
                    "cluster": current_trip["cluster"],
                    "total_gmv": current_trip["total_gmv"],
                    "total_dimension": current_trip["total_dimension"],
                    "total_weight": current_trip["total_weight"],
                    "total_distance": current_trip["total_distance"],
                    "trip_id": current_trip["trip_id"],
                    "avg_cust_lat": current_trip["avg_cust_lat"],
                    "avg_cust_lon": current_trip["avg_cust_lon"],
                    "customer_areas": set(current_trip["customer_areas"]),
                    "warehouse_area": current_trip["warehouse_area"]
                }
                
                remaining_orders = []
                orders_to_merge = list(other_trip["orders"])
                
                # Try to merge orders (same logic as above)
                for order in orders_to_merge:
                    if (new_trip["total_dimension"] + safe_float(order[dimension_idx], DEFAULT_DIMENSION) <= MERGED_DIMENSION_MAX and
                        new_trip["total_weight"] + safe_float(order[weight_idx], DEFAULT_WEIGHT) <= WEIGHT_MAX and
                        check_customer_distance(
                            new_trip, order, distance_cache,
                            cust_lat_idx, cust_lon_idx, id_idx
                        )):
                        
                        new_trip["orders"].append(order)
                        new_trip["total_gmv"] += safe_float(order[gmv_idx])
                        new_trip["total_dimension"] += safe_float(order[dimension_idx], DEFAULT_DIMENSION)
                        new_trip["total_weight"] += safe_float(order[weight_idx], DEFAULT_WEIGHT)
                        
                        area_key = (order[customer_area_idx] if customer_area_idx != -1 and order[customer_area_idx]
                                   else f"{order[cust_lat_idx]},{order[cust_lon_idx]}")
                        new_trip["customer_areas"].add(area_key)
                    else:
                        remaining_orders.append(order)
                
                # Recalculate values (same as above)
                new_trip["avg_cust_lat"] = sum(safe_float(order[cust_lat_idx]) for order in new_trip["orders"]) / len(new_trip["orders"])
                new_trip["avg_cust_lon"] = sum(safe_float(order[cust_lon_idx]) for order in new_trip["orders"]) / len(new_trip["orders"])
                
                new_trip["total_distance"] = max(
                    haversine(
                        supplier_locations[new_trip["supplier"]]["lat"],
                        supplier_locations[new_trip["supplier"]]["lon"],
                        safe_float(order[cust_lat_idx]),
                        safe_float(order[cust_lon_idx])
                    )
                    for order in new_trip["orders"]
                )
                
                # Update order trip mapping
                for order in new_trip["orders"]:
                    order_trip_map[order[id_idx]] = new_trip["trip_id"]
                
                # Handle remaining orders
                if remaining_orders:
                    other_trip["orders"] = remaining_orders
                    other_trip["total_gmv"] = sum(safe_float(order[gmv_idx]) for order in remaining_orders)
                    other_trip["total_dimension"] = sum(safe_float(order[dimension_idx], DEFAULT_DIMENSION) for order in remaining_orders)
                    other_trip["total_weight"] = sum(safe_float(order[weight_idx], DEFAULT_WEIGHT) for order in remaining_orders)
                    
                    # Recalculate averages
                    other_trip["avg_cust_lat"] = sum(safe_float(order[cust_lat_idx]) for order in remaining_orders) / len(remaining_orders)
                    other_trip["avg_cust_lon"] = sum(safe_float(order[cust_lon_idx]) for order in remaining_orders) / len(remaining_orders)
                    other_trip["total_distance"] = max(
                        haversine(
                            supplier_locations[other_trip["supplier"]]["lat"],
                            supplier_locations[other_trip["supplier"]]["lon"],
                            safe_float(order[cust_lat_idx]),
                            safe_float(order[cust_lon_idx])
                        )
                        for order in remaining_orders
                    )
                    
                    # Recalculate customer areas
                    other_trip["customer_areas"] = set(
                        order[customer_area_idx] if customer_area_idx != -1 and order[customer_area_idx]
                        else f"{order[cust_lat_idx]},{order[cust_lon_idx]}"
                        for order in remaining_orders
                    )
                else:
                    merged_trips.pop(best_match["index"])
                
                merged_trips.append(new_trip)
            else:
                merged_trips.append(current_trip)
    
    if iteration_count >= iteration_limit:
        print("Phase 2 merge stopped due to iteration limit.")
    
    # Optimize routes for all merged trips
    for trip in merged_trips:
        trip["orders"] = optimize_route(trip, supplier_locations, cust_lat_idx, cust_lon_idx)
        
        # Recalculate total distance with optimized route
        current_lat = supplier_locations[trip["supplier"]]["lat"]
        current_lon = supplier_locations[trip["supplier"]]["lon"]
        trip["total_distance"] = 0
        
        for order in trip["orders"]:
            distance = haversine(current_lat, current_lon, safe_float(order[cust_lat_idx]), safe_float(order[cust_lon_idx]))
            trip["total_distance"] += distance
            current_lat = safe_float(order[cust_lat_idx])
            current_lon = safe_float(order[cust_lon_idx])
    
    # Update DataFrame with new Trip_ID assignments
    result_df = df.copy()
    for i, row in result_df.iterrows():
        order_id = row[headers[id_idx]]
        if order_id in order_trip_map:
            result_df.at[i, 'Trip_ID'] = order_trip_map[order_id]
    
    print(f"Phase 2 complete. Trips created: {len(merged_trips)}")
    
    return {
        "data": result_df,
        "merged_trips": merged_trips
    }

def phase_3(df, column_indices, supplier_locations, previous_trips):
    """
    Phase 3: Merge trips from nearby suppliers with similar direction
    
    Args:
        df: pandas DataFrame with order data and Trip_ID from phase 2
        column_indices: dictionary mapping column names to their indices
        supplier_locations: dictionary of supplier locations
        previous_trips: list of trips created in phase 2
        
    Returns:
        Updated DataFrame with Merged_Trip_ID assignments
    """
    print("Starting Phase 3...")
    
    # Extract column indices
    id_idx = column_indices.get('id')
    cluster_idx = column_indices.get('cluster')
    sup_place_idx = column_indices.get('sup_place_id')
    gmv_idx = column_indices.get('order_gmv')
    dimension_idx = column_indices.get('order_dimension')
    sup_lon_idx = column_indices.get('supplier_longitude')
    sup_lat_idx = column_indices.get('supplier_latitude')
    cust_lon_idx = column_indices.get('customer_longitude')
    cust_lat_idx = column_indices.get('customer_latitude')
    warehouse_area_idx = column_indices.get('warehouse_location_area')
    weight_idx = column_indices.get('Total Order Weight / KG')
    customer_area_idx = column_indices.get('customer_area')
    trip_id_idx = column_indices.get('Trip_ID')
    merged_trip_id_idx = column_indices.get('Merged_Trip_ID')
    
    # Get headers from DataFrame
    headers = df.columns.tolist()
    
    trips = previous_trips
    order_merged_trip_map = {}
    distance_cache = {}
    merged_trip_counter = 1
    
    # Group trips by supplier
    supplier_trips = {}
    for trip in trips:
        sup_place = trip["supplier"]
        if sup_place not in supplier_trips:
            supplier_trips[sup_place] = []
        supplier_trips[sup_place].append(trip)
    
    merged_trips = []
    trips_to_merge = [trip for trip in trips if trip["total_dimension"] < MERGED_DIMENSION_MAX]
    processed_trips = set()
    
    print(f"Trips to merge in Phase 3: {len(trips_to_merge)}")
    
    for i in range(len(trips_to_merge)):
        current_trip = trips_to_merge[i]
        if current_trip["trip_id"] in processed_trips:
            continue
        
        best_match = None
        min_distance = float('inf')
        
        # Find nearby suppliers
        current_sup = current_trip["supplier"]
        nearby_suppliers = [
            sup for sup in supplier_locations.keys()
            if sup != current_sup and haversine(
                supplier_locations[current_sup]["lat"], supplier_locations[current_sup]["lon"],
                supplier_locations[sup]["lat"], supplier_locations[sup]["lon"]
            ) <= MAX_SUPPLIER_DISTANCE
        ]
        
        print(f"Nearby suppliers for {current_sup}: {', '.join(nearby_suppliers)}")
        
        # Check trips from nearby suppliers
        for other_sup in nearby_suppliers:
            other_trips = supplier_trips.get(other_sup, [])
            
            for j, other_trip in enumerate(other_trips):
                if (other_trip["trip_id"] in processed_trips or 
                    other_trip["total_dimension"] >= MERGED_DIMENSION_MAX):
                    continue
                
                # Check if trips are in similar direction
                current_bearing = calculate_bearing(
                    supplier_locations[current_sup]["lat"], supplier_locations[current_sup]["lon"],
                    current_trip["avg_cust_lat"], current_trip["avg_cust_lon"]
                )
                
                other_bearing = calculate_bearing(
                    supplier_locations[other_sup]["lat"], supplier_locations[other_sup]["lon"],
                    other_trip["avg_cust_lat"], other_trip["avg_cust_lon"]
                )
                
                bearing_diff = abs(current_bearing - other_bearing)
                if bearing_diff > math.pi:
                    bearing_diff = 2 * math.pi - bearing_diff
                
                # Check distance between trip centroids
                centroid_distance = haversine(
                    current_trip["avg_cust_lat"], current_trip["avg_cust_lon"],
                    other_trip["avg_cust_lat"], other_trip["avg_cust_lon"]
                )
                
                print(f"Checking merge: {current_trip['trip_id']} with {other_trip['trip_id']}")
                print(f"  Bearing diff: {bearing_diff}, Centroid distance: {centroid_distance} km")
                print(f"  Total dimension: {current_trip['total_dimension'] + other_trip['total_dimension']}, "
                      f"Total weight: {current_trip['total_weight'] + other_trip['total_weight']}")
                
                if (bearing_diff <= MAX_DIRECTION_DIFF and
                    centroid_distance <= MAX_CUSTOMER_DISTANCE and
                    (current_trip["total_dimension"] + other_trip["total_dimension"]) <= MERGED_DIMENSION_MAX and
                    (current_trip["total_weight"] + other_trip["total_weight"]) <= WEIGHT_MAX):
                    
                    if centroid_distance < min_distance:
                        min_distance = centroid_distance
                        best_match = {"trip": other_trip, "index": j, "supplier": other_sup}
                else:
                    reasons = []
                    if bearing_diff > MAX_DIRECTION_DIFF:
                        reasons.append("Bearing too different")
                    if centroid_distance > MAX_CUSTOMER_DISTANCE:
                        reasons.append("Centroid too far")
                    if current_trip["total_dimension"] + other_trip["total_dimension"] > MERGED_DIMENSION_MAX:
                        reasons.append("Dimension exceeded")
                    if current_trip["total_weight"] + other_trip["total_weight"] > WEIGHT_MAX:
                        reasons.append("Weight exceeded")
                    
                    print(f"  Merge rejected: {', '.join(reasons)}")
        
        if best_match is not None:
            other_trip = best_match["trip"]
            print(f"Merging {current_trip['trip_id']} with {other_trip['trip_id']}")
            
            # Create new merged trip
            new_trip = {
                "orders": list(current_trip["orders"]) + list(other_trip["orders"]),
                "supplier": current_trip["supplier"],  # Keep first supplier for simplicity
                "suppliers": list(set(current_trip["suppliers"] + other_trip["suppliers"])),
                "cluster": current_trip["cluster"],
                "total_gmv": current_trip["total_gmv"] + other_trip["total_gmv"],
                "total_dimension": current_trip["total_dimension"] + other_trip["total_dimension"],
                "total_weight": current_trip["total_weight"] + other_trip["total_weight"],
                "total_distance": 0,
                "trip_id": current_trip["trip_id"],
                "merged_trip_id": f"Merged_Trip_{merged_trip_counter}",
                "avg_cust_lat": 0,
                "avg_cust_lon": 0,
                "customer_areas": set(list(current_trip["customer_areas"]) + list(other_trip["customer_areas"])),
                "warehouse_area": current_trip["warehouse_area"]
            }
            
            # Verify all orders meet customer distance constraint
            valid_orders = []
            temp_trip = {"orders": [], "total_dimension": 0, "total_weight": 0}
            
            for order in new_trip["orders"]:
                if (temp_trip["total_dimension"] + safe_float(order[dimension_idx], DEFAULT_DIMENSION) <= MERGED_DIMENSION_MAX and
                    temp_trip["total_weight"] + safe_float(order[weight_idx], DEFAULT_WEIGHT) <= WEIGHT_MAX and
                    check_customer_distance(
                        temp_trip, order, distance_cache,
                        cust_lat_idx, cust_lon_idx, id_idx
                    )):
                    
                    temp_trip["orders"].append(order)
                    temp_trip["total_dimension"] += safe_float(order[dimension_idx], DEFAULT_DIMENSION)
                    temp_trip["total_weight"] += safe_float(order[weight_idx], DEFAULT_WEIGHT)
                    valid_orders.append(order)
            
            if valid_orders:
                # Update trip with valid orders
                new_trip["orders"] = valid_orders
                new_trip["total_dimension"] = temp_trip["total_dimension"]
                new_trip["total_weight"] = temp_trip["total_weight"]
                
                # Recalculate average location
                new_trip["avg_cust_lat"] = sum(safe_float(order[cust_lat_idx]) for order in new_trip["orders"]) / len(new_trip["orders"])
                new_trip["avg_cust_lon"] = sum(safe_float(order[cust_lon_idx]) for order in new_trip["orders"]) / len(new_trip["orders"])
                
                # Recalculate max distance
                new_trip["total_distance"] = max(
                    haversine(
                        supplier_locations[new_trip["supplier"]]["lat"],
                        supplier_locations[new_trip["supplier"]]["lon"],
                        safe_float(order[cust_lat_idx]),
                        safe_float(order[cust_lon_idx])
                    )
                    for order in new_trip["orders"]
                )
                
                # Update merged trip mapping
                for order in new_trip["orders"]:
                    order_merged_trip_map[order[id_idx]] = new_trip["merged_trip_id"]
                    print(f"Assigned Merged_Trip_ID {new_trip['merged_trip_id']} to order {order[id_idx]}")
                
                processed_trips.add(current_trip["trip_id"])
                processed_trips.add(other_trip["trip_id"])
                
                # Remove other trip from its supplier's trips
                other_sup_trips = supplier_trips[best_match["supplier"]]
                other_sup_trips.pop(best_match["index"])
                
                # Remove from trips_to_merge
                trips_to_merge.pop(i)
                i -= 1  # Adjust index after removal
                
                if new_trip["total_dimension"] < MERGED_DIMENSION_MAX:
                    trips_to_merge.append(new_trip)
                
                merged_trips.append(new_trip)
            else:
                print(f"No valid orders for merge between {current_trip['trip_id']} and {other_trip['trip_id']}")
                merged_trips.append(current_trip)
                processed_trips.add(current_trip["trip_id"])
        else:
            print(f"No merge found for {current_trip['trip_id']}")
            merged_trips.append(current_trip)
            processed_trips.add(current_trip["trip_id"])
    
    # Add unprocessed trips
    for trip in trips:
        if trip["trip_id"] not in processed_trips:
            merged_trips.append(trip)
    
    # Optimize routes
    for trip in merged_trips:
        trip["orders"] = optimize_route(trip, supplier_locations, cust_lat_idx, cust_lon_idx)
        
        # Recalculate total distance with optimized route
        current_lat = supplier_locations[trip["suppliers"][0]]["lat"]
        current_lon = supplier_locations[trip["suppliers"][0]]["lon"]
        trip["total_distance"] = 0
        
        for order in trip["orders"]:
            distance = haversine(current_lat, current_lon, safe_float(order[cust_lat_idx]), safe_float(order[cust_lon_idx]))
            trip["total_distance"] += distance
            current_lat = safe_float(order[cust_lat_idx])
            current_lon = safe_float(order[cust_lon_idx])
    
    # Update DataFrame with Merged_Trip_ID
    result_df = df.copy()
    for i, row in result_df.iterrows():
        order_id = row[headers[id_idx]]
        if order_id in order_merged_trip_map:
            result_df.at[i, 'Merged_Trip_ID'] = order_merged_trip_map[order_id]
    
    print(f"Phase 3 complete. Merged trips created: {len([t for t in merged_trips if 'merged_trip_id' in t])}")
    
    return result_df

# Main function to assign trips
def assign_trips(data_df):
    """
    Main function to assign trips based on the input data.
    
    Args:
        data_df: Pandas DataFrame containing the necessary data
        
    Returns:
        Updated DataFrame with Trip_ID and Merged_Trip_ID assignments
    """
    # Check for required columns
    required_columns = [
        'id', 'cluster', 'sup_place_id', 'order_gmv', 'order_dimension',
        'supplier_longitude', 'supplier_latitude', 'customer_longitude',
        'customer_latitude', 'warehouse_location_area', 'Total Order Weight / KG'
    ]
    
    # Add Trip_ID column if missing
    if 'Trip_ID' not in data_df.columns:
        data_df['Trip_ID'] = ""
    
    # Add Merged_Trip_ID column if missing
    if 'Merged_Trip_ID' not in data_df.columns:
        data_df['Merged_Trip_ID'] = ""
    
    # Map column names to their indices
    column_indices = {col: i for i, col in enumerate(data_df.columns)}
    column_indices['customer_area'] = column_indices.get('customer_area', -1)
    
    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in column_indices]
    if missing_columns:
        error_msg = f"Missing required columns: {', '.join(missing_columns)}"
        st.error(error_msg)
        print(error_msg)
        return data_df
    
    # Phase 1: Initial trip formation
    print("\n=== Phase 1: Initial Trip Formation ===")
    phase1_result = phase_1(data_df, column_indices)
    
    # Phase 2: Secondary merging
    print("\n=== Phase 2: Secondary Merging ===")
    phase2_result = phase_2(
        phase1_result["data"],
        column_indices,
        phase1_result["supplier_locations"],
        phase1_result["merged_trips"],
        phase1_result["supplier_warehouse_areas"],
        phase1_result["cluster_centroids"]
    )
    
    # Phase 3: Cross-supplier merging
    print("\n=== Phase 3: Cross-Supplier Merging ===")
    final_result = phase_3(
        phase2_result["data"],
        column_indices,
        phase1_result["supplier_locations"],
        phase2_result["merged_trips"]
    )
    
    print("Trip assignment complete!")
    return final_result

# Streamlit UI component to integrate with the dashboard
def trip_assignment_ui():
    st.header("Trip Assignment")
    
    if "total_pending_assign_tasks" not in st.session_state:
        st.session_state.total_pending_assign_tasks = None
    
    # Add containers for the buttons in the navbar instead
    header_cols = st.columns([7, 1, 1, 1])
    
    with header_cols[1]:
        theme_toggle = st.toggle("Dark Mode", value=st.session_state.theme == "dark", key="theme_toggle_trip")
        if theme_toggle != (st.session_state.theme == "dark"):
            st.session_state.theme = "dark" if theme_toggle else "light"
            st.rerun()
    
    with header_cols[2]:
        edit_mode = st.toggle("Edit Dashboard", value=st.session_state.edit_mode, key="edit_mode_toggle_trip")
        if edit_mode != st.session_state.edit_mode:
            # Save current layout before switching to edit mode
            if not st.session_state.edit_mode:
                st.session_state.previous_layout = st.session_state.dashboard_layout
            st.session_state.edit_mode = edit_mode
            st.rerun()
    
    # Create a consolidated action area with better spacing
    action_cols = st.columns([1, 2, 1])
    
    with action_cols[0]:
        # Combined Load/Refresh Data button
        if st.button("Load/Refresh Data", key="trip_assignment_load_btn", type="primary"):
            with st.spinner("Loading tasks data..."):
                st.session_state.total_pending_assign_tasks = load_sheet_data("Tasks")
                st.success("Data loaded successfully!")
    
    with action_cols[2]:
        # Only enable when data is loaded
        if st.session_state.total_pending_assign_tasks is not None:
            if st.button("Run Trip Assignment", key="trip_assignment_run_btn", type="primary"):
                with st.spinner("Assigning trips... This may take a moment..."):
                    result_df = assign_trips(st.session_state.total_pending_assign_tasks)
                    st.session_state.trip_assignment_result = result_df
                st.success("Trip assignment completed!")
        else:
            st.button("Run Trip Assignment", key="trip_assignment_disabled_btn", disabled=True)
    
    # Only show the rest if data is loaded
    if st.session_state.total_pending_assign_tasks is not None:
        df = st.session_state.total_pending_assign_tasks
        
        # Show data preview with a single header
        st.subheader("Data Preview")
        
        # Create columns for supplier overview and data preview
        supplier_cols = st.columns([1, 1])
        
        # Supplier Overview Column
        with supplier_cols[0]:
            st.write("### Supplier Overview")
            if 'sup_place_id' in df.columns:
                try:
                    # Convert numeric columns to proper numeric types
                    numeric_cols = ['order_gmv', 'order_dimension', 'Total Order Weight / KG']
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
                    
                    # Simple aggregation to avoid data type issues
                    supplier_data = df.groupby('sup_place_id').agg({
                        'id': 'count'
                    }).reset_index()
                    
                    supplier_data.columns = ['Supplier', 'Order Count']
                    
                    # Use fixed height matching the data preview
                    st.dataframe(supplier_data, use_container_width=True, height=250)
                except Exception as e:
                    st.error(f"Error generating supplier overview: {e}")
            else:
                st.info("No supplier data available in the loaded dataset.")
        
        with supplier_cols[1]:
            st.write("### First 5 Rows")
            st.dataframe(df.head(), key="trip_assignment_data_preview", height=250)
        
        # Add a divider for visual separation
        st.markdown("---")
        
        # Show results if available
        if "trip_assignment_result" in st.session_state:
            st.subheader("Assignment Results")
            
            # Add Overall View with metrics
            result_df = st.session_state.trip_assignment_result
            
            # Create styled container for metrics cards
            st.markdown("""
            <style>
            .metric-container {
                background-color: #f0f2f6;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 20px;
            }
            
            .dark-mode .metric-container {
                background-color: #262730;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            
            # Create metrics cards for overall view with more space
            metric_cols = st.columns(3)
            
            with metric_cols[0]:
                # Count unique Trip_IDs
                trip_count = result_df['Trip_ID'].nunique() if 'Trip_ID' in result_df.columns else 0
                st.metric("Total Trips", trip_count, help="Number of unique trips generated")
            
            with metric_cols[1]:
                # Count total tasks (rows)
                task_count = len(result_df)
                st.metric("Total Tasks", task_count, help="Total number of delivery tasks")
            
            with metric_cols[2]:
                # Count orders
                order_count = result_df['id'].nunique() if 'id' in result_df.columns else len(result_df)
                st.metric("Total Trip Orders", order_count, help="Number of unique orders")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add tabs for different views of the results
            result_tabs = st.tabs(["Trip Map", "Data Table"])
            
            with result_tabs[0]:
                st.subheader("Trip Route Visualization")
                
                # Create dropdowns to select Trip_ID
                trip_ids = result_df['Trip_ID'].unique() if 'Trip_ID' in result_df.columns else []
                if len(trip_ids) > 0:
                    selected_trip = st.selectbox("Select Trip to View", options=trip_ids)
                    
                    # Filter data for the selected trip
                    trip_data = result_df[result_df['Trip_ID'] == selected_trip]
                    
                    # Check if we have the necessary columns for mapping
                    map_cols = ['supplier_latitude', 'supplier_longitude', 'customer_latitude', 'customer_longitude']
                    if all(col in trip_data.columns for col in map_cols) and len(trip_data) > 0:
                        # Create a map
                        try:
                            import folium
                            from streamlit_folium import st_folium
                            import traceback
                            
                            # Get supplier coordinates (first row)
                            supplier_lat = safe_float(trip_data.iloc[0]['supplier_latitude'])
                            supplier_lon = safe_float(trip_data.iloc[0]['supplier_longitude'])
                            
                            # Validate supplier coordinates
                            if pd.isna(supplier_lat) or pd.isna(supplier_lon) or supplier_lat == 0 or supplier_lon == 0:
                                st.warning("Invalid supplier coordinates. Using default map center.")
                                supplier_lat, supplier_lon = 31.2001, 29.9187  # Default to Alexandria
                                
                            # Verify supplier coordinates are in sensible range for Egypt
                            if not (29.0 < supplier_lat < 32.0 and 29.0 < supplier_lon < 33.0):
                                # If coordinates seem reversed, swap them (common GIS error)
                                if 29.0 < supplier_lon < 32.0 and 29.0 < supplier_lat < 33.0:
                                    # Swap coordinates
                                    supplier_lat, supplier_lon = supplier_lon, supplier_lat
                                    st.info("Corrected reversed supplier coordinates")
                            
                            # Create a map centered at the supplier with better tile options
                            m = folium.Map(
                                location=[supplier_lat, supplier_lon],
                                zoom_start=12,
                                tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
                                attr='© OpenStreetMap contributors',
                                max_zoom=19,  # Allow high zoom levels
                                prefer_canvas=True,  # Use canvas for better performance
                                control_scale=True  # Add scale control
                            )
                            
                            # Add supplier marker with coordinate info for debugging
                            folium.Marker(
                                location=[supplier_lat, supplier_lon],
                                popup=f"Supplier Warehouse<br>Data: {trip_data.iloc[0]['supplier_latitude']}, {trip_data.iloc[0]['supplier_longitude']}<br>Used: {supplier_lat}, {supplier_lon}",
                                tooltip="Supplier Warehouse",
                                icon=folium.Icon(color='orange', icon='industry')
                            ).add_to(m)
                            
                            # Create route coordinates starting with supplier
                            route_coords = [[supplier_lat, supplier_lon]]
                            valid_points = 0
                            valid_customer_coords = []  # To store valid customer coordinates
                            
                            # Add customer markers
                            for idx, row in trip_data.iterrows():
                                try:
                                    # Get and validate customer coordinates
                                    customer_lat = safe_float(row['customer_latitude'])
                                    customer_lon = safe_float(row['customer_longitude'])
                                    
                                    # Skip invalid coordinates
                                    if pd.isna(customer_lat) or pd.isna(customer_lon) or customer_lat == 0 or customer_lon == 0:
                                        continue
                                    
                                    # Verify coordinates are in sensible range for Egypt
                                    if not (29.0 < customer_lat < 32.0 and 29.0 < customer_lon < 33.0):
                                        # If coordinates seem reversed, swap them (common GIS error)
                                        if 29.0 < customer_lon < 32.0 and 29.0 < customer_lat < 33.0:
                                            # Swap coordinates for display only (don't change the data)
                                            customer_lat, customer_lon = customer_lon, customer_lat
                                            st.info(f"Corrected reversed coordinates for stop {valid_points+1}")
                                    
                                    valid_points += 1
                                    # Add to route with corrected coordinates
                                    route_coords.append([customer_lat, customer_lon])
                                    # Also save for API call (longitude first for ORS)
                                    valid_customer_coords.append((customer_lon, customer_lat))
                                    
                                    # Add marker with popup showing actual coordinate values for debugging
                                    folium.Marker(
                                        location=[customer_lat, customer_lon],
                                        popup=f"Customer: {row.get('retailer_name', 'Unknown')}<br>Data: {row['customer_latitude']}, {row['customer_longitude']}<br>Used: {customer_lat}, {customer_lon}",
                                        tooltip=f"Stop {valid_points}",
                                        icon=folium.Icon(color='red', icon='home')
                                    ).add_to(m)
                                except Exception as e:
                                    continue
                            
                            # Add the polyline for the route if we have at least one customer
                            if valid_points > 0:
                                try:
                                    # Add a checkbox to toggle between real routes and straight lines
                                    use_real_routes = st.checkbox("Use real road routes",
                                                               value=True,
                                                               key=f"use_real_routes_{selected_trip}")
                                    
                                    total_distance = 0
                                    
                                    if use_real_routes:
                                        # Try to get real routes from OpenRouteService
                                        has_real_routes = False
                                        
                                        # For better route visualization, we'll get individual segments
                                        start_point = (supplier_lon, supplier_lat)  # ORS uses (lon, lat) order
                                        route_segments = []
                                        
                                        for i, customer_point in enumerate(valid_customer_coords):
                                            # Get route from supplier/previous customer to this customer
                                            prev_point = start_point if i == 0 else valid_customer_coords[i-1]
                                            
                                            # Get the route
                                            route_result = get_route(prev_point, customer_point)
                                            
                                            if route_result and 'features' in route_result and len(route_result['features']) > 0:
                                                # Extract route coordinates (needs conversion for folium - swap lat/lon)
                                                coords = route_result['features'][0]['geometry']['coordinates']
                                                route_line_points = [[coord[1], coord[0]] for coord in coords]  # Swap to [lat, lon]
                                                
                                                # Add to segments
                                                route_segments.append(route_line_points)
                                                
                                                # Add segment distance
                                                segment_distance = route_result['features'][0]['properties']['summary']['distance'] / 1000  # km
                                                total_distance += segment_distance
                                                
                                                has_real_routes = True
                                            else:
                                                # Fallback to straight line for this segment
                                                if i == 0:
                                                    segment_coords = [[supplier_lat, supplier_lon], 
                                                                      [customer_point[1], customer_point[0]]]
                                                else:
                                                    prev_customer = valid_customer_coords[i-1]
                                                    segment_coords = [[prev_customer[1], prev_customer[0]], 
                                                                      [customer_point[1], customer_point[0]]]
                                                
                                                route_segments.append(segment_coords)
                                                
                                                # Calculate straight-line distance
                                                if i == 0:
                                                    segment_distance = haversine(supplier_lat, supplier_lon, 
                                                                             customer_point[1], customer_point[0])
                                                else:
                                                    prev_customer = valid_customer_coords[i-1]
                                                    segment_distance = haversine(prev_customer[1], prev_customer[0], 
                                                                             customer_point[1], customer_point[0])
                                                total_distance += segment_distance
                                        
                                        # Now add all segments to the map
                                        for i, segment in enumerate(route_segments):
                                            color = 'blue' if i == 0 else '#22c55e'  # First segment blue, rest green
                                            
                                            # Create route line with different styling for real vs. straight routes
                                            folium.PolyLine(
                                                segment,
                                                color=color,
                                                weight=4,  # Thicker line
                                                opacity=0.8,  # More opaque
                                                dash_array=None,  # Solid line for real routes
                                                tooltip=f"Route Segment {i+1}"
                                            ).add_to(m)
                                        
                                        if has_real_routes:
                                            st.success("✅ Using real road routes from OpenRouteService")
                                    else:
                                        # Use simple straight-line route visualization (original code)
                                        route_line = folium.PolyLine(
                                            route_coords,
                                            color='blue',
                                            weight=4,  # Thicker line
                                            opacity=0.8,  # More opaque
                                            dash_array='5',  # Dashed line
                                            tooltip="Delivery Route"
                                        )
                                        route_line.add_to(m)
                                        
                                        # Calculate straight-line distances
                                        for i in range(len(route_coords)-1):
                                            lat1, lon1 = route_coords[i]
                                            lat2, lon2 = route_coords[i+1]
                                            segment_distance = haversine(lat1, lon1, lat2, lon2)
                                            total_distance += segment_distance
                                    
                                    # Add a layer control (only if we have multiple tile layers)
                                    folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
                                    folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
                                    folium.LayerControl().add_to(m)
                                    
                                    # Add trip info
                                    st.metric("Route Distance", f"{total_distance:.2f} km")
                                except Exception as e:
                                    st.warning(f"Could not draw route line: {e}")
                                    st.code(traceback.format_exc())
                            else:
                                st.warning("No valid customer coordinates found.")
                            
                            # Display the map with better rendering settings
                            st_folium(
                                m,
                                width=700,
                                height=500, 
                                returned_objects=[],
                                feature_group_to_add=None,
                                use_container_width=True,  # Use full width
                                key=f"trip_map_{selected_trip}"  # Unique key for each trip
                            )
                            
                        except Exception as e:
                            st.error(f"Error creating map: {str(e)}")
                            st.code(traceback.format_exc())
                    else:
                        st.warning("Missing location data for trip visualization.")
                else:
                    st.warning("No trips available to visualize.")
            
            with result_tabs[1]:
                # Show the dataframe with results
                st.subheader("Trip Assignment Data")
                st.dataframe(st.session_state.trip_assignment_result, key="trip_assignment_results", height=500)
                
                # Download option
                download_cols = st.columns([3, 1])
                with download_cols[1]:
                    csv = st.session_state.trip_assignment_result.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Results CSV",
                        csv,
                        "trip_assignment_results.csv",
                        "text/csv",
                        key="trip_assignment_download",
                        use_container_width=True
                    )

