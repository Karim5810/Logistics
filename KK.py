# Import needed modules
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from google.auth.transport.requests import Request
from streamlit_elements import elements, mui, dashboard, nivo, sync, lazy
# Import trip_assignment module
import trip_assignment
# Import counter example module
import json
import os

# Import sweetviz for analysis if available
try:
    import sweetviz as sv
except ImportError:
    pass

import folium
from streamlit_folium import st_folium
import traceback
import math
import importlib
import io
# Import OpenRouteService for real road routes
import openrouteservice

# Define the path for dashboard settings file
DASHBOARD_SETTINGS_FILE = "dashboard_settings.json"

# Function to save dashboard settings to file
def save_dashboard_settings():
    """Save dashboard settings to a JSON file"""
    try:
        # Convert dashboard_layout to a serializable format if it exists
        dashboard_layout = None
        if st.session_state.dashboard_layout is not None:
            dashboard_layout = []
            for item in st.session_state.dashboard_layout:
                # Convert to dict if it's not already
                if not isinstance(item, dict):
                    item_dict = {
                        "i": item["i"],
                        "x": item["x"],
                        "y": item["y"],
                        "w": item["w"],
                        "h": item["h"]
                    }
                    dashboard_layout.append(item_dict)
                else:
                    dashboard_layout.append(item)
            
        settings = {
            "dashboard_layout": dashboard_layout,
            "dashboard_column_mapping": st.session_state.dashboard_column_mapping
        }
        with open(DASHBOARD_SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        print(f"Dashboard settings saved to {DASHBOARD_SETTINGS_FILE}")
        return True
    except Exception as e:
        print(f"Error saving dashboard settings: {e}")
        import traceback
        print(traceback.format_exc())
        return False

# Function to load dashboard settings from file
def load_dashboard_settings():
    """Load dashboard settings from a JSON file"""
    try:
        if os.path.exists(DASHBOARD_SETTINGS_FILE):
            with open(DASHBOARD_SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                
                # Load dashboard layout
                if "dashboard_layout" in settings and settings["dashboard_layout"]:
                    # Convert list of dicts to the format expected by the dashboard grid
                    dashboard_layout = []
                    for item in settings["dashboard_layout"]:
                        if isinstance(item, dict) and all(k in item for k in ["i", "x", "y", "w", "h"]):
                            dashboard_layout.append(item)
                    
                    if dashboard_layout:
                        st.session_state.dashboard_layout = dashboard_layout
                        print(f"Loaded dashboard layout with {len(dashboard_layout)} items")
                
                # Load column mapping
                if "dashboard_column_mapping" in settings and settings["dashboard_column_mapping"]:
                    # Ensure all expected keys exist to prevent KeyError
                    base_mapping = {
                        "approve_hours": "",
                        "logistics_48hrs": "",
                        "logistics_24hrs": "",
                        "cartona_48hrs": "",
                        "cartona_24hrs": "",
                        "avg_days_deliver": "",
                        "avg_days_pending": "",
                        "chart_settings": {
                            "supplier_gmv_share": {
                                "title": "Supplier GMV Share",
                                "data_column": "",
                                "chart_type": "pie"
                            },
                            "task_count_runsheet": {
                                "title": "Task Count by Runsheet",
                                "data_column": "",
                                "chart_type": "bar"
                            }
                        },
                        "card_titles": {
                            "avg_hours_approve": "Average Hours to Approve Order",
                            "avg_days_deliver": "Average Days to Deliver",
                            "avg_days_pending": "Average Days Pending",
                            "fast_delivery_logistics": "Fast Delivery Rate - Logistics",
                            "fast_delivery_cartona": "Fast Delivery Rate - Cartona"
                        }
                    }
                    
                    # Update with stored settings, keeping defaults for missing keys
                    loaded_mapping = settings['dashboard_column_mapping']
                    
                    # Handle chart_settings if it exists
                    if 'chart_settings' in loaded_mapping:
                        for chart_id, defaults in base_mapping['chart_settings'].items():
                            if chart_id in loaded_mapping['chart_settings']:
                                # Update with existing values
                                for key in defaults:
                                    if key in loaded_mapping['chart_settings'][chart_id]:
                                        base_mapping['chart_settings'][chart_id][key] = loaded_mapping['chart_settings'][chart_id][key]
                    
                    # Handle card_titles if it exists
                    if 'card_titles' in loaded_mapping:
                        for card_id, default_title in base_mapping['card_titles'].items():
                            if card_id in loaded_mapping['card_titles']:
                                base_mapping['card_titles'][card_id] = loaded_mapping['card_titles'][card_id]
                    
                    # Handle simple key/value pairs
                    for key in base_mapping:
                        if key not in ['chart_settings', 'card_titles'] and key in loaded_mapping:
                            base_mapping[key] = loaded_mapping[key]
                    
                    # Use the merged settings
                    st.session_state.dashboard_column_mapping = base_mapping
                    print(f"Loaded column mapping with structure-preserving merge")
            
            print(f"Dashboard settings loaded successfully from {DASHBOARD_SETTINGS_FILE}")
            return True
        
        print(f"Dashboard settings file not found: {DASHBOARD_SETTINGS_FILE}")
        return False
    except Exception as e:
        print(f"Error loading dashboard settings: {e}")
        import traceback
        print(traceback.format_exc())
        return False

# Reload trip_assignment module to get the latest changes
importlib.reload(trip_assignment)

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

# Initialize ORS API key from config file
ORS_API_KEY = load_ors_config()

# Add ORS API Key handling in the session state
if "ors_api_key" not in st.session_state:
    st.session_state.ors_api_key = ORS_API_KEY

# Initialize OpenRouteService client
try:
    ors_client = openrouteservice.Client(key=ORS_API_KEY)
except Exception as e:
    ors_client = None
    print(f"Error initializing OpenRouteService client: {e}")

def init_ors_client():
    """Initialize or reinitialize the OpenRouteService client with the current API key"""
    global ors_client
    try:
        if st.session_state.ors_api_key and st.session_state.ors_api_key != "YOUR_ORS_API_KEY":
            ors_client = openrouteservice.Client(key=st.session_state.ors_api_key)
            return True
        return False
    except Exception as e:
        print(f"Error initializing OpenRouteService client: {e}")
        ors_client = None
        return False

def get_route(start_coords, end_coords):
    """
    Get route from OpenRouteService between two points.
    
    Args:
        start_coords: Tuple of (longitude, latitude) for start point
        end_coords: Tuple of (longitude, latitude) for end point
        
    Returns:
        GeoJSON of the route or None if there was an error
    """
    try:
        # If the client is None, try to initialize it
        if ors_client is None and not init_ors_client():
            return None
            
        # Get directions from OpenRouteService
        route = ors_client.directions(
            coordinates=[start_coords, end_coords],
            profile='driving-car',
            format='geojson'
        )
        return route
    except Exception as e:
        print(f"Error getting route: {e}")
        return None

def is_valid_egypt_coordinates(lat, lng):
    """Validate if coordinates are within Egypt's boundaries."""
    # Egypt's approximate boundaries
    EGYPT_BOUNDS = {
        'min_lat': 22.0,  # Southernmost point
        'max_lat': 31.9,  # Northernmost point
        'min_lng': 24.7,  # Westernmost point
        'max_lng': 37.0   # Easternmost point
    }
    
    try:
        lat = float(lat)
        lng = float(lng)
        return (EGYPT_BOUNDS['min_lat'] <= lat <= EGYPT_BOUNDS['max_lat'] and 
                EGYPT_BOUNDS['min_lng'] <= lng <= EGYPT_BOUNDS['max_lng'])
    except (ValueError, TypeError):
        return False

def validate_coordinates(df, lat_col, lng_col):
    """Validate coordinates in a dataframe and return valid rows."""
    valid_mask = df.apply(lambda row: is_valid_egypt_coordinates(row[lat_col], row[lng_col]), axis=1)
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        st.warning(f"Found {invalid_count} invalid coordinates that will be filtered out.")
    return df[valid_mask]

# Debug code to list all available sheets
def list_all_sheets():
    try:
        client = init_gspread_client()
        if client is None:
            st.error("Unable to initialize gspread client")
            return
        
        ss = client.open_by_url(SPREADSHEET_URL)
        sheet_names = [sheet.title for sheet in ss.worksheets()]
        st.write("### Available Sheets:")
        for name in sheet_names:
            st.write(f"- {name}")
    except Exception as e:
        st.error(f"Error listing sheets: {e}")
        st.code(traceback.format_exc())

# Define haversine function for calculating distance
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Add safety checks for invalid or empty values
    try:
        # Convert decimal degrees to radians
        lat1 = float(lat1) if lat1 else 0
        lon1 = float(lon1) if lon1 else 0
        lat2 = float(lat2) if lat2 else 0
        lon2 = float(lon2) if lon2 else 0
        
        # Return 0 if coordinates are missing
        if (lat1 == 0 and lon1 == 0) or (lat2 == 0 and lon2 == 0):
            return 0
            
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        return c * r
    except (ValueError, TypeError) as e:
        # Fail gracefully on conversion errors
        print(f"Error calculating distance: {e}")
        return 0

# Initialize session state
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False
if "previous_layout" not in st.session_state:
    st.session_state.previous_layout = None
if "dashboard_layout" not in st.session_state:
    st.session_state.dashboard_layout = None
if "dashboard_column_mapping" not in st.session_state:
    st.session_state.dashboard_column_mapping = {
        "approve_hours": "",
        "logistics_48hrs": "",
        "logistics_24hrs": "",
        "cartona_48hrs": "",
        "cartona_24hrs": "",
        "avg_days_deliver": "",
        "avg_days_pending": "",
        "chart_settings": {
            "supplier_gmv_share": {
                "title": "Supplier GMV Share",
                "data_column": "",
                "chart_type": "pie"
            },
            "task_count_runsheet": {
                "title": "Task Count by Runsheet",
                "data_column": "",
                "chart_type": "bar"
            }
        },
        "card_titles": {
            "avg_hours_approve": "Average Hours to Approve Order",
            "avg_days_deliver": "Average Days to Deliver",
            "avg_days_pending": "Average Days Pending",
            "fast_delivery_logistics": "Fast Delivery Rate - Logistics",
            "fast_delivery_cartona": "Fast Delivery Rate - Cartona"
        }
    }
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "view" not in st.session_state:
    st.session_state.view = "dashboard"
if "selected_retailer" not in st.session_state:
    st.session_state.selected_retailer = None
if "user_name" not in st.session_state:
    st.session_state.user_name = "Guest"
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "counter" not in st.session_state:
    st.session_state.counter = 0

# Load saved dashboard settings after session state is initialized
try:
    load_success = load_dashboard_settings()
    print(f"Load dashboard settings success: {load_success}")
except Exception as e:
    print(f"Error during dashboard settings loading: {e}")
    # If there's any error during loading, we'll just use the default settings
    load_success = False

# Constants
SERVICE_ACCOUNT_FILE = "Credentials_clean.json"
SPREADSHEET_KEY = "138EPgPhd0Ddp1-z9xPCexsm5s--t5EHjPpK7NSTT1LI"
SPREADSHEET_URL = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_KEY}/edit"
WEIGHT_CALCULATOR_SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_KEY}/edit"
ORDERS_SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_KEY}/edit"

# After imports section, add authentication check
# Check if we're being imported by the login module


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
        
        # Handle empty and duplicate headers more robustly
        clean_headers = []
        seen_headers = set()
        for i, header in enumerate(headers):
            if header == '':
                header = f"column_{i}"
            
            # Ensure unique header names
            original_header = header
            counter = 0
            while header in seen_headers:
                counter += 1
                header = f"{original_header}_{counter}"
            
            seen_headers.add(header)
            clean_headers.append(header)
        
        # Create DataFrame with cleaned headers
        return pd.DataFrame(data[1:], columns=clean_headers)
    except gspread.WorksheetNotFound:
        st.error(f"The '{sheet_name}' sheet is not found.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load pending tasks data
def load_pending_tasks():
    # First try loading with "Tasks" sheet name
    return load_sheet_data("Tasks")
        
  

# Load weight calculator data
def load_weight_calculator_data():
    return load_sheet_data("Active Suppliers Products")

# Load orders data
def load_orders_data():
    return load_sheet_data("Runsheet")

# Helper function to load and process dashboard metrics
def load_and_process_dashboard_metrics():
    """
    Load and process data specifically for dashboard metrics.
    Returns a dictionary with processed metrics.
    """
    metrics = {
        "avg_hours_approve": {"value": "12.5", "change": "+8.3%"},
        "avg_days_deliver": {"value": "3.2", "change": "-5.2%"},
        "avg_days_pending": {"value": "1.5", "change": "+2.1%"},
        "logistics_delivery": {"48hrs": "76.49786020", "24hrs": "32.45363766", "change": "+3.8%"},
        "cartona_delivery": {"48hrs": "70.36376605", "24hrs": "25.99857347", "change": "+2.1%"}
    }
    
    try:
        # Load overview data
        overview_data = load_sheet_data("Overview")
        
        if overview_data is not None and not overview_data.empty:
            # Get column mapping from session state if available
            column_mapping = st.session_state.get("dashboard_column_mapping", {})
            
            # Process average hours to approve order
            approve_hours_col = column_mapping.get("approve_hours", "")
            if approve_hours_col and approve_hours_col in overview_data.columns:
                # Use mapped column
                hour_cols = [approve_hours_col]
            else:
                # Fallback to auto-detection
                hour_cols = [col for col in overview_data.columns if "average hours" in col.lower() and "approve" in col.lower()]
            
            if hour_cols and len(overview_data) >= 2:
                metrics["avg_hours_approve"]["value"] = overview_data.iloc[0][hour_cols[0]]
                if len(overview_data) > 1:
                    try:
                        current = float(overview_data.iloc[0][hour_cols[0]])
                        previous = float(overview_data.iloc[1][hour_cols[0]])
                        change_val = ((current - previous) / previous) * 100 if previous != 0 else 0
                        metrics["avg_hours_approve"]["change"] = f"{'+' if change_val >= 0 else ''}{change_val:.1f}%"
                    except (ValueError, TypeError):
                        # Handle non-numeric values
                        metrics["avg_hours_approve"]["change"] = "N/A"
            
            # Process average days to deliver
            days_deliver_col = column_mapping.get("avg_days_deliver", "")
            if days_deliver_col and days_deliver_col in overview_data.columns:
                # Use mapped column
                days_deliver_cols = [days_deliver_col]
            else:
                # Fallback to auto-detection
                days_deliver_cols = [col for col in overview_data.columns if "days" in col.lower() and "deliver" in col.lower()]
            
            if days_deliver_cols and len(overview_data) >= 2:
                metrics["avg_days_deliver"]["value"] = overview_data.iloc[0][days_deliver_cols[0]]
                if len(overview_data) > 1:
                    try:
                        current = float(overview_data.iloc[0][days_deliver_cols[0]])
                        previous = float(overview_data.iloc[1][days_deliver_cols[0]])
                        change_val = ((current - previous) / previous) * 100 if previous != 0 else 0
                        metrics["avg_days_deliver"]["change"] = f"{'+' if change_val >= 0 else ''}{change_val:.1f}%"
                    except (ValueError, TypeError):
                        # Handle non-numeric values
                        metrics["avg_days_deliver"]["change"] = "N/A"
            
            # Process average days pending
            days_pending_col = column_mapping.get("avg_days_pending", "")
            if days_pending_col and days_pending_col in overview_data.columns:
                # Use mapped column
                days_pending_cols = [days_pending_col]
            else:
                # Fallback to auto-detection
                days_pending_cols = [col for col in overview_data.columns if "days" in col.lower() and "pending" in col.lower()]
            
            if days_pending_cols and len(overview_data) >= 2:
                metrics["avg_days_pending"]["value"] = overview_data.iloc[0][days_pending_cols[0]]
                if len(overview_data) > 1:
                    try:
                        current = float(overview_data.iloc[0][days_pending_cols[0]])
                        previous = float(overview_data.iloc[1][days_pending_cols[0]])
                        change_val = ((current - previous) / previous) * 100 if previous != 0 else 0
                        metrics["avg_days_pending"]["change"] = f"{'+' if change_val >= 0 else ''}{change_val:.1f}%"
                    except (ValueError, TypeError):
                        # Handle non-numeric values
                        metrics["avg_days_pending"]["change"] = "N/A"
            
            # Process Logistics delivery rates
            logistics_48hrs_col = column_mapping.get("logistics_48hrs", "")
            logistics_24hrs_col = column_mapping.get("logistics_24hrs", "")
            
            if logistics_48hrs_col and logistics_48hrs_col in overview_data.columns:
                # Use mapped column
                logistics_48_cols = [logistics_48hrs_col]
            else:
                # Fallback to auto-detection
                logistics_48_cols = [col for col in overview_data.columns if "48hrs" in col.lower() and "logistics" in col.lower()]
            
            if logistics_24hrs_col and logistics_24hrs_col in overview_data.columns:
                # Use mapped column
                logistics_24_cols = [logistics_24hrs_col]
            else:
                # Fallback to auto-detection
                logistics_24_cols = [col for col in overview_data.columns if "24hrs" in col.lower() and "logistics" in col.lower()]
            
            if logistics_48_cols and len(overview_data) >= 1:
                metrics["logistics_delivery"]["48hrs"] = overview_data.iloc[0][logistics_48_cols[0]]
                if len(overview_data) > 1:
                    try:
                        current = float(str(overview_data.iloc[0][logistics_48_cols[0]]).replace('%', ''))
                        previous = float(str(overview_data.iloc[1][logistics_48_cols[0]]).replace('%', ''))
                        change = current - previous
                        metrics["logistics_delivery"]["change"] = f"{'+' if change >= 0 else ''}{change:.1f}%"
                    except (ValueError, TypeError):
                        metrics["logistics_delivery"]["change"] = "N/A"
            
            if logistics_24_cols and len(overview_data) >= 1:
                metrics["logistics_delivery"]["24hrs"] = overview_data.iloc[0][logistics_24_cols[0]]
            
            # Process Cartona delivery rates
            cartona_48hrs_col = column_mapping.get("cartona_48hrs", "")
            cartona_24hrs_col = column_mapping.get("cartona_24hrs", "")
            
            if cartona_48hrs_col and cartona_48hrs_col in overview_data.columns:
                # Use mapped column
                cartona_48_cols = [cartona_48hrs_col]
            else:
                # Fallback to auto-detection
                cartona_48_cols = [col for col in overview_data.columns if "48hrs" in col.lower() and "cartona" in col.lower()]
            
            if cartona_24hrs_col and cartona_24hrs_col in overview_data.columns:
                # Use mapped column
                cartona_24_cols = [cartona_24hrs_col]
            else:
                # Fallback to auto-detection
                cartona_24_cols = [col for col in overview_data.columns if "24hrs" in col.lower() and "cartona" in col.lower()]
            
            if cartona_48_cols and len(overview_data) >= 1:
                metrics["cartona_delivery"]["48hrs"] = overview_data.iloc[0][cartona_48_cols[0]]
                if len(overview_data) > 1:
                    try:
                        current = float(str(overview_data.iloc[0][cartona_48_cols[0]]).replace('%', ''))
                        previous = float(str(overview_data.iloc[1][cartona_48_cols[0]]).replace('%', ''))
                        change = current - previous
                        metrics["cartona_delivery"]["change"] = f"{'+' if change >= 0 else ''}{change:.1f}%"
                    except (ValueError, TypeError):
                        metrics["cartona_delivery"]["change"] = "N/A"
            
            if cartona_24_cols and len(overview_data) >= 1:
                metrics["cartona_delivery"]["24hrs"] = overview_data.iloc[0][cartona_24_cols[0]]
    
    except Exception as e:
        print(f"Error processing dashboard metrics: {e}")
    
    return metrics

def calculate_weight_and_cbm(merged_data):
    try:
        # Check for required columns
        required_columns = [
            'product_amount',
            'وزن الكرتونة (كجم)',
            'وزن التغليف (جم)',
            'حجم الكرتونة (م³)'
        ]
        
        missing_cols = [col for col in required_columns if col not in merged_data.columns]
        if missing_cols:
            st.error(f"Missing required columns for calculations: {', '.join(missing_cols)}")
            return None

        # Calculate total weight in kg
        merged_data['total_weight_kg'] = (
            (merged_data['product_amount'] * merged_data['وزن الكرتونة (كجم)']) +
            (merged_data['وزن التغليف (جم)'] * merged_data['product_amount'] / 1000)
        )
        
        # Calculate total CBM (Cubic Meters)
        merged_data['total_cbm'] = merged_data['product_amount'] * merged_data['حجم الكرتونة (م³)']
        
        # Handle any invalid values
        merged_data['total_weight_kg'] = merged_data['total_weight_kg'].fillna(0)
        merged_data['total_cbm'] = merged_data['total_cbm'].fillna(0)
        
        return merged_data
    except Exception as e:
        st.error(f"Error calculating weight and CBM: {e}")
        return None

# Aggregate Order Data
def aggregate_order_data(merged_data):
    try:
        required_columns = ['task_id', 'created_at', 'total_weight_kg', 'total_cbm', 'product_id', 'base_product_id']
        for col in required_columns:
            if col not in merged_data.columns:
                raise KeyError(f"Required column '{col}' is missing")

        return merged_data.groupby(['task_id', 'created_at']).agg(
            total_weight_kg=('total_weight_kg', 'sum'),
            total_cbm=('total_cbm', 'sum'),
            total_products=('product_id', 'count'),
            total_product_types=('base_product_id', 'nunique')
        ).reset_index()
    except Exception as e:
        st.error(f"Error aggregating order data: {e}")
        return None

# Load all available sheets
@st.cache_data(ttl=600)
def get_available_sheets():
    client = init_gspread_client()
    if client is None:
        return ["No sheets available"]
    try:
        ss = client.open_by_url(SPREADSHEET_URL)
        return [sheet.title for sheet in ss.worksheets()]
    except Exception as e:
        st.error(f"Error loading sheets: {e}")
        return ["Error loading sheets"]

# Custom CSS based on theme
def get_css():
    # الحصول على قيمة theme بطريقة آمنة
    current_theme = st.session_state.get("theme", "light")
    
    base_css = """
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    * {
        font-family: 'Roboto', sans-serif;
    }
    
    .header-container {
        display: flex; 
        justify-content: space-between; 
        align-items: center;
        padding: 1rem 2rem; 
        box-shadow: 0 2px 5px 0 rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem; 
        border-bottom: 1px solid #e2e8f0;
        height: 80px;
        width: 100%;
        position: relative;
        left: 0;
        right: 0;
        box-sizing: border-box;
    }
    
    .dashboard-title {
        font-size: 2rem;
        font-weight: 600;
        color: #1e40af;
        font-family: 'Roboto', sans-serif;
        letter-spacing: -0.5px;
        margin: 0;
    }
    
    .stButton > button {
        height: 48px; 
        padding: 0 24px; 
        border-radius: 8px; 
        font-size: 14px; 
        font-weight: 500; 
        display: flex;
        align-items: center; 
        justify-content: center; 
        gap: 8px;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
    }
    
    #MainMenu, footer, header { 
        visibility: hidden; 
    }
    
    .metric-value { 
        font-size: 2rem; 
        font-weight: 600; 
        margin: 0.5rem 0; 
    }
    
    .metric-label { 
        font-size: 0.875rem; 
        opacity: 0.7; 
        margin-bottom: 0.25rem; 
    }
    
    .metric-change { 
        font-size: 0.875rem; 
        display: flex; 
        align-items: center; 
        gap: 0.25rem; 
    }
    
    .dashboard-container {
        max-width: 100% !important;
        padding: 0 !important;
    }
    
    .retailer-card {
        position: absolute;
        top: 10px;
        right: 10px;
        width: 300px;
        background: white;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        z-index: 1000;
        max-height: 80vh;
        overflow-y: auto;
    }
    
    .retailer-card h3 {
        margin-top: 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #eee;
    }
    
    .retailer-card .close-btn {
        position: absolute;
        top: 8px;
        right: 8px;
        background: none;
        border: none;
        font-size: 18px;
        cursor: pointer;
        color: #666;
    }
    
    .retailer-card .info-row {
        display: flex;
        margin-bottom: 8px;
    }
    
    .retailer-card .info-label {
        font-weight: 600;
        width: 40%;
        color: #555;
    }
    
    .retailer-card .info-value {
        width: 60%;
    }
    """
    
    if current_theme == "light":
        return base_css + """
        [data-testid="stAppViewContainer"] { 
            background-color: #f8fafc; 
            color: #1e293b;
        }
        .header-container {
            background: white;
        }
        .positive-change { 
            color: #059669; 
        }
        .negative-change { 
            color: #dc2626; 
        }
        .stButton > button {
            background: white; 
            border: 1px solid #e2e8f0;
            color: #1e293b;
        }
        .stButton > button:hover {
            background: #f8fafc; 
            border-color: #1a237e; 
        }
        .retailer-card {
            background: white;
            color: #1e293b;
        }
        """
    else:
        return base_css + """
        [data-testid="stAppViewContainer"] { 
            background-color: #1e293b; 
            color: #f8fafc;
        }
        .header-container {
            background: #0f172a;
            border-color: #334155;
        }
        .dashboard-title {
            color: #60a5fa;
        }
        .positive-change { 
            color: #4ade80; 
        }
        .negative-change { 
            color: #f87171; 
        }
        .stButton > button {
            background: #334155; 
            border: 1px solid #475569;
            color: #f8fafc;
        }
        .stButton > button:hover {
            background: #475569; 
            border-color: #64748b; 
        }
        .retailer-card {
            background: #0f172a;
            color: #f8fafc;
            border: 1px solid #334155;
        }
        .retailer-card .info-label {
            color: #9ca3af;
        }
        """

# Apply CSS but hide it from the UI
st.markdown(f"<style>{get_css()}</style>", unsafe_allow_html=True)

# Fix layout container width
st.markdown("""
<style>
.block-container {
    max-width: 100% !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* Prevent layout shift when scrollbars appear */
html {
    overflow-y: scroll;
}

/* Fix for app container */
[data-testid="stAppViewContainer"] {
    padding-top: 0 !important;
}

/* Make sure elements stay within their container */
img {
    max-width: 100%;
    height: auto;
}

/* Add more space to the left/right of the main content */
[data-testid="stVerticalBlock"] {
    padding-left: 5px;
    padding-right: 5px;
}

/* Style header buttons for consistency */
.header-btn {
    background-color: #2196F3;
    color: white !important;
    padding: 8px 16px;
    border-radius: 4px;
    cursor: pointer;
    border: none;
    text-align: center;
    display: inline-block;
    transition: background-color 0.3s;
    margin-right: 10px;
    font-weight: 500;
}

.header-btn:hover {
    background-color: #0b7dda;
}

.header-btn.dark {
    background-color: #333;
}

.header-btn.dark:hover {
    background-color: #444;
}

.header-btn.light {
    background-color: #f8f9fa;
    color: #333 !important;
    border: 1px solid #ddd;
}

.header-btn.light:hover {
    background-color: #e9ecef;
}

.header-btn.refresh {
    background-color: #4CAF50;
}

.header-btn.refresh:hover {
    background-color: #45a049;
}
</style>
""", unsafe_allow_html=True)

# Add loading spinner
with st.spinner("Loading dashboard..."):
    # Header with improved styling
    with st.container():
        st.markdown("""
        <div class="header-container">
            <div style="display: flex; align-items: center; justify-content: space-between; width: 100%;">
                <div>
                    <img src="https://i.postimg.cc/HWtyY6jh/output-onlinepngtools.png" width="450">
                </div>
                <div style="display: flex; gap: 12px; align-items: center;">
                    <div id="header-button-container"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a single hidden container for the buttons that will be moved by JavaScript
        button_container = st.container()
        
        # All buttons in one container with custom CSS for positioning
        with button_container:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Edit button with custom CSS to make it look like a button
                edit_mode = st.button("🖊️ Edit Dashboard", key="edit_mode_toggle_header")
                if edit_mode:
                    # Store previous layout before toggling edit mode
                    if not st.session_state.edit_mode:
                        # Turning edit mode ON
                        st.session_state.edit_mode = True
                    else:
                        # Turning edit mode OFF
                        st.session_state.previous_layout = st.session_state.dashboard_layout
                        save_dashboard_settings()
                        st.session_state.edit_mode = False
                    st.rerun()
            
            # Theme toggle button
            with col2:
                current_theme = st.session_state.get("theme", "light")
                theme_label = "🌙 Dark Mode" if current_theme == "light" else "☀️ Light Mode"
                if st.button(theme_label, key="theme_toggle_header"):
                    st.session_state.theme = "dark" if current_theme == "light" else "light"
                    st.rerun()
            
            # Refresh button
            with col3:
                if st.button("🔄 Refresh Data", key="refresh_button_header"):
                    # Refresh all data
                    st.session_state.overview_data = load_sheet_data("Overview")
                    st.session_state.runsheet_data = load_sheet_data("Runsheet")
                    st.session_state.drivers_data = None
                    st.session_state.drivers_map_initialized = False
                    st.session_state.map_initialized = False
                    if "dashboard_metrics" in st.session_state:
                        del st.session_state.dashboard_metrics
                    st.success("All data refreshed successfully!")
                    st.rerun()
        
        # Add JavaScript to move the buttons into the header container and apply custom styling
        st.markdown("""
        <script>
            // Function to set up the header buttons
            function setupHeaderButtons() {
                // Find all buttons in the container
                const headerButtonContainer = document.getElementById('header-button-container');
                const columns = document.querySelectorAll('[data-testid="column"]');
                
                if (headerButtonContainer && columns.length >= 3) {
                    // Clear previous content
                    headerButtonContainer.innerHTML = '';
                    
                    // Add the buttons with proper styling
                    for (let i = 0; i < 3; i++) {
                        const button = columns[i].querySelector('button');
                        if (button) {
                            // Create a wrapper to maintain button functionality
                            const wrapper = document.createElement('div');
                            wrapper.style.display = 'inline-block';
                            wrapper.style.marginRight = '10px';
                            
                            // Clone the button to maintain click handlers
                            const styledButton = button.cloneNode(true);
                            
                            // Add appropriate class based on button content
                            if (styledButton.innerText.includes('Edit')) {
                                styledButton.className = 'header-btn';
                            } else if (styledButton.innerText.includes('Dark')) {
                                styledButton.className = 'header-btn dark';
                            } else if (styledButton.innerText.includes('Light')) {
                                styledButton.className = 'header-btn light';
                            } else if (styledButton.innerText.includes('Refresh')) {
                                styledButton.className = 'header-btn refresh';
                            }
                            
                            // Fix button styling
                            styledButton.style.padding = '8px 16px';
                            styledButton.style.width = 'auto';
                            styledButton.style.height = 'auto';
                            styledButton.style.lineHeight = 'normal';
                            styledButton.style.borderRadius = '4px';
                            styledButton.style.fontSize = '14px';
                            
                            // Replace button in the original location to maintain click handlers
                            columns[i].style.display = 'none';
                            headerButtonContainer.appendChild(styledButton);
                        }
                    }
                }
            }
            
            // Run the setup after a short delay to ensure all elements are available
            setTimeout(setupHeaderButtons, 500);
        </script>
        """, unsafe_allow_html=True)

# Remove running animation icon
st.markdown("""
    <style>
    [data-testid="stStatusWidget"] {display: none !important;}
    
    /* Fix tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        padding-left: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
        padding: 0 16px;
        font-size: 14px;
    }
    
    /* Add padding to tab content */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dashboard", "Map View", "Sheet Data", "Trip Assignment", "CBM & Weight Calculator"])

with tab1:
    # Better header within tab
    tab_header_col1, tab_header_col2 = st.columns([8, 2])
    
    with tab_header_col2:
        # Only show edit button in this section - use a regular button like the others for consistency
        current_edit_mode = st.session_state.edit_mode
        button_label = "✏️ Exit Edit Mode" if current_edit_mode else "✏️ Edit Dashboard"
        button_color = "#4CAF50" if current_edit_mode else "#2196F3"
        
        # Custom styling for the button
        st.markdown(f"""
        <style>
        div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {{
            width: 100%;
            background-color: {button_color};
            color: white;
            border: none;
            transition: all 0.3s;
        }}
        </style>
        """, unsafe_allow_html=True)
        
        if st.button(button_label, key="edit_mode_toggle_tab"):
            # Store previous layout before toggling edit mode
            if st.session_state.edit_mode:
                st.session_state.previous_layout = st.session_state.dashboard_layout
                # Save settings when exiting edit mode
                save_dashboard_settings()
            
            # Toggle edit mode
            st.session_state.edit_mode = not current_edit_mode
            st.rerun()
    
    # Add save button that only appears in edit mode
    if st.session_state.edit_mode:
        save_col1, save_col2 = st.columns(2)
        with save_col1:
            if st.button("💾 Save Layout", key="save_dashboard_layout"):
                save_dashboard_settings()
                st.success("Dashboard layout saved!")
                
        with save_col2:
            if st.button("↩️ Reset to Default", key="reset_dashboard_layout"):
                # Reset column mapping
                st.session_state.dashboard_column_mapping = {
                    "approve_hours": "",
                    "logistics_48hrs": "",
                    "logistics_24hrs": "",
                    "cartona_48hrs": "",
                    "cartona_24hrs": "",
                    "avg_days_deliver": "",
                    "avg_days_pending": "",
                    "chart_settings": {
                        "supplier_gmv_share": {
                            "title": "Supplier GMV Share",
                            "data_column": "",
                            "chart_type": "pie"
                        },
                        "task_count_runsheet": {
                            "title": "Task Count by Runsheet",
                            "data_column": "",
                            "chart_type": "bar"
                        }
                    },
                    "card_titles": {
                        "avg_hours_approve": "Average Hours to Approve Order",
                        "avg_days_deliver": "Average Days to Deliver",
                        "avg_days_pending": "Average Days Pending",
                        "fast_delivery_logistics": "Fast Delivery Rate - Logistics",
                        "fast_delivery_cartona": "Fast Delivery Rate - Cartona"
                    }
                }
                # Reset layout to default
                st.session_state.dashboard_layout = None
                save_dashboard_settings()
                st.success("Dashboard reset to default!")
                st.rerun()
    
    # Add Export/Import options
    st.subheader("Export/Import Dashboard Settings")
    exp_col1, exp_col2 = st.columns(2)
    
    with exp_col1:
        if st.button("📤 Export Settings", key="export_settings"):
            # Get current settings
            dashboard_layout = None
            if st.session_state.dashboard_layout is not None:
                dashboard_layout = []
                for item in st.session_state.dashboard_layout:
                    # Convert to dict if it's not already
                    if not isinstance(item, dict):
                        item_dict = {
                            "i": item["i"],
                            "x": item["x"],
                            "y": item["y"],
                            "w": item["w"],
                            "h": item["h"]
                        }
                        dashboard_layout.append(item_dict)
                    else:
                        dashboard_layout.append(item)
                
            export_data = {
                "dashboard_layout": dashboard_layout,
                "dashboard_column_mapping": st.session_state.dashboard_column_mapping
            }
            
            # Convert to JSON string
            json_str = json.dumps(export_data, indent=2)
            
            # Create download button
            st.download_button(
                label="Download Settings JSON",
                data=json_str,
                file_name="dashboard_settings.json",
                mime="application/json",
                key="download_settings"
            )
    
    with exp_col2:
        uploaded_file = st.file_uploader("Import Settings", type="json", key="import_settings_file")
        if uploaded_file is not None:
            try:
                import_data = json.load(uploaded_file)
                
                # Validate import data
                if "dashboard_layout" in import_data and "dashboard_column_mapping" in import_data:
                    if st.button("✅ Apply Imported Settings", key="apply_imported_settings"):
                        # Apply the imported settings
                        if import_data["dashboard_layout"]:
                            st.session_state.dashboard_layout = import_data["dashboard_layout"]
                        else:
                            st.session_state.dashboard_layout = None
                            
                        st.session_state.dashboard_column_mapping = import_data["dashboard_column_mapping"]
                        
                        # Save to file
                        save_dashboard_settings()
                        st.success("Imported settings applied successfully!")
                        st.rerun()
                else:
                    st.error("Invalid settings file structure. Please upload a valid dashboard settings file.")
            except Exception as e:
                st.error(f"Error importing settings: {str(e)}")

    # Load overview data if not already loaded
    if "overview_data" not in st.session_state:
        with st.spinner("Loading overview data..."):
            st.session_state.overview_data = load_sheet_data("Overview")
    
    # Load runsheet data if not already loaded
    if "runsheet_data" not in st.session_state:
        with st.spinner("Loading runsheet data..."):
            st.session_state.runsheet_data = load_sheet_data("Runsheet")
    
    # Load dashboard metrics if not already processed
    if "dashboard_metrics" not in st.session_state:
        with st.spinner("Processing dashboard metrics..."):
            st.session_state.dashboard_metrics = load_and_process_dashboard_metrics()
    
    # Display column mapping in edit mode
    if st.session_state.edit_mode:
        st.subheader("Dashboard Configuration")
        
        # Create tabs for different configuration categories
        conf_tab1, conf_tab2, conf_tab3 = st.tabs(["Data Mapping", "Chart Configuration", "Card Titles"])
        
        with conf_tab1:
            # Get available columns from overview data
            overview_data = st.session_state.get("overview_data")
            overview_columns = []
            if overview_data is not None and not overview_data.empty:
                overview_columns = list(overview_data.columns)
            
            # Create a form for column mapping
            with st.form(key="column_mapping_form"):
                st.subheader("Metric Data Mapping")
                st.caption("Map dashboard metrics to spreadsheet columns")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.session_state.dashboard_column_mapping["approve_hours"] = st.selectbox(
                        "Column for 'Average Hours to Approve Order':",
                        options=[""] + overview_columns,
                        index=0 if st.session_state.dashboard_column_mapping["approve_hours"] == "" 
                               else overview_columns.index(st.session_state.dashboard_column_mapping["approve_hours"]) + 1 
                               if st.session_state.dashboard_column_mapping["approve_hours"] in overview_columns else 0
                    )
                    
                    st.session_state.dashboard_column_mapping["avg_days_deliver"] = st.selectbox(
                        "Column for 'Average Days to Deliver':",
                        options=[""] + overview_columns,
                        index=0 if st.session_state.dashboard_column_mapping["avg_days_deliver"] == "" 
                               else overview_columns.index(st.session_state.dashboard_column_mapping["avg_days_deliver"]) + 1
                               if st.session_state.dashboard_column_mapping["avg_days_deliver"] in overview_columns else 0
                    )
                    
                    st.session_state.dashboard_column_mapping["avg_days_pending"] = st.selectbox(
                        "Column for 'Average Days Pending':",
                        options=[""] + overview_columns,
                        index=0 if st.session_state.dashboard_column_mapping["avg_days_pending"] == "" 
                               else overview_columns.index(st.session_state.dashboard_column_mapping["avg_days_pending"]) + 1
                               if st.session_state.dashboard_column_mapping["avg_days_pending"] in overview_columns else 0
                    )
                    
                    st.session_state.dashboard_column_mapping["logistics_48hrs"] = st.selectbox(
                        "Column for 'Logistics 48Hrs Delivery Rate':",
                        options=[""] + overview_columns,
                        index=0 if st.session_state.dashboard_column_mapping["logistics_48hrs"] == "" 
                               else overview_columns.index(st.session_state.dashboard_column_mapping["logistics_48hrs"]) + 1
                               if st.session_state.dashboard_column_mapping["logistics_48hrs"] in overview_columns else 0
                    )
                
                with col2:
                    st.session_state.dashboard_column_mapping["logistics_24hrs"] = st.selectbox(
                        "Column for 'Logistics 24Hrs Delivery Rate':",
                        options=[""] + overview_columns,
                        index=0 if st.session_state.dashboard_column_mapping["logistics_24hrs"] == "" 
                               else overview_columns.index(st.session_state.dashboard_column_mapping["logistics_24hrs"]) + 1
                               if st.session_state.dashboard_column_mapping["logistics_24hrs"] in overview_columns else 0
                    )
                    
                    st.session_state.dashboard_column_mapping["cartona_48hrs"] = st.selectbox(
                        "Column for 'Cartona 48Hrs Delivery Rate':",
                        options=[""] + overview_columns,
                        index=0 if st.session_state.dashboard_column_mapping["cartona_48hrs"] == "" 
                               else overview_columns.index(st.session_state.dashboard_column_mapping["cartona_48hrs"]) + 1
                               if st.session_state.dashboard_column_mapping["cartona_48hrs"] in overview_columns else 0
                    )
                    
                    st.session_state.dashboard_column_mapping["cartona_24hrs"] = st.selectbox(
                        "Column for 'Cartona 24Hrs Delivery Rate':",
                        options=[""] + overview_columns,
                        index=0 if st.session_state.dashboard_column_mapping["cartona_24hrs"] == "" 
                               else overview_columns.index(st.session_state.dashboard_column_mapping["cartona_24hrs"]) + 1
                               if st.session_state.dashboard_column_mapping["cartona_24hrs"] in overview_columns else 0
                    )
                
                submitted = st.form_submit_button("Apply Data Mapping")
                if submitted:
                    # Refresh dashboard metrics with the new mapping
                    if "dashboard_metrics" in st.session_state:
                        del st.session_state.dashboard_metrics
                    st.session_state.dashboard_metrics = load_and_process_dashboard_metrics()
                    # Save settings after column mapping is applied
                    save_dashboard_settings()
                    st.success("Data mapping applied!")
        
        with conf_tab2:
            # Configure chart settings
            st.subheader("Chart Configuration")
            st.caption("Customize the charts in your dashboard")
            
            # Get runsheet data columns for chart data options
            runsheet_data = st.session_state.get("runsheet_data")
            runsheet_columns = []
            if runsheet_data is not None and not runsheet_data.empty:
                runsheet_columns = list(runsheet_data.columns)
            
            with st.form(key="chart_config_form"):
                # Supplier GMV Share chart
                st.subheader("Supplier GMV Share Chart")
                
                # Chart title
                st.session_state.dashboard_column_mapping["chart_settings"]["supplier_gmv_share"]["title"] = st.text_input(
                    "Chart Title",
                    value=st.session_state.dashboard_column_mapping["chart_settings"]["supplier_gmv_share"]["title"],
                    key="supplier_gmv_title"
                )
                
                # Chart type
                st.session_state.dashboard_column_mapping["chart_settings"]["supplier_gmv_share"]["chart_type"] = st.selectbox(
                    "Chart Type",
                    options=["pie", "bar", "line"],
                    index=["pie", "bar", "line"].index(st.session_state.dashboard_column_mapping["chart_settings"]["supplier_gmv_share"]["chart_type"]),
                    key="supplier_gmv_chart_type"
                )
                
                # Data column
                st.session_state.dashboard_column_mapping["chart_settings"]["supplier_gmv_share"]["data_column"] = st.selectbox(
                    "Data Column",
                    options=[""] + runsheet_columns,
                    index=0 if st.session_state.dashboard_column_mapping["chart_settings"]["supplier_gmv_share"]["data_column"] == ""
                          else runsheet_columns.index(st.session_state.dashboard_column_mapping["chart_settings"]["supplier_gmv_share"]["data_column"]) + 1
                          if st.session_state.dashboard_column_mapping["chart_settings"]["supplier_gmv_share"]["data_column"] in runsheet_columns else 0,
                    key="supplier_gmv_data_column"
                )
                
                st.divider()
                
                # Task Count by Runsheet chart
                st.subheader("Task Count Chart")
                
                # Chart title
                st.session_state.dashboard_column_mapping["chart_settings"]["task_count_runsheet"]["title"] = st.text_input(
                    "Chart Title",
                    value=st.session_state.dashboard_column_mapping["chart_settings"]["task_count_runsheet"]["title"],
                    key="task_count_title"
                )
                
                # Chart type
                st.session_state.dashboard_column_mapping["chart_settings"]["task_count_runsheet"]["chart_type"] = st.selectbox(
                    "Chart Type",
                    options=["bar", "pie", "line"],
                    index=["bar", "pie", "line"].index(st.session_state.dashboard_column_mapping["chart_settings"]["task_count_runsheet"]["chart_type"]),
                    key="task_count_chart_type"
                )
                
                # Data column
                st.session_state.dashboard_column_mapping["chart_settings"]["task_count_runsheet"]["data_column"] = st.selectbox(
                    "Data Column",
                    options=[""] + runsheet_columns,
                    index=0 if st.session_state.dashboard_column_mapping["chart_settings"]["task_count_runsheet"]["data_column"] == ""
                          else runsheet_columns.index(st.session_state.dashboard_column_mapping["chart_settings"]["task_count_runsheet"]["data_column"]) + 1
                          if st.session_state.dashboard_column_mapping["chart_settings"]["task_count_runsheet"]["data_column"] in runsheet_columns else 0,
                    key="task_count_data_column"
                )
                
                chart_submitted = st.form_submit_button("Apply Chart Settings")
                if chart_submitted:
                    save_dashboard_settings()
                    st.success("Chart settings applied!")
        
        with conf_tab3:
            # Edit card titles
            st.subheader("Card Titles")
            st.caption("Customize the titles of cards in your dashboard")
            
            with st.form(key="card_titles_form"):
                for card_id, default_title in [
                    ("avg_hours_approve", "Average Hours to Approve Order"),
                    ("avg_days_deliver", "Average Days to Deliver"),
                    ("avg_days_pending", "Average Days Pending"),
                    ("fast_delivery_logistics", "Fast Delivery Rate - Logistics"),
                    ("fast_delivery_cartona", "Fast Delivery Rate - Cartona")
                ]:
                    current_title = st.session_state.dashboard_column_mapping["card_titles"].get(card_id, default_title)
                    st.session_state.dashboard_column_mapping["card_titles"][card_id] = st.text_input(
                        f"Title for {default_title}",
                        value=current_title,
                        key=f"title_{card_id}"
                    )
                
                titles_submitted = st.form_submit_button("Apply Title Changes")
                if titles_submitted:
                    save_dashboard_settings()
                    st.success("Card titles updated!")
    
    # Main dashboard using streamlit elements
    with elements("dashboard"):
        # Define layout with adjusted sizes
        layout = [
            dashboard.Item("avg_hours_approve", 0, 0, 4, 2),
            dashboard.Item("avg_days_deliver", 4, 0, 4, 2),
            dashboard.Item("avg_days_pending", 8, 0, 4, 2),
            dashboard.Item("fast_delivery_logistics", 0, 2, 6, 2),
            dashboard.Item("fast_delivery_cartona", 6, 2, 6, 2),
            dashboard.Item("supplier_gmv_share", 0, 4, 6, 2),
            dashboard.Item("task_count_runsheet", 0, 6, 6, 2),
        ]

        # If we have a saved layout, use it
        current_dashboard_layout = st.session_state.get("dashboard_layout")
        if current_dashboard_layout is not None:
            layout = current_dashboard_layout

        # Function to handle layout changes
        def handle_layout_change(updated_layout):
            if st.session_state.edit_mode:
                st.session_state.dashboard_layout = updated_layout
                # Avoid saving on every small change as it could affect performance
                # The save will happen when exiting edit mode or clicking Save Layout

        # Use the draggable and resizable properties based on edit mode
        with dashboard.Grid(
            layout,
            draggableHandle=".draggable" if st.session_state.edit_mode else None,
            draggable=st.session_state.edit_mode,
            resizable=st.session_state.edit_mode,
            onLayoutChange=handle_layout_change
        ):
            # Average Hours to Approve Order
            with mui.Paper(key="avg_hours_approve", elevation=2, className="draggable", sx={
                "p": "16px 20px", 
                "borderRadius": 2, 
                "background": "linear-gradient(135deg, #1e40af 0%, #3b82f6 100%)",
                "color": "white", 
                "height": "100%", 
                "display": "flex", 
                "flexDirection": "column", 
                "justifyContent": "space-between"
            }):
                # Use custom title if available
                custom_title = st.session_state.dashboard_column_mapping["card_titles"].get("avg_hours_approve", "Average Hours to Approve Order")
                mui.Typography(custom_title, className="metric-label")
                
                # Get data from processed metrics
                metrics = st.session_state.get("dashboard_metrics", {})
                approve_metrics = metrics.get("avg_hours_approve", {"value": "12.5", "change": "+8.3%"})
                
                # Display the value
                mui.Typography(str(approve_metrics["value"]), className="metric-value", variant="h3")
                with mui.Box(sx={"display": "flex", "alignItems": "center", "gap": 1}):
                    mui.Typography(approve_metrics["change"], className="positive-change")
                    mui.Typography("vs last month", sx={"opacity": 0.7, "fontSize": "0.875rem", "color": "white"})
            
            # Average Days to Deliver
            with mui.Paper(key="avg_days_deliver", elevation=2, className="draggable", sx={
                "p": "16px 20px", 
                "borderRadius": 2, 
                "background": "linear-gradient(135deg, #065f46 0%, #10b981 100%)",
                "color": "white", 
                "height": "100%", 
                "display": "flex", 
                "flexDirection": "column", 
                "justifyContent": "space-between"
            }):
                # Use custom title if available
                custom_title = st.session_state.dashboard_column_mapping["card_titles"].get("avg_days_deliver", "Average Days to Deliver")
                mui.Typography(custom_title, className="metric-label")
                
                # Get data from processed metrics
                metrics = st.session_state.get("dashboard_metrics", {})
                deliver_metrics = metrics.get("avg_days_deliver", {"value": "3.2", "change": "-5.2%"})
                
                # Display the value
                mui.Typography(str(deliver_metrics["value"]), className="metric-value", variant="h3")
                with mui.Box(sx={"display": "flex", "alignItems": "center", "gap": 1}):
                    mui.Typography(deliver_metrics["change"], className="positive-change")
                    mui.Typography("vs last month", sx={"opacity": 0.7, "fontSize": "0.875rem", "color": "white"})
            
            # Average Days Pending
            with mui.Paper(key="avg_days_pending", elevation=2, className="draggable", sx={
                "p": "16px 20px", 
                "borderRadius": 2, 
                "background": "linear-gradient(135deg, #7c2d12 0%, #ea580c 100%)",
                "color": "white", 
                "height": "100%", 
                "display": "flex", 
                "flexDirection": "column", 
                "justifyContent": "space-between"
            }):
                # Use custom title if available
                custom_title = st.session_state.dashboard_column_mapping["card_titles"].get("avg_days_pending", "Average Days Pending")
                mui.Typography(custom_title, className="metric-label")
                
                # Get data from processed metrics
                metrics = st.session_state.get("dashboard_metrics", {})
                pending_metrics = metrics.get("avg_days_pending", {"value": "1.5", "change": "+2.1%"})
                
                # Display the value
                mui.Typography(str(pending_metrics["value"]), className="metric-value", variant="h3")
                with mui.Box(sx={"display": "flex", "alignItems": "center", "gap": 1}):
                    mui.Typography(pending_metrics["change"], className="positive-change")
                    mui.Typography("vs last month", sx={"opacity": 0.7, "fontSize": "0.875rem", "color": "white"})

            # Fast Delivery Rate - Logistics
            with mui.Paper(key="fast_delivery_logistics", elevation=2, className="draggable", sx={
                "p": "16px 20px", 
                "borderRadius": 2, 
                "background": "linear-gradient(135deg, #be123c 0%, #f43f5e 100%)",
                "color": "white", 
                "height": "100%", 
                "display": "flex", 
                "flexDirection": "column", 
                "justifyContent": "space-between"
            }):
                # Use custom title if available
                custom_title = st.session_state.dashboard_column_mapping["card_titles"].get("fast_delivery_logistics", "Fast Delivery Rate - Logistics")
                mui.Typography(custom_title, className="metric-label")
                
                # Get data from processed metrics
                metrics = st.session_state.get("dashboard_metrics", {})
                logistics_metrics = metrics.get("logistics_delivery", {
                    "48hrs": "76.49786020", 
                    "24hrs": "32.45363766", 
                    "change": "+3.8%"
                })
                
                # Display the value
                mui.Typography(f"48Hrs: {logistics_metrics['48hrs']} | 24Hrs: {logistics_metrics['24hrs']}", 
                               className="metric-value", variant="h4")
                with mui.Box(sx={"display": "flex", "alignItems": "center", "gap": 1}):
                    mui.Typography(logistics_metrics["change"], className="positive-change")
                    mui.Typography("vs last month", sx={"opacity": 0.7, "fontSize": "0.875rem", "color": "white"})

            # Fast Delivery Rate - Cartona
            with mui.Paper(key="fast_delivery_cartona", elevation=2, className="draggable", sx={
                "p": "16px 20px", 
                "borderRadius": 2, 
                "background": "linear-gradient(135deg, #0f766e 0%, #14b8a6 100%)",
                "color": "white", 
                "height": "100%", 
                "display": "flex", 
                "flexDirection": "column", 
                "justifyContent": "space-between"
            }):
                # Use custom title if available
                custom_title = st.session_state.dashboard_column_mapping["card_titles"].get("fast_delivery_cartona", "Fast Delivery Rate - Cartona")
                mui.Typography(custom_title, className="metric-label")
                
                # Get data from processed metrics
                metrics = st.session_state.get("dashboard_metrics", {})
                cartona_metrics = metrics.get("cartona_delivery", {
                    "48hrs": "70.36376605", 
                    "24hrs": "25.99857347", 
                    "change": "+2.1%"
                })
                
                # Display the value
                mui.Typography(f"48Hrs: {cartona_metrics['48hrs']} | 24Hrs: {cartona_metrics['24hrs']}", 
                               className="metric-value", variant="h4")
                with mui.Box(sx={"display": "flex", "alignItems": "center", "gap": 1}):
                    mui.Typography(cartona_metrics["change"], className="positive-change")
                    mui.Typography("vs last month", sx={"opacity": 0.7, "fontSize": "0.875rem", "color": "white"})

            # Supplier GMV Share
            with mui.Paper(key="supplier_gmv_share", elevation=2, className="draggable", sx={
                "p": "16px 20px", 
                "borderRadius": 2,
                "height": "100%"
            }):
                # Get chart settings
                chart_settings = st.session_state.dashboard_column_mapping["chart_settings"]["supplier_gmv_share"]
                chart_title = chart_settings.get("title", "Supplier GMV Share")
                chart_type = chart_settings.get("chart_type", "pie")
                data_column = chart_settings.get("data_column", "")
                
                # Display title with draggable class
                mui.Typography(chart_title, variant="h6", className="draggable")
                
                # Process and display chart data if available
                runsheet_data = st.session_state.get("runsheet_data")
                if runsheet_data is not None and not runsheet_data.empty and data_column and data_column in runsheet_data.columns:
                    # Process data for the selected chart type
                    try:
                        if chart_type == "pie":
                            # For pie chart, aggregate data by the selected column
                            chart_data = runsheet_data[data_column].value_counts().reset_index()
                            chart_data.columns = ['id', 'value']
                            
                            # Create a pie chart with nivo
                            with mui.Box(sx={"height": "calc(100% - 40px)"}):
                                nivo.Pie(
                                    data=chart_data.to_dict('records'),
                                    margin={"top": 40, "right": 80, "bottom": 80, "left": 80},
                                    innerRadius=0.5,
                                    padAngle=0.7,
                                    cornerRadius=3,
                                    activeOuterRadiusOffset=8,
                                    borderWidth=1,
                                    borderColor={"from": "color", "modifiers": [["darker", 0.2]]},
                                    arcLinkLabelsSkipAngle=10,
                                    arcLinkLabelsTextColor="#333333",
                                    arcLinkLabelsThickness=2,
                                    arcLinkLabelsColor={"from": "color"},
                                    arcLabelsSkipAngle=10,
                                    arcLabelsTextColor={"from": "color", "modifiers": [["darker", 2]]},
                                    legends=[
                                        {
                                            "anchor": "bottom",
                                            "direction": "row",
                                            "justify": False,
                                            "translateX": 0,
                                            "translateY": 56,
                                            "itemsSpacing": 0,
                                            "itemWidth": 100,
                                            "itemHeight": 18,
                                            "itemTextColor": "#999",
                                            "itemDirection": "left-to-right",
                                            "itemOpacity": 1,
                                            "symbolSize": 18,
                                            "symbolShape": "circle"
                                        }
                                    ]
                                )
                        elif chart_type == "bar":
                            # For bar chart, aggregate data by the selected column
                            chart_data = runsheet_data[data_column].value_counts().reset_index()
                            chart_data.columns = ['category', 'value']
                            chart_data_list = [{"category": row["category"], "value": row["value"]} for _, row in chart_data.iterrows()]
                            
                            # Create a bar chart with nivo
                            with mui.Box(sx={"height": "calc(100% - 40px)"}):
                                nivo.Bar(
                                    data=chart_data_list,
                                    keys=["value"],
                                    indexBy="category",
                                    margin={"top": 50, "right": 50, "bottom": 50, "left": 60},
                                    padding=0.3,
                                    valueScale={"type": "linear"},
                                    indexScale={"type": "band", "round": True},
                                    colors={"scheme": "nivo"},
                                    axisTop=None,
                                    axisRight=None,
                                    axisBottom={
                                        "tickSize": 5,
                                        "tickPadding": 5,
                                        "tickRotation": 45,
                                        "legendPosition": "middle",
                                        "legendOffset": 32
                                    },
                                    axisLeft={
                                        "tickSize": 5,
                                        "tickPadding": 5,
                                        "tickRotation": 0,
                                        "legendPosition": "middle",
                                        "legendOffset": -40
                                    },
                                    labelSkipWidth=12,
                                    labelSkipHeight=12,
                                    labelTextColor={"from": "color", "modifiers": [["darker", 1.6]]},
                                    role="application"
                                )
                        elif chart_type == "line":
                            # For line chart, we need time-based data
                            # Try to find a date column for the x-axis
                            date_cols = [col for col in runsheet_data.columns if 'date' in col.lower()]
                            if date_cols:
                                date_col = date_cols[0]
                                # Group by date and count values in the selected column
                                runsheet_data[date_col] = pd.to_datetime(runsheet_data[date_col], errors='coerce')
                                runsheet_data['date_str'] = runsheet_data[date_col].dt.strftime('%Y-%m-%d')
                                chart_data = runsheet_data.groupby('date_str')[data_column].count().reset_index()
                                chart_data.columns = ['x', 'y']
                                
                                line_data = [{
                                    "id": data_column,
                                    "data": chart_data.to_dict('records')
                                }]
                                
                                # Create a line chart with nivo
                                with mui.Box(sx={"height": "calc(100% - 40px)"}):
                                    nivo.Line(
                                        data=line_data,
                                        margin={"top": 50, "right": 50, "bottom": 50, "left": 60},
                                        xScale={"type": "point"},
                                        yScale={"type": "linear", "min": 0, "max": "auto"},
                                        axisTop=None,
                                        axisRight=None,
                                        axisBottom={
                                            "tickSize": 5,
                                            "tickPadding": 5,
                                            "tickRotation": 45,
                                            "legendPosition": "middle",
                                            "legendOffset": 32
                                        },
                                        axisLeft={
                                            "tickSize": 5,
                                            "tickPadding": 5,
                                            "tickRotation": 0,
                                            "legendPosition": "middle",
                                            "legendOffset": -40
                                        },
                                        pointSize=10,
                                        pointColor={"theme": "background"},
                                        pointBorderWidth=2,
                                        pointBorderColor={"from": "serieColor"},
                                        pointLabelYOffset=-12,
                                        useMesh=True,
                                        legends=[
                                            {
                                                "anchor": "bottom-right",
                                                "direction": "column",
                                                "justify": False,
                                                "translateX": 100,
                                                "translateY": 0,
                                                "itemsSpacing": 0,
                                                "itemDirection": "left-to-right",
                                                "itemWidth": 80,
                                                "itemHeight": 20,
                                                "itemOpacity": 0.75,
                                                "symbolSize": 12,
                                                "symbolShape": "circle",
                                                "symbolBorderColor": "rgba(0, 0, 0, .5)"
                                            }
                                        ]
                                    )
                            else:
                                mui.Typography("No date column found for line chart", 
                                            sx={"color": "#666", "display": "flex", "justifyContent": "center", 
                                                "alignItems": "center", "height": "80%", "fontStyle": "italic"})
                    except Exception as e:
                        mui.Typography(f"Error creating chart: {str(e)}", 
                                    sx={"color": "#666", "display": "flex", "justifyContent": "center", 
                                        "alignItems": "center", "height": "80%", "fontStyle": "italic"})
                else:
                    # Display "No data available" message
                    mui.Typography("No data available. Select a data column in the Chart Configuration tab.", 
                                sx={"color": "#666", "display": "flex", "justifyContent": "center", 
                                    "alignItems": "center", "height": "80%", "fontStyle": "italic"})

            # Task Count by Runsheet
            with mui.Paper(key="task_count_runsheet", elevation=2, className="draggable", sx={
                "p": "16px 20px", 
                "borderRadius": 2,
                "height": "100%"
            }):
                # Get chart settings
                chart_settings = st.session_state.dashboard_column_mapping["chart_settings"]["task_count_runsheet"]
                chart_title = chart_settings.get("title", "Task Count by Runsheet")
                chart_type = chart_settings.get("chart_type", "bar")
                data_column = chart_settings.get("data_column", "")
                
                # Display title with draggable class
                mui.Typography(chart_title, variant="h6", className="draggable")
                
                # Process and display chart data if available
                runsheet_data = st.session_state.get("runsheet_data")
                if runsheet_data is not None and not runsheet_data.empty and data_column and data_column in runsheet_data.columns:
                    # Process data for the selected chart type
                    try:
                        if chart_type == "bar":
                            # For bar chart, aggregate data by the selected column
                            chart_data = runsheet_data[data_column].value_counts().reset_index()
                            chart_data.columns = ['category', 'value']
                            chart_data_list = [{"category": row["category"], "value": row["value"]} for _, row in chart_data.iterrows()]
                            
                            # Create a bar chart with nivo
                            with mui.Box(sx={"height": "calc(100% - 40px)"}):
                                nivo.Bar(
                                    data=chart_data_list,
                                    keys=["value"],
                                    indexBy="category",
                                    margin={"top": 50, "right": 50, "bottom": 50, "left": 60},
                                    padding=0.3,
                                    valueScale={"type": "linear"},
                                    indexScale={"type": "band", "round": True},
                                    colors={"scheme": "nivo"},
                                    axisTop=None,
                                    axisRight=None,
                                    axisBottom={
                                        "tickSize": 5,
                                        "tickPadding": 5,
                                        "tickRotation": 45,
                                        "legendPosition": "middle",
                                        "legendOffset": 32
                                    },
                                    axisLeft={
                                        "tickSize": 5,
                                        "tickPadding": 5,
                                        "tickRotation": 0,
                                        "legendPosition": "middle",
                                        "legendOffset": -40
                                    },
                                    labelSkipWidth=12,
                                    labelSkipHeight=12,
                                    labelTextColor={"from": "color", "modifiers": [["darker", 1.6]]},
                                    role="application"
                                )
                        elif chart_type == "pie":
                            # For pie chart, aggregate data by the selected column
                            chart_data = runsheet_data[data_column].value_counts().reset_index()
                            chart_data.columns = ['id', 'value']
                            
                            # Create a pie chart with nivo
                            with mui.Box(sx={"height": "calc(100% - 40px)"}):
                                nivo.Pie(
                                    data=chart_data.to_dict('records'),
                                    margin={"top": 40, "right": 80, "bottom": 80, "left": 80},
                                    innerRadius=0.5,
                                    padAngle=0.7,
                                    cornerRadius=3,
                                    activeOuterRadiusOffset=8,
                                    borderWidth=1,
                                    borderColor={"from": "color", "modifiers": [["darker", 0.2]]},
                                    arcLinkLabelsSkipAngle=10,
                                    arcLinkLabelsTextColor="#333333",
                                    arcLinkLabelsThickness=2,
                                    arcLinkLabelsColor={"from": "color"},
                                    arcLabelsSkipAngle=10,
                                    arcLabelsTextColor={"from": "color", "modifiers": [["darker", 2]]},
                                    legends=[
                                        {
                                            "anchor": "bottom",
                                            "direction": "row",
                                            "justify": False,
                                            "translateX": 0,
                                            "translateY": 56,
                                            "itemsSpacing": 0,
                                            "itemWidth": 100,
                                            "itemHeight": 18,
                                            "itemTextColor": "#999",
                                            "itemDirection": "left-to-right",
                                            "itemOpacity": 1,
                                            "symbolSize": 18,
                                            "symbolShape": "circle"
                                        }
                                    ]
                                )
                        elif chart_type == "line":
                            # For line chart, we need time-based data
                            # Try to find a date column for the x-axis
                            date_cols = [col for col in runsheet_data.columns if 'date' in col.lower()]
                            if date_cols:
                                date_col = date_cols[0]
                                # Group by date and count values in the selected column
                                runsheet_data[date_col] = pd.to_datetime(runsheet_data[date_col], errors='coerce')
                                runsheet_data['date_str'] = runsheet_data[date_col].dt.strftime('%Y-%m-%d')
                                chart_data = runsheet_data.groupby('date_str')[data_column].count().reset_index()
                                chart_data.columns = ['x', 'y']
                                
                                line_data = [{
                                    "id": data_column,
                                    "data": chart_data.to_dict('records')
                                }]
                                
                                # Create a line chart with nivo
                                with mui.Box(sx={"height": "calc(100% - 40px)"}):
                                    nivo.Line(
                                        data=line_data,
                                        margin={"top": 50, "right": 50, "bottom": 50, "left": 60},
                                        xScale={"type": "point"},
                                        yScale={"type": "linear", "min": 0, "max": "auto"},
                                        axisTop=None,
                                        axisRight=None,
                                        axisBottom={
                                            "tickSize": 5,
                                            "tickPadding": 5,
                                            "tickRotation": 45,
                                            "legendPosition": "middle",
                                            "legendOffset": 32
                                        },
                                        axisLeft={
                                            "tickSize": 5,
                                            "tickPadding": 5,
                                            "tickRotation": 0,
                                            "legendPosition": "middle",
                                            "legendOffset": -40
                                        },
                                        pointSize=10,
                                        pointColor={"theme": "background"},
                                        pointBorderWidth=2,
                                        pointBorderColor={"from": "serieColor"},
                                        pointLabelYOffset=-12,
                                        useMesh=True,
                                        legends=[
                                            {
                                                "anchor": "bottom-right",
                                                "direction": "column",
                                                "justify": False,
                                                "translateX": 100,
                                                "translateY": 0,
                                                "itemsSpacing": 0,
                                                "itemDirection": "left-to-right",
                                                "itemWidth": 80,
                                                "itemHeight": 20,
                                                "itemOpacity": 0.75,
                                                "symbolSize": 12,
                                                "symbolShape": "circle",
                                                "symbolBorderColor": "rgba(0, 0, 0, .5)"
                                            }
                                        ]
                                    )
                            else:
                                mui.Typography("No date column found for line chart", 
                                            sx={"color": "#666", "display": "flex", "justifyContent": "center", 
                                                "alignItems": "center", "height": "80%", "fontStyle": "italic"})
                    except Exception as e:
                        mui.Typography(f"Error creating chart: {str(e)}", 
                                    sx={"color": "#666", "display": "flex", "justifyContent": "center", 
                                        "alignItems": "center", "height": "80%", "fontStyle": "italic"})
                else:
                    # Display "No data available" message
                    mui.Typography("No data available. Select a data column in the Chart Configuration tab.", 
                                sx={"color": "#666", "display": "flex", "justifyContent": "center", 
                                    "alignItems": "center", "height": "80%", "fontStyle": "italic"})

with tab2:
    # Map View Tab
    st.header("Location Map")
    
    # Create columns for layout
    map_col, details_col = st.columns([7, 3])
    
    # Help info
    
    # Load the pending tasks data for the map
    pending_tasks = load_pending_tasks()
    
    # Initialize session state for map view
    if "selected_retailer" not in st.session_state:
        st.session_state.selected_retailer = None
    if "map_last_clicked" not in st.session_state:
        st.session_state.map_last_clicked = None
    if "show_routes" not in st.session_state:
        st.session_state.show_routes = False
    
    # Add a checkbox to toggle route display
    with map_col:
        show_routes = st.checkbox("Show delivery routes on map", value=st.session_state.show_routes)
        if show_routes != st.session_state.show_routes:
            st.session_state.show_routes = show_routes
            st.rerun()
        
        if pending_tasks is not None:
            # Check if required columns exist
            required_cols = ['customer_longitude', 'customer_latitude', 'customer_area', 'supplier_name', 'retailer_name']
            missing_cols = [col for col in required_cols if col not in pending_tasks.columns]
            
            if missing_cols:
                st.error(f"Missing required columns for map: {', '.join(missing_cols)}")
            else:
                # Handle empty values
                pending_tasks['customer_latitude'] = pending_tasks['customer_latitude'].replace('', None)
                pending_tasks['customer_longitude'] = pending_tasks['customer_longitude'].replace('', None)
                
                # Also handle supplier coordinates for routes
                if st.session_state.show_routes:
                    pending_tasks['supplier_latitude'] = pending_tasks['supplier_latitude'].replace('', None)
                    pending_tasks['supplier_longitude'] = pending_tasks['supplier_longitude'].replace('', None)
                
                # Validate coordinates
                valid_tasks = validate_coordinates(pending_tasks, 'customer_latitude', 'customer_longitude')
                
                # Calculate average coordinates
                try:
                    avg_lat = pd.to_numeric(valid_tasks['customer_latitude'], errors='coerce').mean()
                    avg_lng = pd.to_numeric(valid_tasks['customer_longitude'], errors='coerce').mean()
                    if pd.isna(avg_lat) or pd.isna(avg_lng):
                        avg_lat, avg_lng = 30.0444, 31.2357  # Cairo
                except:
                    avg_lat, avg_lng = 30.0444, 31.2357  # Cairo
                
                # Create the map
                m = folium.Map(location=[avg_lat, avg_lng], zoom_start=12)
                
                # Add customer markers - simplified with just tooltips, no popups
                for idx, row in valid_tasks.iterrows():
                    try:
                        # Skip empty values
                        if pd.isna(row['customer_latitude']) or pd.isna(row['customer_longitude']):
                            continue
                            
                        cust_lat = float(row['customer_latitude'])
                        cust_lng = float(row['customer_longitude'])
                        
                        # Add marker with only a tooltip, no popup
                        folium.Marker(
                            location=[cust_lat, cust_lng],
                            tooltip=row['retailer_name'],
                            icon=folium.Icon(color='red', icon='home')
                        ).add_to(m)
                        
                        # If showing routes and we have supplier coordinates, add supplier markers and routes
                        if st.session_state.show_routes and 'supplier_latitude' in row and 'supplier_longitude' in row:
                            try:
                                if not pd.isna(row['supplier_latitude']) and not pd.isna(row['supplier_longitude']):
                                    supp_lat = float(row['supplier_latitude'])
                                    supp_lng = float(row['supplier_longitude'])
                                    
                                    # Add supplier marker
                                    folium.Marker(
                                        location=[supp_lat, supp_lng],
                                        tooltip=f"Supplier: {row['supplier_name']}",
                                        icon=folium.Icon(color='blue', icon='industry')
                                    ).add_to(m)
                                    
                                    # Get real route if possible
                                    route_geojson = get_route(
                                        (supp_lng, supp_lat),  # OpenRouteService uses (lon, lat) order
                                        (cust_lng, cust_lat)
                                    )
                                    
                                    if route_geojson and 'features' in route_geojson and len(route_geojson['features']) > 0:
                                        # Add real route as GeoJSON
                                        style = {
                                            'color': '#22c55e',
                                            'weight': 3,
                                            'opacity': 0.7
                                        }
                                        
                                        # Add GeoJSON route to map
                                        folium.GeoJson(
                                            route_geojson,
                                            style_function=lambda x: style,
                                            tooltip=f"Route to {row['retailer_name']}"
                                        ).add_to(m)
                                    else:
                                        # Fallback to a simple line if route not available
                                        folium.PolyLine(
                                            locations=[[supp_lat, supp_lng], [cust_lat, cust_lng]],
                                            color='#22c55e',
                                            weight=2,
                                            opacity=0.7,
                                            dash_array='10,10',
                                            tooltip=f"Straight line to {row['retailer_name']}"
                                        ).add_to(m)
                            except Exception as route_error:
                                print(f"Error adding route: {route_error}")
                    except:
                        continue
                
                # Display the map
                output = st_folium(m, width="100%", height=600, key="location_map")
                
                # Handle the selection from the map - directly show details when clicked
                if output and "last_object_clicked" in output:
                    clicked = output["last_object_clicked"]
                    
                    # Check if this is a new click
                    current_click = f"{clicked}" if clicked else None
                    if (clicked and "lat" in clicked and "lng" in clicked and 
                        current_click != st.session_state.map_last_clicked):
                        
                        # Store the current click to avoid processing it again
                        st.session_state.map_last_clicked = current_click
                        
                        # Find the closest retailer
                        clicked_lat = clicked["lat"]
                        clicked_lng = clicked["lng"]
                        
                        min_dist = float('inf')
                        closest_idx = None
                        
                        for idx, row in pending_tasks.iterrows():
                            try:
                                if row['customer_latitude'] == '' or row['customer_longitude'] == '':
                                    continue
                                
                                r_lat = float(row['customer_latitude'])
                                r_lng = float(row['customer_longitude'])
                                
                                # Calculate distance
                                dist = ((r_lat - clicked_lat)**2 + (r_lng - clicked_lng)**2)**0.5
                                
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_idx = idx
                            except:
                                continue
                        
                        if closest_idx is not None and min_dist < 0.01:
                            # Directly update the selected retailer
                            st.session_state.selected_retailer = pending_tasks.loc[closest_idx]

        else:
            st.error("Unable to load pending tasks data for the map")
    
    # Show retailer details if selected
    if st.session_state.selected_retailer is not None:
        retailer = st.session_state.selected_retailer
        
        # Display retailer info
        with details_col:
            st.subheader(retailer.get('retailer_name', 'Retailer'))
            st.write(f"**Area:** {retailer.get('customer_area', 'N/A')}")
            st.write(f"**Supplier:** {retailer.get('supplier_name', 'N/A')}")
            st.write(f"**supplier_lat:** {retailer.get('supplier_latitude', 'N/A')}")
            st.write(f"**supplier_lng:** {retailer.get('supplier_longitude', 'N/A')}")
            st.write(f"**customer_lat:** {retailer.get('customer_latitude', 'N/A')}")
            st.write(f"**customer_lng:** {retailer.get('customer_longitude', 'N/A')}")
            # Display more fields if available
            additional_fields = ['order_id', 'created_at', 'delivery_date']
            for field in additional_fields:
                if field in retailer and retailer[field]:
                    st.write(f"**{field.replace('_', ' ').title()}:** {retailer[field]}")
            
            # Show a small map with route if supplier coordinates exist
            if ('supplier_latitude' in retailer and retailer['supplier_latitude'] and 
                'supplier_longitude' in retailer and retailer['supplier_longitude']):
                try:
                    cust_lat = float(retailer['customer_latitude'])
                    cust_lng = float(retailer['customer_longitude'])
                    supp_lat = float(retailer['supplier_latitude'])
                    supp_lng = float(retailer['supplier_longitude'])
                    
                    st.subheader("Delivery Route")
                    
                    # Get actual road route using OpenRouteService
                    actual_route = get_route(
                        (supp_lng, supp_lat),  # OpenRouteService uses (lon, lat) order
                        (cust_lng, cust_lat)
                    )
                    
                    # For Leaflet we need a different format - convert if route was successfully retrieved
                    route_coords_js = None
                    if actual_route and 'features' in actual_route and len(actual_route['features']) > 0:
                        coords = actual_route['features'][0]['geometry']['coordinates']
                        # Convert to JavaScript string for the Leaflet map
                        # Leaflet uses [lat, lng] order, OpenRouteService uses [lng, lat]
                        route_coords_js = str([[coord[1], coord[0]] for coord in coords]).replace("'", "")
                    
                    # Create a map using HTML/JavaScript directly - more reliable than folium in some cases
                    map_html = f"""
                    <div id="map" style="height:300px;width:100%;border-radius:10px;margin-bottom:15px;"></div>
                    <script>
                        // Initialize the map
                        var map = L.map('map').setView([{(cust_lat + supp_lat) / 2}, {(cust_lng + supp_lng) / 2}], 10);
                        
                        // Add tile layer (OpenStreetMap)
                        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                        }}).addTo(map);
                        
                        // Add customer marker
                        var customerMarker = L.marker([{cust_lat}, {cust_lng}], {{
                            icon: L.divIcon({{
                                className: 'custom-div-icon',
                                html: '<div style="background-color: #c30b82; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; justify-content: center; align-items: center;">C</div>',
                                iconSize: [30, 30],
                                iconAnchor: [15, 15]
                            }})
                        }}).addTo(map);
                        customerMarker.bindTooltip("Customer: {retailer['retailer_name']}");
                        
                        // Add supplier marker
                        var supplierMarker = L.marker([{supp_lat}, {supp_lng}], {{
                            icon: L.divIcon({{
                                className: 'custom-div-icon',
                                html: '<div style="background-color: #0b77c3; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; justify-content: center; align-items: center;">S</div>',
                                iconSize: [30, 30],
                                iconAnchor: [15, 15]
                            }})
                        }}).addTo(map);
                        supplierMarker.bindTooltip("Supplier: {retailer['supplier_name']}");
                        
                        // Add route line - either use real route or fallback to straight line
                        {f"var routeCoords = {route_coords_js};" if route_coords_js else ""}
                        
                        {'var routeLine = L.polyline(routeCoords, {' if route_coords_js else 'var routeLine = L.polyline([[' + str(cust_lat) + ', ' + str(cust_lng) + '], [' + str(supp_lat) + ', ' + str(supp_lng) + ']], {'}
                            color: '#22c55e',
                            weight: 5,
                            opacity: 0.7,
                            {'' if route_coords_js else 'dashArray: "10, 10",'}
                            lineJoin: 'round'
                        }}).addTo(map);
                        
                        // Fit bounds to show the entire route
                        {f"map.fitBounds(routeLine.getBounds(), {{ padding: [50, 50] }});" if route_coords_js else "var bounds = [[" + str(cust_lat) + ", " + str(cust_lng) + "], [" + str(supp_lat) + ", " + str(supp_lng) + "]]; map.fitBounds(bounds, { padding: [50, 50] });"}
                    </script>
                    """
                    
                    # Need to include Leaflet CSS and JS
                    leaflet_css = '<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.3/dist/leaflet.css" integrity="sha256-kLaT2GOSpHechhsozzB+flnD+zUyjE2LlfWPgU04xyI=" crossorigin=""/>'
                    leaflet_js = '<script src="https://unpkg.com/leaflet@1.9.3/dist/leaflet.js" integrity="sha256-WBkoXOwTeyKclOHuWtc+i2uENFpDZ9YPdf5Hf+D7ewM=" crossorigin=""></script>'
                    
                    # Display the map using HTML component
                    st.components.v1.html(leaflet_css + leaflet_js + map_html, height=320)
                    
                    # Display distance information
                    if actual_route and 'features' in actual_route and len(actual_route['features']) > 0:
                        road_distance = actual_route['features'][0]['properties']['summary']['distance'] / 1000  # Convert to km
                        road_duration = actual_route['features'][0]['properties']['summary']['duration'] / 60  # Convert to minutes
                        st.info(f"Road distance: {road_distance:.2f} km (approx. {road_duration:.0f} min driving time)")
                    else:
                        # Fallback to straight-line distance
                        distance = haversine(cust_lat, cust_lng, supp_lat, supp_lng)
                        st.info(f"Straight-line distance: {distance:.2f} km (Note: Road route unavailable)")
                    
                except Exception as e:
                    st.error(f"Could not display route map: {str(e)}")
                    st.code(traceback.format_exc())
            
            # Clear selection button
            if st.button("Clear Selection", key="clear_retailer_selection"):
                st.session_state.selected_retailer = None
                st.session_state.map_last_clicked = None

# Add the counter example to the fourth tab
with tab3:
    st.header("Real-time Sheet Data")
    # Sheet selection
    sheet_name_tab4 = st.selectbox(
        "Select Sheet to View", 
        ["Active Suppliers Products", "Runsheet", "Tasks"],
        key="sheet_selector_tab4"
    )
    
    # Load the selected sheet with a spinner
    with st.spinner(f"Loading {sheet_name_tab4} data...", show_time=True):
        sheet_data_tab4 = load_sheet_data(sheet_name_tab4)
    
    if sheet_data_tab4 is not None:
        # Display the data
        st.dataframe(sheet_data_tab4, use_container_width=True, height=500)
        
        # Download button
        csv = sheet_data_tab4.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Data",
            csv,
            f"{sheet_name_tab4}.csv",
            "text/csv",
            key=f"download_tab4_{sheet_name_tab4}"
        )
    else:
        st.error(f"Unable to load {sheet_name_tab4} data")

# Add the new Trip Assignment tab
with tab4:
    try:
        # Import and reload the trip assignment module to ensure latest changes
        import importlib
        # Explicitly reload the module to get the latest changes
        importlib.reload(trip_assignment)
        # Run the trip assignment UI
        trip_assignment.trip_assignment_ui()
    except Exception as e:
        st.error(f"Error loading Trip Assignment module: {str(e)}")
        st.code(traceback.format_exc())
        st.info("Please make sure the trip_assignment.py file is in the same directory as KK.py")
        
        # Provide more detailed error information and potential fixes
        error_message = str(e).lower()
        if "convert" in error_message and "float" in error_message:
            st.warning("This appears to be a type conversion error. Check that all latitude/longitude values in your data are valid numbers.")
            st.info("Tip: Make sure to use safe_float() or explicit float() conversion for all coordinate values.")
        elif "import" in error_message:
            st.warning("Module import error. Make sure trip_assignment.py exists in the same directory.")
        
        # Show a button to check if trip_assignment exists
        if st.button("Check Trip Assignment Module"):
            try:
                import os
                if os.path.exists("trip_assignment.py"):
                    st.success("trip_assignment.py file exists.")
                    # Check the file size to make sure it's not empty
                    file_size = os.path.getsize("trip_assignment.py")
                    st.info(f"File size: {file_size} bytes")
                else:
                    st.error("trip_assignment.py file does not exist in the current directory.")
            except Exception as check_error:
                st.error(f"Error checking for module: {str(check_error)}")

# Add this at the end of your file, after the Trip Assignment tab code
with tab5:
    # --- CBM & Weight Calculator Tab ---
    st.header("CBM & Weight Calculator")
    
    # Helper function for CBM & Weight calculation
    def get_cbm_weight_appscript_style(row, fallback_df):
        """
        Calculate CBM and weight for a product based on fallback data, following the AppScript logic.
        Returns (confidence, cbm, weight)
        """
        try:
            # Convert measure and unit count to numeric values
            measure_value = pd.to_numeric(row.get('measurement_value',0), errors='coerce')
            unit_count_value = pd.to_numeric(row.get('unit_count', 0), errors='coerce')
            
            # 1. Exact match: brand, category, measure, unit_count
            exact = fallback_df[
                (fallback_df['BRAND_NAME'] == row['brand_name']) &
                (fallback_df['CATEGORY'] == row['category']) &
                (pd.to_numeric(fallback_df['measure'], errors='coerce') == measure_value) &
                (pd.to_numeric(fallback_df['unit count'], errors='coerce') == unit_count_value)
            ]
            if not exact.empty:
                # Handle comma as decimal separator
                cbm_str = str(exact['CBM'].iloc[0]).replace(',', '.')
                weight_str = str(exact['Weight'].iloc[0]).replace(',', '.')
                cbm = pd.to_numeric(cbm_str, errors='coerce')
                weight = pd.to_numeric(weight_str, errors='coerce')
                return 100, cbm, weight

            # 2. Category+measure+unit_count (ignore brand)
            cat_match = fallback_df[
                (fallback_df['CATEGORY_mid'] == row['category']) &
                (pd.to_numeric(fallback_df['measure_mid'], errors='coerce') == measure_value) &
                (pd.to_numeric(fallback_df['unit_count_mid'], errors='coerce') == unit_count_value)
            ]
            if not cat_match.empty:
                # Handle comma as decimal separator for all values
                cbm_str = str(cat_match['CBM_mid'].iloc[0]).replace(',', '.')
                weight_str = str(cat_match['Weight_mid'].iloc[0]).replace(',', '.')
                cbm = pd.to_numeric(cbm_str, errors='coerce')
                weight = pd.to_numeric(weight_str, errors='coerce')
                return 70, cbm, weight

            # 3. Category only
            cat_avg = fallback_df[fallback_df['CATEGORY_AVG'] == row['category']]
            if not cat_avg.empty:
                # Handle comma as decimal separator for all values
                cbm_str = str(cat_avg['CBM_AVG'].iloc[0]).replace(',', '.')
                weight_str = str(cat_avg['Weight_AVG'].iloc[0]).replace(',', '.')
                cbm = pd.to_numeric(cbm_str, errors='coerce')
                weight = pd.to_numeric(weight_str, errors='coerce')
                return 30, cbm, weight

            # 4. No match
            return 0, 0, 0
        except Exception as e:
            st.error(f"Error processing row {row.get('product_name', '')}: {str(e)}")
            return 0, 0, 0

    def apply_cbm_weight_to_runsheet(runsheet_df, fallback_df):
        # Create result columns
        result_df = runsheet_df.copy()
        result_df['cbm_confidence'] = 0
        result_df['calculated_cbm'] = 0.0
        result_df['calculated_weight'] = 0.0
        
        # Process each row
        for idx, row in result_df.iterrows():
            try:
                conf, cbm, weight = get_cbm_weight_appscript_style(row, fallback_df)
                product_amount = float(row.get('product_amount', 1))
                result_df.at[idx, 'cbm_confidence'] = conf
                result_df.at[idx, 'calculated_cbm'] = cbm * product_amount
                result_df.at[idx, 'calculated_weight'] = weight * product_amount
            except Exception as e:
                st.error(f"Error processing row {idx}: {str(e)}")
        
        return result_df
    
    # 1. Upload Runsheet
    runsheet_file = st.file_uploader("Upload Runsheet Excel/CSV", type=["xlsx", "csv"], key="cbm_runsheet_upload")
    
    if runsheet_file:
        try:
            if runsheet_file.name.endswith('.csv'):
                runsheet_df = pd.read_csv(runsheet_file)
            else:
                runsheet_df = pd.read_excel(runsheet_file)
            st.success(f"✅ Runsheet loaded with {len(runsheet_df)} rows!")
            
            # Load Fallback data from Google Sheet
            with st.spinner("Loading Fallback data from Google Sheet..."):
                fallback_df = load_sheet_data("Fallback")
                if fallback_df is not None:
                    st.success(f"✅ Fallback data loaded with {len(fallback_df)} rows!")
                else:
                    st.error("Could not load the 'Fallback' sheet from Google Sheet.")
                    st.stop()
            
            # Check required columns
            required_cols = ['brand_name', 'category', 'measurement_unit', 'unit_count', 'product_amount']
            missing_cols = [col for col in required_cols if col not in runsheet_df.columns]
            
            if missing_cols:
                st.warning(f"⚠️ Missing columns in Runsheet: {', '.join(missing_cols)}")
                
                # Offer column mapping
                st.subheader("Column Mapping")
                st.write("Map the required columns to your data:")
                
                col_mapping = {}
                for req_col in missing_cols:
                    col_mapping[req_col] = st.selectbox(
                        f"Which column contains '{req_col}'?",
                        options=["None"] + list(runsheet_df.columns),
                        key=f"map_{req_col}"
                    )
                
                if st.button("Apply Mapping"):
                    # Rename columns based on mapping
                    for req_col, mapped_col in col_mapping.items():
                        if mapped_col != "None":
                            runsheet_df[req_col] = runsheet_df[mapped_col]
                    
                    # Check again after mapping
                    missing_cols = [col for col in required_cols if col not in runsheet_df.columns]
                    if missing_cols:
                        st.error(f"⚠️ Still missing columns after mapping: {', '.join(missing_cols)}")
                    else:
                        st.success("✅ Column mapping successful!")
            
            # Preview Table
            st.subheader("Preview of Runsheet")
            st.dataframe(runsheet_df.head(5))
            
            # Preview Fallback Data
            with st.expander("Preview Fallback Data"):
                st.dataframe(fallback_df.head(5))
            
            # Action Button
            if not missing_cols and st.button("Calculate CBM & Weight", key="calculate_cbm_button"):
                with st.spinner("Processing... This may take a moment for large datasets."):
                    result_df = apply_cbm_weight_to_runsheet(runsheet_df, fallback_df)
                
                st.success("✅ Calculation complete!")
                
                # Show Results Table
                st.subheader("Runsheet with Calculated CBM & Weight")
                st.dataframe(result_df)
                
                # Summary Stats
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total CBM", f"{result_df['calculated_cbm'].sum():.2f}")
                with col2:
                    st.metric("Total Weight", f"{result_df['calculated_weight'].sum():.2f} kg")
                with col3:
                    unmatched = (result_df['cbm_confidence'] == 0).sum()
                    matched = len(result_df) - unmatched
                    st.metric("Matched Products", f"{matched} ({matched/len(result_df)*100:.1f}%)")
                with col4:
                    st.metric("Unmatched Products", f"{unmatched} ({unmatched/len(result_df)*100:.1f}%)")
                with col5:
                    median_confidence = result_df['cbm_confidence'].median()
                    st.metric("Median Confidence", f"{median_confidence:.0f}%")
                
                # Download Buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    # Convert to Excel format
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        result_df.to_excel(writer, sheet_name='Results', index=False)
                    excel_data = excel_buffer.getvalue()
                    st.download_button(
                        "📥 Download Complete Results (XLSX)",
                        excel_data,
                        "runsheet_with_cbm_weight.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_complete"
                    )
                
                with col2:
                    if unmatched > 0:
                        unmatched_df = result_df[result_df['cbm_confidence'] == 0]
                        # Convert to Excel format
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                            unmatched_df.to_excel(writer, sheet_name='Unmatched', index=False)
                        excel_data = excel_buffer.getvalue()
                        st.download_button(
                            "📥 Download Unmatched Products (XLSX)",
                            excel_data,
                            "unmatched_products.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_unmatched"
                        )
                
                with col3:
                    if unmatched > 0:
                        # Create unique brand-category combinations for unmatched products
                        unmatched_brand_cat = unmatched_df[['brand_name', 'category']].drop_duplicates()
                        unmatched_brand_cat = unmatched_brand_cat.sort_values(['brand_name', 'category'])
                        # Convert to Excel format
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                            unmatched_brand_cat.to_excel(writer, sheet_name='Unmatched_Brand_Category', index=False)
                        excel_data = excel_buffer.getvalue()
                        st.download_button(
                            "📥 Download Unmatched Brand-Category (XLSX)",
                            excel_data,
                            "unmatched_brand_category.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_unmatched_brand_cat"
                        )
    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.code(traceback.format_exc())
    else:
        st.info("Please upload a Runsheet file to begin.")
        
        # Show example data
        with st.expander("How to use this tool"):
            st.write("""
            ### How to use the CBM & Weight Calculator
            
            1. **Upload your Runsheet** Excel or CSV file.
            2. **Map columns** if your column names don't match the required ones.
            3. **Click Calculate** to process the data.
            4. **Download results** when processing is complete.
            
            The calculator will try to match products in this order:
            1. Exact match (brand, category, measure, unit count) - 100% confidence
            2. Category match (ignoring brand) - 70% confidence
            3. Category average - 30% confidence
            """)

def get_product_cbm_and_weight(product_id, quantity, fallback_df, category=None, value=None, unit=None, brand_category_key=None, product_name=None):
    """
    Calculate CBM and weight for a product based on fallback data.
    """
    # 1. Try exact match by product_id
    if product_id:
        exact = fallback_df[fallback_df['product_id'] == product_id]
        if not exact.empty:
            carton_weight = pd.to_numeric(exact['carton_weight'], errors='coerce').fillna(0).iloc[0]
            packaging_weight = pd.to_numeric(exact['packaging_weight'], errors='coerce').fillna(0).iloc[0]
            carton_volume = pd.to_numeric(exact['carton_volume'], errors='coerce').fillna(0).iloc[0]
            confidence = 100
            return carton_volume * quantity, (carton_weight + packaging_weight) * quantity, confidence

    # 2. Try category + unit + value
    if category and unit and value is not None:
        avg = fallback_df[
            (fallback_df['category'] == category) &
            (fallback_df['unit'] == unit) &
            (pd.to_numeric(fallback_df['value'], errors='coerce') == float(value))
        ]
        if not avg.empty:
            carton_weight = pd.to_numeric(avg['median_weight'], errors='coerce').fillna(0).iloc[0]
            packaging_weight = pd.to_numeric(avg['packaging_weight'], errors='coerce').fillna(0).iloc[0]
            carton_volume = pd.to_numeric(avg['median_cbm'], errors='coerce').fillna(0).iloc[0]
            confidence = 70
            return carton_volume * quantity, (carton_weight + packaging_weight) * quantity, confidence

    # 3. Try brand-category key
    if brand_category_key:
        bc = fallback_df[fallback_df['brand_category_key'] == brand_category_key]
        if not bc.empty:
            carton_weight = pd.to_numeric(bc['median_weight'], errors='coerce').fillna(0).iloc[0]
            packaging_weight = pd.to_numeric(bc['packaging_weight'], errors='coerce').fillna(0).iloc[0]
            carton_volume = pd.to_numeric(bc['median_cbm'], errors='coerce').fillna(0).iloc[0]
            confidence = 50
            return carton_volume * quantity, (carton_weight + packaging_weight) * quantity, confidence

    # 4. Try category only (by product name)
    if product_name:
        cat_only = fallback_df[fallback_df['product_name'] == product_name]
        if not cat_only.empty:
            carton_volume = pd.to_numeric(cat_only['fallback_volume'], errors='coerce').fillna(0).iloc[0]
            packaging_weight = pd.to_numeric(cat_only['packaging_weight'], errors='coerce').fillna(0).iloc[0]
            confidence = 25
            return carton_volume * quantity, packaging_weight * quantity, confidence

    # 5. Default fallback
    default_packaging_weight = 20  # or whatever your default is
    return 0, default_packaging_weight * quantity, 0

def cbm_weight_calculator():
    """
    Streamlit interface for the CBM & Weight Calculator.
    """
    st.title("CBM & Weight Calculator")
    
    # File uploader for Runsheet
    uploaded_file = st.file_uploader("Upload Runsheet Excel/CSV", type=["xlsx", "csv"])
    
    if uploaded_file is None:
        st.info("Please upload a Runsheet file to begin.")
        
        # Help section
        with st.expander("How to use this tool"):
            st.markdown("""
            ## How to use the CBM & Weight Calculator
            1. Upload your Runsheet Excel or CSV file.
            2. Map columns if your column names don't match the required ones.
            3. Click Calculate to process the data.
            4. Download results when processing is complete.
            
            The calculator will try to match products in this order:
            
            * Exact match (brand, category, measure, unit count) - 100% confidence
            * Category match (ignoring brand) - 70% confidence  
            * Category average - 30% confidence
            """)
        return
    
    # Load data based on file type
    try:
        if uploaded_file.name.endswith('.csv'):
            runsheet_df = pd.read_csv(uploaded_file)
        else:
            runsheet_df = pd.read_excel(uploaded_file)
        
        # Convert date columns to string to avoid PyArrow errors
        date_cols = runsheet_df.select_dtypes(include=['datetime64']).columns
        for col in date_cols:
            runsheet_df[col] = runsheet_df[col].astype(str)
            
        # Also convert any object columns that might contain dates
        for col in runsheet_df.columns:
            if runsheet_df[col].dtype == 'object':
                # Try to detect if this column has date-like strings
                sample = runsheet_df[col].dropna().iloc[:10] if len(runsheet_df) > 10 else runsheet_df[col].dropna()
                if len(sample) > 0:
                    date_patterns = [r'\d{2}/\d{2}/\d{2}', r'\d{2}-\d{2}-\d{2}', r'\d{4}-\d{2}-\d{2}']
                    if any(sample.astype(str).str.match('|'.join(date_patterns)).any()):
                        runsheet_df[col] = runsheet_df[col].astype(str)
            
        st.success(f"Successfully loaded Runsheet data with {len(runsheet_df)} rows.")
        
        # Show sample of the data
        with st.expander("Preview Runsheet Data"):
            st.dataframe(runsheet_df.head())
            
        # Option to use Google Sheet or uploaded file for Fallback data
        fallback_source = st.radio(
            "Fallback Data Source",
            ["Google Sheet", "Upload File"],
            horizontal=True
        )
        
        fallback_df = None
        
        if fallback_source == "Google Sheet":
            fallback_df = load_sheet_data("Fallback")
            if fallback_df is None:
                st.error("Could not load the 'Fallback' sheet from Google Sheets.")
                return
        else:
            fallback_file = st.file_uploader("Upload Fallback Data (Excel/CSV)", type=["xlsx", "csv"])
            if fallback_file is None:
                st.info("Please upload a Fallback data file.")
                return
                
            if fallback_file.name.endswith('.csv'):
                fallback_df = pd.read_csv(fallback_file)
            else:
                fallback_df = pd.read_excel(fallback_file)
        
        # Ensure required columns exist in fallback_df
        required_fallback_cols = ['BRAND_NAME', 'CATEGORY', 'measure', 'unit count', 'Weight', 'CBM']
        missing_fallback_cols = [col for col in required_fallback_cols if col not in fallback_df.columns]
        
        if missing_fallback_cols:
            st.warning(f"Missing required columns in Fallback data: {', '.join(missing_fallback_cols)}")
            
            # Column mapping for fallback data
            fallback_mapping = {}
            for missing_col in missing_fallback_cols:
                fallback_mapping[missing_col] = st.selectbox(
                    f"Map '{missing_col}' to existing column in Fallback data:",
                    options=[""] + list(fallback_df.columns),
                    key=f"fallback_map_{missing_col}"
                )
            
            # Apply mapping
            for target_col, source_col in fallback_mapping.items():
                if source_col:
                    fallback_df[target_col] = fallback_df[source_col]
        
        # Ensure required columns exist in runsheet
        required_cols = ['brand_name', 'category', 'unit_count', 'measurement_value', 'product_amount']
        missing_cols = [col for col in required_cols if col not in runsheet_df.columns]
        
        if missing_cols:
            st.warning(f"Missing required columns in Runsheet: {', '.join(missing_cols)}")
            
            # Column mapping
            col_mapping = {}
            for missing_col in missing_cols:
                col_mapping[missing_col] = st.selectbox(
                    f"Map '{missing_col}' to existing column:",
                    options=[""] + list(runsheet_df.columns),
                    key=f"map_{missing_col}"
                )
            
            # Apply mapping
            for target_col, source_col in col_mapping.items():
                if source_col:
                    runsheet_df[target_col] = runsheet_df[source_col]
            
            # Check if we still have missing columns
            still_missing = [col for col in required_cols if col not in runsheet_df.columns]
            if still_missing:
                st.error(f"Still missing required columns: {', '.join(still_missing)}")
                return
        
        # Calculate button
        if st.button("Calculate CBM & Weight"):
            with st.spinner("Processing..."):
                # Create results DataFrame
                results_df = runsheet_df.copy()
                
                # Add columns for results
                results_df['confidence'] = 0
                results_df['cbm'] = 0.0
                results_df['weight'] = 0.0
                results_df['total_cbm'] = 0.0
                results_df['total_weight'] = 0.0
                
                # Process each row
                for idx, row in results_df.iterrows():
                    confidence, cbm, weight = get_cbm_weight_appscript_style(row, fallback_df)
                    
                    # Get product amount (quantity)
                    product_amount = pd.to_numeric(row.get('product_amount', 1), errors='coerce')
                    if pd.isna(product_amount) or product_amount <= 0:
                        product_amount = 1
                    
                    # Update results
                    results_df.at[idx, 'confidence'] = confidence
                    results_df.at[idx, 'cbm'] = cbm
                    results_df.at[idx, 'weight'] = weight
                    results_df.at[idx, 'total_cbm'] = cbm * product_amount
                    results_df.at[idx, 'total_weight'] = weight * product_amount
                
                # Display results
                st.success("Calculation complete!")
                
                # Convert all columns to string to avoid PyArrow errors
                for col in results_df.columns:
                    if results_df[col].dtype == 'datetime64[ns]':
                        results_df[col] = results_df[col].astype(str)
                
                st.dataframe(results_df)
                
                # Summary statistics
                total_cbm = results_df['total_cbm'].sum()
                total_weight = results_df['total_weight'].sum()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total CBM", f"{total_cbm:.4f}")
                with col2:
                    st.metric("Total Weight (kg)", f"{total_weight:.2f}")
                
                # Confidence breakdown
                confidence_counts = results_df['confidence'].value_counts().sort_index(ascending=False)
                
                st.subheader("Match Confidence Breakdown")
                confidence_df = pd.DataFrame({
                    'Confidence Level': confidence_counts.index,
                    'Product Count': confidence_counts.values
                })
                
                # Create a horizontal bar chart
                fig = px.bar(
                    confidence_df, 
                    x='Product Count',
                    y='Confidence Level',
                    orientation='h',
                    color='Confidence Level',
                    color_continuous_scale=px.colors.sequential.Viridis,
                    title="Match Confidence Distribution"
                )
                st.plotly_chart(fig)
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="cbm_weight_results.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.exception(e)

