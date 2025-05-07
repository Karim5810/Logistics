import pandas as pd
import numpy as np
import streamlit as st
import io
import traceback
from typing import Dict, List, Tuple, Any, Optional

# Helper function for safe numeric conversion
def safe_float(value, default=0.0):
    """Safely convert a value to float, returning default if conversion fails."""
    if value is None or value == '':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def calculate_cbm_from_dimensions(length: float, width: float, height: float) -> float:
    """Calculate CBM (Cubic Meter) from dimensions in meters."""
    return length * width * height

def process_excel_for_cbm(uploaded_file) -> Tuple[Optional[Dict], str]:
    """
    Process the uploaded Excel file and calculate CBM for each item.
    
    Args:
        uploaded_file: The uploaded Excel file object
        
    Returns:
        Tuple containing:
        - Dictionary with calculated CBM values or None if error
        - Status message (success or error message)
    """
    try:
        # Read the Excel file
        df = pd.read_excel(uploaded_file)
        
        # Check for required columns
        required_columns = []
        
        # Option 1: Direct CBM column
        if 'CBM' in df.columns:
            required_columns = ['Order ID', 'Product ID', 'Quantity', 'CBM']
        # Option 2: Dimensions columns
        elif all(col in df.columns for col in ['Length (m)', 'Width (m)', 'Height (m)']):
            required_columns = ['Order ID', 'Product ID', 'Quantity', 'Length (m)', 'Width (m)', 'Height (m)']
        # Option 3: Dimensions in cm
        elif all(col in df.columns for col in ['Length (cm)', 'Width (cm)', 'Height (cm)']):
            required_columns = ['Order ID', 'Product ID', 'Quantity', 'Length (cm)', 'Width (cm)', 'Height (cm)']
        # Option 4: Dimensions without units (assume cm)
        elif all(col in df.columns for col in ['Length', 'Width', 'Height']):
            required_columns = ['Order ID', 'Product ID', 'Quantity', 'Length', 'Width', 'Height']
        else:
            return None, "Missing required columns. Please ensure your file has either 'CBM' column or dimensions columns (Length, Width, Height)."
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return None, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Calculate CBM based on available columns
        if 'CBM' in df.columns:
            # CBM already provided, just ensure it's numeric
            df['CBM'] = df['CBM'].apply(lambda x: safe_float(x))
        elif 'Length (m)' in df.columns:
            # Dimensions in meters
            df['CBM'] = df.apply(
                lambda row: calculate_cbm_from_dimensions(
                    safe_float(row['Length (m)']),
                    safe_float(row['Width (m)']),
                    safe_float(row['Height (m)'])
                ),
                axis=1
            )
        elif 'Length (cm)' in df.columns:
            # Dimensions in cm, convert to meters (divide by 100)
            df['CBM'] = df.apply(
                lambda row: calculate_cbm_from_dimensions(
                    safe_float(row['Length (cm)']) / 100,
                    safe_float(row['Width (cm)']) / 100,
                    safe_float(row['Height (cm)']) / 100
                ),
                axis=1
            )
        else:
            # Dimensions without units (assume cm)
            df['CBM'] = df.apply(
                lambda row: calculate_cbm_from_dimensions(
                    safe_float(row['Length']) / 100,
                    safe_float(row['Width']) / 100,
                    safe_float(row['Height']) / 100
                ),
                axis=1
            )
        
        # Calculate total CBM per item (CBM * Quantity)
        df['Total CBM'] = df['CBM'] * df['Quantity'].apply(lambda x: safe_float(x, 1))
        
        # Calculate total CBM per order
        order_summary = df.groupby('Order ID').agg(
            Total_CBM=('Total CBM', 'sum'),
            Total_Products=('Product ID', 'count'),
            Total_Quantity=('Quantity', 'sum')
        ).reset_index()
        
        # Calculate grand total
        grand_total = df['Total CBM'].sum()
        
        # Add grand total to the results
        df_with_totals = df.copy()
        df_with_totals.loc['Grand Total'] = ['', '', df['Quantity'].sum(), '', '', '', '', grand_total]
        
        return {
            'detailed_data': df_with_totals,
            'order_summary': order_summary,
            'grand_total': grand_total
        }, "Success"
        
    except Exception as e:
        error_msg = f"Error processing Excel file: {str(e)}"
        st.error(error_msg)
        st.code(traceback.format_exc())
        return None, error_msg

def cbm_calculator_ui():
    """
    UI function for CBM calculator page.
    """
    st.title("CBM Calculator")
    st.write("Upload an Excel file with product dimensions to calculate CBM (Cubic Meter).")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
    
    with col2:
        calculate_button = st.button("Calculate CBM", use_container_width=True)
    
    if uploaded_file and calculate_button:
        try:
            with st.spinner("Calculating CBM..."):
                # Read the Excel file
                df = pd.read_excel(uploaded_file)
                
                # Check if required columns exist
                required_cols = ['Length (cm)', 'Width (cm)', 'Height (cm)', 'Quantity']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    st.write("Your Excel file should have the following columns:")
                    st.write(", ".join(required_cols))
                else:
                    # Calculate CBM for each row
                    df['CBM'] = (df['Length (cm)'] * df['Width (cm)'] * df['Height (cm)'] * df['Quantity']) / 1000000
                    
                    # Calculate total CBM
                    total_cbm = df['CBM'].sum()
                    
                    # Display results
                    st.subheader("Results")
                    
                    # Show total CBM prominently
                    st.markdown(f"<h2 style='text-align: center; color: #1e88e5;'>Total CBM: {total_cbm:.4f} m³</h2>", unsafe_allow_html=True)
                    
                    # Display the dataframe with calculated CBM
                    st.dataframe(df, use_container_width=True)
                    
                    # Provide download buttons for the results
                    csv = df.to_csv(index=False).encode('utf-8')
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        df.to_excel(writer, sheet_name='CBM_Results', index=False)
                    excel_buffer.seek(0)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="cbm_results.csv",
                            mime="text/csv",
                        )
                    with col2:
                        st.download_button(
                            label="Download Results as Excel",
                            data=excel_buffer,
                            file_name="cbm_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    # Display template download option
    st.subheader("Need a template?")
    template_buffer = get_template_download()
    st.download_button(
        label="Download Template",
        data=template_buffer,
        file_name="cbm_calculator_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# Sample Excel template creation function
def create_sample_template():
    """Create a sample Excel template for CBM calculation."""
    df = pd.DataFrame({
        'Order ID': ['ORD001', 'ORD001', 'ORD002', 'ORD002', 'ORD003'],
        'Product ID': ['P001', 'P002', 'P001', 'P003', 'P002'],
        'Quantity': [5, 2, 3, 1, 4],
        'Length (cm)': [50, 30, 50, 100, 30],
        'Width (cm)': [40, 20, 40, 80, 20],
        'Height (cm)': [30, 15, 30, 60, 15],
    })
    
    return df

# Function to provide a downloadable template
def get_template_download():
    # Create a sample Excel template
    df = pd.DataFrame({
        'Product Name': ['Product A', 'Product B'],
        'Length (cm)': [10, 20],
        'Width (cm)': [5, 10],
        'Height (cm)': [2, 5],
        'Quantity': [1, 2]
    })
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='CBM_Template', index=False)
    buffer.seek(0)
    return buffer 