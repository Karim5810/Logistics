# Logistics Admin Dashboard

A comprehensive logistics administration dashboard built with Streamlit, featuring trip assignment optimization and CBM (Cubic Meter) calculation tools.

## Features

- **Trip Assignment**: Optimize delivery trips based on location, weight, and volume constraints
- **CBM Calculator**: Calculate the total cubic meter volume of orders from Excel uploads
- **Dashboard**: View key logistics metrics and data visualizations

## CBM Calculator

The CBM Calculator allows you to upload an Excel file containing order information and calculates the total cubic meter volume for each order and all orders combined.

### Excel File Format

Your Excel file should contain the following columns:

- **Order ID**: Unique identifier for each order
- **Product ID**: Unique identifier for each product
- **Quantity**: Number of items for each product

And either:
- **CBM**: Direct cubic meter value per item

OR dimensions:
- **Length (m)**, **Width (m)**, **Height (m)**: Dimensions in meters
- **Length (cm)**, **Width (cm)**, **Height (cm)**: Dimensions in centimeters
- **Length**, **Width**, **Height**: Dimensions (assumed to be in centimeters)

### How to Use

1. Navigate to the "CBM Calculator" section in the sidebar
2. Upload your Excel file using the file uploader
3. View the calculated CBM values for each order and the grand total
4. Download the results as CSV files if needed

A sample template can be downloaded from the Settings page.

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   streamlit run app.py
   ```

## Requirements

- Python 3.7+
- Streamlit 1.22.0
- Pandas 1.5.3
- Other dependencies listed in requirements.txt

## Google Sheets Integration

To use Google Sheets integration:
1. Create a service account in Google Cloud Console
2. Generate and download credentials JSON file
3. Rename it to `Credentials_clean.json` and place it in the project root
4. Share your Google Sheets with the service account email

## Setup Instructions

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```