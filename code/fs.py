import pandas as pd
import requests
from bs4 import BeautifulSoup

# Define the URL of the website
url = "https://www.zenius.net/5ec044d03df711ef827d51b557cee7df/44917b603e9c11ef8ab5b1db2849fec3/9da681503e9c11ef8ab5b1db2849fec3/9da744bf3e9c11ef8ab5b1db2849fec3"

# Send an HTTP request to fetch the webpage content
response = requests.get(url)
html_content = response.content

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, "html.parser")

# Find the relevant data (you may need to inspect the webpage to identify the correct elements)
# For example, if there's a table, you can use soup.find("table") to locate it.

# Create a pandas DataFrame with the extracted data (replace this with your actual data)
data = {"Column1": ["Value1", "Value2", "Value3"],
        "Column2": ["ValueA", "ValueB", "ValueC"]}
df = pd.DataFrame(data)

# Specify the output Excel file path
output_file = "zenius_data.xlsx"

# Export the DataFrame to Excel
df.to_excel(output_file, index=False)

print(f"Data exported to {output_file}")
