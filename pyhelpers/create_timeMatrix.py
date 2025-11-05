# requires: requests

import requests
import json
import time

def get_osrm_time_matrix(coordinates):
    """
    Fetches a time matrix from the OSRM API for a given list of coordinates.

    Args:
        coordinates: A list of [longitude, latitude] pairs.

    Returns:
        A dictionary representing the JSON response from the OSRM API,
        which includes the duration matrix.
    """
    # Format the coordinates into a string for the URL
    locations = ";".join([f"{lon},{lat}" for lon, lat in coordinates])
    
    # OSRM API endpoint for the table service (time matrix)
    # Using the public demo server. For production, you should run your own OSRM server.
    url = f"http://router.project-osrm.org/table/v1/driving/{locations}"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        
        
        if data['code'] == 'Ok':
            return data['durations']
        else:
            print(f"OSRM API Error: {data.get('message', 'Unknown error')}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

