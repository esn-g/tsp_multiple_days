from mapbox import Geocoder
import requests
MAPBOX_TOKEN = "pk.eyJ1Ijoib3B0LWRldiIsImEiOiJjbWhsMDI4aWoxbW55MmpxcXgycDR3a2xyIn0.4wATU3zYEWQIIxnwl6w0oQ"

def forward_geocode(address):
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{address}.json"
    params = {
        "access_token": MAPBOX_TOKEN,
        "limit": 1
    }
    response = requests.get(url, params=params)
    data = response.json()
    if data["features"]:
        print(data["features"])
        return data["features"][0]["geometry"]["coordinates"]

    else:
        return None

def reverse_geocode(lon, lat):
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{lon},{lat}.json"
    params = {
        "access_token": MAPBOX_TOKEN
    }
    response = requests.get(url, params=params)
    data = response.json()
    if data["features"]:
        return data["features"][0]["place_name"]
    else:
        return None

def main():
    # Example usage
    coords = forward_geocode("Villa Horn 1, 72592 Västerås, Sweden")
    print("Coordinates:", coords)

    address = reverse_geocode(coords[0], coords[1])
    print("Address:", address)
    
if __name__== "__main__":
    main()
    
    
    
    

