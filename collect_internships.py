import requests
import json

url = "https://internships-api.p.rapidapi.com/active-jb-7d"

HEADERS = {
    "x-rapidapi-key": "a98a8c6046msh3d169abb1084bfdp1fe883jsna0c75baffca6",
    "x-rapidapi-host": "internships-api.p.rapidapi.com"
}

def fetch_internships(title_filter="", location_filter="", description_filter=""):
    params = {
        "title_filter": title_filter,
        "location_filter": location_filter,
        "description_filter": description_filter,
        "description_type": "text",
    }

    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()  # raises HTTPError for 4xx/5xx
        return response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Request failed: {e}")

def collect_and_save_internships(title="", location="", description="", output_file=""):
    try:
        listings = fetch_internships(title, location, description)
        if not listings:
            print("⚠️ No job listings found or response was empty.")
            return
    except RuntimeError as e:
        print(f"❌ Error fetching data: {e}")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(listings, f, indent=2)

    print(f"✅ Saved {len(listings)} job listings to {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch internships using the Fantastic.jobs API")

    parser.add_argument("--title", type=str, default="", help="Job title or keywords to filter")
    parser.add_argument("--location", type=str, default="", help="Job location to filter")
    parser.add_argument("--description", type=str, default="", help="Description keywords (e.g. tools or skills)")
    parser.add_argument("--output", type=str, default="fantastic_internships_dataset.json", help="Output file name")

    args = parser.parse_args()

    collect_and_save_internships(
        title=args.title,
        location=args.location,
        description=args.description,
        output_file=args.output
    )
