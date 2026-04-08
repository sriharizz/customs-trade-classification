import os
import time
import json
import csv
import requests

def fetch_hts_data():
    url = "https://www.usitc.gov/sites/default/files/tata/hts/hts_2026_revision_2_json.json"
    output_path = os.path.join(os.path.dirname(__file__), "hts_data.json")
    
    chapters_we_need = {"39", "61", "62", "72", "84", "85", "87", "90"}
    all_entries = []

    try:
        print("Downloading HTS 2026 data...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()

        for item in data:
            htsno = str(item.get("htsno") or "").strip()
            description = str(item.get("description") or "").strip()
            general = str(item.get("general") or "").strip()
            chapter = htsno[:2] if len(htsno) >= 2 else ""

            if chapter in chapters_we_need and htsno and general:
                all_entries.append({
                    "htsno": htsno,
                    "description": description,
                    "general": general
                })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_entries, f, indent=2)

        print(f"Saved {len(all_entries)} HTS entries.")

    except Exception as e:
        print(f"Failed to fetch HTS data: {e}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([], f)

def fetch_ofac_data():
    url = "https://www.treasury.gov/ofac/downloads/sdn.csv"
    output_path = os.path.join(os.path.dirname(__file__), "ofac_sdn.csv")
    unique_countries = set()

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        decoded_content = response.content.decode("utf-8")
        reader = csv.reader(decoded_content.splitlines())
        
        for row in reader:
            if len(row) > 3:
                country = row[3].strip()
                if country and country != "-":
                    unique_countries.add(country)
                    
        unique_countries_list = sorted(list(unique_countries))
        
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["country"])
            for c in unique_countries_list:
                writer.writerow([c])
                
        print(f"Saved {len(unique_countries_list)} sanctioned countries.")

    except Exception as e:
        print(f"Failed to fetch OFAC data: {e}")
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["country"])

if __name__ == "__main__":
    fetch_hts_data()
    fetch_ofac_data()
    print("Data loading complete")
