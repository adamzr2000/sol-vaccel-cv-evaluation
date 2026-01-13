import os
import json
import glob

def main():
    output_filename = "combined_latency_summary.json"
    combined_data = []

    # Find all summary files recursively
    search_pattern = os.path.join(".", "*", "latency_results_summary.json")
    summary_files = glob.glob(search_pattern)

    print(f"Found {len(summary_files)} summary files. Processing...")

    for file_path in summary_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                combined_data.append(data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # SORTING: 
    # 1. Model Name (Groups 'resnet50' and 'resnet50_sol' together alphabetically)
    # 2. Device (Separates CPU vs GPU results)
    # 3. Backend (Stock vs vAccel)
    combined_data.sort(key=lambda x: (
        x.get('model', ''), 
        x.get('device', ''), 
        x.get('backend', '')
    ))

    # Write the sorted, grouped data to the new file
    with open(output_filename, 'w') as f:
        json.dump(combined_data, f, indent=4)

    print(f"Done! Combined results saved to '{output_filename}' (Sorted by Model -> Device)")

if __name__ == "__main__":
    main()