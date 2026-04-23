import os
import csv
import statistics
import glob

def generate_summary():
    output_file = "summary.csv"
    results = []

    # Find all CSV files in the current directory ending with 'idle.csv'
    csv_files = glob.glob("*-idle.csv")
    
    if not csv_files:
        print("No '-idle.csv' files found in the current directory.")
        return

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        # Use the filename (without extension) as the host identifier
        host = filename.replace(".csv", "")
        
        power_values = []
        
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Detect the correct column based on whether it's a CPU or GPU file
                if 'cpu_watts' in row:
                    val = row['cpu_watts']
                elif 'power_draw_w' in row:
                    val = row['power_draw_w']
                else:
                    # If neither column exists, skip to the next row
                    continue 
                    
                try:
                    power_values.append(float(val))
                except ValueError:
                    # Skip rows with missing or malformed data
                    pass
                    
        # Calculate stats if we found valid data
        if power_values:
            mean_val = statistics.mean(power_values)
            # stdev requires at least two data points
            std_val = statistics.stdev(power_values) if len(power_values) > 1 else 0.0
            results.append([host, round(mean_val, 3), round(std_val, 3)])
        else:
            print(f"Warning: No valid power data found in {filename}")

    # Write out the results to summary.csv
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['host', 'mean', 'std dev'])
        writer.writerows(results)

    print(f"Success! Parsed {len(results)} files. Summary saved to {output_file}")
    
    # Print it to the console as well for a quick preview
    print("\n--- PREVIEW ---")
    print("host, mean, std dev")
    for row in results:
        print(f"{row[0]}, {row[1]}, {row[2]}")

if __name__ == "__main__":
    generate_summary()
