
import pandas as pd
import numpy as np
file_path = r"C:\Users\anmol\Downloads\archive (1)\US_Accidents_March23.csv"

USE_COLS = [
    "Start_Time",
    "Temperature(F)",
    "Precipitation(in)",
    "Visibility(mi)",
    "Weather_Condition",
    "Distance(mi)",
    "Severity",
]

def main():
    print("Reading first chunk of US_Accidents_March23.csv ...")
    chunk_iter = pd.read_csv(
        file_path,
        usecols=USE_COLS,
        chunksize=100_000,   # read 100,000 rows at a time
    )

    first_chunk = next(chunk_iter)
    
    print("Chunk shape:", first_chunk.shape)
        # Convert units and create model features

    # Temperature: F -> C
    first_chunk["temperature_c"] = (first_chunk["Temperature(F)"] - 32) * 5.0 / 9.0

    # Rainfall: inches -> mm, missing as 0
    first_chunk["Precipitation(in)"] = first_chunk["Precipitation(in)"].fillna(0.0)
    first_chunk["rainfall_mm"] = first_chunk["Precipitation(in)"] * 25.4

    # Visibility: miles -> km
    first_chunk["visibility_km"] = first_chunk["Visibility(mi)"] * 1.60934

    # Distance: miles -> km
    first_chunk["distance_km"] = first_chunk["Distance(mi)"] * 1.60934

    # Time of day bucket from Start_Time
    dt = pd.to_datetime(first_chunk["Start_Time"])
    hours = dt.dt.hour

    def map_hour_to_time_of_day(h):
        if 5 <= h < 12:
            return "morning"
        elif 12 <= h < 17:
            return "afternoon"
        elif 17 <= h < 21:
            return "evening"
        else:
            return "night"

    first_chunk["time_of_day"] = hours.apply(map_hour_to_time_of_day)

    # Risk score from Severity (1..4 -> 0..1)
    first_chunk["risk_score"] = (first_chunk["Severity"] - 1) / 3.0

    # Add synthetic rider experience (0–20 years)
    n = len(first_chunk)

    # Generate according to desired distribution
    experience = np.zeros(n)

    # 40%: 0-2 years
    mask1 = np.random.rand(n) < 0.40
    experience[mask1] = np.random.randint(0, 3, size=mask1.sum())

    # 35%: 3-7 years
    remaining = ~mask1
    mask2 = np.random.rand(n) < (0.35 / 0.60)  # conditional
    mask2 = remaining & mask2
    experience[mask2] = np.random.randint(3, 8, size=mask2.sum())

    # Remaining 25%: 8–20 years
    mask3 = ~(mask1 | mask2)
    experience[mask3] = np.random.randint(8, 21, size=mask3.sum())

    first_chunk["experience"] = experience.astype(int)


    # Show only the columns we care about now
    print("Transformed preview:")
    print(
        first_chunk[
            [
                "temperature_c",
                "rainfall_mm",
                "visibility_km",
                "distance_km",
                "time_of_day",
                "experience",
                "risk_score",
            ]
        ].head()
    )

    # Build a clean DataFrame with only the model features
    model_df = first_chunk[
        [
            "temperature_c",
            "rainfall_mm",
            "visibility_km",
            "distance_km",
            "time_of_day",
            "experience",
            "risk_score",
        ]
    ].dropna()

    print("After dropna, rows:", len(model_df))

    # Take a sample of up to 20,000 rows for our initial ML dataset
    sample_size = min(20_000, len(model_df))
    model_sample = model_df.sample(sample_size, random_state=42)

    # Save to CSV inside the project
    output_path = "data/processed/us_accidents_risk_sample.csv"
    model_sample.to_csv(output_path, index=False)
    print(f"Saved sample of {sample_size} rows to {output_path}")


if __name__ == "__main__":
    main()