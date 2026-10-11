import pandas as pd
import numpy as np
from pathlib import Path

def generate_sensor_data(rows=1000):
    np.random.seed(42)

    data = pd.DataFrame({
        "plot_id": [f"PLOT-{i:04d}" for i in range(1, rows + 1)],
        "soil_moisture": np.random.uniform(10, 80, rows),
        "temperature": np.random.uniform(20, 42, rows),
        "humidity": np.random.uniform(30, 95, rows),
        "rainfall_mm": np.random.uniform(0, 25, rows),
        "crop_stage": np.random.choice(
            ["Germination", "Tillering", "Grand Growth", "Maturity"],
            rows
        )
    })

    # Clearly labelled rule-based baseline for the prototype.
    data["irrigation_needed"] = (
        (data["soil_moisture"] < 30) &
        (data["rainfall_mm"] < 5)
    ).astype(int)

    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "sensor_data.csv"
    data.to_csv(output_file, index=False)

    print(f"Generated {len(data)} synthetic sensor records.")
    print(f"Saved to: {output_file.resolve()}")
    print("Note: This is synthetic demonstration data, not real farm data.")

if __name__ == "__main__":
    generate_sensor_data()