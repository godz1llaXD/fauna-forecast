import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===========================================================
# 1. Define Historical Anchor Points
# ===========================================================
anchors = {
    1860: 45000000,   # Pre-collapse population
    1884: 325,        # Near extinction
    1889: 1091,
    1905: 1091,
    1910: 2108,
    1920: 225,        # Placeholder
    1990: 237500,
    2017: 500000
}

# Start and end years
start_year = 1800
end_year = 2017

# ===========================================================
# 2. Define Piecewise Phases and Models
# ===========================================================
years = []
populations = []

# --- Phase 1: Pre-collapse (1800–1860) ---
pre_start, pre_end = 1800, 1860
pre_start_pop, pre_end_pop = 60000000, anchors[1860]  # assume 60M in 1800
for year in range(pre_start, pre_end + 1):
    frac = (year - pre_start) / (pre_end - pre_start)
    pop = pre_start_pop + frac * (pre_end_pop - pre_start_pop)
    years.append(year)
    populations.append(pop)

# --- Phase 2: Collapse (1860–1884) ---
collapse_start, collapse_end = 1860, 1884
P0 = anchors[collapse_start]
P_end = anchors[collapse_end]
k = np.log(P0 / P_end) / (collapse_end - collapse_start)
for year in range(collapse_start, collapse_end + 1):
    pop = P0 * np.exp(-k * (year - collapse_start))
    years.append(year)
    populations.append(pop)

# --- Phase 3: Early Recovery (1884–1920) ---
early_start, early_end = 1884, 1920
K1 = 5000
P_start = anchors[early_start]
A = (K1 - P_start) / P_start
target = anchors[early_end]
r = np.log((K1 / target) - 1) / -(early_end - early_start)
for year in range(early_start, early_end + 1):
    pop = K1 / (1 + A * np.exp(-r * (year - early_start)))
    years.append(year)
    populations.append(pop)

# --- Phase 4: Mid-century Recovery (1920–1990) ---
mid_start, mid_end = 1920, 1990
K2 = 300000
P_start = anchors[mid_start]
A = (K2 - P_start) / P_start
target = anchors[mid_end]
r = np.log((K2 / target) - 1) / -(mid_end - mid_start)
for year in range(mid_start, mid_end + 1):
    pop = K2 / (1 + A * np.exp(-r * (year - mid_start)))
    years.append(year)
    populations.append(pop)

# --- Phase 5: Modern Recovery (1990–2017) ---
modern_start, modern_end = 1990, 2017
K3 = 500000
P_start = anchors[modern_start]
A = (K3 - P_start) / P_start
r = 0.1  # small for smooth curve
for year in range(modern_start, modern_end + 1):
    pop = K3 / (1 + A * np.exp(-r * (year - modern_start)))
    years.append(year)
    populations.append(pop)

# ===========================================================
# 3. Build DataFrame and Save
# ===========================================================
df = pd.DataFrame({"Year": years, "Population": populations})

# Save CSV
save_path = r"E:\file_main\Major_1\fauna-forecast\data\raw\revised_bison_population.csv"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
df.to_csv(save_path, index=False)
print(f"Revised dataset saved to: {save_path}")

# ===========================================================
# 4. Visualization
# ===========================================================
plt.figure(figsize=(12, 6))
plt.plot(df["Year"], df["Population"], label="Modeled Population", color="blue")
plt.scatter(list(anchors.keys()), list(anchors.values()), color="red", label="Historical Anchors")
plt.yscale("log")
plt.xlabel("Year")
plt.ylabel("Population (log scale)")
plt.title("American Bison Population: Piecewise Modeled Trend (Smooth)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()
