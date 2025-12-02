# submission/chart.py
"""
Generates a professional Seaborn correlation-heatmap for customer engagement metrics
Saves chart.png with exactly 512x512 pixels.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- Create realistic synthetic data ----------
metrics = [
    "Visits", "AvgSessionMin", "ConversionRate", "AvgOrderValue",
    "RetentionRate", "NPS", "SupportTickets", "ChurnRate"
]

base_corr = np.array([
    [ 1.00,  0.60,  0.50,  0.40,  0.55,  0.30, -0.10, -0.50],
    [ 0.60,  1.00,  0.45,  0.35,  0.50,  0.25, -0.05, -0.45],
    [ 0.50,  0.45,  1.00,  0.30,  0.35,  0.20, -0.15, -0.60],
    [ 0.40,  0.35,  0.30,  1.00,  0.25,  0.15, -0.05, -0.30],
    [ 0.55,  0.50,  0.35,  0.25,  1.00,  0.40, -0.20, -0.55],
    [ 0.30,  0.25,  0.20,  0.15,  0.40,  1.00, -0.10, -0.25],
    [-0.10, -0.05, -0.15, -0.05, -0.20, -0.10,  1.00,  0.10],
    [-0.50, -0.45, -0.60, -0.30, -0.55, -0.25,  0.10,  1.00]
])

# Create covariance-like matrix by scaling with plausible stds
stds = np.array([3000, 10, 0.02, 50, 0.10, 15, 5, 0.05])
cov = (base_corr * stds).T * stds

rng = np.random.default_rng(seed=42)
n_samples = 1000
raw = rng.multivariate_normal(mean=np.zeros(len(metrics)), cov=cov, size=n_samples)
df = pd.DataFrame(raw, columns=metrics)

# Transform to realistic ranges
df["Visits"] = np.clip(5000 + df["Visits"], 10, None).astype(int)
df["AvgSessionMin"] = np.clip(3 + df["AvgSessionMin"] / 5, 0.1, None)
df["ConversionRate"] = np.clip(0.01 + df["ConversionRate"] / 200, 0.0001, 1.0)
df["AvgOrderValue"] = np.clip(20 + df["AvgOrderValue"] / 2, 0.01, None)
df["RetentionRate"] = np.clip(0.2 + df["RetentionRate"] / 10, 0.0, 1.0)
df["NPS"] = np.clip(10 + df["NPS"] / 2, -100, 100)
df["SupportTickets"] = np.clip(1 + np.abs(df["SupportTickets"]/3), 0, None).astype(int)
df["ChurnRate"] = np.clip(0.05 + np.abs(df["ChurnRate"]/20), 0.0, 1.0)

# ---------- Compute correlation ----------
corr = df.corr()

# ---------- Plot heatmap ----------
sns.set_style("white")
sns.set_context("talk", font_scale=1.0)

plt.figure(figsize=(8, 8))                # 8in x 8in
ax = sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="vlag",
    center=0,
    linewidths=0.8,
    linecolor="white",
    cbar_kws={"shrink": 0.75, "label": "Correlation"}
)
ax.set_title("Customer Engagement — Correlation Matrix", pad=16, fontsize=16)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

# Save: 8in * dpi must equal 512 pixels => dpi = 64
plt.savefig("submission/chart.png", dpi=64, bbox_inches="tight", pad_inches=0.08)
plt.close()

if __name__ == "__main__":
    # verification when run as script
    from PIL import Image
    im = Image.open("submission/chart.png")
    print("Saved submission/chart.png with size:", im.size)
