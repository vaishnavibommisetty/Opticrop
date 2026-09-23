import pandas as pd
import matplotlib.pyplot as plt

# Load the actual OptiCrop dataset
df = pd.read_csv("dataset.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# Agricultural features
features = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

# Create 2 x 2 EDA figure
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

fig.suptitle(
    "Exploratory Data Analysis of OptiCrop Dataset",
    fontsize=18,
    fontweight="bold"
)

# --------------------------------------------------
# 1. Crop Distribution
# --------------------------------------------------

crop_counts = df["crop"].value_counts()

axes[0, 0].bar(
    crop_counts.index,
    crop_counts.values
)

axes[0, 0].set_title(
    "Crop Distribution",
    fontsize=13,
    fontweight="bold"
)

axes[0, 0].set_xlabel("Crop Type")
axes[0, 0].set_ylabel("Number of Samples")

axes[0, 0].tick_params(
    axis="x",
    rotation=90
)

# --------------------------------------------------
# 2. Agricultural Parameter Distribution
# --------------------------------------------------

axes[0, 1].boxplot(
    [df[col] for col in features],
    labels=features
)

axes[0, 1].set_title(
    "Distribution of Agricultural Parameters",
    fontsize=13,
    fontweight="bold"
)

axes[0, 1].set_xlabel("Parameters")
axes[0, 1].set_ylabel("Values")

axes[0, 1].tick_params(
    axis="x",
    rotation=45
)

# --------------------------------------------------
# 3. Nitrogen vs Phosphorus
# --------------------------------------------------

axes[1, 0].scatter(
    df["N"],
    df["P"],
    alpha=0.6
)

axes[1, 0].set_title(
    "Nitrogen vs Phosphorus",
    fontsize=13,
    fontweight="bold"
)

axes[1, 0].set_xlabel("Nitrogen (N)")
axes[1, 0].set_ylabel("Phosphorus (P)")

# --------------------------------------------------
# 4. Correlation Matrix
# --------------------------------------------------

correlation = df[features].corr()

axes[1, 1].imshow(correlation)

axes[1, 1].set_title(
    "Feature Correlation Matrix",
    fontsize=13,
    fontweight="bold"
)

axes[1, 1].set_xticks(range(len(features)))
axes[1, 1].set_yticks(range(len(features)))

axes[1, 1].set_xticklabels(
    features,
    rotation=45,
    ha="right"
)

axes[1, 1].set_yticklabels(features)

# Show correlation values
for i in range(len(features)):
    for j in range(len(features)):
        axes[1, 1].text(
            j,
            i,
            f"{correlation.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

# --------------------------------------------------
# Save Figure 5.2
# --------------------------------------------------

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(
    "Figure_5_2_EDA_OptiCrop.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()