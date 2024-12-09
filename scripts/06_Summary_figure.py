import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Ensure the output directory exists
output_dir = "../figures/"
os.makedirs(output_dir, exist_ok=True)

# Data for the chart
accuracy = [81.7, 83.4, 81.4, 82.7, 80.3, 77.3, 55.8]
labels = [
    "Fine-tuned gpt-4o-mini",
    "Fine-tuned Llama-3.1-8B",
    "Fine-tuned Mistral-Nemo-12B",
    "Fine-tuned Qwen-2.5-72B",
    "SentenceBERT nearest",
    "String Edit nearest",
    "Guess majority",
]

# Reverse data to display bars from top to bottom
accuracy = accuracy[::-1]
labels = labels[::-1]

# Use a beautiful Seaborn color palette
colors = sns.color_palette("viridis", len(labels))

# Plot the bar chart
fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(labels, accuracy, color=colors, edgecolor="black", linewidth=0.7)

# Add data labels
for bar in bars:
    width = bar.get_width()
    ax.text(
        width + 0.8,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.1f}%",
        va="center",
        ha="left",
        fontsize=11,
        color="black",
    )

# Add title and labels
ax.set_title("Model Accuracy Comparison", fontsize=18, fontweight="bold", pad=20)
ax.set_xlabel("% Accuracy", fontsize=14, labelpad=10)
ax.set_xlim(0, 90)
ax.grid(axis="x", linestyle="--", alpha=0.6, linewidth=0.8)

# Customize the chart appearance
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#555555")
ax.spines["bottom"].set_color("#555555")
ax.tick_params(axis="y", labelsize=12)
ax.tick_params(axis="x", labelsize=12)

# Enhance the overall aesthetics using Seaborn's style
sns.set_style("whitegrid")
sns.despine(left=True)

# Save the chart as an image
plt.tight_layout()
output_path = os.path.join(output_dir, "summary_barchart.png")
plt.savefig(output_path, dpi=300)
plt.show()
