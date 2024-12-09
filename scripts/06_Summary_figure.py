import matplotlib.pyplot as plt
import numpy as np

# Data for the chart
accuracy = [81.7, 83.4, 81.4, 83.4, 75.9, 77.3, 55.8]
labels = [
    "Fine-tuned gpt-4o-mini",
    "Fine-tuned Llama-3.1-8B",
    "Fine-tuned Mistral-Nemo-12B",
    "Fine-tuned Qwen-2.5-72B",
    "SentenceBERT nearest",
    "String Edit nearest",
    "Guess majority",
]

# Custom color scheme
colors = [
    "#4B83CC",  # Muted Blue
    "#6DBD6B",  # Soft Green
    "#D9A440",  # Golden Yellow
    "#B97399",  # Muted Purple
    "#999999",  # Neutral Gray
    "#7E9AD0",  # Light Blue
    "#C19163",  # Warm Brown
]

# Reverse data to display bars from top to bottom
accuracy = accuracy[::-1]
labels = labels[::-1]
colors = colors[::-1]

# Plot the bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(labels, accuracy, color=colors, edgecolor="black")

# Add data labels
for bar in bars:
    width = bar.get_width()
    ax.text(
        width + 1,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.1f}%",
        va="center",
        ha="left",
        fontsize=10,
    )

# Add title and labels
ax.set_title("Model Accuracy Comparison", fontsize=16, fontweight="bold", pad=15)
ax.set_xlabel("% Accuracy", fontsize=12)
ax.set_xlim(0, 90)
ax.grid(axis="x", linestyle="--", alpha=0.7)

# Customize the chart appearance
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#444444")
ax.spines["bottom"].set_color("#444444")
ax.tick_params(axis="y", labelsize=10)
ax.tick_params(axis="x", labelsize=10)

# Save the chart as an image
plt.tight_layout()
plt.savefig("../figures/summary_barchart.png", dpi=300)
plt.show()
