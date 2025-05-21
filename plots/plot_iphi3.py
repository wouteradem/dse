import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("roots.csv")
plt.figure(figsize=(6,6))
plt.scatter(data["real"], data["imag"], color="blue", marker="o")

exact_value = -0.72901113
plt.scatter([0], [exact_value], color='red', s=80, zorder=5)

plt.axhline(0, color="gray", linestyle="--")
plt.axvline(0, color="gray", linestyle="--")
plt.xlabel("Re")
plt.ylabel("Im")
plt.title("Roots for iPHI 3")
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()