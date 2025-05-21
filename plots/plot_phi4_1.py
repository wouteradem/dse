import matplotlib.pyplot as plt

# All real roots rounded to 4 decimal places
roots = [
    0.5851, 0.6441, 0.4561, 0.2519, -0.2519, -0.5851, -0.6619, -0.6441, -0.4561, 0.0, 0.6619,
    0.6615, 0.6387, 0.5602, -0.1418, -0.3924, -0.6387, -0.6615, -0.5602, 0.1418, 0.3924,
    0.6610, 0.6306, 0.3028, -0.3028, -0.5232, -0.6610, -0.6306, 0.0, 0.5232,
    0.6603, 0.6181, 0.1763, -0.4664, -0.6603, -0.6181, -0.1763, 0.4664,
    0.6592, 0.5973, 0.0, -0.5973, -0.6592, -0.3768, 0.3768,
    0.6575, 0.5602, -0.2321, -0.6575, -0.5602, 0.2321,
    0.6543, 0.4884, -0.4884, -0.6543, 0.0,
    0.6480, 0.3367, -0.6480, -0.3367,
    0.6325, 0.0, -0.6325,
    0.5774, -0.5774
]

# Exact value
exact_value = 0.675978

# Plot roots
plt.figure(figsize=(12, 2))
plt.scatter(roots, [0] * len(roots), color='blue', s=30)

# Mark the exact value
plt.scatter([exact_value], [0], color='red', s=80, zorder=5)

# Axis formatting
plt.axhline(0, color='black', linewidth=0.5)
plt.yticks([])
plt.xlabel("Real axis")
plt.title("All computed roots and exact value (red dot)")

plt.tight_layout()
plt.show()
