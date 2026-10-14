import matplotlib.pyplot as plt
import pandas as pd


data = pd.DataFrame(
    {
        "superficie": [50, 60, 70, 80, 90, 100, 110, 120],
        "chambres": [1, 2, 2, 3, 3, 3, 4, 4],
        "anciennete": [20, 15, 12, 10, 8, 5, 4, 2],
        "distance_centre_km": [12, 10, 9, 7, 6, 5, 4, 3],
        "prix": [400, 480, 530, 650, 700, 820, 850, 960],
    }
)

print("Jeu de données (prix en kDH) :")
print(data.to_string(index=False))
print("\nStatistiques descriptives :")
print(data.describe().round(2))
print(f"\nVariance échantillon des prix : {data['prix'].var():.2f} (kDH)²")
print(f"Écart-type échantillon des prix : {data['prix'].std():.2f} kDH")

data["prix_m2"] = data["prix"] * 1000 / data["superficie"]
print("\nPrix au mètre carré (DH/m²) :")
print(data[["superficie", "prix", "prix_m2"]].round(2).to_string(index=False))
print(f"\nCorrélation superficie/prix : {data['superficie'].corr(data['prix']):.3f}")

plt.hist(data["prix"], bins=5, edgecolor="white", color="#4472C4")
plt.xlabel("Prix (milliers de DH)")
plt.ylabel("Nombre de maisons")
plt.title("Distribution des prix des maisons")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()
