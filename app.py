import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


st.set_page_config(page_title="Estimation immobilière", page_icon="🏠", layout="wide")
st.title("Analyse et prédiction des prix des maisons")
st.write("Exemple pédagogique : les prix sont fictifs et exprimés en milliers de dirhams (kDH).")

# La distance est une caractéristique illustrative, ajoutée au jeu fictif.
data = pd.DataFrame(
    {
        "superficie": [50, 60, 70, 80, 90, 100, 110, 120],
        "chambres": [1, 2, 2, 3, 3, 3, 4, 4],
        "anciennete": [20, 15, 12, 10, 8, 5, 4, 2],
        "distance_centre_km": [12, 10, 9, 7, 6, 5, 4, 3],
        "prix": [400, 480, 530, 650, 700, 820, 850, 960],
    }
)
features = ["superficie", "chambres", "anciennete", "distance_centre_km"]

st.header("1. Explorer les données")
st.dataframe(data, use_container_width=True)

st.header("2. Prix : statistiques et distribution")
prices = data["prix"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Moyenne", f"{prices.mean():.2f} kDH")
c2.metric("Médiane", f"{prices.median():.2f} kDH")
c3.metric("Variance (échantillon)", f"{prices.var():.2f} (kDH)²")
c4.metric("Écart-type (échantillon)", f"{prices.std():.2f} kDH")
st.caption("La variance et l’écart-type affichés utilisent la formule échantillon (ddof=1).")

fig, ax = plt.subplots()
ax.hist(prices, bins=5, edgecolor="white", color="#4472C4")
ax.set(title="Distribution des prix des maisons", xlabel="Prix (kDH)", ylabel="Nombre de maisons")
ax.grid(axis="y", alpha=0.25)
st.pyplot(fig)
plt.close(fig)

st.header("3. Comparer les modèles")
X = data[features]
y = data["prix"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
models = {
    "Régression linéaire": LinearRegression(),
    "Arbre de décision": DecisionTreeRegressor(max_depth=3, random_state=42),
}
results = {}
comparison = pd.DataFrame({"Prix réel (kDH)": y_test})
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    results[name] = {
        "MAE (kDH)": mean_absolute_error(y_test, predictions),
        "R²": r2_score(y_test, predictions),
    }
    comparison[f"Prédit — {name} (kDH)"] = predictions

st.dataframe(pd.DataFrame(results).T.style.format({"MAE (kDH)": "{:.1f}", "R²": "{:.2f}"}), use_container_width=True)
st.caption("Un MAE plus faible indique des erreurs absolues moyennes plus petites. Un R² proche de 1 indique un meilleur ajustement; il peut être négatif sur un petit jeu test.")
st.subheader("Prix prédits et prix réels sur le jeu de test")
st.dataframe(comparison.sort_index().round(1), use_container_width=True)

st.header("4. Estimer une maison")
col1, col2 = st.columns(2)
with col1:
    superficie = st.slider("Superficie (m²)", 30, 200, 85)
    chambres = st.slider("Nombre de chambres", 1, 6, 3)
with col2:
    anciennete = st.slider("Ancienneté (années)", 0, 50, 10)
    distance = st.slider("Distance au centre-ville (km)", 0, 30, 7)
maison = pd.DataFrame([{
    "superficie": superficie,
    "chambres": chambres,
    "anciennete": anciennete,
    "distance_centre_km": distance,
}])
if st.button("Estimer le prix"):
    left, right = st.columns(2)
    for col, (name, model) in zip((left, right), models.items()):
        predicted = model.predict(maison)[0]
        col.success(f"{name} : {predicted:.1f} kDH (environ {predicted * 1000:,.0f} DH)")

st.header("5. Comprendre le machine learning")
st.markdown(
    """
- **X** contient les variables explicatives fournies au modèle : superficie, chambres, ancienneté et distance au centre-ville.
- **y** contient la cible à prédire : ici, le prix.
- **`fit(X_train, y_train)`** apprend les relations entre les caractéristiques des maisons et leurs prix à partir des exemples d’entraînement.
- **`predict(X_test)`** utilise ces relations pour estimer les prix des maisons réservées au test (ou d’une nouvelle maison).
- **MAE** (erreur absolue moyenne) donne en kDH l’écart absolu moyen entre prix estimés et prix réels; plus bas est préférable.
- **R²** mesure la part de variation des prix expliquée par le modèle par rapport à une prédiction constante par la moyenne; plus proche de 1 est généralement préférable. Il peut être négatif.

La régression linéaire ajuste une relation additive; l’arbre de décision sépare les données par seuils et peut représenter des relations non linéaires. Avec seulement huit exemples fictifs, les résultats dépendent fortement du partage train/test et ne permettent pas de conclure qu’un modèle est meilleur en général.
"""
)

st.header("6. Données utiles pour un modèle plus réaliste")
st.markdown(
    """
- Localisation plus précise (quartier, coordonnées, accessibilité aux transports et services).
- Date de vente et évolution du marché, ainsi que des transactions réelles et vérifiées en nombre suffisant.
- État général, rénovations, qualité de construction, étage, ascenseur, parking, balcon et surface du terrain.
- Nombre de salles de bain, type de bien, orientation, vue, chauffage et efficacité énergétique.
- Charges, statut juridique et contexte de la transaction, en respectant la confidentialité et la réglementation applicable.
"""
)
st.caption("Données fictives très limitées : cette démonstration est pédagogique et ne constitue pas une estimation immobilière fiable.")
