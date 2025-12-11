import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils2 import get_feature_importance, tune_parameters, create_pipeline, custom_metric

def pipeline_generation_train_test_split(csv_path, target_col='status'):
    """
    Fonction principale (Question 8) qui orchestre tout:
    Chargement -> Split -> Sélection -> Tuning -> Pipeline -> Sauvegarde
    """
    print("=== DEBUT DE L'ORCHESTRATION ===")
    
    # 1. Chargement des données
    print("1. Chargement des donnees...")
    df = pd.read_csv(csv_path, sep=';')
    data = df.values
    X = data[:, :-1]
    y = data[:, -1]
    feature_names = df.columns[:-1].tolist()
    
    # 2. Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=1
    )
    
    # 3. Sélection des variables
    indices_opt = get_feature_importance(X_train, y_train, feature_names, top_n=8)
    print(f"   Variables retenues indices: {indices_opt}")
    
    # 4. Optimisation (GridSearch)
    scaler_temp = StandardScaler()
    X_train_reduced = X_train[:, indices_opt]
    X_train_norm = scaler_temp.fit_transform(X_train_reduced)
    
    best_params, best_score = tune_parameters(X_train_norm, y_train)
    print(f"   Meilleurs parametres trouves: {best_params}")
    
    # 5. Création et Entraînement du Pipeline Final
    print("5. Creation du pipeline final...")
    final_pipeline = create_pipeline(indices_opt, best_params)
    
    final_pipeline.fit(X_train, y_train)
    
    # 6. Sauvegarde (Pickle)

    filename = "credit_scoring_pipeline.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(final_pipeline, f)
    
    print(f" Pipeline sauvegarde sous : {filename}")
    
    # Validation rapide
    print("\n--- Validation sur le Test Set ---")
    y_pred = final_pipeline.predict(X_test)
    score = custom_metric(y_test, y_pred)
    print(f"Score final ((Acc+Prec)/2) sur le test set : {score:.4f}")

if __name__ == "__main__":
    pipeline_generation_train_test_split("credit_scoring.csv")