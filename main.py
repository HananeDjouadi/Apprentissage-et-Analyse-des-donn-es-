"""
Main script for Credit Scoring Analysis
Uses the utility functions from utils.py to train and evaluate credit scoring models
"""

import pandas as pd
import numpy as np
import pickle
from utils2 import (
    run_classifiers_train_test,
    get_feature_importance,
    tune_parameters,
    create_and_save_pipeline,
    pipeline_generation_train_test_split,
    evaluate_model
)
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """
    Load and prepare the credit scoring dataset.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    tuple : (X, y) features and target
    """
    print("=== CHARGEMENT DES DONNÉES ===")
    
    # Try to read with semicolon delimiter first
    df = pd.read_csv(filepath, sep=';')
    
    # If still only one column, the delimiter might be different
    if df.shape[1] == 1:
        print("Tentative avec délimiteur virgule...")
        df = pd.read_csv(filepath, sep=',')
    
    print(f"Shape des données: {df.shape}")
    print(f"Colonnes: {list(df.columns)}")
    print(f"\nPremières lignes:")
    print(df.head())
    
    # Check for missing values
    print(f"\nValeurs manquantes par colonne:")
    print(df.isnull().sum())
    
    # Séparation des features et de la cible
    # La dernière colonne est 'Status' (la cible)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(int)
    
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Distribution de la cible: {np.bincount(y)}")
    print(f"  - Classe 0 (mauvais): {np.sum(y == 0)}")
    print(f"  - Classe 1 (bon): {np.sum(y == 1)}")
    
    return X, y

def test_basic_classifiers(X, y):
    """
    Test des classifieurs de base (Question 2).
    """
    print("\n" + "="*60)
    print("QUESTION 2: Test des classifieurs de base")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=1, stratify=y
    )
    
    # Sans normalisation
    print("\n--- Sans Normalisation ---")
    best_name, best_score = run_classifiers_train_test(
        X_train, X_test, y_train, y_test, normalize=False
    )
    print(f"\nMeilleur modèle: {best_name} (Score: {best_score:.3f})")

def test_with_normalization(X, y):
    """
    Test avec normalisation (Question 3).
    """
    print("\n" + "="*60)
    print("QUESTION 3: Test avec normalisation")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=1, stratify=y
    )
    
    # Avec normalisation
    print("\n--- Avec Normalisation ---")
    best_name, best_score = run_classifiers_train_test(
        X_train, X_test, y_train, y_test, normalize=True
    )
    print(f"\nMeilleur modèle: {best_name} (Score: {best_score:.3f})")

def test_with_pca(X, y):
    """
    Test avec ACP (Question 4).
    """
    print("\n" + "="*60)
    print("QUESTION 4: Test avec ACP")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=1, stratify=y
    )
    
    # Avec normalisation + PCA
    print("\n--- Avec Normalisation + PCA ---")
    best_name, best_score = run_classifiers_train_test(
        X_train, X_test, y_train, y_test, normalize=True, apply_pca=True
    )
    print(f"\nMeilleur modèle: {best_name} (Score: {best_score:.3f})")

def test_feature_selection(X, y):
    """
    Sélection de variables (Question 5).
    """
    print("\n" + "="*60)
    print("QUESTION 5: Sélection de variables")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=1, stratify=y
    )
    
    # Trouver les meilleures variables
    top_indices = get_feature_importance(X_train, y_train, top_n=8)
    
    # Test avec seulement les variables sélectionnées
    X_train_selected = X_train[:, top_indices]
    X_test_selected = X_test[:, top_indices]
    
    print("\n--- Test avec variables sélectionnées ---")
    best_name, best_score = run_classifiers_train_test(
        X_train_selected, X_test_selected, y_train, y_test, normalize=True
    )
    print(f"\nMeilleur modèle: {best_name} (Score: {best_score:.3f})")
    
    return top_indices

def test_hyperparameter_tuning(X, y, top_indices):
    """
    Optimisation des hyperparamètres (Question 6).
    """
    print("\n" + "="*60)
    print("QUESTION 6: Optimisation des hyperparamètres")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=1, stratify=y
    )
    
    # Utiliser seulement les variables importantes
    X_train_selected = X_train[:, top_indices]
    
    # Optimisation
    best_params = tune_parameters(X_train_selected, y_train)
    
    return best_params

def generate_final_pipeline(X, y):
    """
    Génération du pipeline final (Questions 7 & 8).
    """
    print("\n" + "="*60)
    print("QUESTIONS 7 & 8: Pipeline complet")
    print("="*60)
    
    # Lance tout le processus automatisé
    pipeline = pipeline_generation_train_test_split(X, y)
    
    return pipeline

def use_saved_pipeline(filepath, X_new):
    """
    Utilisation d'un pipeline sauvegardé pour faire des prédictions.
    
    Parameters:
    -----------
    filepath : str
        Chemin vers le fichier pickle du pipeline
    X_new : array-like
        Nouvelles données à prédire
        
    Returns:
    --------
    array : Prédictions
    """
    print("\n" + "="*60)
    print("UTILISATION DU PIPELINE SAUVEGARDÉ")
    print("="*60)
    
    # Charger le pipeline
    with open(filepath, 'rb') as f:
        pipeline = pickle.load(f)
    
    print(f"Pipeline chargé depuis '{filepath}'")
    
    # Faire des prédictions
    predictions = pipeline.predict(X_new)
    probabilities = pipeline.predict_proba(X_new)
    
    print(f"\nNombre de prédictions: {len(predictions)}")
    print(f"Distribution des prédictions: {np.bincount(predictions)}")
    
    return predictions, probabilities

def main():
    """
    Fonction principale pour exécuter l'analyse complète.
    """
    print("="*60)
    print("CREDIT SCORING - ANALYSE COMPLÈTE")
    print("="*60)
    
    # 1. Charger les données
    # REMPLACER 'credit_scoring.csv' par le chemin de votre fichier
    try:
        X, y = load_data('credit_scoring.csv')
    except FileNotFoundError:
        print("\nERREUR: Fichier 'credit_scoring.csv' non trouvé!")
        print("Veuillez placer votre fichier de données dans le même répertoire.")
        print("Ou modifier le chemin dans la fonction main().")
        return
    
    # Menu interactif
    print("\n" + "="*60)
    print("MENU D'ANALYSE")
    print("="*60)
    print("1. Test des classifieurs de base (Q2)")
    print("2. Test avec normalisation (Q3)")
    print("3. Test avec ACP (Q4)")
    print("4. Sélection de variables (Q5)")
    print("5. Optimisation des hyperparamètres (Q6)")
    print("6. Génération du pipeline complet (Q7 & Q8)")
    print("7. Analyse complète (toutes les questions)")
    print("8. Utiliser un pipeline sauvegardé")
    print("0. Quitter")
    
    choice = input("\nVotre choix (0-8): ")
    
    if choice == '1':
        test_basic_classifiers(X, y)
    elif choice == '2':
        test_with_normalization(X, y)
    elif choice == '3':
        test_with_pca(X, y)
    elif choice == '4':
        test_feature_selection(X, y)
    elif choice == '5':
        top_indices = test_feature_selection(X, y)
        test_hyperparameter_tuning(X, y, top_indices)
    elif choice == '6':
        generate_final_pipeline(X, y)
    elif choice == '7':
        # Analyse complète
        print("\n" + "="*60)
        print("ANALYSE COMPLÈTE - TOUTES LES QUESTIONS")
        print("="*60)
        
        test_basic_classifiers(X, y)
        test_with_normalization(X, y)
        test_with_pca(X, y)
        top_indices = test_feature_selection(X, y)
        best_params = test_hyperparameter_tuning(X, y, top_indices)
        pipeline = generate_final_pipeline(X, y)
        
        print("\n" + "="*60)
        print("ANALYSE TERMINÉE")
        print("="*60)
        print(f"Pipeline sauvegardé: credit_scoring.pkl")
        
    elif choice == '8':
        pipeline_file = input("Nom du fichier pipeline (défaut: credit_scoring.pkl): ")
        if not pipeline_file:
            pipeline_file = "credit_scoring.pkl"
        
        # Utiliser le jeu de test comme exemple
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1, stratify=y
        )
        
        predictions, probabilities = use_saved_pipeline(pipeline_file, X_test)
        
        # Évaluation
        print("\n--- Évaluation sur le jeu de test ---")
        evaluate_model(y_test, predictions, probabilities[:, 1])
        
    elif choice == '0':
        print("Au revoir!")
    else:
        print("Choix invalide!")

if __name__ == "__main__":
    main()