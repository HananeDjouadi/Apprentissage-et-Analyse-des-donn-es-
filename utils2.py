# **Trinome du TP :**
# - CHARIF Taha Khalil : p2512972
# - DJOUADI Hanane : p2512986
# - MEZAOUROU Sidali : p2515000

# Importation des bibliothèques principales
import pandas as pd  # Pour la manipulation de données tabulaires
import numpy as np   # Pour le calcul numérique et les tableaux multidimensionnels
import matplotlib.pyplot as plt  # Pour les visualisations graphiques
import pickle        # Pour la sérialisation/désérialisation d'objets Python
import time          # Pour mesurer les temps d'exécution

# Importations Scikit-learn pour l'apprentissage automatique
from sklearn.model_selection import GridSearchCV, train_test_split  # Recherche d'hyperparamètres et division des données
from sklearn.preprocessing import StandardScaler, OneHotEncoder      # Prétraitement des données
from sklearn.ensemble import RandomForestClassifier                  # Random Forest pour l'importance des variables
from sklearn.neighbors import KNeighborsClassifier                   # K-plus proches voisins
from sklearn.tree import DecisionTreeClassifier                      # Arbre de décision
from sklearn.neural_network import MLPClassifier                     # Perceptron multicouche
from sklearn.pipeline import Pipeline                                # Création de pipelines de traitement
from sklearn.compose import ColumnTransformer                        # Transformations conditionnelles par colonnes
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc, make_scorer  # Métriques d'évaluation
from sklearn.decomposition import PCA                                # Analyse en Composantes Principales

# --- 1. FONCTIONS D'ÉVALUATION & MÉTRIQUES ---

def custom_metric(y_true, y_pred):
    """
    Calcule (Accuracy + Precision) / 2 ou (Accuracy + Rappel) / 2 selon le métier.
    
    Parameters:
    -----------
    y_true : array-like
        Étiquettes réelles
    y_pred : array-like
        Étiquettes prédites
        
    Returns:
    --------
    float : Score combiné (moyenne entre accuracy et précision)
    """
    # Calcul de l'accuracy (proportion de prédictions correctes)
    acc = accuracy_score(y_true, y_pred)
    # Pour le crédit scoring, le Rappel (Recall) est souvent plus critique que la précision
    # car on veut éviter de rater les mauvais payeurs. 
    # Mais suivons votre consigne précédente (Acc + Prec / 2)
    prec = precision_score(y_true, y_pred, zero_division=0)
    return (acc + prec) / 2

# Création du scorer pour GridSearchCV à partir de la métrique personnalisée
custom_scorer = make_scorer(custom_metric)

def evaluate_model(y_test, y_pred, y_proba=None):
    """
    Affiche les métriques principales d'évaluation.
    
    Parameters:
    -----------
    y_test : array-like
        Étiquettes réelles de test
    y_pred : array-like
        Étiquettes prédites
    y_proba : array-like, optional
        Probabilités prédites pour le calcul de l'AUC
        
    Returns:
    --------
    tuple : (score_final, auc_score)
        Score personnalisé et AUC
    """
    # Calcul de la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    
    # Calcul des métriques de base
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    score_final = (acc + prec) / 2  # Score personnalisé
    
    # Affichage des métriques
    print(f"   Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f}")
    print(f"   Score Perso: {score_final:.3f}")
    # print(f"   Confusion Matrix:\n{cm}")  # Optionnel
    
    # Calcul de l'AUC si les probabilités sont disponibles
    auc_score = 0
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = auc(fpr, tpr)
        print(f"   AUC: {auc_score:.3f}")
        
    return score_final, auc_score

# --- 2. FONCTIONS D'APPRENTISSAGE (Questions 2, 3, 4) ---

def run_classifiers_train_test(X_train, X_test, y_train, y_test, normalize=False, apply_pca=False):
    """
    Fonction générique pour entraîner et tester les algos.
    Gère la normalisation et l'ACP si demandé.
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Caractéristiques d'entraînement et de test
    y_train, y_test : array-like
        Étiquettes d'entraînement et de test
    normalize : bool, default=False
        Si True, applique une normalisation StandardScaler
    apply_pca : bool, default=False
        Si True, applique PCA et concatène les composantes
        
    Returns:
    --------
    tuple : (best_model_name, best_score)
        Nom et score du meilleur modèle
    """
    
    # 1. Prétraitement (Normalisation)
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    # 2. Prétraitement (ACP - après normalisation si applicable)
    if apply_pca:
        # On garde 3 composantes comme demandé dans le TP
        pca = PCA(n_components=3)
        # Concaténation des variables ACP avec les données normalisées
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        X_train = np.concatenate((X_train, X_train_pca), axis=1)
        X_test = np.concatenate((X_test, X_test_pca), axis=1)

    # Dictionnaire des algorithmes (Comme demandé Q2)
    clfs = {
        'CART': DecisionTreeClassifier(random_state=1),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'MLP': MLPClassifier(hidden_layer_sizes=(40, 20), random_state=1, max_iter=1000)
    }
    
    best_score = 0
    best_model_name = ""
    
    print(f"--- Résultat (Normalize={normalize}, PCA={apply_pca}) ---")
    
    # Évaluation de chaque classifieur
    for name, clf in clfs.items():
        print(f"\nAlgorithme: {name}")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Probabilités pour AUC (si disponible)
        y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
        
        score, _ = evaluate_model(y_test, y_pred, y_proba)
        
        # Mise à jour du meilleur modèle
        if score > best_score:
            best_score = score
            best_model_name = name
            
    return best_model_name, best_score

# --- 3. SÉLECTION DE VARIABLES (Question 5) ---

def get_feature_importance(X, y, top_n=8):
    """
    Utilise Random Forest pour trouver les meilleures variables.
    Retourne les indices des 'top_n' meilleures colonnes.
    
    Parameters:
    -----------
    X : array-like
        Caractéristiques
    y : array-like
        Étiquettes
    top_n : int, default=8
        Nombre de variables à sélectionner
        
    Returns:
    --------
    array : Indices des top_n variables les plus importantes
    """
    print(f"\n-> Sélection des variables (RandomForest)...")
    clf = RandomForestClassifier(n_estimators=100, random_state=1)
    clf.fit(X, y)
    
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]  # Tri décroissant
    
    top_indices = sorted_idx[:top_n]
    print(f"   Indices conservés : {top_indices}")
    return top_indices

# --- 4. RECHERCHE DES PARAMÈTRES (Question 6) ---

def tune_parameters(X, y):
    """
    Cherche les meilleurs paramètres pour le KNN via GridSearchCV.
    Attention: X doit contenir uniquement les colonnes sélectionnées.
    
    Parameters:
    -----------
    X : array-like
        Caractéristiques (déjà réduites aux variables importantes)
    y : array-like
        Étiquettes
        
    Returns:
    --------
    dict : Meilleurs paramètres trouvés
    """
    print("\n-> Optimisation des hyperparamètres (GridSearchCV)...")
    
    # Grille de paramètres à tester pour KNN
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    # On utilise KNN car c'est souvent celui choisi dans ce TP, 
    # mais on pourrait passer l'algo en paramètre
    knn = KNeighborsClassifier()
    
    # Configuration de la recherche sur grille
    grid = GridSearchCV(
        knn, 
        param_grid, 
        scoring=custom_scorer, 
        cv=5,  # Validation croisée 5-fold
        n_jobs=-1  # Utilisation de tous les cœurs CPU
    )
    
    # Exécution de la recherche
    grid.fit(X, y)
    print(f"   Meilleurs paramètres: {grid.best_params_}")
    print(f"   Meilleur score CV: {grid.best_score_:.3f}")
    
    return grid.best_params_

# --- 5. CRÉATION DU PIPELINE (Question 7) ---

def create_and_save_pipeline(X, y, indices_opt, best_params, filename="credit_model.pkl"):
    """
    Crée le pipeline: Sélection -> Normalisation -> Classification
    Et le sauvegarde dans un fichier pickle.
    
    Parameters:
    -----------
    X : array-like
        Caractéristiques complètes
    y : array-like
        Étiquettes
    indices_opt : array-like
        Indices des variables à conserver
    best_params : dict
        Meilleurs paramètres pour le classifieur
    filename : str, default="credit_model.pkl"
        Nom du fichier de sauvegarde
        
    Returns:
    --------
    Pipeline : Pipeline sklearn entraîné
    """
    print("\n-> Création et sauvegarde du Pipeline...")
    
    # 1. Sélecteur de colonnes (ne garde que les indices importants)
    # ColumnTransformer s'applique sur les colonnes par index ou nom
    feature_selector = ColumnTransformer(
        transformers=[('selector', 'passthrough', indices_opt)],
        remainder='drop'  # Jette les autres colonnes
    )
    
    # 2. Pipeline complet
    pipeline = Pipeline([
        ('feature_selection', feature_selector),  # Garde les variables top_n
        ('scaler', StandardScaler()),             # Normalise
        ('classifier', KNeighborsClassifier(**best_params))  # Classifie avec les meilleurs paramètres
    ])
    
    # Entraînement final sur l'ensemble des données train
    pipeline.fit(X, y)
    
    # Sauvegarde dans un fichier pickle
    with open(filename, 'wb') as f:
        pickle.dump(pipeline, f)
        
    print(f"   Pipeline sauvegardé sous '{filename}'")
    return pipeline

# --- 6. ORCHESTRATION (Question 8) ---

def pipeline_generation_train_test_split(X, y):
    """
    Fonction principale qui orchestre tout le processus.
    
    Parameters:
    -----------
    X : array-like
        Caractéristiques complètes
    y : array-like
        Étiquettes
        
    Returns:
    --------
    Pipeline : Pipeline sklearn complet et entraîné
    """
    print("=== DÉBUT DE L'ORCHESTRATION ===")
    
    # 1. Split des données (50% comme demandé dans la partie 1, ou 70/30 standard)
    # Le PDF mentionne 50% dans la partie 1.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)
    
    # 2. (Optionnel) Test rapide des algos de base
    # run_classifiers_train_test(X_train, X_test, y_train, y_test, normalize=True)
    
    # 3. Sélection de variables (sur X_train)
    # On choisit arbitrairement de garder les 8 meilleures par exemple
    top_indices = get_feature_importance(X_train, y_train, top_n=8)
    
    # 4. Optimisation des paramètres (sur X_train réduit)
    X_train_reduced = X_train[:, top_indices]
    best_params = tune_parameters(X_train_reduced, y_train)
    
    # 5. Création et sauvegarde du Pipeline
    # On repasse X_train complet, le pipeline se chargera de filtrer les colonnes
    pipeline = create_and_save_pipeline(X_train, y_train, top_indices, best_params, filename="credit_scoring.pkl")
    
    # 6. Validation finale sur le jeu de test
    print("\n-> Validation finale sur le jeu de Test...")
    y_pred_test = pipeline.predict(X_test)
    evaluate_model(y_test, y_pred_test)
    
    print("=== FIN DE L'ORCHESTRATION ===")
    return pipeline