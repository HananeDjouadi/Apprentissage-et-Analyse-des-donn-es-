import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, make_scorer

# --- FONCTION 1: CRITÈRE D'ÉVALUATION (Question 2 & 6) ---
def custom_metric(y_true, y_pred):
    """Calcule (Accuracy + Precision) / 2"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    return (acc + prec) / 2

custom_scorer = make_scorer(custom_metric)

# --- FONCTION 2: SÉLECTION DE VARIABLES (Question 5) ---
def get_feature_importance(X, y, feature_names, top_n=8):
    """
    Utilise Random Forest pour trouver les meilleures variables.
    Retourne les indices des 'top_n' meilleures colonnes.
    """
    print(f"   -> Sélection des {top_n} meilleures variables...")
    clf = RandomForestClassifier(n_estimators=100, random_state=1)
    clf.fit(X, y)
    
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1] # Tri décroissant
    
    top_indices = sorted_idx[:top_n]
    return top_indices

# --- FONCTION 3: RECHERCHE DES PARAMÈTRES (Question 6) ---
def tune_parameters(X, y):
    """
    Cherche les meilleurs paramètres pour le KNN via GridSearchCV.
    """
    print("   -> Optimisation des hyperparametres (GridSearchCV)...")
    
    # Paramètres à tester (adaptés pour KNN comme vu en Q6)
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    knn = KNeighborsClassifier()
    
    grid = GridSearchCV(
        knn, 
        param_grid, 
        scoring=custom_scorer, 
        cv=5, 
        n_jobs=-1
    )
    
    grid.fit(X, y)
    return grid.best_params_, grid.best_score_

# --- FONCTION 4: CRÉATION DU PIPELINE (Question 7) ---
def create_pipeline(indices_opt, best_params):
    """
    Crée le pipeline: Sélection -> Normalisation -> Classification
    """
    # 1. Sélecteur de colonnes (ne garde que les indices importants)
    feature_selector = ColumnTransformer(
        transformers=[('selector', 'passthrough', indices_opt)],
        remainder='drop'
    )
    
    # 2. Pipeline complet
    pipeline = Pipeline([
        ('feature_selection', feature_selector), # Garde les 8 variables
        ('scaler', StandardScaler()),            # Normalise
        ('classifier', KNeighborsClassifier(**best_params)) # Classifie
    ])
    
    return pipeline