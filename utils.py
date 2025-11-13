import numpy as np
np.set_printoptions(threshold=10000, suppress=True)
import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score, precision_score, recall_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


algorithms = {
    'CART': DecisionTreeClassifier(random_state=1),
    'KNN-5': KNeighborsClassifier(n_neighbors=5),
    'MLP': MLPClassifier(hidden_layer_sizes=(40, 20), random_state=1, max_iter=1000)
}

def evaluate_model(model, X_test, y_test, model_name):
    """
    Évalue un modèle et retourne les métriques
    """
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcul des métriques
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    

    best_metric = recall  # Rappel choisi comme critère métier
    
    # Score final = moyenne entre accuracy et rappel
    final_score = (acc + best_metric) / 2
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    
    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'best_metric': best_metric,
        'final_score': final_score,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

# Une fonction dapprentissage et de test des algorithmes supervisés
def run_classifiers_train_test(X_train, X_test, y_train, y_test):
    """
    Entraîne et évalue tous les classifieurs du dictionnaire
    """
    results = {}
    best_model = None
    best_score = -1
    best_model_name = ""
    
    print("=== APPRENTISSAGE ET ÉVALUATION DES MODÈLES ===")
    print("=" * 60)
    
    for name, model in algorithms.items():
        print(f"\n {name}")
        print("-" * 30)
        
        # Entraînement du modèle
        model.fit(X_train, y_train)
        
        # Évaluation
        eval_results = evaluate_model(model, X_test, y_test, name)
        results[name] = {
            'model': model,
            'metrics': eval_results
        }
        
        # Affichage des résultats
        print(f"Accuracy:    {eval_results['accuracy']:.4f}")
        print(f"Precision:   {eval_results['precision']:.4f}")
        print(f"Recall:      {eval_results['recall']:.4f}")
        print(f"Score final: {eval_results['final_score']:.4f} (Accuracy + Recall)/2")
        print(f"AUC:         {eval_results['roc_auc']:.4f}")
        
        # Mise à jour du meilleur modèle
        if eval_results['final_score'] > best_score:
            best_score = eval_results['final_score']
            best_model = model
            best_model_name = name
    
    # Affichage comparatif
    print("\n" + "=" * 60)
    print("COMPARAISON DES MODÈLES")
    print("=" * 60)
    for name in results:
        score = results[name]['metrics']['final_score']
        print(f"{name:10} : {score:.4f}")
    
    print(f"\n MEILLEUR MODÈLE: {best_model_name} avec un score de {best_score:.4f}")
    
    return best_model, best_model_name, results

# Une fonction dapprentissage et de test des algorithmes supervisés sur des données normalisees
def normalize_data(X_train, X_test, method='standard'):
    """
    Normalise les données d'entraînement et de test
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax': 
        scaler = MinMaxScaler()
    else:
        raise ValueError("Méthode doit être 'standard' ou 'minmax'")
    
    # Fit seulement sur le train, transform sur train et test
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    
    return X_train_normalized, X_test_normalized, scaler

def train_test_with_normalization(X_train, X_test, y_train, y_test):
    """AVEC model.fit() - IMPORTANTE"""  
    X_train_norm, X_test_norm = normalize_data(X_train, X_test)
    return run_classifiers_train_test(X_train_norm, X_test_norm, y_train, y_test)

#une fonction dapprentissage et de test des algorithmes supervisés sur des données incluant des nouvelles variable
def apply_pca_and_concat(X_train, X_test, n_components=3):
    """
    Applique l'ACP et concatène les composantes aux données originales
    """
    # Application de l'ACP
    pca = PCA(n_components=n_components, random_state=1)
    
    # Fit seulement sur le train
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Concaténation avec les données originales
    X_train_combined = np.concatenate([X_train, X_train_pca], axis=1)
    X_test_combined = np.concatenate([X_test, X_test_pca], axis=1)
    
    X_train_pca_combined, X_test_pca_combined, pca, X_train_pca, X_test_pca = apply_pca_and_concat(
    X_train_std, X_test_std, n_components=3)
    return X_train_combined, X_test_combined, pca, X_train_pca, X_test_pca

def train_test_with_pca(X_train, X_test, y_train, y_test):
    """AVEC model.fit() - IMPORTANTE"""
    X_train_pca, X_test_pca = apply_pca_and_concat(X_train, X_test)
    return run_classifiers_train_test(X_train_pca, X_test_pca, y_train, y_test)


