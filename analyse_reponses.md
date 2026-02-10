# Analyse du Code - Réponses aux Questions

## 1. Étiquetage des données

### Réponse :
Les données sont étiquetées de manière **supervisée** pour un apprentissage sur séries temporelles.

### Implémentation :
- **Fichier** : `src/neural_network.py`, méthode `_create_sequences()` (lignes 230-295)
- **Mécanisme** : 
  - Pour chaque échantillon, on utilise une fenêtre glissante de taille `lookback_window`
  - **Features (X)** : valeurs de commits des semaines `[t-lookback, ..., t-1]`
  - **Label (y)** : valeur de commits à la semaine `t`
  
### Code pertinent :
```python
# src/neural_network.py, ligne 294-295
X = np.hstack(feature_list)
y = np.array([commits[i + lookback] for i in range(n_samples)])
```

### Conclusion :
Il s'agit d'un **apprentissage supervisé sur séries temporelles** sans labels externes. Le label est automatiquement généré à partir de la valeur suivante dans la série temporelle.

---

## 2. Normalisation des données

### Réponse :
La méthode de normalisation appliquée est **Min-Max Scaling**, **PAS** une normalisation statistique.

### Formule exacte implémentée :
```
Valeur normalisée = (Valeur originale - Minimum) / (Maximum - Minimum)
```

Cette formule ramène les valeurs dans l'intervalle **[0, 1]**.

### Implémentation :
- **Fichier** : `src/neural_network.py`, méthode `_normalize_data()` (lignes 298-315)
- **Code** :
```python
from sklearn.preprocessing import MinMaxScaler

def _normalize_data(self, data: np.ndarray, key: str, fit: bool = True) -> np.ndarray:
    """Normalize data using min-max scaling."""
    from sklearn.preprocessing import MinMaxScaler
    
    data = data.reshape(-1, 1) if len(data.shape) == 1 else data
    
    if fit:
        self.scalers[key] = MinMaxScaler()
        return self.scalers[key].fit_transform(data).flatten()
    else:
        if key not in self.scalers:
            return data.flatten()
        return self.scalers[key].transform(data).flatten()
```

### Vérification :
**NON**, cela ne correspond **PAS** à une normalisation statistique (moyenne nulle, variance unitaire).
- Normalisation statistique : `(X - μ) / σ` → intervalle non borné
- Min-Max scaling (utilisée) : `(X - min) / (max - min)` → intervalle [0, 1]

**Note** : Pour les features engineered, il y a une utilisation de `StandardScaler` (normalisation statistique) dans certains cas (ligne 414 de `neural_network.py`), mais pour les valeurs de commits brutes, c'est bien MinMaxScaler qui est utilisé.

---

## 3. Fenêtre temporelle (sliding window)

### Réponse :
La taille de la fenêtre varie selon la configuration :
- **Par défaut dans `neural_network.py`** : `lookback_window = 24` semaines
- **Par défaut dans `train_neural_network.py`** : `lookback_window = 12` semaines  
- **Dans le notebook** : `LOOKBACK_WINDOW = 12` semaines

### Implémentation :
- **Fichier** : `src/neural_network.py`, méthode `_create_sequences()` (lignes 230-295)

### Code :
```python
def _create_sequences(
    self, 
    commits: np.ndarray, 
    prs: Optional[np.ndarray] = None,
    lookback: int = 24
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input sequences for the neural network with feature engineering.
    """
    n_samples = len(commits) - lookback
    
    # Raw lookback features
    raw_features = []
    for i in range(n_samples):
        raw_features.append(commits[i:i + lookback])  # Fenêtre glissante
    raw_features = np.array(raw_features)
    
    # Target values
    y = np.array([commits[i + lookback] for i in range(n_samples)])
    
    return X, y
```

### Mécanisme :
Pour créer les séquences :
1. On parcourt la série temporelle avec une fenêtre glissante de taille `lookback_window`
2. Pour chaque position `i` :
   - **Input** : `commits[i:i+lookback]` (ex: semaines 0 à 11)
   - **Output** : `commits[i+lookback]` (ex: semaine 12)
3. La fenêtre se déplace d'un pas à chaque itération

---

## 4. Méthodes de référence (baselines)

### Réponse :
**NON**, les deux méthodes de baseline ne sont **PAS implémentées** dans le code.

### Recherche effectuée :
- Recherche de mots-clés : `persistence`, `last value`, `baseline`, `moving average`, `naive`
- Fichiers examinés : tous les fichiers Python du projet
- Résultat : Aucune implémentation explicite trouvée

### Conclusion :
- **Méthode 1 (Persistance/Last Value)** : ❌ Non implémentée
- **Méthode 2 (Moyenne mobile/Moving Average)** : ❌ Non implémentée

**Note** : Il y a bien une mention de "Last value" dans les features engineered (ligne 206 de `neural_network.py`), mais ce n'est pas une méthode baseline de prédiction, c'est simplement une feature ajoutée au modèle.

---

## 5. Architecture du modèle

### Réponse IMPORTANTE :
Le modèle implémenté n'est **PAS un GRU**, c'est un **MLP (Multi-Layer Perceptron)** simple.

### Architecture complète :

#### Version PyTorch (fichier `train_neural_network.py`, lignes 304-312) :
```python
# Build PyTorch model
layers = []
input_size = X_train.shape[1]  # Dépend du lookback et des features
for hidden_size in hidden_layers:
    layers.append(nn.Linear(input_size, hidden_size))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(self.config.dropout_rate))
    input_size = hidden_size
layers.append(nn.Linear(input_size, 1))
model = nn.Sequential(*layers)
```

#### Structure typique :
```
Input Layer (size = lookback_window × nombre_de_séries)
    ↓
Linear(input_size → hidden_layer[0])
    ↓
ReLU()
    ↓
Dropout(rate=0.2-0.4)
    ↓
Linear(hidden_layer[0] → hidden_layer[1])
    ↓
ReLU()
    ↓
Dropout(rate=0.2-0.4)
    ↓
[... répété pour chaque couche cachée ...]
    ↓
Linear(hidden_layer[-1] → 1)
    ↓
Output (1 valeur : prédiction des commits)
```

### Dimensions des couches cachées :
Plusieurs configurations possibles selon les paramètres :

#### Configuration 1 (par défaut auto-scale avec peu de données) :
- **Couches** : [64, 32]
- **Exemple complet** :
  - Input: 24 (si lookback=24, sans PRs)
  - Hidden 1: 64 neurones
  - Hidden 2: 32 neurones  
  - Output: 1 neurone

#### Configuration 2 (notebook) :
- **Couches** : [128, 64, 32]
- **Exemple complet** :
  - Input: 12 (si lookback=12, sans PRs)
  - Hidden 1: 128 neurones
  - Hidden 2: 64 neurones
  - Hidden 3: 32 neurones
  - Output: 1 neurone

### Nombre total de paramètres entraînables :
Pour calculer le nombre de paramètres, on utilise la formule pour chaque couche Linear :
```
Paramètres = (input_size × output_size) + output_size (bias)
```

#### Exemple concret (configuration [64, 32] avec lookback=24, sans PRs) :
1. **Linear(24 → 64)** : (24 × 64) + 64 = **1,600 paramètres**
2. **Dropout** : 0 paramètres
3. **Linear(64 → 32)** : (64 × 32) + 32 = **2,080 paramètres**
4. **Dropout** : 0 paramètres
5. **Linear(32 → 1)** : (32 × 1) + 1 = **33 paramètres**

**Total : 3,713 paramètres entraînables**

#### Exemple avec PRs (input = 24×2 = 48 features) :
1. **Linear(48 → 64)** : (48 × 64) + 64 = **3,136 paramètres**
2. **Linear(64 → 32)** : (64 × 32) + 32 = **2,080 paramètres**
3. **Linear(32 → 1)** : (32 × 1) + 1 = **33 paramètres**

**Total : 5,249 paramètres entraînables**

---

## 6. Implémentation du modèle

### Réponse :
Le modèle n'est **PAS un GRU**. C'est un **MLP feedforward simple**.

### Caractéristiques :
- **Type** : MLP simple (Multi-Layer Perceptron)
- **Non empilé** : Pas de stacked layers (contrairement aux stacked LSTM/GRU)
- **Non bidirectionnel** : Unidirectionnel (feedforward uniquement)

### Mécanismes supplémentaires présents :

#### 1. Dropout (fichier `neural_network.py`, ligne 336)
```python
layers.append(nn.Dropout(self.config.dropout_rate))
```
- Taux par défaut : 0.4 (ligne 44 de `neural_network.py`)
- Active pendant l'entraînement, désactivée pendant l'inférence

#### 2. Batch Normalization (optionnel)
- **Configuration** : `use_batch_norm: bool = True` (ligne 50)
- Mais **non implémenté** dans le code PyTorch actuel
- Implémenté uniquement dans le notebook (lignes 89-90) :
```python
layers.append(nn.BatchNorm1d(hidden_size))
```

#### 3. Ensemble de modèles (ligne 521 de `neural_network.py`)
```python
# Train ensemble with different random seeds
for i in range(self.config.n_ensemble):
    torch.manual_seed(42 + i)
    np.random.seed(42 + i)
    model = self._train_torch_model(X, y, hidden_layers)
    self.models.append(model)
```
- **n_ensemble = 5** par défaut
- Utilisé pour calculer des intervalles de confiance

#### 4. L2 Regularization
- **Configuration** : `l2_regularization: float = 0.01` (ligne 49)
- Appliqué dans sklearn MLPRegressor (ligne 428 de `neural_network.py`) :
```python
nn_model = MLPRegressor(
    ...
    alpha=self.config.l2_regularization,  # L2 regularization
    ...
)
```

#### 5. Noise Injection (ligne 417)
```python
if self.config.noise_injection > 0:
    noise = np.random.normal(0, self.config.noise_injection, X_scaled.shape)
    X_noisy = X_scaled + noise
```
- Taux par défaut : 0.05
- Appliqué pendant l'entraînement pour robustesse

#### 6. Learning Rate Scheduling (ligne 317 de `train_neural_network.py`)
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
```

#### 7. Early Stopping (ligne 322)
```python
best_val_loss = float('inf')
patience_counter = 0
# ...
if patience_counter >= self.config.early_stopping_patience:
    logger.info(f"Early stopping at epoch {epoch + 1}")
    break
```
- Patience par défaut : 30 epochs

### Version fallback (sklearn) :
Quand PyTorch n'est pas disponible, le code utilise un **ensemble de modèles** (ligne 439-450) :
- MLPRegressor (neural network)
- GradientBoostingRegressor
- Ridge regression

---

## 7. Détail de la normalisation

### Formule :
```
Valeur normalisée = (Valeur originale - Minimum) / (Maximum - Minimum)
```

**Note** : Ce n'est pas la formule demandée `(X - μ) / σ` car le code utilise MinMaxScaler, pas StandardScaler.

### Sur quelles données sont calculés min et max ?

#### Réponse : **Sur le jeu d'entraînement uniquement**

### Implémentation détaillée :

#### Fichier : `src/train_neural_network.py`, lignes 233-240
```python
# Normalize data
commit_scaler = MinMaxScaler()
train_commits_norm = commit_scaler.fit_transform(
    train_commits.reshape(-1, 1)
).flatten()

all_commits_norm = commit_scaler.transform(
    all_commits.reshape(-1, 1)
).flatten()
```

### Processus :
1. **Entraînement** :
   - `fit_transform()` sur `train_commits` → calcule min et max sur les données d'entraînement
   - Les valeurs min et max sont stockées dans `commit_scaler`

2. **Validation/Test** :
   - `transform()` sur les données de validation → utilise les min et max calculés sur l'entraînement
   - **PAS de nouveau calcul** de statistiques

3. **Inférence** :
   - Les mêmes min et max de l'entraînement sont utilisés (ligne 550 de `neural_network.py`) :
```python
commits_norm = self._normalize_data(commits, "commits", fit=False)
```

### Pourquoi c'est important :
- Évite la **fuite d'information (data leakage)** des données de validation vers l'entraînement
- Garantit que le modèle voit des données dans la même échelle en production
- Standard en machine learning pour les séries temporelles

### Stockage des scalers :
Les scalers sont sauvegardés avec le modèle (ligne 432 de `train_neural_network.py`) :
```python
"scalers": {
    "commits": commit_scaler,
    "prs": prs_scaler
},
```

---

## Résumé des Points Clés

### ✅ Ce qui est implémenté :
1. Apprentissage supervisé sur séries temporelles
2. Min-Max Scaling (normalisation [0,1])
3. Fenêtre glissante de 12 ou 24 semaines
4. **MLP simple** (pas de GRU)
5. Dropout, L2 regularization, noise injection
6. Ensemble de modèles pour intervalles de confiance
7. Early stopping et learning rate scheduling

### ❌ Ce qui N'est PAS implémenté :
1. **GRU ou LSTM** (c'est un MLP)
2. **Méthodes baseline** (Persistence, Moving Average)
3. Normalisation statistique (μ=0, σ=1) pour les commits
4. Architecture bidirectionnelle
5. Stacked/empilé layers (au sens RNN)

### 📊 Configuration typique :
```
Input: 24 features (lookback=24 weeks)
   ↓
Dense(24 → 64) + ReLU + Dropout(0.4)
   ↓
Dense(64 → 32) + ReLU + Dropout(0.4)
   ↓
Dense(32 → 1)
   ↓
Output: 1 prediction

Total: ~3,713 paramètres
Normalisation: MinMaxScaler fit sur train
Early stopping: patience=30
Learning rate: 0.0005
```

---

## Sources du code analysé

- `src/neural_network.py` - Architecture du modèle MLP
- `src/train_neural_network.py` - Script d'entraînement
- `notebooks/neural_network_training.ipynb` - Notebook de démonstration
- Configuration par défaut : `NeuralNetworkConfig` (lignes 31-91 de `neural_network.py`)
