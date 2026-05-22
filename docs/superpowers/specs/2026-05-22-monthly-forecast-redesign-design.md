# Repo-Pulse — Refonte vers prévision mensuelle de l'activité GitHub

**Date** : 2026-05-22
**Statut** : Design validé, prêt pour planning
**Auteur** : Naif Asswiel
**Type** : Refonte profonde (même repo, code largement remplacé)

---

## 1. Objectif

Démontrer, via un site web déployé, la capacité à concevoir, fine-tuner et opérer un modèle de prédiction de séries temporelles appliqué à l'activité des dépôts GitHub publics.

Concrètement : un utilisateur colle l'URL d'un repo GitHub, choisit un horizon (1–12 mois), clique "Predict", et obtient une prévision mensuelle du nombre de commits accompagnée d'un backtest comparatif sur les 12 derniers mois.

Le projet vit toujours dans le repo actuel `RepoPulse-`, mais la majeure partie du code existant sera supprimée au profit d'une architecture nettement plus ciblée.

## 2. Critères de succès

- Le site est en ligne sur Hugging Face Spaces et fonctionne pour n'importe quel repo GitHub public.
- Le modèle fine-tuné bat Chronos zero-shot sur les 4 horizons (1, 3, 6, 12 mois) sur 30 repos de validation jamais vus à l'entraînement.
- Le tableau de résultats du README est reproductible en une commande.
- Le README raconte clairement : problème → approche → résultats → limitations.

## 3. Décisions clés

| Décision | Choix | Justification |
|---|---|---|
| Granularité | Mensuelle (vs hebdomadaire) | Réduit le bruit, suffisant pour le signal "activité", aligné avec l'horizon utilisateur. |
| Métrique cible | `commits` | Proxy le plus direct de l'activité, signal le plus fort statistiquement. |
| Horizon | Paramétrable 1–12 mois (UI slider) | Souplesse côté démo, le modèle est entraîné sur 12 mois max. |
| Scope repos | N'importe quelle URL GitHub publique | "Wow effect" pour la démo, contrainte de latence acceptée. |
| Modèle | Fine-tuning de `amazon/chronos-t5-small` via LoRA | Le seul moyen réaliste d'avoir "notre" modèle avec ~60 points par série. Petit adapter (~10 Mo). Comparaison naturelle avec Chronos zero-shot. |
| Comparaison | Ours (FT) vs Chronos zero-shot vs baseline saisonnière naive | Le triplet rend la valeur ajoutée du fine-tuning lisible. |
| Stack web | Gradio + Hugging Face Spaces | Convient au profil ML, déploiement git-push, look moderne avec thème custom, GPU possible. |
| Approche du repo existant | Refonte profonde (suppression agressive du code non lié) | Le code MLOps actuel (Prefect, Dask, A/B testing, registry, 2 dashboards) dilue le message. |

## 4. Architecture

### Vue d'ensemble

```
┌──────────────────────────────────────────────────────────────┐
│  ENTRAÎNEMENT (offline, exécuté une fois sur Colab)          │
│                                                              │
│  training/repos.yaml  (liste curatée ~150 repos)             │
│         │                                                    │
│         ▼                                                    │
│  training/build_dataset.py  ── GitHub API ── parquet         │
│         │                                                    │
│         ▼                                                    │
│  HF Datasets : naifasswiel/github-monthly-commits            │
│         │                                                    │
│         ▼                                                    │
│  training/train.py  ── LoRA fine-tune Chronos-small          │
│         │                                                    │
│         ▼                                                    │
│  HF Hub : naifasswiel/chronos-github-commits (adapter)       │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  INFÉRENCE (live, sur Hugging Face Spaces)                   │
│                                                              │
│  User colle URL ──▶ Gradio app (app.py)                      │
│                          │                                   │
│                          ▼                                   │
│                     src/github_fetch.py                      │
│                          │  (commits paginés, cache 24h)     │
│                          ▼                                   │
│                     src/aggregate.py                         │
│                          │  (série mensuelle UTC)            │
│                          ▼                                   │
│                     src/forecast.py                          │
│                       ├─ Chronos zero-shot                   │
│                       ├─ Chronos fine-tuné (LoRA loaded)     │
│                       └─ Naive seasonal baseline             │
│                          │                                   │
│                          ▼                                   │
│                     src/metrics.py + src/plotting.py         │
│                          │  (backtest 12 derniers mois)      │
│                          ▼                                   │
│                     Réponse Gradio : chart + tableau         │
└──────────────────────────────────────────────────────────────┘
```

### Découpage en unités

Chaque fichier sous `src/` a une responsabilité unique et une interface explicite. Aucun couplage circulaire.

- **`src/github_fetch.py`** — `fetch_commits(owner, repo, token) -> pd.DataFrame[date, sha]`. Gère pagination, rate limit, cache parquet sous `data/cache/{owner}__{repo}.parquet` avec TTL 24h.
- **`src/aggregate.py`** — `to_monthly(commits_df) -> pd.Series[index=Period('M'), values=int]`. Aucune dépendance réseau. Pure transformation. Pad les mois sans commit à 0.
- **`src/forecast.py`** — `forecast(series, horizon, model_id) -> ForecastResult`. Charge le modèle au premier appel (lazy + cached). `ForecastResult` contient `mean`, `lower`, `upper`, `latency_ms`.
- **`src/metrics.py`** — `backtest(series, horizon, models) -> pd.DataFrame[model, smape, mae, latency_ms]`. Held-out sur les 12 derniers mois. Pas de fuite : on tronque la série avant de la passer aux forecasters.
- **`src/plotting.py`** — `make_figure(series, forecasts) -> plotly.Figure`. Aucune logique métier, juste rendering.
- **`app.py`** — Gradio Blocks. Orchestre les appels, gère les états UI (loading, erreur, warning historique court). Pas de logique métier dedans.

### Dépendances entre unités

```
app.py ──▶ github_fetch
       ──▶ aggregate
       ──▶ forecast
       ──▶ metrics
       ──▶ plotting

forecast indépendant des autres (sauf types pandas).
metrics dépend uniquement de forecast.
plotting dépend uniquement de types pandas/plotly.
```

## 5. Structure du code

### Suppressions

```
src/api_server.py
src/dashboard.py
src/dashboard_lite.py
src/inference_dashboard.py
src/inference_dashboard.py.bak
src/ab_testing.py
src/distributed.py
src/orchestration.py
src/model_registry.py
src/model_engine.py
src/model_selection.py
src/neural_network.py
src/train_neural_network.py
src/data_validation.py
visualize_training_losses.py
notebooks/neural_network_training.ipynb
analyse_reponses.md
docs/project_documentation.tex
docs/project_summary.tex
Dockerfile
compose.yaml
.github/workflows/ci-cd.yaml  (à simplifier, voir §10)
```

### Conservés et adaptés

- `src/data_ingestion.py` → renommé `src/github_fetch.py`. On garde la couche `GitHubAPIClient` (auth, retries, pagination), on jette tout le tracking multi-repo, l'incrémental, le concurrent fetching.
- `src/etl.py` → renommé `src/aggregate.py`. On passe en granularité mensuelle (`MS` pandas freq), on n'agrège que `commits`.

### Cible finale

```
repo-pulse/
├── app.py
├── src/
│   ├── __init__.py
│   ├── github_fetch.py
│   ├── aggregate.py
│   ├── forecast.py
│   ├── metrics.py
│   └── plotting.py
├── training/
│   ├── repos.yaml
│   ├── build_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── train.ipynb
├── scripts/
│   └── reproduce.sh
├── data/
│   └── cache/                        (gitignored)
├── tests/
│   ├── test_github_fetch.py
│   ├── test_aggregate.py
│   ├── test_forecast.py
│   └── test_metrics.py
├── docs/
│   └── superpowers/specs/2026-05-22-monthly-forecast-redesign-design.md
├── images/
│   └── shot_demo.png
├── pyproject.toml
├── requirements.txt
├── README.md
├── .gitignore
└── .env.example
```

## 6. Pipeline d'entraînement

### Dataset

- **Source** : 150 repos GitHub publics curés dans `training/repos.yaml`.
- **Critères de curation** : diversité de domaines (ML, web, infra, langages, libs scientifiques), diversité de tailles (1k–100k+ stars), âge ≥ 3 ans pour garantir ≥36 points mensuels par série, exclusion des hard forks.
- **Schéma** : parquet `{repo: str, month: date, commits: int, months_since_start: int}`. Une ligne par mois par repo.
- **Taille attendue** : ~150 × ~80 = ~12k lignes.
- **Publication** : push sur HF Datasets `{HF_USERNAME}/github-monthly-commits`, public. (`{HF_USERNAME}` à confirmer — placeholder remplacé à la première étape du plan.)

### Fine-tuning

- **Modèle de base** : `amazon/chronos-t5-small` (T5 encoder-decoder, 8M paramètres, tokenisation native des valeurs numériques).
- **Méthode** : LoRA via `peft`. Rang `r=8`, `alpha=16`, dropout 0.05. Target modules : couches `q`, `v` de l'attention dans encoder et decoder. ~99% des poids gelés. Adapter résultant ~5–10 Mo.
- **Splits** :
  - Train : 120 repos
  - Validation : 30 repos (jamais vus pendant l'entraînement, fixés dans `repos.yaml`).
- **Génération d'exemples** : sliding window par série. Context length = 36 mois, prediction length = 12 mois, stride = 1.
- **Loss** : cross-entropy native Chronos sur les tokens quantifiés.
- **Optimiseur** : AdamW, lr 1e-4, weight decay 0.01.
- **Schedule** : 3 epochs, batch size effectif 32, warmup 10%.
- **Hardware cible** : Colab T4 gratuit, ~1–2h.
- **Logging** : un seul fichier `training_log.json` (loss train/val par epoch, hyperparams, durée). Pas de MLflow/W&B.
- **Publication** : push de l'adapter sur HF Hub `{HF_USERNAME}/chronos-github-commits`.

### Évaluation

`training/evaluate.py` calcule sur les 30 repos val :

- SMAPE moyen aux horizons 1, 3, 6, 12 mois
- MAE moyen aux mêmes horizons
- Latence d'inférence moyenne (ms)
- Comparaison Ours FT vs Chronos zero-shot vs baseline saisonnière naive

Sortie : `training/results.md` qui sert directement de section "Results" du README.

**Critère de release** : ours bat Chronos zero-shot en SMAPE sur tous les horizons. Sinon, on retravaille la config LoRA avant de publier.

## 7. Application Gradio

### Layout (style hero, validé en brainstorming)

```
Header : repo-pulse | [GitHub link] [HF Spaces link]

Hero :
  H1  "Forecast any GitHub repo"
  Sub "Fine-tuned Chronos transformer · monthly horizon"
  Input  [ github.com/owner/repo            ]  [Predict →]
  Slider Horizon: 1 ──●──── 12 months  (default 6)
  Exemples cliquables : pytorch/pytorch · facebook/react · rust-lang/rust

Results (révélé après clic, fade-in) :
  Subtitle : "{owner}/{repo} · {N} months of history"
  Plotly chart full-width (historique gris, 3 forecasts colorés, IC ombré pour ours)
  Tableau backtest 12 derniers mois (3 lignes × 3 colonnes : SMAPE, MAE, Latency)
  Boutons secondaires : [Download forecast CSV] [Share permalink]

Footer : How it works · Limitations · Source code
```

### Comportement

- **Validation URL** : regex `github\.com/([\w.-]+)/([\w.-]+)/?`. Si invalide, message inline rouge sous le champ.
- **Loading state** : skeleton chart + texte progressif "Fetching commits… → Aggregating → Forecasting (3 models)". Important pour repos de 30s+.
- **Repos avec <24 mois d'historique** : prédiction quand même, mais bandeau d'avertissement "Limited history, forecast confidence is low".
- **Repos privés / 404** : message clair "Repo not found or private — repo-pulse only works on public repos".
- **GitHub rate limit** : exposer "GitHub API limit reached, retry in {minutes}m".
- **Cache hit** : badge discret "cached" si le repo a été fetché dans les 24 dernières heures.
- **Permalink** : URL Spaces avec query params `?repo=owner/repo&horizon=6`.

### Configuration HF Spaces

- SDK : Gradio 4.x
- Hardware : `cpu-basic` par défaut (free), upgradable `cpu-upgrade` si latence inacceptable.
- Secrets : `GITHUB_TOKEN` configuré dans Settings → Secrets, accédé via `os.getenv("GITHUB_TOKEN")`. Jamais commit.
- Header YAML du README pour Spaces : title, emoji, colorFrom/To, sdk, sdk_version, app_file, pinned.

## 8. README

Structure imposée :

1. **Hero** — capture animée du site (gif/webp), lien live Spaces, badges build/license.
2. **What** — un paragraphe : objectif POC, approche fine-tune Chronos, ce que démontre le projet.
3. **Results** — tableau benchmark généré par `evaluate.py`, jamais retouché à la main.
4. **Architecture** — diagramme ASCII de §4.
5. **Reproduce** — trois commandes : `python training/build_dataset.py`, `python training/train.py`, `python app.py`. Plus `bash scripts/reproduce.sh` pour ne re-jouer que l'évaluation.
6. **Limitations & next steps** — repos jeunes, domain shift, pas de multivariate, etc.

## 9. Évaluation publique & reproductibilité

- `scripts/reproduce.sh` : (1) télécharge le dataset depuis HF, (2) charge l'adapter, (3) re-run `evaluate.py`, (4) print la table de résultats. Vérification possible en ~5 minutes par n'importe qui.
- Tag git `v1.0` marque l'état exact correspondant aux chiffres publiés.

## 10. CI/CD

Le workflow actuel `ci-cd.yaml` est sur-dimensionné pour ce POC. On le remplace par un workflow minimal :

- **lint** : ruff + black check sur `src/`, `training/`, `app.py`.
- **tests** : pytest sur `tests/` avec mocks GitHub API.

Pas de training en CI (Colab one-shot), pas de Docker build (HF Spaces gère le déploiement), pas de deploy stages (push sur Spaces se fait par sync de branche).

## 11. Dépendances

`pyproject.toml` réduit drastiquement :

```toml
[project]
name = "repo-pulse"
requires-python = ">=3.10"
dependencies = [
    "gradio>=4.0",
    "chronos-forecasting>=1.4",
    "transformers>=4.40",
    "peft>=0.10",
    "torch>=2.1",
    "pandas>=2.0",
    "pyarrow>=15.0",
    "plotly>=5.18",
    "requests>=2.31",
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
]
[project.optional-dependencies]
dev = ["pytest", "pytest-mock", "ruff", "black"]
```

`requirements.txt` généré pour HF Spaces (sous-ensemble runtime uniquement, pas dev).

## 12. Limites connues du design

- **Domain shift** : entraîné sur des repos OSS populaires, biais vers ce profil. Repos perso à très faible volume → résultats moins bons. Documenté dans README.
- **Pas de multivariate** : la série commits est forecastée sans conditionner sur stars/issues/PRs. Choix conscient pour rester focus. Mentionné comme next step.
- **Horizon max 12 mois** : au-delà, l'incertitude exploserait visuellement et décrédibiliserait la démo.
- **Repos très jeunes (<24 mois)** : forecast affiché mais avec warning UI.
- **Latence sur gros repos** : un repo avec 200k+ commits peut prendre 30s+ à fetcher. Mitigé par cache 24h et UI loading explicite. Ne sera pas résolu davantage.

## 13. Sécurité

- `GITHUB_TOKEN` lu via `os.getenv`, jamais loggé, jamais affiché dans l'UI.
- `.env.example` commité avec placeholders, `.env` dans `.gitignore`.
- HF Spaces Secrets pour le token en production.
- Pas d'auth utilisateur côté app (POC public).
- Validation stricte de l'URL côté serveur avant tout appel API.

## 14. Hors scope (explicite)

- Authentification utilisateur, comptes, historique de prédictions sauvegardé.
- Prédiction multi-métriques (stars/issues/PRs).
- Modèle entraîné from-scratch (rejeté : insuffisant de données par repo, jugé inférieur à un fine-tune Chronos).
- A/B testing, model registry, orchestration Prefect, distributed training. Tout supprimé.
- Dockerfile et docker-compose : remplacés par déploiement HF Spaces.
- Internationalisation de l'UI.
