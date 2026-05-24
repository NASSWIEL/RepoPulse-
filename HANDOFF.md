# Repo-Pulse — Session Handoff

> **Pour reprendre ce projet sur une autre machine (avec GPU pour le training).**

---

## TL;DR

- Refonte profonde de `RepoPulse-` en **app Gradio sur Hugging Face Spaces** qui forecast l'activité mensuelle (commits) d'un repo GitHub via un **Chronos-T5-small fine-tuné en LoRA**.
- Code complet et testé sur `main`.
- **Étape en cours** : fetch de 2454 repos (2000 train + 454 val) — peut encore tourner sur la machine Windows, ou à reprendre sur GPU machine.
- 30+ commits, 27 tests passent, lint clean.

---

## Objectifs

**POC** : démontrer la capabilité de bout en bout — concevoir un dataset, fine-tuner un foundation model time-series, opérer une app déployée. Critère de succès : un site live où on colle une URL GitHub et on obtient un forecast mensuel, avec backtest comparant 3 modèles.

**Output final attendu** :
- Site déployé sur HF Spaces : https://huggingface.co/spaces/NaifSaleem/repo-pulse
- Backtest sur 120 repos de validation : **Ours (FT) bat Chronos zero-shot** sur les 4 horizons (1, 3, 6, 12 mois)

---

## État au moment du handoff

### Code

Branche `feat/monthly-forecast-redesign`, 28 commits depuis `main`. Tout est commité.

```
src/
├── github_fetch.py   ── fetch commits + cache 24h + auto-wait sur rate limit
├── aggregate.py      ── commits DataFrame -> monthly Period Series
├── forecast.py       ── ForecastResult, NaiveSeasonal, ChronosZeroShot, ChronosFineTuned
├── metrics.py        ── SMAPE, MAE, backtest (strict train/test split)
└── plotting.py       ── Plotly figure : history + 3 forecasts + IC band

training/
├── repos.yaml                ── 2000 train + 454 val (lambda repos 50-5000 stars, big-tech filtrés)
├── discover_small_repos.py   ── génère repos.yaml via 12 search buckets GitHub (4 star × 3 date ranges)
├── build_dataset.py          ── fetch_monthly_commits_fast (since=5ans, max_pages=30) -> parquet
├── train.py                  ── LoRA fine-tuning de Chronos-T5-small
└── evaluate.py               ── benchmark Ours vs ZS vs naive sur val split

app.py                ── Gradio Blocks (à la racine pour HF Spaces convention)
scripts/
├── push_to_hf.py    ── upload dataset + adapter sur HF Hub
└── reproduce.sh     ── pull from HF + re-run eval

docs/superpowers/
├── specs/2026-05-22-monthly-forecast-redesign-design.md
└── plans/2026-05-22-monthly-forecast-redesign.md
```

### Fetch en cours (ou interrompu)

- **2454 repos** dans `training/repos.yaml` (2000 train + 454 val)
- Filtrés : big-tech orgs exclus (Azure, Google, AWS, Intel...), bots exclus (>300 commits/mois)
- Script optimisé : `fetch_monthly_commits_fast` avec `since=5ans, max_pages=30` → ~5-10s/repo
- Cache : `data/cache/<owner>__<repo>.monthly.parquet` (TTL 24h)
- Le cache N'EST PAS commité (gitignored). Le fetch repart de zéro sur une nouvelle machine.
- **ETA fetch complet** : ~16h sur réseau corporate (2 repos/min), ~3-4h sur réseau rapide

Pour reprendre le fetch :
```bash
python training/build_dataset.py --split train
python training/build_dataset.py --split validation
```

### Tests

```
pytest tests/ -m "not slow"  ──>  27 passed, 2 deselected
ruff check src/ training/ app.py  ──>  All checks passed
black --check src/ training/ app.py  ──>  11 files unchanged
```

Les 2 tests `slow` (Chronos zero-shot + Chronos fine-tuned) nécessitent download huggingface.co.

---

## Setup sur la nouvelle machine (GPU)

### 1. Clone et checkout

```bash
git clone https://github.com/NASSWIEL/RepoPulse-.git
cd RepoPulse-
git checkout main
```

### 2. Python env

Linux/Mac avec Python 3.10+ :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,training]"
```

Vérification :

```bash
pytest tests/ -m "not slow" -v
# attendu: 27 passed
```

### 3. Créer `.env` (à la racine)

```bash
cat > .env << 'EOF'
GITHUB_TOKEN=<ton_token_github_pat>
HF_USERNAME=NaifSaleem
HF_TOKEN=<ton_token_hugging_face>
GITHUB_VERIFY_SSL=true
EOF
```

**Notes** :
- `GITHUB_VERIFY_SSL=true` (par défaut) sur la nouvelle machine, sauf si elle est aussi derrière un proxy d'entreprise.
- `HF_TOKEN` : crée-le sur https://huggingface.co/settings/tokens avec scope `write` (besoin pour push dataset + model).
- **Le token GitHub de cette session devrait être rotaté** — il est passé dans des transcripts.

### 4. (Optionnel) Vérifier l'accès GitHub API

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
import os, requests
r = requests.get('https://api.github.com/rate_limit',
                 headers={'Authorization': f'Bearer {os.getenv(\"GITHUB_TOKEN\")}'})
print(r.status_code, r.json()['resources']['core'])
"
# attendu: 200, remaining 5000/5000
```

---

## Étapes à faire (dans l'ordre)

### Étape 1 — Fetch dataset (CPU, ~1h pour 480 + ~15 min pour 120)

```bash
# Train split (480 repos)
python -u training/build_dataset.py --split train \
    --output data/training_dataset_train.parquet \
    2>&1 | tee fetch_train.log

# Validation split (120 repos)
python -u training/build_dataset.py --split validation \
    --output data/training_dataset_validation.parquet \
    2>&1 | tee fetch_val.log
```

**Outputs attendus :**
- `data/training_dataset_train.parquet` (~12-15k lignes)
- `data/training_dataset_validation.parquet` (~3-4k lignes)
- Le script log un message par repo et écrit un checkpoint parquet tous les 25 repos.

### Étape 2 — Push dataset sur HF Datasets

```bash
python scripts/push_to_hf.py
```
Crée le repo `NaifSaleem/github-monthly-commits` sur HF Datasets. Si tu préfères régénérer ailleurs, c'est optionnel.

### Étape 3 — Fine-tuning (GPU, ~1-2h sur T4)

```bash
python training/train.py \
    --train-parquet data/training_dataset_train.parquet \
    --val-parquet data/training_dataset_validation.parquet \
    --output-dir models/chronos-github \
    --epochs 3
```

**Outputs :**
- `models/chronos-github/adapter_model.safetensors` (~10MB LoRA adapter)
- `models/chronos-github/adapter_config.json`
- `models/chronos-github/training_log.json` (loss train/val par epoch)

**Critère de check** : la val_loss doit baisser d'epoch en epoch. Sinon, tuner `lora_r`, `learning_rate`, `epochs`.

### Étape 4 — Évaluation

```bash
python training/evaluate.py \
    --val-parquet data/training_dataset_validation.parquet \
    --adapter models/chronos-github \
    --out training/results.md
```

**Critère de release** : dans `training/results.md`, `Ours (FT)` SMAPE < `Chronos ZS` SMAPE sur les 4 horizons (1, 3, 6, 12 mois). Sinon, retraine avec un meilleur config.

### Étape 5 — Push adapter sur HF Hub

Re-run `python scripts/push_to_hf.py` après que `models/chronos-github/` existe. Crée `NaifSaleem/chronos-github-commits` (modèle).

### Étape 6 — Coller les vraies métriques dans README

Édite `README.md`, remplace le tableau "Results" avec les 4 lignes de `training/results.md`.

### Étape 7 — Déployer le Space

1. Sur https://huggingface.co/new-space → name `repo-pulse` → Gradio SDK → CPU basic → Public.
2. Settings → Variables and secrets :
   - `GITHUB_TOKEN` (avec scope `public_repo`)
   - `HF_USERNAME=NaifSaleem`
3. Add Space remote :
   ```bash
   git remote add space https://huggingface.co/spaces/NaifSaleem/repo-pulse
   ```
4. Push :
   ```bash
   git push space feat/monthly-forecast-redesign:main
   ```
5. Attendre que le Space build (5-10 min). Tester avec `pytorch/pytorch`.

### Étape 8 — Demo screenshot

Take screenshot of the live Space → save as `images/shot_demo.png` (overwrite).

```bash
git add images/shot_demo.png
git commit -m "docs: live deploy screenshot"
git push origin feat/monthly-forecast-redesign
git push space feat/monthly-forecast-redesign:main
```

### Étape 9 — Merge sur main + tag v1.0

```bash
git checkout main
git merge --no-ff feat/monthly-forecast-redesign -m "Merge: monthly forecast redesign v1.0"
git tag -a v1.0 -m "Initial public release"
git push origin main --tags
```

---

## Documents de référence (à lire avant de continuer)

- **Spec** : `docs/superpowers/specs/2026-05-22-monthly-forecast-redesign-design.md` — toutes les décisions techniques, l'architecture, les contraintes.
- **Plan** : `docs/superpowers/plans/2026-05-22-monthly-forecast-redesign.md` — décomposition en tâches avec TDD pour chaque.
- **README** : à mettre à jour avec les métriques après training.

---

## Décisions techniques importantes (rappel)

| Sujet | Décision | Pourquoi |
|---|---|---|
| Granularité | Mensuelle (vs hebdo) | Réduit le bruit, suffisant pour le signal d'activité. |
| Métrique cible | `commits` | Proxy direct, signal fort. |
| Horizon | Paramétrable 1-12 mois | Slider Gradio. Training sur 12 mois max. |
| Modèle base | `amazon/chronos-t5-small` (T5, 8M params) | Foundation model time-series, tokenisation native des nombres. |
| Méthode FT | LoRA via `peft`, r=8, alpha=16, q+v projections | Adapter ~10MB, GPU léger, garde 99% des poids gelés. |
| Dataset | 600 small repos (200-600 stars, 3+ ans) | Évite les méga-repos qui prennent des heures à fetcher et capèrent le `max_pages`. |
| Comparaison | Ours FT vs Chronos ZS vs naive seasonal | 3 modèles côte à côte rend la valeur ajoutée du FT lisible. |
| Stack web | Gradio + HF Spaces | Convention ML, déploiement git-push, look moderne. |

---

## Points d'attention / gotchas

- **`max_pages=250`** dans `build_dataset.py` cap les méga-repos à ~25k commits = ~20 mois min, ce qui peut faire skip certains (filter MIN=24 mois). C'est OK pour les small repos qu'on a curé maintenant (200-600 stars).
- **Cache TTL = 24h** dans `github_fetch.py`. Si tu re-runs build_dataset dans la même journée, les repos déjà fetchés sont skipped (lecture parquet).
- **`GITHUB_VERIFY_SSL=false`** était nécessaire sur la machine corporate de la session précédente (proxy d'entreprise). Sur ta nouvelle machine GPU, laisse à `true` (défaut).
- **HF Spaces `app_file=app.py`** est défini dans le front-matter YAML du README. Ne pas renommer `app.py`.
- **Train ne peut pas tourner CPU** raisonnablement — il faut un GPU (T4 Colab gratuit suffit).

---

## Stats globales de la session

- 28 commits sur la branche
- Suppression : 13 481 lignes (legacy MLOps : MLflow, Prefect, Dask, A/B testing, dashboards, neural net hand-coded, etc.)
- Ajout : ~3 500 lignes (src + training + app + tests + docs)
- 27 tests unitaires verts
- 6 modules `src/` indépendants, single-responsibility chacun
- Spec + plan détaillés en français dans `docs/superpowers/`

---

Bon training sur GPU.
