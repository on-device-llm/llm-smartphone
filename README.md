# PFE — LLMs Embarqués sur Smartphone

> Mémoire de Master Informatique — Intelligence Artificielle
> Solutions Google (Gemini Nano, ML Kit GenAI) et alternatives open source (llama.cpp, MLC-LLM, Gemma)

---

## Structure du dépôt

```
llm-smartphone/
├── docs/
│   ├── chapitre_1_etat_art.md          # Chapitre 1 — État de l'art
│   ├── chapitre_2_mise_en_oeuvre.md    # Chapitre 2 — Mise en œuvre : déploiement et analyse des performances
│   ├── chapitre_3_prototype.md         # Chapitre 3 — Prototype minimal de chatbot embarqué
│   ├── chapitre_4_retour_critique.md   # Chapitre 4 — Retour critique
│   ├── chapitre_*.pdf                  # Versions PDF des chapitres 1 à 4
│   ├── annexe_chapitre2.md             # Annexes du chapitre 2 (extraits ; code complet dans prototype-android/)
│   ├── annexe_chapitre3.md             # Annexes du chapitre 3 (extraits ; code complet dans prototype-cli/)
│   ├── journal_validation_prototype.md # Journal de validation du prototype (.pdf inclus)
│   └── protocole_validation_chatbot.md # Protocole de validation du chatbot (.pdf inclus)
├── results/
│   └── metrics_s26_2026-09-20.json     # Métriques brutes du test du 20/09/2026 (Galaxy S26 Ultra, Llama 3.2 1B Q4_K_M)
├── prototype-cli/
│   ├── chatbot.py                   # Chatbot CLI interactif (pilotage de llama-cli en sous-processus)
│   ├── benchmark.py                 # Script de mesure latence/mémoire
│   ├── utils.py                     # Fonctions utilitaires
│   └── requirements.txt             # Dépendances Python
├── prototype-android/                # App Android (ML Kit GenAI / Gemini Nano)
│   └── app/src/main/
│       ├── java/com/pfe/llmchat/    # Code Kotlin (MainActivity, LlmViewModel, ChatAdapter)
│       └── res/layout/              # Layouts XML
├── scripts/
│   ├── setup_test_termux.sh         # Installation llama.cpp sur Termux (paramétrable par appareil)
│   ├── download_model.sh            # Téléchargement d'un modèle GGUF seul
│   ├── benchmark_complet.sh         # Benchmark complet (latence, RAM, batterie, throttling)
│   └── throttling_rigoureux.sh      # Protocole thermique rigoureux (warm-up + charge 5 min)
└── .github/workflows/
    └── build-apk.yml                # Build manuel de l'APK (déclenchement via workflow_dispatch)
```

> Le reste des livrables (page de garde, résumé, bibliographie, etc.) figure dans le mémoire complet et n'est pas reproduit dans ce dépôt.

---

## Démarrage rapide — Prototype CLI

### Prérequis
- Python 3.9+
- 4 Go de RAM disponibles minimum
- Un modèle GGUF (voir `scripts/download_model.sh`)

### Installation

```bash
cd prototype-cli
pip install -r requirements.txt

# Télécharger un modèle (exemple : Gemma 2 2B Q4)
bash ../scripts/download_model.sh gemma2-2b

# Lancer le chatbot
python chatbot.py --model ../models/gemma-2-2b-it-q4_k_m.gguf

# Mode benchmark
python benchmark.py --model ../models/gemma-2-2b-it-q4_k_m.gguf
```

### Démo rapide (mode mock — sans modèle)

```bash
python chatbot.py --mock
```

---

## Démarrage rapide — Android (Termux)

```bash
# Installation seule
bash scripts/setup_test_termux.sh <nom_appareil>

# Installation + benchmark automatique
bash scripts/setup_test_termux.sh <nom_appareil> --with-benchmark
```

Voir `docs/chapitre_2_mise_en_oeuvre.md` pour le tutoriel complet pas-à-pas, et `scripts/throttling_rigoureux.sh` pour le protocole de mesure thermique détaillé.

---

## Solutions couvertes

| Solution | Type | Framework | Appareil requis |
|---|---|---|---|
| llama.cpp + Gemma 2 2B | Open source | llama.cpp | Tout Android ARM64 |
| ML Kit GenAI + Gemini Nano | Propriétaire | AICore | Pixel 9/10, Galaxy S25/S26 |

---

## Livrables du PFE

- `docs/chapitre_1_etat_art.md` à `docs/chapitre_4_retour_critique.md` (et PDF) — chapitres 1 à 4 du mémoire (version du 20/09)
- `docs/annexe_chapitre2.md`, `docs/annexe_chapitre3.md` — annexes correspondantes (extraits ; le code complet est dans `prototype-android/` et `prototype-cli/`)
- `docs/journal_validation_prototype.md`, `docs/protocole_validation_chatbot.md` — journal et protocole de validation
- `results/` — métriques brutes archivées
- Ce dépôt Git — code + tutoriels reproductibles

---

## Références principales

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — Gerganov, 2023
- [MLC-LLM](https://github.com/mlc-ai/mlc-llm) — MLC AI Contributors, 2023
- [ML Kit GenAI](https://developers.google.com/ml-kit/genai) — Google, 2025
- [Gemma](https://arxiv.org/abs/2403.08295) — Google DeepMind, 2024
- [Xu et al. COTS Benchmark](https://arxiv.org/abs/2410.03613) — 2024
