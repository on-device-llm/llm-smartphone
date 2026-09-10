# Annexes du Chapitre 3 : Prototype Minimal de Chatbot Embarqué

**Annexes A à C : Code source du prototype (`chatbot.py`, `benchmark.py`, `utils.py`)**

Le code source complet du prototype décrit dans ce chapitre est publié sur le dépôt public du projet, qui constitue la référence à jour et exécutable :

```
git clone https://github.com/on-device-llm/llm-smartphone
```

Ce dépôt est privilégié ici à une reproduction statique du code parce qu'il reste la version courante et exécutable (y compris les corrections apportées après la rédaction de ce mémoire, par exemple l'incident du flag `--no-conversation` documenté en section 3.8.3), alors qu'un extrait figé en annexe se serait périmé dès la première correction. Il comprend notamment :

- `prototype-cli/chatbot.py` (sections 3.3, 3.4) : point d'entrée du prototype, chargement du modèle, gestion des trois modes (chat, résumé, classification), système de prompt par tâche.
- `prototype-cli/benchmark.py` (section 3.5) : module de benchmark automatisé, basé sur les bindings `llama-cpp-python`.
- `prototype-cli/utils.py` (section 3.6) : structure `InferenceMetrics` et fonctions de mesure/sauvegarde des résultats.
- `docs/protocole_validation_chatbot.md` (section 3.8) : protocole détaillé de validation (Parties A à D), dont l'Annexe E ci-dessous synthétise les résultats.

**Annexe D : Dépendances Python** (`requirements.txt`)

```
# Prototype CLI — LLM Embarqué sur Smartphone
# Testé avec Python 3.9, 3.10, 3.11

# Inférence LLM locale (llama.cpp bindings Python)
llama-cpp-python>=0.2.90

# Mesures de performance
psutil>=5.9.0

# Interface CLI améliorée
rich>=13.7.0
prompt_toolkit>=3.0.43

# Optionnel : export des résultats
pandas>=2.0.0
```

**Annexe E : Synthèse des résultats de validation du prototype**

Résultats bruts obtenus en rejouant le protocole de validation (Parties A à D : questions factuelles et de raisonnement, résumé, classification, observations de session) sur les six appareils du corpus. Modèle : LLaMA 3.2 1B Q4_K_M dans tous les cas. Pour les cinq premiers appareils, le détail question par question est documenté dans le journal de validation associé à ce PFE. Pour le Galaxy A73, sixième et dernier appareil testé, le tableau ci-dessous est reconstitué à partir des observations rapportées dans le corps du chapitre 3 (sections 3.8.1 à 3.8.3), qui n'ont pas fait l'objet d'un journal séparé question par question au moment de la rédaction.

**Galaxy S26 Ultra (Snapdragon 8 Elite) :**

| **Partie** | **Résultat** |
| --- | --- |
| A.1 Factuelles (7) | Run 1 : 5 correctes / 1 partielle / 1 incorrecte — Run 2 : 5 correctes / 0 partielle / 2 incorrectes |
| A.2 Raisonnement (5) | Run 1 : 0 correcte / 1 partielle / 4 incorrectes — Run 2 : 0 correcte / 1 partielle / 4 incorrectes — Run 3 (post-correctif) : 1 correcte / 0 partielle / 4 incorrectes |
| B. Résumé | Run 1 : Bon — Run 2 : Bon — Run 3 : Partiel |
| C. Classification (6) | Run 1 : 6/6 — Run 2 : 6/6 (dont ironie #5 correcte aux deux runs) |
| Saturation du contexte | Confirmée après ~13 échanges (2226 tokens vs 2048) |
| Bugs identifiés | `pandas`, `numpy`/Termux, `llama-cpp-python` Android, `psutil.cpu_percent`, `save_metrics()` — tous corrigés en cours de PIR |

**Galaxy A16 (Exynos 1330) :**

| **Partie** | **Résultat** |
| --- | --- |
| A.1 Factuelles (7) | 5 correctes / 0 partielle / 2 incorrectes |
| A.2 Raisonnement (5) | 0 correcte / 0 partielle / 5 incorrectes |
| B. Résumé | Bon (2 points complets, 2 partiels) |
| C. Classification (6) | 4/6 — échecs sur l'ironie (#5) et le neutre mitigé (#6) |
| Vitesse | Prefill 12,1-52,0 tok/s — Decode 3,9-7,8 tok/s |

**Infinix Hot 60i 5G (Dimensity 6400) :**

| **Partie** | **Résultat** |
| --- | --- |
| A.1 Factuelles (7) | 4 correctes / 1 partielle / 2 incorrectes |
| A.2 Raisonnement (5) | 0 correcte / 0 partielle / 4 incorrectes + 1 non évaluable (saturation du contexte) |
| B. Résumé | Bon (2 points complets, 2 partiels) |
| C. Classification (6) | 1/6 — biais systématique vers NÉGATIF, y compris sur des textes clairement positifs |
| Saturation du contexte | Dès la 12ᵉ question (2117 tokens) |
| Incident technique majeur | `--no-conversation is not supported` (build `llama-cli` `b9-13e6738`) → rappel intégral de l'historique à chaque tour en mode chat ; premier appareil touché par cet incident (voir section 3.8.3) |
| Vitesse | Prefill ~44-56 tok/s — Decode très variable (0,8 à 13,8 tok/s selon la charge de contexte accumulée) |

**Galaxy A26 (Exynos 1280) :**

| **Partie** | **Résultat** |
| --- | --- |
| A.1 Factuelles (7) | 3 correctes / 0 partielle / 4 incorrectes |
| A.2 Raisonnement (5) | 0 correcte / 1 partielle / 4 incorrectes |
| B. Résumé | Partiel (1 point complet, 2 partiels, 1 absent) |
| C. Classification (6) | 3/6 — contradictions internes raisonnement/label, pas de biais directionnel comme sur l'Infinix |
| Saturation du contexte | Non atteinte sur cette session |
| Incident technique majeur | Même incident `--no-conversation` que sur l'Infinix (build `llama-cli` `b4-0eca4d4`) — deuxième appareil consécutif touché |

**Galaxy A71 (Snapdragon 730) :**

| **Partie** | **Résultat** |
| --- | --- |
| A.1 Factuelles (7) | 4 correctes / 0 partielle / 3 incorrectes |
| A.2 Raisonnement (5) | 0 correcte / 1 partielle / 4 incorrectes |
| B. Résumé | Bon (2 points complets, 2 partiels — run de référence en session fraîche) |
| C. Classification (6) | 2/6 — biais total vers NÉGATIF (6/6), combiné à des contradictions internes justification/label sur 4 cas |
| Saturation du contexte | Non atteinte sur cette session |
| Incident technique majeur | Même incident `--no-conversation` (build `llama-cli` `b1-b820cc8`) — troisième appareil consécutif touché |

```{=latex}
\clearpage
```

**Galaxy A73 (Snapdragon 778G) :**

| **Partie** | **Résultat** |
| --- | --- |
| A.1 Factuelles (7) | 2 correctes / 1 partielle / 4 incorrectes — score le plus faible du corpus |
| A.2 Raisonnement (5) | Non isolé par appareil dans le corps du chapitre 3 ; intégré à l'agrégat des sept échantillons (29 réponses incorrectes et 5 partielles sur 35 tentatives, aucune réponse pleinement correcte) |
| B. Résumé | Partiel (1 point complet, 2 partiels, 1 absent — même profil que l'A26) |
| C. Classification (6) | 2/6 — biais total vers NÉGATIF (6/6), combiné à des contradictions internes justification/label sur 3 des 6 cas (même profil que l'A71) |
| Saturation du contexte | Non atteinte sur cette session |
| Incident technique majeur | `--no-conversation is not supported: invalid argument: -no-cnv` (build `llama-cli` `b10739-d08c7872d`) → rejet bloquant du flag (variante plus sévère que sur les trois occurrences précédentes), corrigé par retrait du flag `-no-cnv` dans `chatbot.py` ; cinquième appareil consécutif touché |
| Vitesse | Prefill ~49-64 tok/s — Decode ~7-13 tok/s |

Le protocole complet, incluant l'énoncé exact de chaque question, la réponse générée par le modèle et le verdict associé pour les cinq premiers appareils, ainsi que le détail des incidents techniques rencontrés lors de l'installation (compilation `numpy`/`psutil` sous Termux, compatibilité `llama-cpp-python`/Python 3.14, bug `save_metrics()`), est documenté dans le journal de validation associé à ce PFE. Pour le Galaxy A73, le tableau ci-dessus reprend les observations déjà rapportées dans le corps du chapitre 3 (sections 3.8.1 à 3.8.3), les dix-neuf interactions de sa session ayant été journalisées sans exception dans `results/metrics.json`.

