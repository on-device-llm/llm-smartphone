# Protocole de validation du prototype chatbot (chapitre 3)

> Objectif : obtenir des chiffres réels (pas des estimations) pour combler la section 8.2 "Qualité des réponses" et le tableau 8.1 du chapitre 3, en rejouant exactement le même script sur chaque appareil testé.

## Comment l'utiliser

1. Lancer `chatbot.py` sur l'appareil à tester, avec le modèle standard du corpus (Llama 3.2 1B Q4_K_M).
2. Poser les questions de la Partie A une par une, dans l'ordre, sans redémarrer le programme entre les questions (pour observer aussi la saturation du contexte — voir Partie D).
3. Noter pour chaque question : correcte / partielle / incorrecte, et les métriques automatiques (prefill, decode, latence, RAM) si `save_metrics()` fonctionne.
4. Faire de même pour le résumé (Partie B) et la classification (Partie C).
5. Reporter les résultats dans le tableau de synthèse en bas de ce document, un tableau par appareil.

---

## Partie A — Questions Q/R (12 questions : 7 factuelles simples + 5 raisonnement multi-étapes)

### A.1 — Factuelles simples (réponse courte, vérifiable objectivement)

| # | Question | Réponse attendue |
|---|---|---|
| 1 | Quelle est la capitale de la France ? | Paris |
| 2 | Combien y a-t-il de continents sur Terre ? | 7 (ou 6 selon convention — accepter les deux) |
| 3 | Quel est le symbole chimique de l'eau ? | H2O |
| 4 | En quelle année a eu lieu la Révolution française ? | 1789 |
| 5 | Qui a écrit "Le Petit Prince" ? | Antoine de Saint-Exupéry |
| 6 | Quelle est la plus grande planète du système solaire ? | Jupiter |
| 7 | Combien de jours y a-t-il dans une année bissextile ? | 366 |

### A.2 — Raisonnement multi-étapes (évalue la limite documentée en 1.1 du chapitre 4)

| # | Question | Élément(s) attendu(s) dans la réponse |
|---|---|---|
| 8 | Un train part de Paris vers Lyon (512 km), roulant à 160 km/h. Un autre train part de Lyon vers Paris au même moment à 140 km/h. Après combien de temps se croisent-ils ? | Vitesse combinée 300 km/h → 512/300 ≈ 1h42 (1,71h) |
| 9 | Si un modèle 7B en FP16 nécessite 14 Go de RAM, combien nécessite-t-il approximativement en quantification Q4 (facteur de réduction ~4×) ? | ≈ 3,5 Go |
| 10 | Explique en 2-3 phrases pourquoi le decode d'un LLM est généralement plus lent que le prefill. | Notion memory-bound (decode, un token à la fois) vs compute-bound (prefill, traitement parallèle du prompt) |
| 11 | Un smartphone a 8 Go de RAM, dont 3 Go utilisés en permanence par l'OS/apps. Un modèle 2B quantifié Q4 nécessite ~1,5 Go, et un contexte de 2048 tokens ~500 Mo. Reste-t-il assez de RAM ? Justifie. | 8 − 3 − 1,5 − 0,5 = 3 Go restants → oui, marge suffisante |
| 12 | Compare en 2 phrases les avantages et inconvénients d'une architecture hybride edge+cloud par rapport à une exécution 100% locale. | Doit citer au moins : qualité/latence réseau (hybride) vs confidentialité/hors-ligne (local) |

**Grille de notation** : Correcte (réponse juste et complète) / Partielle (élément juste mais incomplet ou imprécis) / Incorrecte (faux ou hors-sujet).

---

## Partie B — Résumé de texte

**Texte source à soumettre en mode `/résumé` :**

> L'inférence de modèles de langage directement sur smartphone repose sur trois piliers techniques complémentaires. Le premier est la quantification, qui réduit la précision numérique des poids du modèle (de 32 ou 16 bits à 4 bits typiquement), divisant la taille du modèle par 4 à 8 avec une perte de qualité limitée à quelques points de pourcentage sur les benchmarks standards. Le second pilier est l'optimisation des frameworks d'inférence : des projets comme llama.cpp exploitent les instructions ARM NEON spécifiques aux processeurs mobiles pour accélérer les calculs matriciels, tandis que des solutions propriétaires comme ML Kit GenAI exploitent directement le NPU (Neural Processing Unit) présent sur les puces récentes pour un gain de performance supplémentaire. Le troisième pilier est la gestion de la mémoire : la technique du memory-mapping (mmap) permet de charger le modèle depuis le stockage sans dupliquer entièrement son contenu en RAM, ce qui est crucial sur des appareils disposant rarement de plus de 8 à 12 Go de mémoire vive partagée entre le système, les applications et le modèle lui-même. Ensemble, ces trois techniques rendent viable l'exécution de modèles de 1 à 3 milliards de paramètres sur la majorité des smartphones Android commercialisés depuis 2020, ouvrant la voie à des applications conversationnelles fonctionnant entièrement hors ligne.

**Points clés attendus dans le résumé (grille de notation) :**

| Point clé | Présent ? |
|---|---|
| Quantification → réduction taille modèle (÷4 à ÷8), perte qualité limitée | |
| Optimisation frameworks (ARM NEON / NPU) | |
| Gestion mémoire via memory-mapping (mmap) | |
| Conclusion : viabilité de modèles 1-3B sur smartphones depuis 2020 | |

**Notation** : Bon résumé = 3-4 points présents. Partiel = 2 points. Insuffisant = 0-1 point.

---

## Partie C — Classification de sentiment (6 textes, difficulté croissante)

| # | Texte | Label attendu |
|---|---|---|
| 1 | Ce smartphone est excellent, la batterie tient toute la journée. | POSITIF |
| 2 | L'écran s'est cassé après une semaine, je suis très déçu. | NÉGATIF |
| 3 | Le téléphone fonctionne, rien de particulier à signaler. | NEUTRE |
| 4 | Le livreur est arrivé à l'heure prévue et le colis était en bon état. | POSITIF |
| 5 | Alors ça, c'est vraiment ce qu'on appelle un service "rapide"... trois semaines d'attente. | NÉGATIF (ironie — cas difficile, teste la limite documentée en 1.3/8.3) |
| 6 | Le produit correspond à la description, sans plus. | NEUTRE |

**Notation** : label prédit = label attendu → correct ; sinon incorrect. Noter séparément le résultat du #5 (ironie) car c'est un cas de test volontairement piégeux.

---

## Partie D — Observations de session (à noter une seule fois par appareil, sur toute la session)

| Mesure | Valeur observée |
|---|---|
| Temps de chargement initial du modèle (s) | |
| Nombre d'échanges avant saturation du contexte (2048 tokens) | |
| Comportement observé à la saturation (le modèle "oublie" le début ?) | |
| RAM totale utilisée en régime stable | |

---

## Tableau de synthèse — à dupliquer une fois par appareil testé

**Appareil :** _______________  **Date :** _______________  **Modèle :** Llama 3.2 1B Q4_K_M

| Partie | Score |
|---|---|
| A.1 Factuelles (X/7 correctes) | |
| A.2 Raisonnement (X/5 correctes ou partielles) | |
| B. Résumé (points clés présents X/4) | |
| C. Classification (X/6 correctes, dont #5 ironie) | |
| Temps chargement modèle | |
| Échanges avant saturation contexte | |

Une fois rempli pour au moins 1-2 appareils, ces chiffres remplacent les pourcentages estimés ("~80 %", "~87 %") de la section 8.2 du chapitre 3, avec une vraie source citable.
