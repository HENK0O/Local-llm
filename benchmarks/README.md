# Mesures des optimisations

Les rapports conservés comparent des versions ou des options de local-llm sur
les mêmes poids SmolLM2-360M-Instruct Q8_0, sur un MacBook Air Apple M5.
Ils ne démontrent pas une supériorité sur LM Studio ou llama.cpp.
Les temps dépendent de la charge, de la température et des fréquences du CPU.

| Mesure | Données brutes | Contrôle |
|---|---|---|
| Traitement du prompt | [inference-q8.json](inference-q8.json) | 3 essais alternés, mêmes tokens gloutons |
| Décodage natif | [decode-before.json](decode-before.json), [decode-after.json](decode-after.json) | 5 essais, mêmes 64 tokens gloutons |
| Réutilisation du contexte | [prefix-cache.json](prefix-cache.json) | 5 essais alternés, mêmes tokens gloutons |

## Prefill

Comparaison de `model.py` et `generation.py` avant/après les optimisations
de l’attention et de la projection du vocabulaire à la dernière position :

| Tokens du prompt | Avant | Après | Accélération |
|---:|---:|---:|---:|
| 25 | 0,145 s | 0,118 s | 1,23× |
| 97 | 0,707 s | 0,452 s | 1,56× |
| 193 | 1,868 s | 0,951 s | 1,96× |

Pour reproduire le protocole depuis la racine du dépôt :

```bash
.venv/bin/python scripts/benchmark_inference.py \
  models/SmolLM2-360M-Instruct.official.Q8_0.gguf \
  --baseline-ref 14994f4b8401f7efcc757587ffb7225c4a9d5452 \
  --runs 3 --tokens 16 --output /tmp/inference-comparison.json
```

Le script exécute les deux modules du commit de référence : choisir uniquement
un commit de confiance. Les poids et les kernels sont partagés avec la version
courante. Les réponses peuvent s’arrêter avant la borne si le modèle émet EOS.
Ces mesures portent sur la préparation, pas sur le débit de génération.

## Decode

Le 1er octobre 2026, le prompt `Once upon a time` produit les mêmes 64 tokens
avant/après. Le débit médian passe de **89,73 à 106,14 tok/s (+18,28 %)**.
Le cache KV garde la même taille : 5 570 560 octets.

La référence décrit l’arbre de travail au début de cette session, déjà modifié,
et non un commit isolé. Les kernels ont été reconstruits entre les mesures.
Le débit mesure les 63 passages de décodage, hors prefill, sampling et HTTP.
Les arrondis des réductions peuvent différer : l’identité observée des tokens
ne garantit pas une identité sur tous les prompts et checkpoints.

Pour mesurer la version actuelle :

```bash
python -m local_llm benchmark \
  models/SmolLM2-360M-Instruct.official.Q8_0.gguf \
  --prompt 'Once upon a time' --tokens 64 --runs 5 \
  --output /tmp/current.json --json
```

Ajouter `--compare /chemin/before.json` pour comparer à un rapport obtenu sur la
même machine. La commande vérifie les poids, le prompt, les tokens, le nombre
d’essais et l’environnement. Les minimums et maximums restent dans le rapport.

## Contexte

Le 2 octobre 2026, le second tour d’une conversation réutilise **517 des 546
tokens d’entrée**. Le prefill médian passe de **2,778 s à 0,159 s (-94,3 %)**,
et le temps total de 3,144 s à 0,533 s. Tous les tokens de sortie sont identiques.
La référence est le même moteur avec la réutilisation du préfixe désactivée.
Le cache conservé est borné à 64 Mio.

```bash
.venv/bin/python scripts/benchmark_prefix_cache.py \
  models/SmolLM2-360M-Instruct.official.Q8_0.gguf \
  --runs 5 --tokens 32 --output /tmp/prefix-cache.json
```

Cela accélère la préparation d’un contexte déjà calculé. Aucun gain universel
de décodage n’en est déduit.

## Mesures dans l’interface

Le chat affiche le débit global observé dans le navigateur : tokens générés
divisés par la durée complète de la requête, préparation et transport inclus.
Le libellé « N tok réutilisés » indique les tokens du contexte dont le recalcul
a été évité par le moteur, sans attribuer ce mécanisme à local-llm ni annoncer
une accélération face à un autre moteur.

Les comparaisons sont manuelles, dans l’onglet Performances. Le témoin NumPy
déquantifie les blocs lors de chaque projection ; il peut être très lent et ne
représente pas un moteur standard. Aucun pourcentage d’accélération n’en est
déduit. La comparaison LM Studio reste indicative tant que les poids, la
quantification, le placement CPU/GPU et la définition du débit ne concordent pas.

## Ling Tiny : chargement et absence de gain validé

Le 3 octobre 2026, Ling-3.0-tiny-Heretic-NX-PRIME-Q8_0 (7,83 Gio de fichier)
charge sur Apple M5 / 24 Gio avec un budget déclaré inférieur à 7,9 Gio. La
tentative contrôlée utilise 2 048 tokens de contexte, batch 256 / ubatch 128,
KV F16, un emplacement et aucun cache hôte. Le moteur déclare **8,15 Gio de
buffers** ; il ne s’agit pas d’une mesure de RAM physique exclusive. Le plan
prudent estimait environ 10 Gio avant marge système et refusait ce budget.

Le [parcours de chat](ling-direct-lifecycle.json), avec les paramètres de
raisonnement par défaut et 512 tokens de sortie maximum, vérifie deux réponses
visibles, le rappel du mot demandé et la reprise après interruption. L’état de
test est temporaire ; le serveur de l’utilisateur n’est pas utilisé.
Un essai supplémentaire avec le budget Standard de l’interface (2 048 tokens
de sortie maximum) vérifie l’agrandissement automatique de 2 048 à 4 096 tokens
de contexte, puis une réponse visible, sans suppression de message.

La [calibration ciblée](ling-direct-calibration.json) compare trois configurations :
référence GPU compacte, réglages-1024 et motifs-adaptatifs-64. Les six workloads
de sélection sont suivis des six workloads indépendants et des contrôles normaux.
**Aucune variante n’est retenue** : leurs gains sont inférieurs au seuil ou
négatifs. La référence conservée mesure 52,36 tok/s de décodage agrégé sur la
validation indépendante ; aucun gain de décodage n’est annoncé. La suite montre
11,24 % de variation de durée entre passages. Ce résultat est une absence de
gain sur ces trois configurations, pas une recherche exhaustive ni une mesure
de supériorité sur LM Studio.

Les trois paires du test de cache retrouvent 1 189 tokens identiques et un délai
médian avant le premier token de 1,625 s à froid contre 0,035 s en réutilisation.
C’est le cache de préfixe du même moteur llama.cpp, pas une accélération du
décodage ni une preuve d’avantage propre à l’application. La calibration ciblée
a été réalisée sur la version de développement du correctif ; les deux courts
tours de son parcours utilisent `enable_thinking=false`, tandis que le rapport
de cycle de vie séparé teste les paramètres par défaut de la version 0.23.0.

Pour reproduire le parcours et cette recherche ciblée depuis la racine du dépôt :

```bash
.venv/bin/python scripts/verify_direct_runtime.py /chemin/vers/ling.gguf \
  --available-gib 7.9 --calibrate --output /tmp/ling-verification.json
```

Le plafond ne peut jamais augmenter la RAM mesurée. Sans `--calibrate`, la
commande vérifie seulement le parcours de chat. `local-llm calibrate` et le
bouton Optimiser de l’app restent la recherche complète des candidats disponibles.
