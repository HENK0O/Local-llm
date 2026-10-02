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
Le compteur vert indique les tokens du contexte dont le recalcul a été évité.

Les comparaisons sont manuelles, dans l’onglet Performances. Le témoin NumPy
déquantifie les blocs lors de chaque projection ; il peut être très lent et ne
représente pas un moteur standard. Aucun pourcentage d’accélération n’en est
déduit. La comparaison LM Studio reste indicative tant que les poids, la
quantification, le placement CPU/GPU et la définition du débit ne concordent pas.
