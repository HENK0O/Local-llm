# Débit de génération — 1er octobre 2026

SmolLM2-360M-Instruct officiel Q8_0, même fichier, prompt `Once upon a time`,
64 tokens demandés, 5 essais sur la même machine. Le prompt et les 64 tokens
obtenus sont identiques avant/après. Les médianes sont :

| Mesure | Avant | Après | Évolution |
|---|---:|---:|---:|
| Décodage | 89,73 tok/s | 106,14 tok/s | +18,28 % |
| Prefill | 134,02 tok/s | 144,34 tok/s | +7,70 % |
| Cache KV | 5 570 560 octets | 5 570 560 octets | identique |

`before.json` décrit le moteur tel qu’il était dans l’arbre de travail au début
de cette session, incluant les optimisations précédentes du prefill. Ce n’est
pas une mesure du commit HEAD seul : l’arbre avait déjà des modifications non
commitées. `after.json` conserve la comparaison effectuée par la commande CLI.
Les kernels ont été reconstruits entre les mesures.

Les changements retenus : lookup immuable des échelles FP16 (256 Kio), projections
Q/K/V groupées dans un dispatch sans recopier les poids, RMSNorm native,
facteurs RoPE partagés à travers les couches et suppression de la copie F64
pour l’argmax glouton. La variante SIMD testée qui ralentissait le modèle a été
retirée. Les arrondis des réductions RMSNorm peuvent différer légèrement ; les
réponses identiques de ce benchmark ne garantissent pas une identité des tokens
sur tous les prompts et tous les checkpoints.

## Reproduire la mesure actuelle

Depuis un environnement du projet avec NumPy et l’extension native :

```bash
python setup.py build_ext --inplace
python -m local_llm benchmark \
  models/SmolLM2-360M-Instruct.official.Q8_0.gguf \
  --prompt 'Once upon a time' --tokens 64 --runs 5 \
  --output /tmp/current.json --json
```

Pour comparer à un rapport obtenu sur la même machine, ajouter
`--compare /chemin/before.json`. La commande vérifie les poids, le prompt, les
tokens générés, le nombre d’essais et l’environnement. Le premier essai n’est
pas exclu ; le rapport expose aussi minimum et maximum. Le débit mesure les
63 passages de décodage, hors prefill, sampling, rendu et transport HTTP.
La mesure courante peut varier avec la charge, la température et les fréquences
CPU. Ces rapports ne démontrent pas une supériorité sur llama.cpp ou LM Studio.

## Comparaison dans l’interface

L’interface mesure séparément le débit de la réponse et un rejeu CPU de ses
premières étapes. Le témoin NumPy conserve le cache KV et les mêmes poids ;
seules les projections quantifiées sont remplacées. Le calcul du delta est
bloqué si les logits divergent au-delà de la tolérance. Un modèle déjà exécuté
par les mêmes projections BLAS n’affiche aucun gain attribuable aux kernels.

Le témoin LM Studio utilise ses timings publiés par l’API v0. L’interface affiche
un écart indicatif, car l’identité des poids et le placement CPU/GPU n’ont pas
été vérifiés. Le serveur LM Studio n’était pas lancé sur la machine lors de
cette session ; la découverte de fichiers a été vérifiée en réel, et le client
API v1/v0 a été testé avec des réponses simulées.

Mise à jour du 2 octobre : le diagnostic NumPy est désormais manuel, dans
l’onglet Performances. Aucun pourcentage de gain face à un moteur standard
n’en est déduit. Voir [l’audit](../2026-10-02/README.md).
