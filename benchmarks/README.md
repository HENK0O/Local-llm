# Mesures des optimisations d’inférence

Comparaison du moteur avant/après sur les mêmes poids SmolLM2-360M-Instruct Q8_0, sur cette machine. Trois mesures alternées par variante, après échauffement. Les tokens gloutons sont identiques pour chaque paire ; le script échoue si ce contrôle échoue.

| Tokens du prompt | Prefill avant | Prefill après | Accélération |
|---:|---:|---:|---:|
| 25 | 0.145 s | 0.118 s | 1.23× |
| 97 | 0.707 s | 0.452 s | 1.56× |
| 193 | 1.868 s | 0.951 s | 1.96× |

Ces mesures concernent le traitement du prompt, pas une multiplication identique du débit de génération. Les réponses peuvent se terminer avant les 16 tokens demandés si le modèle émet EOS. Les durées brutes, tokens, empreinte du modèle, environnement et commit de référence sont conservés dans `inference-q8.json`.

## Reproduire

Depuis la racine du dépôt, avec l’environnement Python du projet :

```bash
.venv/bin/python scripts/benchmark_inference.py \
  models/SmolLM2-360M-Instruct.official.Q8_0.gguf \
  --baseline-ref 14994f4b8401f7efcc757587ffb7225c4a9d5452 \
  --runs 3 --tokens 16 --output /tmp/inference-comparison.json
```

Le script exécute `model.py` et `generation.py` du commit de référence : choisir uniquement un commit de confiance. Le chargement des poids et les kernels sont partagés avec la version courante afin d’isoler ces deux modules. Ce benchmark ne compare pas le moteur à Transformers ou llama.cpp.

## Changements

- Attention par produits matriciels BLAS, avec diffusion des têtes KV sans réplication.
- En génération, normalisation finale et projection du vocabulaire uniquement à la dernière position du prompt. Le cache conserve toutes les positions. `forward()` conserve par défaut tous les logits pour les évaluations.
- Streaming : chaque token est livré avant le calcul du suivant ; fermer le générateur évite ce calcul devenu inutile.
- Demander zéro nouveau token ne lance plus une inférence dont le résultat serait jeté.

Ces optimisations visent la latence et les allocations. Elles ne changent ni les poids, ni le prompt, ni la stratégie de sampling. Les arrondis BLAS peuvent légèrement modifier les logits ; la parité des tokens observée ici ne garantit pas des tokens identiques sur tous les prompts possibles.
