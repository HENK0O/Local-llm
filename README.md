# local-llm

Un moteur d’inférence Llama minimal écrit en Python et NumPy. Le passage avant est
entièrement implémenté dans ce dépôt : aucune bibliothèque d’inférence et aucun
appel à Transformers ne sont utilisés.

La v0.1 pédagogique comprend :

- embeddings, RMSNorm, RoPE, attention causale multi-têtes/GQA et SwiGLU ;
- prefill et décodage token par token avec cache KV préalloué ;
- tokenizer UTF-8 par octets, réversible et sans dépendance ;
- génération gloutonne, température, top-k, top-p et graine reproductible ;
- streaming, mode interactif, débit prefill/décodage et taille du cache KV ;
- tests comparant les logits et la génération avec une voie lente sans cache.

## Démarrage rapide

Python 3.9 ou supérieur et NumPy sont les seules dépendances d’exécution.

```bash
python3 -m local_llm create-toy /tmp/local-llm-toy
python3 -m local_llm run --model /tmp/local-llm-toy --prompt "Bonjour" --max-new-tokens 32
python3 -m local_llm run /tmp/local-llm-toy --interactive --temperature 0.8 --top-p 0.9
```

Le modèle jouet possède des poids aléatoires déterministes : il sert à examiner le
runtime et ne produit donc pas de texte utile.

Pour installer la commande `local-llm` dans un environnement virtuel :

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
local-llm run /tmp/local-llm-toy --prompt "Bonjour"
```

## Format de modèle v0.1

Un modèle est un dossier contenant :

```text
model/
├── config.json
├── tokenizer.json
└── weights.npz
```

`config.json` reprend les champs Llama usuels (`hidden_size`,
`intermediate_size`, `num_hidden_layers`, `num_attention_heads`,
`num_key_value_heads`, etc.). Les matrices dans `weights.npz` suivent la
convention `[sortie, entrée]` et les noms Hugging Face usuels :

```text
model.embed_tokens.weight
model.layers.0.input_layernorm.weight
model.layers.0.self_attn.{q,k,v,o}_proj.weight
model.layers.0.post_attention_layernorm.weight
model.layers.0.mlp.{gate,up,down}_proj.weight
model.norm.weight
lm_head.weight
```

Le tokenizer v0.1 utilise `{ "type": "byte", "byte_offset": 3 }`. C’est un
choix volontaire : il permet de valider le Transformer indépendamment des
détails SentencePiece/BPE. Un modèle préentraîné doit avoir été entraîné avec ce
vocabulaire, ou ses poids doivent être accompagnés du tokenizer correspondant.

## Vérification

```bash
python3 -m unittest discover -s tests -v
```

Les tests vérifient chaque primitive, la sérialisation, la mémoire du cache, la
parité des logits entre passage complet et décodage incrémental, puis l’identité
des tokens gloutons avec une référence recalculant toute la séquence.

## Limites et feuille de route

Cette étape établit le socle correct et testable. Elle ne prétend pas encore
charger directement un modèle Hugging Face ou GGUF. L’ordre conseillé pour la
suite est :

1. ajouter un tokenizer SentencePiece/BPE et un importeur SafeTensors de référence ;
2. comparer activations et logits couche par couche avec un petit modèle Llama ;
3. lire GGUF F32/F16 via `mmap`, puis mapper ses noms de tenseurs ;
4. ajouter Q8 et ses kernels matrice-vecteur, avec benchmarks et seuils d’erreur ;
5. porter les kernels stables en C++/Accelerate, puis seulement explorer Metal.

Le modèle jouet permet de développer chacune de ces étapes sans confondre les
erreurs de format, de tokenizer, de quantification et de calcul.

