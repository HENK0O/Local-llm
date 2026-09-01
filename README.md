# local-llm

Un moteur d’inférence Llama minimal écrit en Python et NumPy. Le passage avant est
entièrement implémenté dans ce dépôt : aucune bibliothèque d’inférence et aucun
appel à Transformers ne sont utilisés.

Le runtime comprend :

- embeddings, RMSNorm, RoPE, attention causale multi-têtes/GQA et SwiGLU ;
- prefill et décodage token par token avec cache KV préalloué ;
- tokenizer jouet UTF-8 et tokenizer GPT-2 byte-level BPE réel ;
- lecteur SafeTensors natif F32/F16/BF16, mono-fichier ou shardé ;
- lecteur GGUF v3 natif F32/F16/BF16/Q8_0 avec métadonnées, tokenizer et `mmap` ;
- génération gloutonne, température, top-k, top-p et graine reproductible ;
- streaming, mode interactif, débit prefill/décodage et taille du cache KV ;
- tests comparant les logits et la génération avec une voie lente sans cache.

## Démarrage rapide

Python 3.9 ou supérieur, NumPy et `regex` sont les seules dépendances d’exécution.

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
python -m pip install --upgrade pip
pip install -e .
local-llm run /tmp/local-llm-toy --prompt "Bonjour"
```

## Exécuter un modèle préentraîné réel

Le checkpoint de validation est
[HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M),
un modèle Apache-2.0 de type Llama. Télécharge `config.json`, `tokenizer.json`,
`tokenizer_config.json` et `model.safetensors` dans
`models/SmolLM2-135M/`. Le dossier `models/` et les poids sont ignorés par Git.

```bash
python -m local_llm run models/SmolLM2-135M \
  --prompt "Bonjour, comment ça va ?" \
  --max-new-tokens 32
```

Un GGUF Llama non quantifié se lance directement, sans dossier annexe :

```bash
python -m local_llm run models/SmolLM2-135M.official.F16.gguf \
  --prompt "Bonjour, comment ça va ?" \
  --max-new-tokens 32
```

Pour inspecter son contenu ou mesurer les performances :

```bash
python -m local_llm inspect models/SmolLM2-135M.official.F16.gguf
python -m local_llm inspect models/SmolLM2-135M.official.F16.gguf --tensors
python -m local_llm benchmark models/SmolLM2-135M.official.F16.gguf --tokens 32 --runs 3
```

Q8_0 est également exécutable sans déquantifier le modèle complet :

```bash
python -m local_llm run models/SmolLM2-135M.official.Q8_0.gguf \
  --prompt "Bonjour, comment ça va ?" \
  --max-new-tokens 32
```

## Formats de modèles

Un modèle est un dossier contenant :

```text
model/
├── config.json
├── tokenizer.json
└── weights.npz                  # format jouet
```

ou, pour un checkpoint Hugging Face :

```text
model/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
└── model.safetensors            # ou model.safetensors.index.json + shards
```

Le troisième format accepté est un fichier unique `model.gguf`. Le runtime lit
lui-même les métadonnées, le vocabulaire BPE et les tenseurs Llama
`token_embd`, `blk.N.*`, `output_norm` et `output`. Les vues F32/F16 du lecteur
sont memory-mappées. Pour les calculs CPU, les matrices F16 sont promues en F32
au chargement : sur Apple Silicon, cela évite les kernels NumPy F16 très lents.
Les matrices Q8_0 restent compressées en blocs de 32 valeurs avec une échelle
FP16 par bloc ; embeddings et produits matrice-vecteur sont calculés directement
depuis ces blocs.

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

Le tokenizer jouet utilise `{ "type": "byte", "byte_offset": 3 }`. C’est un
choix volontaire : il permet de valider le Transformer indépendamment des
détails SentencePiece/BPE. Un modèle préentraîné doit avoir été entraîné avec ce
vocabulaire. Les checkpoints réels peuvent utiliser un `tokenizer.json` BPE
byte-level de style GPT-2, avec tokens spéciaux et pré-tokenisation Unicode.

## Vérification

```bash
python3 -m unittest discover -s tests -v
```

Le test d’intégration du checkpoint, ignoré quand les poids ne sont pas présents,
se lance avec :

```bash
LOCAL_LLM_TEST_MODEL=models/SmolLM2-135M \
  python -m unittest tests.test_real_model -v

LOCAL_LLM_TEST_GGUF=models/SmolLM2-135M.official.F16.gguf \
  python -m unittest tests.test_real_gguf -v
```

Les tests vérifient chaque primitive, la sérialisation, la mémoire du cache, la
parité des logits entre passage complet et décodage incrémental, puis l’identité
des tokens gloutons avec une référence recalculant toute la séquence. Ils testent
aussi le lecteur SafeTensors, BF16, les shards et le BPE.

Transformers reste une dépendance de développement optionnelle. Pour exporter
une référence couche par couche puis la comparer :

```bash
pip install -e '.[reference]'
python scripts/export_transformers_reference.py \
  --model models/SmolLM2-135M \
  --prompt "Bonjour, comment ça va ?" \
  --output /tmp/smollm2-reference.npz
python -m local_llm verify \
  --model models/SmolLM2-135M \
  --reference /tmp/smollm2-reference.npz
```

Transformers n’est importé que par le script d’export. `local_llm run` et
`local_llm verify` effectuent leurs calculs avec le runtime NumPy du projet.

Sur le checkpoint de validation, le tokenizer produit exactement les mêmes IDs,
les tokens gloutons sont identiques et l’écart moyen mesuré sur les logits est
d’environ `1.23e-5` en SafeTensors et `9.19e-6` avec le GGUF F16 officiel,
en calcul CPU F32. Les 16 tokens gloutons de référence sont identiques dans les
deux formats.

Mesures indicatives sur la machine de développement pour SmolLM2‑135M :

| Format | Taille | Decode | Écart moyen des logits | Tokens gloutons |
|---|---:|---:|---:|---|
| GGUF F16, calcul F32 | 269 Mo | ~102 tok/s | `9.19e-6` | identiques |
| GGUF Q8_0, kernel NumPy | 143 Mo | ~12 tok/s | `2.40e-1` | identiques |

Le kernel Q8_0 actuel privilégie la lisibilité et la correction. Sa taille est
réduite, mais il est plus lent que BLAS F32 : l’accélération Q8 nécessite le
prochain kernel natif C++/Accelerate.

## Limites et feuille de route

Le chargement direct Llama SafeTensors et GGUF F32/F16/BF16/Q8_0 est opérationnel.
La suite est :

1. porter le kernel Q8_0 en C++/Accelerate et le benchmarker ;
2. ajouter une quantification 4 bits simple après stabilisation de Q8 ;
3. mesurer perplexité et consommation mémoire résidente par format ;
4. ajouter un tokenizer SentencePiece pour les modèles qui n’utilisent pas BPE ;
5. explorer Metal seulement après les kernels CPU natifs.

Le modèle jouet permet de développer chacune de ces étapes sans confondre les
erreurs de format, de tokenizer, de quantification et de calcul.
