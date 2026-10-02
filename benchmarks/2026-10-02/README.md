# Audit de l’affichage des performances — 2 octobre 2026

L’ancien écran présentait environ 105 tok/s avec les projections natives,
contre 5 tok/s pour un témoin NumPy, puis annonçait presque 2 000 % de gain.
Cet écart a été reproduit sur le SmolLM2 360M Q8 local : les chronomètres et
le calcul arithmétique concordent. La présentation était trompeuse, car le
témoin déquantifie les blocs et effectue des projections NumPy pendant chaque
étape ; ce n’est ni LM Studio ni llama.cpp.

`measurement-audit.json` conserve une nouvelle exécution, l’empreinte des poids,
les tokens, les temps et les trois essais alternés du diagnostic. Une identité
numérique des logits ne rend pas le témoin représentatif d’un moteur standard.
Ce rapport ne démontre donc aucun gain face aux moteurs établis.

Le débit de décodage natif mesure `(N - 1) / secondes de décodage`. Le premier
token provient du prefill ; prefill, sampling et HTTP sont exclus. Dans le chat,
le débit est maintenant global : `N / durée complète de la requête` observée
dans le navigateur. Il inclut la préparation et la transmission. Il n’est pas
comparable directement au débit natif ou au débit global de LM Studio lorsque
les définitions, les poids et le placement CPU/GPU diffèrent.

La nouvelle interface conserve la comparaison dans l’onglet Performances, avec
un déclenchement manuel. Le diagnostic interne présente les débits, les temps
bruts et la tolérance numérique, sans pourcentage d’accélération. Un essai LM
Studio reste un écart indicatif tant que les conditions ne sont pas vérifiées.
Le gain de 18,28 % documenté le 1er octobre compare deux versions du moteur local
sur cette machine ; c’est une autre comparaison, avec une autre référence.

Une nouvelle optimisation conserve le préfixe KV entre les requêtes, avec une
borne de 64 Mio. `prefix-cache.json` mesure une conversation en deux tours :
517 des 546 tokens d’entrée du second tour sont réutilisés. Sur cinq essais
alternés, le prefill médian passe de 2,778 s à 0,159 s (-94,3 %), et le temps
total de 3,144 s à 0,533 s. Tous les tokens de sortie sont identiques.
La référence est le même moteur avec la réutilisation désactivée. Cela accélère
la préparation d’un contexte déjà calculé ; ce n’est pas une comparaison à
LM Studio, et aucun gain de décodage universel n’est annoncé.

Pour reproduire le protocole sur le même fichier :

```bash
.venv/bin/python scripts/benchmark_prefix_cache.py \
  models/SmolLM2-360M-Instruct.official.Q8_0.gguf \
  --runs 5 --tokens 32 --output /tmp/prefix-cache.json
```
