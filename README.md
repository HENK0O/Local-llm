# local-llm

[![Tests](https://github.com/HENK0O/local-llm/actions/workflows/tests.yml/badge.svg)](https://github.com/HENK0O/local-llm/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](pyproject.toml)

**Une app d’inférence locale qui choisit ses réglages à partir de mesures sur votre machine.**

L’objectif est de **trouver et appliquer les réglages validés pour votre modèle
et votre PC**. Aucun gain universel n’est promis : une référence déjà rapide
peut être conservée. Le [premier parcours vérifié sur Ling Tiny](benchmarks/README.md#ling-tiny--chargement-et-absence-de-gain-validé)
documente le chargement, les réponses et les essais sans accélération retenue.

Le moteur direct utilise **llama.cpp** pour les GGUF, **MLX-LM** pour les dossiers
MLX sur Apple Silicon et, si son environnement est installé, **MTPLX** pour les
checkpoints avec têtes MTP compatibles. L’app lit les poids déjà présents, y
compris dans `~/.mtplx/models`. Chaque réponse indique le moteur utilisé.
local-llm calibre ses réglages sur votre machine et conserve une configuration
uniquement si le benchmark confirme un gain reproductible sans changer ses
sorties. Les poids ne sont ni modifiés ni téléchargés automatiquement.

Le moteur CPU Python/NumPy reste disponible avec `--engine native`. Il expose la
chaîne d’inférence et ses kernels Q8/Q4 en C++/NEON, vérifiés face aux logits et
tokens du chemin de référence. L’accélération GPU provient des runtimes amont ; la
couche local-llm apporte la calibration, le choix de la spéculation, le suivi du
contexte et des conversations. Relayer une requête à LM Studio n’accélère pas
ses calculs.

## Checkpoints existants, sans serveur LM Studio

Les GGUF détectés dans la bibliothèque LM Studio, son dossier personnalisé ou
le cache Hugging Face sont chargés **sur place par le moteur direct**. Aucun
poids n’est copié et LM Studio peut rester fermé. Une bibliothèque contient des
fichiers ; un moteur doit encore les charger en RAM et exécuter leurs calculs.
local-llm gère son propre worker privé llama.cpp automatiquement ; il n’est pas
nécessaire de démarrer un second serveur à la main.

Si un chargement échoue, le sélecteur conserve le modèle demandé, affiche la
raison et propose « Réessayer le chargement ». La discussion reste bloquée pour
éviter d’envoyer le message au modèle précédent. L’intention est conservée dans
la conversation, y compris après rechargement de la page. Une erreur de RAM
indique le budget disponible et l’estimation minimale avec son contexte ; les
modèles chargés dans un autre moteur doivent être déchargés par leur application.
Les réponses d’un ancien polling sont ignorées pendant un changement de modèle.

Dans **Bibliothèque**, « Ajouter un checkpoint existant » accepte le chemin
absolu d’un GGUF ou d’un dossier natif. Seul ce chemin est enregistré dans
`LOCAL_LLM_STATE_DIR/library.json` (par défaut `~/.cache/local-llm`), jamais les
poids. Les liens symboliques des snapshots Hugging Face conservent leur nom et
pointent vers le même fichier de cache. `POST /v1/local-models/import` reçoit
`{"path":"/chemin/existant/modele.gguf"}` ; il applique la protection de même
origine des autres mutations. Les anciennes conversations LM Studio passent au
moteur direct seulement si leur clé correspond exactement à un unique fichier
détecté, sans association par nom d’affichage.

**Performances → Une variante mieux adaptée ?** vérifie automatiquement les
quantifications du checkpoint sélectionné dans son dépôt Hugging Face. La
recherche conserve le nom précis du modèle, sa taille et sa variante fine-tunée.
Elle exclut les projecteurs, les auxiliaires et les fichiers partiels. Chaque
lien correspond à un fichier réel vérifié via l’API publique, fixé à sa révision
Git. Un dépôt inconnu ou inaccessible ne produit aucune recommandation devinée.
Aucun téléchargement n’est lancé. Le lien identifie le dépôt déjà installé ou
une origine connue ; il ne certifie pas son auteur ni la qualité de ses poids.

Les suggestions distinguent RAM disponible maintenant et RAM totale après
réserve système, pour le contexte affiché. Elles indiquent l’économie de poids
ou la précision à comparer ; **une autre quantification n’est pas une promesse
de tok/s** et peut changer les réponses. Les gains de calibration du checkpoint
actuel ne sont jamais transférés à d’autres poids. Le matériel, les chemins et
les messages restent locaux ; seul l’identifiant du dépôt est envoyé à
`huggingface.co`. Deux recherches au maximum tournent en arrière-plan ; le
catalogue vérifié est mis en cache six heures ; un échec réseau peut être retenté
après trente secondes avec « Actualiser les conseils ». En cas de panne réseau, l’inférence locale
continue. `GET /v1/model-advice?model_id=identifiant-du-catalogue` expose cet état.

Les **têtes MTP déjà intégrées au GGUF** sont détectées par leurs métadonnées et
leurs tenseurs. Si le build llama.cpp expose `draft-mtp`, la calibration teste
les profondeurs 2, 4, 8 et 16 avec les mêmes vérifications de sortie et de gain
indépendant que les autres candidats. Le runtime partage les poids du modèle
principal : aucun deuxième checkpoint à stocker, mais un contexte supplémentaire
consomme de la RAM. Aucun gain MTP n’est déclaré avant validation. Le protocole
6 invalide les anciens profils pour permettre cette nouvelle recherche.
Voir la [documentation officielle de la spéculation llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md).

## Moteur GPU et optimisation automatique

Sur macOS, installez le runtime officiel, puis démarrez l’app :

```bash
brew install llama.cpp
.venv/bin/python -m local_llm serve
```

Sur Linux, utilisez un build officiel de [llama.cpp](https://github.com/ggml-org/llama.cpp)
avec votre backend GPU et `llama-server` dans le PATH. La détection vérifie les
options requises ; un build ancien affiche une erreur explicite. Le runtime
validé localement est le build **11146 / 7fe450e19**. Le chemin de l’exécutable
peut être défini par `LOCAL_LLM_LLAMA_SERVER`.

`serve --engine auto` est le mode par défaut : il préfère un petit GGUF déjà
compatible avec le moteur CPU quand llama.cpp est installé. Sans ce runtime,
le moteur CPU reste utilisable. `--engine gpu` impose le moteur direct et
`--engine native` conserve le moteur CPU. La bibliothèque permet aussi de
charger directement un GGUF que le moteur CPU ne prend pas en charge ; sa
compatibilité finale dépend du build llama.cpp installé. Les modèles auxiliaires
DFlash/DSpark et les modèles d’embeddings ne sont pas proposés comme modèles de chat.

Dans **Performances**, choisissez **Optimiser ce modèle**. L’app explore les
threads de génération et de préparation séparément, les batches physiques de
préparation (`ubatch`), les tailles de batch, le cache KV Q8 et, si le build le
propose, le sampling GPU. Elle teste les motifs `ngram-simple` (4/16/48 tokens,
avec une recherche de motifs courts), `ngram-map-k` et `ngram-mod` adaptatif
(jusqu’à 64 tokens). Les options absentes du build installé sont ignorées.
En mode automatique, elle cherche au maximum deux auxiliaires locaux compatibles
qui rentrent dans la RAM disponible et teste 2/3/4/6/8/12/16 tokens anticipés.
Pour le MTP intégré et l’auxiliaire classique, le seuil de confiance est aussi
exploré quand le build installé le permet. Le MTP partage les poids de la cible,
mais son contexte et les états de rollback restent compris dans le budget RAM. Un
auxiliaire classique doit avoir exactement les mêmes tokens et tokens spéciaux.
Vous pouvez choisir un auxiliaire précis ou désactiver cette recherche. Aucun
modèle n’est téléchargé. Le modèle cible valide les propositions spéculatives.

Une **présélection** de six workloads publics, avec au plus 128 tokens de sortie,
retient au maximum six configurations en préservant le meilleur candidat par
usage. Ces mesures rapides ne servent jamais à annoncer un gain. La **sélection**
utilise six workloads : deux formes par catégorie (discussion, code, contexte
long), avec des plafonds de 128/192/128/512/512/256 tokens et deux passages dans
des ordres opposés après échauffement. Six workloads différents sont ensuite
joués **trois fois** pour vérifier les candidats retenus. Le tableau indique les
longueurs d’entrée réellement tokenisées et les sorties obtenues, le premier
token mesuré en streaming, le débit moteur et le temps total. Les journaux publics
sont raccourcis explicitement pour les petites capacités et les limites de sortie
bornées au quart du contexte ; aucun prompt utilisateur n’est utilisé ou tronqué.
La vérification indépendante ne cherche jamais de candidat de remplacement : en
cas d’échec, cet usage garde le standard. Les sorties gloutonnes doivent être
identiques dans les **trois catégories**, y compris pour un profil spécialisé.

Quatre profils sont proposés : **Équilibré, Discussion, Code, Contexte long**.
Chaque passage doit améliorer le temps d’au moins 5 % et préserver le débit de
décodage. Aucun workload du périmètre retenu ne doit ralentir de plus de 10 %,
même si un autre workload de sa catégorie progresse. Le premier token ne doit
pas ralentir simultanément de plus de 20 % et de 10 ms. Les profils spécialisés
évaluent les performances de leur catégorie uniquement : leur gain ne constitue pas un gain général.
Le sélecteur dans Performances applique le profil aux prochaines réponses du
modèle. Si ses réglages diffèrent, le worker est rechargé ; un cache disque
compatible peut retrouver un état précédent et les conversations restent
disponibles. Ce coût de changement est affiché séparément
et n’est pas inclus dans le gain de génération.

Les chargements et échauffements sont exclus du temps des suites ; préparation,
génération et appel HTTP local d’inférence sont inclus, ainsi que le travail
spéculatif. La mise en forme et le comptage du prompt public précèdent le
chronomètre ; le temps des réponses de chat inclut aussi ces étapes.
Les compteurs réels de tokens anticipés proposés et acceptés sont affichés
lorsque le runtime les expose ; une valeur absente reste inconnue. Le coût de
l’auxiliaire seul n’est pas fourni par cette API : un taux d’acceptation élevé ne
suffit donc pas à valider un gain. Les médianes et le débit pondéré par le travail
sont affichés. Les tests couvrent deux formes par catégorie, répétées trois fois : ils ne
garantissent pas les mêmes gains sur toute requête.

Les poids et leur quantification ne changent jamais. La capacité du contexte et
le nombre de slots sont les mêmes pour tous les essais. Le standard utilise un
cache KV F16 ; **le cache Q8 est une modification de précision distincte**,
rejetée dès qu’une sortie diffère sur la sélection ou la vérification. Même si
ces tests passent, ils ne garantissent pas la qualité sur tous les prompts ni
avec d’autres températures. Le diagnostic indique explicitement cette différence.

L’écran détaille les réglages retenus, la mémoire estimée, le pic de RSS échantillonné pendant
les essais (toutes les 100 ms), les résultats par longueur et par catégorie et la raison de chaque rejet. Le gain
affiché vient uniquement des prompts indépendants ; ce n’est pas une promesse
pour chaque réponse. Si aucun réglage ne passe, le gain affiché est zéro.
Une mesure séparée compare trois paires du même contexte sans cache puis avec
cache : premier token et préparation réellement économisée. Le cache doit avoir
réutilisé des tokens pour qu’un gain de préparation soit déclaré. Ce test ne
constitue pas une comparaison avec LM Studio. Les réponses rapportent aussi le
délai du premier token, réflexion incluse, et le délai du premier texte visible,
mesurés à la réception par le serveur, dans **Performances**.

Le profil reste local (`~/.cache/local-llm`, ou `LOCAL_LLM_STATE_DIR`) et ne
contient aucun texte de conversation. Il est lié à l’empreinte des poids, au
matériel, au build llama.cpp, au contexte et aux slots. Un ancien profil ou un
profil dont la validation ne peut pas être reproduite depuis ses mesures est
ignoré. Un gain CPU→GPU, un gain contre LM Studio et un gain contre le standard
GPU sont des observations différentes.

Le worker GPU est lancé sur un port privé de boucle locale, avec un jeton
aléatoire temporaire. L’app arrête uniquement le processus qu’elle a créé. Les
poids restent chargés entre les messages et sont libérés au déchargement/changement
de moteur ou à la fermeture de l’app. Au chargement, une estimation prudente des
poids, du KV et des buffers démarre avec **un seul emplacement GPU et au plus
4096 tokens** de contexte. Avoir davantage de RAM libre ne provoque plus une
réservation automatique de quatre caches GPU. Un cache de préfixes en RAM hôte,
**borné à 64/128/256 Mio selon la marge disponible** (ou désactivé si elle est
insuffisante), permet de retrouver les états des conversations récentes sans
réserver un emplacement GPU par fil. Le runtime ne restaure que des préfixes de
tokens identiques. Ce cache en RAM est volatile. Le plan distingue les poids,
le KV, les états récurrents et les buffers de calcul : Ling `bailingmoe3` (KDA/MLA)
et Qwen `qwen35` (attention/delta) ont des géométries spécifiques, y compris
l’exclusion des couches MTP du KV principal. Les métadonnées inconnues ou
incomplètes gardent une estimation prudente. Si le runtime refuse une allocation
au chargement initial, la capacité est réduite jusqu’à 512 tokens et le cache hôte
désactivé avant toute conversation ; les autres erreurs ne déclenchent pas cette
réduction. Les allocations CPU/GPU déclarées par le runtime sont détaillées dans
Performances. Leur somme et le RSS ne sont pas une mesure de RAM physique
exclusive et ne remplacent pas les capteurs.
Si une conversation dépasse la capacité chargée, l’app peut réallouer jusqu’à
32 768 tokens si le modèle et la RAM le permettent ; les messages sont conservés,
le cache est vidé et le profil de vitesse est invalidé. Une impossibilité est
signalée, sans tronquer ni compresser silencieusement la conversation.

Sur macOS, un refus de cette estimation peut déclencher une **tentative contrôlée**
si la pression mémoire est normale, la géométrie du modèle connue et le plan
prudent tient dans la RAM physique avec 3 Gio réservés. Le chargement commence à
2 048 tokens maximum, un seul emplacement, batch 256, ubatch 128, sans cache hôte.
Le contrôle borne à 2 Gio (ou 10 % de la RAM) la récupération potentielle de mémoire
par rapport aux poids, états, KV, 512 Mio de calcul et 512 Mio de réserve. Ce budget
réduit est un seuil de tentative, pas une consommation garantie : l’estimation
prudente reste visible et les allocations réelles sont lues après chargement.
Le worker privé est surveillé chaque seconde et arrêté si la pression devient
critique ou si sa lecture échoue. Aucun autre moteur n’est arrêté. Une pression
déjà élevée, une architecture inconnue ou un budget trop faible restent bloquants.
Les agrandissements du contexte passent les mêmes contrôles et ne suppriment
aucun message. La lecture de RAM précédant une allocation est rafraîchie plutôt
que reprise du cache de trois secondes des capteurs.

Un **cache KV sur disque de 512 Mio maximum de fichiers retenus**, activé par défaut si le runtime
expose `--slot-save-path`, conserve les états des réponses terminées dans
`~/.cache/local-llm/kv-cache` (ou sous `LOCAL_LLM_STATE_DIR`). Il évite de recalculer
un préfixe après redémarrage ou changement de modèle. Il ne duplique pas les poids.
Les fichiers binaires contiennent des **états privés de conversation** ; les
métadonnées ne contiennent que des empreintes, tailles et coûts. Dossier en mode
700, fichiers en mode 600 ; aucun fichier n’est envoyé à GitHub. Le budget évince
les états les plus anciens. L’interface Performances permet de désactiver le cache
(et d’effacer les fichiers) ou d’effacer uniquement les snapshots disque.

Chaque snapshot est lié au SHA-256 du modèle et de son auxiliaire, au build et
aux appareils du runtime, à tous ses réglages et à l’identité de la conversation.
Les messages précédents doivent être inchangés ; le runtime vérifie ensuite le
préfixe de tokens exact. Une corruption ou incompatibilité impose un recalcul.
La première estimation de restauration utilise 1,5 fois le coût mesuré de
sauvegarde et de hash ; le coût réellement observé la remplace ensuite et est
ajusté à la taille du prochain snapshot. La restauration est évitée si elle est
estimée plus coûteuse que la préparation connue du préfixe. Cette estimation
n’est pas une garantie de gain. Sauvegarde/restauration sont exclues des essais
de réglages, mais incluses dans le temps total de vos requêtes ; leur coût est
rapporté séparément. Les interruptions suppriment le snapshot du fil concerné.

Les interruptions imposent une requête sans réutilisation de cache au prochain
message du même fil. Le moteur CPU conserve jusqu’à huit
préfixes ; son budget est limité à 1/32 de la RAM disponible, avec un plafond de
512 Mio (64 Mio si la RAM est inconnue). Ce budget borne les **caches retenus**,
pas le KV de la génération active. La RAM système inclut LM Studio ; charger
une seconde copie d’un grand modèle peut demander de le décharger dans LM Studio.

La même calibration est disponible sans serveur HTTP :

```bash
local-llm calibrate /chemin/modele.gguf --output /tmp/profil.json
# Facultatif : --draft /chemin/auxiliaire-compatible.gguf
```

L’API expose `GET /v1/accelerator`, `GET /v1/accelerator/drafts` et les POST
`/v1/accelerator/load` (`{"id":"id-du-catalogue"}`), `/optimize` (facultatif
`{"draft_id":"id-du-catalogue"}` ou `{"draft_mode":"off"}`), `/cancel`, `/unload` et
`/profile` (par exemple `{"profile":"code"}` ; valeurs : `balanced`, `discussion`,
`code`, `long_context`) et `/cache` (`{"enabled":false}` ou `{"clear":true}`).
L’optimisation retourne
202 ; son état et ses mesures sont accessibles dans `GET /v1/accelerator`.
Pour le chat direct, envoyez `backend: "llamacpp"`, `model` correspondant à
l’identifiant chargé, `conversation_id` propre au fil et `stream: true` à
`/v1/chat/completions`. La capacité de réponse est bornée à 4096 tokens et par le
contexte chargé. Le débit moteur et les tokens d’entrée effectivement réutilisés
sont rapportés séparément des gains mesurés pendant la calibration.

### Auxiliaires spécialisés

EAGLE-3, DFlash et DSpark nécessitent un auxiliaire entraîné pour la cible précise.
Le vocabulaire ou le nom affiché ne suffit pas. Un GGUF auxiliaire spécialisé
peut être déclaré par un fichier adjacent portant le même nom, avec le suffixe
`.local-llm-draft.json` (pour `auxiliaire.gguf`, `auxiliaire.local-llm-draft.json`) :

```json
{
  "method": "draft-dflash",
  "target_sha256": "SHA256_DES_POIDS_CIBLES",
  "draft_sha256": "SHA256_DES_POIDS_AUXILIAIRES",
  "source": "https://adresse-de-la-fiche-amont-documentant-la-paire"
}
```

Méthodes autorisées : `draft-eagle3`, `draft-dflash`, `draft-dspark`. Les empreintes
se calculent sur chaque fichier entier (par exemple `shasum -a 256 fichier.gguf`).
Cette déclaration est à renseigner **uniquement si la compatibilité de la paire
est documentée en amont**. Elle lie les poids exacts ; elle ne prouve ni la
qualité ni le gain. Les mêmes tests indépendants s’appliquent, et le build
llama.cpp doit proposer la méthode. Sans déclaration, un DFlash/DSpark n’est
jamais associé automatiquement à un modèle de chat. Les fichiers auxiliaires
restent ignorés par Git.

### MLX et MTPLX dans le chat (Apple Silicon, expérimental)

La bibliothèque détecte les dossiers MLX dans les emplacements existants,
notamment `~/.mtplx/models`. On peut aussi ajouter leur chemin dans Bibliothèque.
La compatibilité est confirmée au chargement, jamais à partir du seul nom.
Il faut des poids Safetensors, un tokenizer et un template locaux. Un GGUF reste
exécuté par llama.cpp ; il n’est pas converti en MLX. Aucun poids n’est copié ou
téléchargé et aucun serveur LM Studio n’est requis.

MLX-LM récent demande Python 3.11+ et une version de Transformers différente du
backend de référence. Utilisez donc un environnement séparé ; n’installez pas
MLX dans la `.venv` du moteur Python 3.9 :

```bash
python3.12 -m venv /chemin/venv-mlx
/chemin/venv-mlx/bin/python -m pip install 'mlx-lm>=0.31,<0.33'
LOCAL_LLM_MLX_PYTHON=/chemin/venv-mlx/bin/python .venv/bin/python -m local_llm serve
```

Un environnement MTPLX déjà installé est détecté sans installation automatique.
L’intégration Python a été préparée pour MTPLX 2.12.2, MLX 0.32.2 et MLX-LM
0.31.3. Elle utilise explicitement le profil **Sustained**, sans changer les
réglages de l’app MTPLX ou la ventilation. Elle ne représente pas le mode Turbo
de MTPLX. **Powered by MTPLX** — [MTPLX, Youssof Altoukhi](https://github.com/youssofal/MTPLX).

Dans **Performances → Le moteur de ce modèle**, Automatique utilise un profil
vérifié pour ces poids, versions et matériel ; sinon il garde la référence
MLX-LM, ou l’exécution MTPLX sans spéculation pour les formats propriétaires.
Les poids restent chargés jusqu’au déchargement. Une interruption arrête le
worker concerné ; il faut recharger le modèle avant la requête suivante.

**Comparer les moteurs sur ce Mac** teste la préparation MLX à 2048 et 512
tokens par étape et, quand les têtes compatibles existent, les profondeurs MTP
1/2/3. Les candidats utilisent **exactement le même dossier de poids**, le même
template, la même tokenisation et un sampling glouton avec raisonnement
désactivé. Le cache de requête est désactivé pour tous les essais. Deux passages
de sélection précèdent trois passages sur des prompts indépendants. Les tokens
de sortie doivent être identiques ; un gain stable d’au moins 5 % doit passer
les contrôles de durée, débit et délai du premier token. Sinon la référence est
conservée et l’app affiche **Aucun gain validé**. Un gain de calibration décrit
ces workloads ; il ne mesure pas un gain sur chaque réponse du chat.

Le rapport exportable conserve les mesures brutes, empreintes, versions et
décisions. Les profils sont revérifiés avant réutilisation. Une quantification
GGUF différente n’est jamais comparée à ce dossier MLX pour annoncer un gain.
Le contexte MLX est actuellement borné à 4096 tokens, ou à la capacité inférieure
du modèle ; un dépassement est refusé sans supprimer des messages. Le cache de
préfixe et le cache disque ne sont pas encore intégrés pour MLX/MTPLX.

Avant chaque chargement, l’app vérifie un budget estimé et la pression mémoire.
Elle arrête son worker dès que la pression devient élevée ou que le capteur
devient indisponible. La limite configurée dans MLX reste une indication,
**pas un plafond matériel garanti**. Le budget et les capteurs réduisent le
risque de swap sans garantir son absence.

La génération sur Metal, le contexte, l’interruption, la reprise et la
calibration ont été vérifiés avec un checkpoint synthétique de moins de 1 Mio.
Ce test ne démontre ni la qualité d’un modèle utile ni un gain face à MTPLX.
L’inférence MTPLX sur un checkpoint réel reste à vérifier ; aucun résultat de
performance MTPLX n’est publié pour cette intégration.

Le parcours réel peut se vérifier sans ouvrir le serveur de l’app :

```bash
.venv/bin/python scripts/verify_mac_runtime.py /chemin/modele-mlx-local \
  --engine mlx --calibrate --output /tmp/mesures-mac.json
```

L’ancienne commande `mlx-experiment` reste disponible pour la préparation seule
(128/512/2048 tokens par étape). Sources : [MLX-LM](https://github.com/ml-explore/mlx-lm),
[limite mémoire MLX](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_memory_limit.html)
et [MTPLX](https://github.com/youssofal/MTPLX).

## Essai rapide avec un vrai modèle

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
local-llm run models/SmolLM2-360M-Instruct.official.Q8_0.gguf \
  --chat --prompt "Explique simplement le cache KV." --max-new-tokens 80
```

Le fichier GGUF doit être placé dans `models/` ; les poids ne sont jamais ajoutés
au dépôt. Pour une conversation avec streaming et statistiques :

```bash
local-llm serve models/SmolLM2-360M-Instruct.official.Q8_0.gguf
```

Puis ouvre [http://127.0.0.1:8080](http://127.0.0.1:8080).
Tu peux aussi lancer `local-llm serve` sans chemin : les modèles compatibles des
bibliothèques locales sont détectés et sélectionnables dans l’interface.

### Support actuel

| Élément       | Support                                                         |
| ------------- | --------------------------------------------------------------- |
| Architectures | Llama (MHA/GQA), Baguette non hybride                           |
| Poids         | SafeTensors F32/F16/BF16, GGUF v3 F32/F16/BF16/Q8_0/Q4_0        |
| Tokenizers    | UTF-8 pédagogique, GPT-2 byte-level BPE                         |
| Inférence     | prefill, cache KV préalloué, décodage autoregressif, sampling   |
| CPU           | NumPy/BLAS ; C++ multithread ; SIMD NEON Apple Silicon          |
| Validation    | logits, tokens gloutons, cache contre recalcul, traces externes |
| Interfaces    | CLI, chat interactif, streaming SSE, API HTTP locale            |

### Performances indicatives

Mesures sur un MacBook Air Apple M5, modèle SmolLM2-360M-Instruct :

| Backend                  |   Poids |     Decode | Résultat glouton                    |
| ------------------------ | ------: | ---------: | ----------------------------------- |
| F16 / BLAS               | 692 Mio |  ~46 tok/s | référence                           |
| Q8_0 / C++ NEON fusionné | 369 Mio | ~106 tok/s | tokens identiques sur le cas mesuré |
| Q8_0 / NumPy             | 369 Mio |   ~5 tok/s | tokens identiques                   |

Les chiffres dépendent du prompt, de la longueur générée et de la machine. Les
commandes reproductibles et la méthodologie sont détaillées plus bas.
La session du 1er octobre mesure **89,7 → 106,1 tok/s (+18,3 %)** par rapport à
l’état du moteur au début de cette session, avec les mêmes 64 tokens gloutons
sur cinq essais. Ce gain ne compare pas local-llm à un moteur tiers ; les
[rapports bruts et limites](benchmarks/README.md#decode) sont conservés.

## Démarrage rapide

Python 3.9 ou supérieur, NumPy, `regex` et Jinja2 sont les seules dépendances
d’exécution.

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

Sur macOS, l'installation tente également de compiler les kernels Q8/Q4 C++ avec
Clang. Si aucun compilateur n'est disponible, le runtime reste utilisable avec
le fallback NumPy. `python setup.py build_ext --inplace` permet de reconstruire
explicitement l'extension pendant le développement.

## Exécuter un modèle préentraîné réel

### Poser des questions à un modèle Instruct

Pour obtenir des réponses d'assistant, il faut un checkpoint **Instruct**. Le
modèle `SmolLM2-135M` utilisé pour vérifier les calculs est un modèle **Base** :
il complète du texte, mais n'a pas été entraîné à répondre à une conversation.

Avec `SmolLM2-360M-Instruct`, le runtime lit le template ChatML attendu
directement depuis `tokenizer_config.json` ou le GGUF :

```bash
python -m local_llm run models/SmolLM2-360M-Instruct \
  --chat \
  --prompt "Quelle est la capitale de la France ?" \
  --max-new-tokens 80
```

Une conversation interactive conserve l'historique des messages :

```bash
python -m local_llm run models/SmolLM2-360M-Instruct.official.F16.gguf \
  --chat --interactive --temperature 0.7 --top-p 0.9
```

Le message système est personnalisable avec
`--system "Réponds brièvement en français."`.

Le moteur n'impose plus le format de SmolLM : `--chat` rend automatiquement le
template Jinja embarqué par le modèle. Les variables standard `messages`,
`bos_token`, `eos_token`, `pad_token` et `add_generation_prompt` sont prises en
charge dans un environnement sandboxé. Si un GGUF propose plusieurs variantes,
`inspect` les affiche et `--chat-template <nom>` permet d'en choisir une :

```bash
python -m local_llm inspect model.gguf
python -m local_llm run model.gguf --chat-template default --interactive
```

Un modèle dépourvu de template peut toujours être exécuté en complétion sans
`--chat`, mais le runtime refuse de deviner son format de conversation.
Cette amélioration règle la mise en forme du dialogue ; le modèle doit toujours
utiliser une architecture Llama et un tokenizer BPE actuellement pris en charge.

### Charger Baguette

Le checkpoint [HENK0O/baguette](https://github.com/HENK0O/baguette) utilise une
architecture proche de Llama avec QK-Norm, RoPE partiel et une porte de sortie
sur l'attention. Le runtime implémente ces opérations directement. La conversion
emploie PyTorch uniquement pour lire le conteneur `.pt` ; l'inférence obtenue
reste entièrement exécutée par `local-llm`.

```bash
python -m local_llm convert-baguette \
  /chemin/baguette-123m-sft.pt \
  --tokenizer /chemin/tokenizer.json \
  --output models/baguette-123m-sft
```

Le dossier de sortie contient la configuration, le tokenizer, le template
ChatML, les informations de provenance et environ 235 Mio de poids SafeTensors.
Les poids RMSNorm zéro-centrés sont convertis en gains ordinaires F32 ; les
grandes matrices restent en F16 sur disque et sont promues par le runtime au
chargement. Le checkpoint source n'est jamais modifié et une sortie existante
n'est jamais écrasée.

Pour utiliser la version SFT en conversation :

```bash
python -m local_llm run models/baguette-123m-sft \
  --chat --interactive --temperature 0.5 --top-k 20

python -m local_llm serve models/baguette-123m-sft
```

Puis ouvre [http://127.0.0.1:8080](http://127.0.0.1:8080). Le convertisseur
refuse volontairement les checkpoints Baguette `hybrid: true` utilisant
DeltaNet, qui demanderaient un second type de cache et de nouvelles opérations.

### Serveur HTTP local

Le même runtime peut rester chargé en mémoire et recevoir plusieurs requêtes de
chat, sans recharger le GGUF à chaque question :

```bash
python -m local_llm serve \
  models/SmolLM2-360M-Instruct.official.Q8_0.gguf \
  --host 127.0.0.1 --port 8080
```

Ouvre ensuite [http://127.0.0.1:8080](http://127.0.0.1:8080) dans un navigateur.
L'interface permet de discuter avec le modèle, conserve l'historique, affiche
les tokens en direct et permet de régler la température et la longueur maximale.
Le bouton carré interrompt une génération. « Nouvelle conversation » crée un
autre fil ; le précédent reste accessible dans la liste de gauche.

Le sélecteur de modèles permet de rechercher parmi les modèles exécutables par
local-llm ou déjà chargés dans LM Studio. Les fichiers incompatibles restent dans
la bibliothèque, sans entrées grisées en double dans le sélecteur. Un voyant
vert et « En cours » signalent la génération.

Le modèle se choisit dans la barre supérieure. La barre de gauche reste fixe,
avec une liste paginée de conversations, les onglets et les relevés de la machine.
Chaque conversation conserve son modèle, ses messages et son brouillon dans le
stockage local du navigateur. Revenir à une conversation restitue son contexte ;
changer de modèle conserve les messages. Les options permettent de renommer ou
d’archiver un fil, puis de le restaurer depuis les archives. Un export JSON de
l’historique est disponible dans les archives. Si le navigateur refuse la
sauvegarde, un message le signale ; les messages restent disponibles en session.
Les conversations ne sont pas écrites dans le dépôt Git ni envoyées à GitHub.

Le bouton **Contexte** ouvre une vue séparée des échanges conservés pour la
prochaine réponse et des messages envoyés à la dernière requête. Elle s’actualise
à la fin de chaque requête, y compris après un arrêt ou une erreur ; les
brouillons et les échanges échoués sans réponse sont exclus du contexte futur.
Pour les moteurs natif et direct, le détail affiche les instructions système, les balises du
modèle et le nombre exact de tokens. Le texte de la dernière requête provient
de la trace réellement utilisée, conservée parmi les huit dernières traces
pendant quinze minutes et effacée au changement de modèle. L’aperçu du contexte
futur est préparé à la demande sans inférence ni modification du cache.
`POST /v1/context` accepte un corps de requête de chat pour cet aperçu, ou
`{"completion_id": "…"}` pour consulter une trace. Le moteur ne résume ni ne
compresse les échanges actuellement ; la réutilisation du cache KV conserve leur
contenu. Pour LM Studio, seuls les messages transmis sont visibles : ses
instructions et sa gestion interne du contexte restent hors de portée de l’app.

Sous chaque réponse figurent les **input tok**, **output tok**, le débit global,
les tokens de contexte réutilisés et le bouton Copier. Les tokens d’entrée
comprennent le contexte complet et le template, pas uniquement le dernier
message. L’onglet **Performances** rassemble les statistiques des huit dernières
réponses du fil, les comparaisons manuelles et les mesures de la machine.
Aucune comparaison ne démarre automatiquement après une réponse.

**Réutilisation du contexte.** Le moteur natif conserve au maximum un cache KV
de préfixe, borné selon la RAM disponible (512 Mio maximum), entre les requêtes. Seuls les tokens identiques sont
réutilisés ; la fin du prompt est évaluée pour obtenir de nouveaux logits.
L’API indique `reused_prompt_tokens` et le libellé **N tok réutilisés** compte les
tokens d’entrée dont le recalcul a été évité par le moteur. Ce compteur n’est
pas présenté comme un gain propre à local-llm et ne prouve aucune accélération
face à LM Studio. Sans mesure, le chat affiche **Cache non mesuré**. Dans
Performances, une calibration qui conserve la référence affiche **Aucun gain
validé**, sans transformer un gain nul en succès. Les statistiques sauvegardées survivent au rechargement de
la page, mais un nouvel essai est nécessaire pour comparer une réponse dont le
serveur ne conserve plus la trace.

Les tests comparent les sorties avec et sans réutilisation, y compris après un
changement de préfixe, un agrandissement du cache et une interruption. Une mesure
sur SmolLM2 360M Q8, pour la seconde requête d’une conversation de 546 tokens
d’entrée, réutilise 517 tokens : prefill médian de 2,778 s à 0,159 s, mêmes tokens
de sortie sur cinq essais alternés. Voir [les mesures](benchmarks/prefix-cache.json).
Ce résultat porte sur cette machine et ce contexte ; le débit de décodage n’est
pas accéléré par cette optimisation.

**Mémoire du modèle.** Les poids restent chargés entre les réponses. Les GGUF
utilisent des fichiers mappés en mémoire : les pages lues deviennent résidentes,
sans forcément copier tous les poids à l’ouverture. Le cache de génération est
conservé pour réutilisation s’il respecte la borne ci-dessus ; sinon il est libéré
à la fin de la requête. Le bouton « Décharger le modèle local » dans la bibliothèque
libère les références aux poids et au cache ; les messages du navigateur restent
enregistrés. LM Studio conserve et libère ses propres modèles dans son processus.

Le débit visible est **global et observé dans le navigateur** : tokens générés
/ durée de la requête entière. Il inclut préparation, génération et transport.
Les détails du moteur indiquent séparément le débit de décodage natif :
`(tokens générés - 1) / temps des passages de décodage`, hors prefill, sampling
et transport. Les deux nombres ne mesurent donc pas la même durée.
Le débit de prefill rapporte uniquement les tokens réellement recalculés au
temps de préparation ; les tokens réutilisés ne gonflent pas cette mesure.

Les réglages se trouvent dans « Réglages de la discussion ». **Longueur de
réponse** propose Courte (128 tokens), Standard (256) et Détaillée (512) pour le
moteur natif ; via LM Studio, les plafonds sont 512, 2048 et 4096 tokens, plus
un plafond personnalisé borné par la configuration du serveur. Il s’agit d’un
maximum ; EOS peut arrêter la réponse plus tôt. Même en local, la génération
consomme du calcul et doit tenir dans le contexte. Une explication est intégrée
à la fenêtre ; si le plafond est atteint, le chat propose de demander la suite
ou de choisir une réponse plus détaillée.

Les modèles LM Studio peuvent émettre leur réflexion dans `reasoning_content`
ou `reasoning`, avant le canal de réponse `content`. L’app affiche cette
réflexion dans un détail replié, la conserve dans l’historique et ne la renvoie
pas comme texte de réponse dans le contexte suivant. Les tokens de sortie et
le débit incluent la réflexion comptée par LM Studio. Si le budget est consommé
avant toute réponse, une explication remplace le message vide. Le plafond LM
Studio est indépendant du moteur natif : `--max-lmstudio-tokens` vaut 8192 par
défaut ; `--max-request-tokens` conserve la borne native de 512. Tant qu’un ancien
serveur tourne, l’interface respecte sa borne et signale qu’il faut le relancer.

**Détection des modèles.** Le chemin du modèle est désormais facultatif :

```bash
local-llm serve
local-llm serve --model-dir /chemin/vers/mes-modeles
local-llm serve --model-dir /premier/dossier --model-dir /second/dossier
```

Le serveur examine `./models`, `~/.lmstudio/models`, l’ancien dossier
`~/.cache/lm-studio/models` et le cache Hugging Face (`HF_HUB_CACHE` ou `HF_HOME`
s’ils sont définis). `LOCAL_LLM_MODEL_DIRS` ajoute des bibliothèques, séparées par
le séparateur de chemins du système. Le scan est limité en profondeur et à 256
modèles ; il ne parcourt pas tout le disque et ne télécharge aucun poids.
Le moteur direct privilégie un petit GGUF compatible ; le mode natif privilégie les petits GGUF Q8.
La bibliothèque affiche aussi les modèles incompatibles avec leur raison. Un
changement de modèle invalide les traces comparatives du serveur et le cache de
préfixe. L’historique et les statistiques des conversations du navigateur restent.
Les poids restent à leur emplacement d’origine.

**Diagnostic CPU.** Le témoin NumPy utilise les mêmes poids quantifiés et le
même cache KV, avec des projections NumPy au lieu des projections natives.
On rejoue jusqu’à huit étapes de la réponse avec les mêmes tokens d’entrée,
en alternant les deux chemins sur trois essais. Les logits sont contrôlés et
les temps bruts restent accessibles. Ce chemin NumPy convertit les blocs de
poids pendant chaque projection : il peut être beaucoup plus lent que le code
natif. Il ne représente pas llama.cpp, LM Studio ou un GPU. Le présenter comme
un gain face à un moteur standard serait trompeur.

L’interface n’affiche donc aucun pourcentage d’accélération pour ce diagnostic.
L’API précise `scope: "projection_diagnostic"` et
`validated_engine_gain: false`. `delta_tokens_per_second` et `speedup` restent
nuls ; les valeurs techniques sont conservées sous
`diagnostic_delta_tokens_per_second` et `diagnostic_ratio` pour l’audit.
Voir [le périmètre des comparaisons](benchmarks/README.md#mesures-dans-linterface).

**Bibliothèque et apparence.** Tous les fichiers détectés sont visibles dans le
bibliothèque ; le sélecteur conserve uniquement les choix exécutables par un
moteur disponible. La recherche permet de retrouver Ling, Qwen et les autres modèles
par nom. Les modèles hors du moteur restent explicitement indiqués ; ils ne
sont jamais chargés silencieusement avec des opérations manquantes. Le dossier
personnalisé `downloadsFolder` de LM Studio est également recherché et relu lors
d’une actualisation. Le thème sombre est activé par défaut ; le bouton Clair /
Sombre conserve le choix dans le navigateur.

**Suivi de la machine.** `GET /v1/system` fournit un relevé local mis en cache
pendant trois secondes, sans verrouiller l’inférence. La RAM du système et la
mémoire résidente du processus local-llm sont distinctes ; la mémoire de LM
Studio apparaît dans le total système, pas dans le processus de cette app.
Sur macOS, la RAM utilisée exclut les pages de fichiers et les pages purgeables
pour rester proche de l’affichage du Moniteur d’activité. Linux utilise
`MemTotal - MemAvailable` ; Windows utilise la RAM physique disponible.

Sur les Mac qui l’exposent, la température est lue directement dans les capteurs
SMC, sans helper ni droits administrateur, puis moyennée sur les capteurs CPU
identifiés. Ces noms de capteurs ne constituent pas une API publique Apple ;
une valeur absente reste indisponible. Linux lit les capteurs CPU hwmon, en
millidegrés Celsius ; Windows affiche indisponible sans fournisseur de capteurs.
Aucune température n’est estimée à partir de la charge. Les sources, la méthode,
l’heure et le périmètre figurent dans Performances. Le polling se suspend lorsque
la page est masquée ; si le serveur est hors ligne, les anciennes valeurs sont
effacées de l’affichage en direct.
Si un ancien serveur est encore lancé après une mise à jour, l’interface affiche
« Serveur à relancer » et une explication. Arrête-le puis relance la commande
habituelle pour charger le nouveau code Python ; actualiser la page ne suffit pas.

**Suggestions pour la machine.** Le bouton « Modèles conseillés » détecte le
processeur, les cœurs logiques et la mémoire physique du serveur local sur macOS,
Linux et Windows, sans envoyer ces informations sur Internet. Une sélection
hors ligne de modèles ouverts Apache 2.0, vérifiée le 2 octobre 2026, est filtrée
par un budget prudent de mémoire : 40 % de la RAM, avec au moins 3 Gio, restent
réservés au système. Les estimations concernent un contexte court, les poids et
le cache ; elles ne prédisent aucun tok/s. Le processeur oriente le choix vers le
petit modèle avec moins de quatre cœurs logiques. Le GPU et la RAM disponible
instantanée ne déterminent pas le classement. Une mémoire inconnue reste
explicitement inconnue ; les gros modèles au-delà du budget sont exclus.

Les fiches officielles de [SmolLM2 360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct),
[SmolLM2 1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) et
[Qwen3 8B GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) sont accessibles depuis
les cartes. Choisir Q8_0 pour les SmolLM2 avec local-llm ; les Q4_K_M et Qwen sont utilisables avec le moteur direct llama.cpp. Le moteur
Python natif garde son périmètre Llama F32/F16/BF16/Q8_0/Q4_0. Aucun modèle
n’est téléchargé automatiquement. `GET /v1/recommendations` expose le matériel,
le budget et les suggestions avec leurs limites.

**LM Studio.** Le serveur détecte sa bibliothèque locale même lorsque l’application
est arrêtée. Pour voir les modèles via son API et mesurer un écart face à son
runtime, active son serveur local (port 1234 par défaut), puis clique sur
« Actualiser » dans **Bibliothèque → Connexion LM Studio optionnelle**. Le tutoriel y explique les
étapes. Choisis les mêmes poids et la même quantification dans les deux moteurs.
Un port différent se configure au lancement :

```bash
local-llm serve --lm-studio http://127.0.0.1:1235
```

Lorsque son serveur local est actif, la bibliothèque distingue les modèles
**téléchargés** de ceux **chargés en mémoire**. Le sélecteur propose uniquement
les checkpoints utilisables par le moteur direct, les modèles natifs et les
instances chargées dans LM Studio après ouverture de sa connexion optionnelle. Le bouton « Charger dans LM Studio » dans la bibliothèque charge les
poids déjà installés via `POST /v1/lmstudio/load` (`{"model": "clé-du-modèle"}`),
sans téléchargement. Les modèles restent en mémoire selon les réglages de
LM Studio ; passer d’une conversation à une autre ne déclenche pas de chargement
LM Studio automatique.

L’état LM Studio est actualisé toutes les cinq secondes lorsque la page est visible
et que cette connexion est utilisée (conversation externe, panneau optionnel ou comparaison),
à l’ouverture du sélecteur et après chaque réponse. Le chat est relayé en
streaming via `/v1/chat/completions` avec `backend: "lmstudio"`, `model` (clé du
catalogue), `model_instance_id` (identifiant exact de l’instance chargée) et
`stream: true`. Si une seule instance est chargée, les clients API peuvent
omettre `model_instance_id`. Une instance déchargée ou appartenant à un autre
modèle est rejetée avant génération. L’identifiant renvoyé dans le flux LM Studio
est vérifié avant de transmettre le texte ; une réponse d’un autre modèle n’est
pas affichée. Le nom du modèle confirmé apparaît sur chaque nouvelle réponse.
Les anciennes réponses conservent leur libellé d’origine.

Un fichier présent sur le PC n’est pas forcément exécutable par notre moteur :
son architecture, son tokenizer et sa quantification doivent être pris en charge.
La bibliothèque conserve les fichiers incompatibles avec leur raison précise.
Ils peuvent être utilisés via LM Studio lorsqu’ils sont exposés comme LLM par
son serveur. Les modèles d’embeddings ne sont pas des modèles de discussion et
ne sont pas proposés dans le chat.
Le débit de ce parcours est le débit global observé, préparation et transport
inclus, calculé seulement si LM Studio fournit son usage en tokens. Il n’est pas
assimilé au débit de décodage natif et aucun gain local-llm n’y est attribué.
Les comparaisons de référence conservent les timings moteur de l’API v0.

Si LM Studio exige une authentification, définis `LM_STUDIO_API_TOKEN` dans
l’environnement du serveur. Le jeton n’est pas envoyé au navigateur. Le client
utilise [la liste des modèles v1](https://lmstudio.ai/docs/developer/rest/list),
avec repli sur v0 pour les versions antérieures, puis les
[complétions brutes v0 et leurs statistiques de moteur](https://lmstudio.ai/docs/developer/rest/endpoints)
pour conserver le prompt déjà rendu. La requête de référence utilise la
température 0. Ce mode affiche un **écart indicatif**, éventuellement négatif :
l’identité des poids, les réglages CPU/GPU et les définitions du débit ne sont
pas vérifiés automatiquement. Aucun résultat de LM Studio n’est présenté comme
une accélération de ses propres kernels par local-llm.

La prise en charge d’une bibliothèque n’ajoute pas celle de nouvelles
architectures. Le moteur CPU natif exécute actuellement Llama et Baguette non hybride,
avec les formats du tableau de support. Les modèles Qwen, MoE, hybrides, les
quantifications K/IQ et les modèles MLX peuvent être détectés sans pouvoir être
exécutés par ce runtime CPU. Les configurations avec RoPE scaling ou biais de
projection non implémentés sont refusées. Les moteurs directs llama.cpp,
MLX-LM et MTPLX ont leurs propres architectures prises en charge, confirmées
au chargement.

La référence PyTorch/Transformers optionnelle reste disponible via l’API avec
`"backend": "reference"` après un lancement avec `--reference` et, si nécessaire,
`--reference-repo`. L’ancien endpoint `/v1/benchmark` conserve ses contrôles
numériques ; son témoin « recalcul complet » est pédagogique et distinct de la
comparaison CPU avec cache KV de la nouvelle interface.

Une réponse JSON contient le texte, l'usage en tokens, le débit du prefill et du
décodage ainsi que la taille du cache KV. `curl` reste utile pour tester l'API
directement, mais n'est pas nécessaire pour utiliser l'interface :

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Quelle est la capitale de la France ?"}],
    "max_tokens": 48,
    "temperature": 0
  }'
```

Pour recevoir le texte au fil de la génération, ajoute `"stream": true`. Le
serveur envoie alors des événements SSE et termine par `data: [DONE]` :

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Explique simplement ce qu est GGUF."}],
    "max_tokens": 80,
    "temperature": 0.7,
    "top_p": 0.9,
    "stream": true
  }'
```

`GET /health` vérifie que le serveur répond et `GET /v1/models` indique le modèle
chargé. L'API reprend la structure principale de Chat Completions, sans prétendre
encore en couvrir toutes les options. Elle n'emploie aucune bibliothèque serveur
externe pour sa façade HTTP ; les générations sont sérialisées pour borner les ressources. Sans `backend`, les appels visent le moteur CPU ; utilisez `backend: "llamacpp"` pour le moteur direct et son identifiant chargé.

Le serveur n'a pas d'authentification et reste donc local par défaut. Les appels
provenant d'une autre origine web sont refusés, le corps doit être du JSON, une
requête est limitée à 512 tokens générés et huit connexions peuvent être ouvertes
simultanément. Ces bornes sont réglables avec `--max-request-tokens` et
`--max-connections`.

Une adresse non locale est refusée sans confirmation explicite. Pour écouter sur
le réseau, il faut ajouter `--host 0.0.0.0 --allow-remote`. Ne le fais que sur un
réseau de confiance ou derrière une couche d'authentification adaptée.

### Modèle Base de validation

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

Q8_0 et Q4_0 sont également exécutables sans déquantifier le modèle complet :

```bash
python -m local_llm run models/SmolLM2-135M.official.Q8_0.gguf \
  --prompt "Bonjour, comment ça va ?" \
  --max-new-tokens 32
```

```bash
python -m local_llm run models/SmolLM2-135M.official.Q4_0.gguf \
  --prompt "Bonjour, comment ça va ?" \
  --max-new-tokens 32
```

Ce second exemple suppose qu'un fichier Q4_0 a été placé dans `models/` ; le
projet n'en télécharge pas automatiquement afin de ne pas consommer d'espace
disque sans confirmation.

La commande `inspect` indique le backend réellement utilisé :

```text
tensor types: F32=65, Q8_0=225
Q8 backend: native-cpp

tensor types: F32=65, Q4_0=225
Q4 backend: native-cpp
```

La variable `LOCAL_LLM_DISABLE_NATIVE=1` force le chemin NumPy pour établir une
baseline ou diagnostiquer le kernel C++.

Sur Apple Silicon, le runtime utilise automatiquement les cœurs performance
pour le décodage à un token et tous les cœurs disponibles pour le prefill. Le
réglage peut être forcé pour mesurer une machine particulière :

```bash
LOCAL_LLM_THREADS=4 python -m local_llm benchmark model.gguf --tokens 48 --runs 5
```

### Profiler le passage avant

La commande `profile` chronomètre séparément les grandes opérations du
Transformer, sans activer cette instrumentation pendant une exécution normale :

```bash
python -m local_llm profile \
  models/SmolLM2-360M-Instruct.official.Q8_0.gguf \
  --prompt "Bonjour, explique le cache KV." \
  --tokens 16
```

Elle affiche le nombre d'appels, le temps total, le temps moyen par appel et la
part de chaque opération (`qkv_projections`, `ffn_gate_up`, attention,
projection vocabulaire, etc.). `--json` produit une sortie exploitable par un
script ou un benchmark automatisé.

## Formats de modèles

### Qu'est-ce que GGUF ?

GGUF est un **format de fichier pour distribuer et charger des modèles**. Ce
n'est ni un LLM particulier, ni un moteur d'inférence, ni une méthode
d'entraînement. Là où un checkpoint Hugging Face est généralement un dossier
contenant plusieurs fichiers, un GGUF regroupe dans un seul fichier :

- les métadonnées de l'architecture (dimensions, couches, RoPE, etc.) ;
- le vocabulaire et les informations du tokenizer ;
- tous les tenseurs de poids, avec leur type (`F32`, `F16`, `Q8_0`, etc.) ;
- éventuellement le template de conversation du modèle.

Cette disposition permet de retrouver rapidement chaque tenseur et de lire les
poids avec un _memory mapping_ (`mmap`) sans copier immédiatement tout le fichier
en mémoire. Un fichier GGUF n'est pas forcément quantifié : le même modèle peut
exister en F16, Q8 ou Q4. La quantification réduit sa taille et sa consommation
mémoire, au prix d'une approximation numérique et avec une vitesse qui dépend
de la qualité des kernels utilisés.

Dans ce projet, notre propre lecteur analyse l'en-tête GGUF, reconstruit la
configuration et le tokenizer, récupère `tokenizer.chat_template` et ses
variantes nommées, associe les noms `blk.N.*` aux couches Llama, puis fournit
les tenseurs au passage avant NumPy. Aucune bibliothèque d'inférence externe
n'exécute le modèle.

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
Les matrices Q8_0 restent compressées en blocs de 32 octets signés avec une
échelle FP16 par bloc. En Q4_0, les 32 valeurs sont stockées dans 16 octets :
chaque demi-octet représente une valeur comprise entre -8 et 7, avec la même
échelle FP16 par bloc. Les embeddings et produits matrice-vecteur sont calculés
directement depuis ces blocs, sans créer une copie déquantifiée du modèle entier.
Un bloc Q4_0 occupe 18 octets contre 34 en Q8_0.

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

## Mesurer les progrès sans changer les règles

La qualité du **modèle** et les performances du **moteur** sont mesurées
séparément. Passer de 135M à 360M peut améliorer les réponses, mais ne constitue
pas une optimisation du runtime. Pour mesurer une optimisation, la commande de
benchmark enregistre l'empreinte SHA-256 exacte des poids, les tokens du prompt,
les tokens générés, l'environnement, le débit et le cache KV :

```bash
python -m local_llm benchmark \
  models/SmolLM2-360M-Instruct.official.F16.gguf \
  --prompt "Bonjour" --tokens 32 --runs 5 \
  --output /tmp/local-llm-baseline.json
```

Après une modification du moteur, la comparaison se lance avec exactement les
mêmes paramètres :

```bash
python -m local_llm benchmark \
  models/SmolLM2-360M-Instruct.official.F16.gguf \
  --prompt "Bonjour" --tokens 32 --runs 5 \
  --compare /tmp/local-llm-baseline.json
```

La comparaison est refusée si le modèle, le prompt ou les tokens gloutons ont
changé. On distingue ainsi trois axes : parité des logits pour la correction,
tokens par seconde pour le moteur, et jeux de questions séparés pour la qualité
du modèle.

### Tester les corrections du moteur

`evaluate` contrôle la correction numérique, indépendamment de la qualité du
modèle. La commande compare le décodage avec cache KV à un recalcul complet,
mesure les écarts de logits, vérifie les tokens gloutons et affiche aussi le
débit et la mémoire du cache :

```bash
python -m local_llm evaluate models/baguette-123m-sft \
  --chat --prompt "Bonjour, comment vas-tu ?" --tokens 8
```

Pour comparer directement le runtime NumPy au checkpoint PyTorch original de
Baguette :

```bash
python -m local_llm evaluate models/baguette-123m-sft \
  --chat --prompt "Bonjour, comment vas-tu ?" --tokens 8 \
  --reference /chemin/baguette-123m-sft.pt \
  --reference-repo /chemin/Baguette \
  --output /tmp/baguette-evaluation.json
```

PyTorch n'est utilisé que pour cette voie de référence. Le passage avant testé
reste celui de `local-llm`. Pour conserver une version connue comme correcte
avant une optimisation, il est également possible d'enregistrer une trace
compressée puis de la rejouer :

```bash
python -m local_llm evaluate models/baguette-123m-sft \
  --chat --prompt "Bonjour, comment vas-tu ?" --tokens 8 \
  --save-reference /tmp/baguette-baseline.npz

# Après la modification du moteur :
python -m local_llm evaluate models/baguette-123m-sft \
  --chat --prompt "Bonjour, comment vas-tu ?" --tokens 8 \
  --reference /tmp/baguette-baseline.npz
```

La commande retourne le code `0` lorsque les logits respectent les tolérances
et que tous les tokens gloutons sont identiques, sinon le code `1`. Une trace
est refusée si les fichiers du modèle ou les tokens du prompt diffèrent : cela
évite de présenter deux expériences différentes comme une régression du moteur.

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

LOCAL_LLM_TEST_BAGUETTE=models/baguette-123m-sft \
  python -m unittest tests.test_real_baguette -v

LOCAL_LLM_TEST_INSTRUCT=models/SmolLM2-360M-Instruct.official.F16.gguf \
  python -m unittest tests.test_real_instruct -v
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

Ajoute `--chat` à l'export pour comparer une invite mise en forme comme une
conversation Instruct.

Transformers n’est importé que par le script d’export. `local_llm run` et
`local_llm verify` effectuent leurs calculs avec le runtime NumPy du projet.

Sur le checkpoint de validation, le tokenizer produit exactement les mêmes IDs,
les tokens gloutons sont identiques et l’écart moyen mesuré sur les logits est
d’environ `1.23e-5` en SafeTensors et `9.19e-6` avec le GGUF F16 officiel,
en calcul CPU F32. Les 16 tokens gloutons de référence sont identiques dans les
deux formats.

Sur le même modèle Q8_0, le chemin C++ accélère ici le décodage d'environ `16×`
par rapport au fallback NumPy. Par rapport à l'ancien kernel C++ scalaire, le
benchmark reproductible passe de `29,1` à `80,2 tok/s` et le prefill de `46,6`
à `170,5 tok/s`. Le gain vient du SIMD NEON, des projections K/V et SwiGLU
fusionnées, des résidus natifs, du choix automatique des threads et de la
suppression d'allocations dans l'attention GQA. Les tokens gloutons restent
identiques et l'écart maximal contre la trace pré-optimisation est `5,15e-5`.
F16 reste rapide grâce à BLAS, tandis que Q8 réduit presque de moitié la taille
des poids. Le kernel natif libère le GIL et distribue les lignes avec Grand
Central Dispatch sur Apple Silicon.

Un cache KV F16 a également été mesuré : il divisait bien la mémoire du cache
par deux, mais ralentissait ce backend d'environ `7,5 %`. Le chemin rapide garde
donc le cache F32 ; une optimisation n'est conservée que lorsqu'elle améliore
réellement la métrique visée.

## Limites et feuille de route

Le chargement direct Llama SafeTensors et GGUF F32/F16/BF16/Q8_0/Q4_0 est opérationnel.
La suite est :

1. mesurer les écarts de logits, la perplexité et la mémoire résidente de Q4_0 ;
2. ajouter un convertisseur F16 vers Q4_0 pour produire nos propres GGUF ;
3. explorer les activations Q8 et la fusion des projections avec validation des logits ;
4. ajouter un tokenizer SentencePiece pour les modèles qui n’utilisent pas BPE ;
5. explorer Metal seulement après les kernels CPU natifs.

Le modèle jouet permet de développer chacune de ces étapes sans confondre les
erreurs de format, de tokenizer, de quantification et de calcul.
