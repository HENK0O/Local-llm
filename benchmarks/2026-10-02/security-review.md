# Vérification du contenu public — 2 octobre 2026

Périmètre : 12 références publiques récupérées depuis `origin`, 19 commits et
198 versions uniques de fichiers, jusqu’au commit `907529d`. Vérification des
formats courants de clés API (OpenAI, GitHub, AWS, Google, Slack), clés privées,
JWT, secrets littéraux et identifiants dans les URL, puis revue des résultats.

Aucun secret réel n’a été détecté. Les deux résultats d’URL avec identifiants
proviennent d’un exemple fictif `user:password@localhost` dans deux versions
d’un test qui vérifie précisément le rejet de ces URL. Aucun fichier `.env`,
clé privée ou fichier d’identifiants n’était suivi.

Des exemples du README contenaient des chemins locaux avec le nom d’utilisateur.
La version actuelle les remplace par `/chemin/...`. Les anciennes versions
restent dans l’historique Git ; aucun historique public n’a été réécrit.

Les métadonnées publiques des anciens commits contiennent une adresse Gmail
d’auteur. Les nouveaux commits de cette intervention utilisent l’adresse
GitHub `noreply` correspondant à l’identifiant public du propriétaire. L’adresse
des anciens commits reste visible ; le nettoyage du README ne la supprime pas.

Le jeton de LM Studio est lu depuis l’environnement du serveur et n’est pas
enregistré par l’application dans les conversations ou le frontend. Les chats
sont sauvegardés dans le navigateur, pas dans le dépôt. `.gitignore` exclut les
fichiers d’environnement, clés usuelles, logs et exports de conversations.

Cette revue décrit le contenu récupéré et les motifs vérifiés ; elle ne constitue
pas une garantie d’absence de toute information sensible sous une forme inconnue.
