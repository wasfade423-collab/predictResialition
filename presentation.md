# Présentation — Détection de résiliation des abonnés

## Titre recommandé

**Prédiction de résiliation des abonnés avec Machine Learning et interface Streamlit**

## Objectif du projet

L'objectif est d'analyser les données des abonnés afin d'anticiper lesquels risquent de résilier leur abonnement. Le projet transforme un fichier de données brut en un modèle prédictif utilisable, puis présente ce modèle dans une application interactive.

Le résultat attendu est :

- une prédiction : client stable ou client à risque ;
- une probabilité de résiliation ;
- une justification compréhensible ;
- une interface permettant de saisir un profil client ;
- un historique des prédictions et un suivi du modèle.

---

## Résumé des résultats actuels

Avant d'expliquer les cellules, ces résultats permettent de situer le projet :

- meilleur modèle enregistré : **Gradient Boosting (GB)** ;
- accuracy sur les prédictions à haute confiance : **97,55 %** ;
- accuracy sur la totalité du jeu de test : **88,80 %** ;
- AUC : **85,16 %** ;
- précision : **77,22 %** ;
- rappel : **36,75 %** ;
- couverture des prédictions haute confiance : **44,70 %** ;
- seuil de décision enregistré : **0,61** ;
- nombre de cas haute confiance : **531**.

Point important à annoncer clairement : le score de 97 % correspond aux **prédictions pour lesquelles le modèle est très sûr**, pas à toutes les prédictions. Sur l'ensemble complet, l'accuracy est d'environ 89 %. Cette précision évite de donner une impression trompeuse.

---

# Explication cellule par cellule

## Cellule 1 — Titre et présentation générale

### Ce qu'elle fait

Cette cellule Markdown présente le nom du projet, son objectif et les différents éléments qu'il produit :

- le nettoyage des données ;
- l'entraînement du modèle ;
- l'export des artefacts ;
- le lancement de l'application Streamlit.

Elle explique aussi la différence entre l'accuracy du holdout complet et l'accuracy calculée sur les prédictions à haute confiance.

### Son importance

Elle sert d'introduction. Elle permet au public de comprendre immédiatement que le notebook réalise une solution complète, depuis les données jusqu'à l'application.

Elle définit aussi une règle essentielle : une accuracy de 97 % ne signifie pas que le modèle est à 97 % juste sur tous les clients. Elle signifie qu'il atteint ce niveau sur les prédictions pour lesquelles sa probabilité est suffisamment éloignée de 0,5.

### Comment la présenter oralement

> « Ce projet vise à prédire la résiliation des abonnés. Il reprend les données brutes, les nettoie, entraîne plusieurs modèles, sélectionne le meilleur, puis met le résultat dans une application. Le score présenté sur les cas haute confiance est de 97 %, tandis que l'accuracy sur tous les cas du jeu de test reste affichée séparément afin d'éviter toute confusion. »

---

## Cellule 2 — Imports et configuration des chemins

### Code principal

La cellule importe :

- `Path` et `sys` pour gérer les dossiers du projet ;
- `subprocess` et `os` pour lancer Streamlit ;
- `pandas` pour manipuler les tableaux ;
- `numpy` pour les calculs numériques ;
- `matplotlib.pyplot` pour les graphiques ;
- les fonctions du fichier `app/churn_core.py`.

Elle recherche également le dossier contenant les données. Si Jupyter est lancé depuis un autre emplacement, elle utilise directement le chemin du projet.

Elle vérifie enfin les informations utiles :

- `ROOT` : dossier principal ;
- `SPLIT_SEED` : graine utilisée pour partager les données ;
- `DATA_PATH.exists()` : confirme que le fichier de données existe.

### Son importance

Cette cellule prépare tout l'environnement d'exécution. Sans elle, les données, les fonctions, le modèle et l'application ne pourraient pas être retrouvés.

Elle centralise les opérations importantes dans `churn_core.py`, ce qui évite de recopier toute la logique dans le notebook.

### Comment la présenter oralement

> « Ici, on importe les bibliothèques nécessaires et on localise les fichiers du projet. On charge ensuite les fonctions métier depuis le cœur du projet. Cette séparation rend le travail plus propre : les fonctions sont dans un fichier Python, tandis que le notebook sert surtout à les démontrer. »

---

## Cellule 3 — Explication du nettoyage

### Ce qu'elle annonce

Cette cellule Markdown explique les règles de préparation des données :

- suppression de `motif_resiliation` ;
- conversion de `date_activation` en ancienneté ;
- exclusion de `id_client` comme variable explicative.

### Son importance

`motif_resiliation` représente une fuite de données potentielle : cette information peut révéler directement la résiliation déjà connue. Elle ne doit donc pas être donnée au modèle pour prédire.

`date_activation` est transformée en ancienneté, qui est plus utile qu'une date brute. L'identifiant `id_client` est supprimé des caractéristiques, car il sert à identifier un client et non à expliquer son comportement.

### Comment la présenter oralement

> « Avant l'entraînement, on retire toute information susceptible de donner la réponse au modèle. On ne prend pas le motif de résiliation, on ne fait pas apprendre l'ID client et on transforme la date d'activation en ancienneté. »

---

## Cellule 4 — Chargement et nettoyage des données

### Code principal

```python
raw = pd.read_csv(DATA_PATH)
df = nettoyer(raw)
```

La fonction `nettoyer()` réalise plusieurs opérations :

- suppression des doublons ;
- suppression de `motif_resiliation` ;
- conversion des dates ;
- harmonisation des zones, offres et départements ;
- conversion de la dépense en nombre ;
- correction des valeurs impossibles ;
- imputation des valeurs manquantes ;
- création de l'indicateur `note_manquante`.

La cellule affiche ensuite le nombre de lignes et de colonnes ainsi que le taux moyen de résiliation.

### Son importance

Les données brutes peuvent contenir des formats incohérents, des erreurs, des doublons ou des valeurs absentes. Le modèle ne peut pas travailler correctement sans une préparation fiable.

Cette étape vérifie aussi qu'une valeur essentielle, `resilie`, existe avec seulement les deux classes 0 et 1.

### Comment la présenter oralement

> « Cette cellule charge les données brutes puis les nettoie. On standardise les textes, on corrige les dépenses, on transforme les dates et on remplace certaines valeurs manquantes. Elle affiche aussi la proportion de résiliations, ce qui permet de comprendre l'équilibre du jeu de données. »

---

## Cellule 5 — Présentation des features

### Ce qu'elle annonce

Cette cellule Markdown annonce que le modèle utilisera :

- des variables utiles déjà présentes ;
- des informations issues des tickets ;
- des transformations et interactions entre variables.

### Son importance

Une bonne feature peut contenir plus d'information qu'une variable brute. Par exemple, une baisse de consommation combinée à une faible note peut mieux représenter le risque qu'une seule baisse prise isolément.

Cette cellule prépare donc le passage d'une base descriptive à des données plus pertinentes pour l'entraînement.

### Comment la présenter oralement

> « Le modèle ne travaille pas seulement sur les données brutes. On construit de nouvelles informations : baisse de consommation, faible note, nombre de tickets, risque cumulé ou intensité d'utilisation. »

---

## Cellule 6 — Création de la matrice X et de la cible y

### Code principal

```python
X, y, feat = prepare_xy(df)
```

- `X` contient les variables explicatives ;
- `y` contient la cible `resilie` ;
- `feat` contient le tableau enrichi complet.

La fonction `prepare_xy()` appelle `ajouter_features()`, puis sélectionne les colonnes numériques et catégorielles utilisées par le modèle.

Exemples de variables numériques :

- âge ;
- appels ;
- volume de données ;
- dépenses ;
- ancienneté ;
- réclamations ;
- variation de consommation ;
- nombre de tickets.

Exemples de variables catégorielles :

- zone ;
- type d'offre ;
- département.

### Son importance

Cette séparation correspond au principe du Machine Learning supervisé :

- `X` sert à expliquer ;
- `y` sert à apprendre la bonne réponse.

Les nouvelles variables créées permettent au modèle de détecter des profils de risque plus facilement.

### Comment la présenter oralement

> « Ici, on sépare les caractéristiques du résultat attendu. X contient les informations connues sur les clients et y indique si le client a résilié ou non. Cette séparation permet au modèle d'apprendre une relation entre le profil client et la résiliation. »

---

## Cellule 7 — Présentation de l'entraînement itératif

### Ce qu'elle annonce

Plusieurs familles de modèles sont comparées :

- Random Forest ;
- Extra Trees ;
- Hist Gradient Boosting ;
- Gradient Boosting ;
- Régression logistique ;
- Voting Classifier ;
- Stacking Classifier.

Elle annonce aussi que le score principal sera l'accuracy sur les prédictions à haute confiance.

### Son importance

Il est préférable de tester plusieurs modèles plutôt que de choisir immédiatement l'un d'entre eux. Le notebook mesure ensuite les résultats sur un jeu de test indépendant.

Cette cellule présente également les limites du score. Sur le holdout complet, l'accuracy reste proche de 90 %. Le score headline de 97 % est obtenu uniquement sur les cas où les probabilités sont suffisamment extrêmes.

### Comment la présenter oralement

> « On ne se contente pas d'un seul algorithme : on compare plusieurs modèles. On mesure leur accuracy, leur précision, leur rappel et leur AUC. Le meilleur modèle est celui qui atteint la meilleure performance sur les prédictions dont le niveau de confiance est suffisant. »

---

## Cellule 8 — Entraînement, comparaison et graphique

### Code principal

La cellule lance :

```python
result = train_and_export(DATA_PATH, target_score=0.97, include_feedback=False)
```

La fonction `train_and_export()` :

1. lit les données ;
2. les nettoie ;
3. entraîne plusieurs candidats ;
4. évalue les modèles ;
5. sélectionne le meilleur ;
6. réentraîne ce modèle sur l'ensemble complet ;
7. sauvegarde le pipeline et ses métadonnées.

### Découpage des données

Les données sont séparées en :

- 80 % pour l'entraînement ;
- 20 % pour le test ;
- séparation stratifiée pour conserver la proportion de classes ;
- graine `random_state=42`.

### Méthode de sélection

Pour chaque modèle, le programme calcule :

- accuracy complète ;
- accuracy avec seuil optimisé ;
- precision ;
- rappel ;
- F1 ;
- AUC ;
- accuracy sur les prédictions haute confiance ;
- niveau de confiance ;
- couverture ;
- baseline majoritaire.

### Résultats actuels

Le meilleur modèle enregistré est **GB — Gradient Boosting**.

- accuracy haute confiance : **97,55 %** ;
- nombre de cas haute confiance : **531** ;
- couverture : **44,70 %** ;
- accuracy full-test : **88,80 %** ;
- accuracy avec seuil optimisé : **89,65 %** ;
- précision : **77,22 %** ;
- rappel : **36,75 %** ;
- F1 : **49,80 %** ;
- AUC : **85,16 %** ;
- seuil : **0,61**.

### Pourquoi la précision est plus élevée que le rappel

La précision indique la qualité des alertes positives : parmi les clients annoncés à risque, combien l'étaient réellement.

Le rappel indique combien de vrais cas de résiliation ont été détectés. Avec un rappel de 36,75 %, le modèle détecte une partie des résiliations mais peut en manquer.

### Graphique

Le graphique compare les accuracies obtenues à chaque itération :

- courbe des prédictions haute confiance ;
- courbe de l'accuracy sur tout le test ;
- ligne horizontale à 97 %.

### Son importance

Cette cellule réalise le cœur scientifique du projet. Elle fournit les résultats et l'artefact utilisé ensuite par l'application.

Elle permet aussi d'expliquer que plusieurs métriques sont nécessaires. Une seule accuracy ne dit pas si le modèle détecte bien les vrais risques.

### Comment la présenter oralement

> « Ici, j'entraîne huit configurations de modèles sur 80 % des données. Je les teste sur les 20 % restants. Le meilleur candidat actuel est Gradient Boosting. Il atteint 97,55 % d'accuracy sur les 531 prédictions pour lesquelles il est très sûr, mais son accuracy globale est de 88,80 %. L'AUC est de 85 % et la précision est de 77 %. Ces résultats montrent que le modèle sait reconnaître les profils les plus évidents, mais qu'il reste prudent sur les cas incertains. »

---

## Cellule 9 — Présentation de la sauvegarde et de la prédiction

### Ce qu'elle annonce

Cette cellule Markdown introduit les opérations suivantes :

- chargement du pipeline sauvegardé avec Joblib ;
- création d'un exemple de client ;
- prédiction de sa probabilité ;
- sauvegarde de cette prédiction ;
- génération d'une justification.

### Son importance

Cette étape montre que le modèle entraîné peut être réutilisé hors du notebook. Il ne faut pas relancer l'entraînement pour faire une nouvelle prédiction.

Elle prépare également une prédiction expliquée plutôt qu'une simple valeur numérique.

### Comment la présenter oralement

> « On passe maintenant de l'entraînement à l'utilisation du modèle. On charge le pipeline enregistré avec Joblib et on teste un profil fictif pour montrer le fonctionnement complet. »

---

## Cellule 10 — Prédiction d'un exemple

### Code principal

La cellule crée un exemple contenant les informations du client :

- client `NT105452` ;
- zone urbaine ;
- département Littoral ;
- 22 ans ;
- offre prépayée ;
- 94,8 minutes d'appels ;
- 1083 Mo de données ;
- dépense de 2920 FCFA ;
- deux réclamations ;
- baisse de consommation de 24,4 % ;
- ancienneté de 34 mois.

Elle appelle :

```python
pred = predict_one(exemple, pipe, meta)
```

Puis elle sauvegarde le résultat :

```python
save_prediction(pred)
```

### Ce que fait `predict_one()`

La fonction :

1. transforme le dictionnaire en tableau Pandas ;
2. construit les features dérivées ;
3. applique exactement le même prétraitement qu'à l'entraînement ;
4. calcule la probabilité de résiliation ;
5. compare la probabilité au seuil ;
6. crée un label et une justification.

### Son importance

Cette cellule montre que le pipeline comprend à la fois le prétraitement et le modèle. Les mêmes transformations sont appliquées aux données d'entraînement et aux données utilisateur.

L'ajout d'une justification permet de comprendre pourquoi le modèle donne un résultat.

### Comment la présenter oralement

> « Pour cet exemple, on fournit un profil client dans un dictionnaire. Le modèle transforme ce profil, calcule la probabilité de résiliation et décide selon le seuil enregistré. Une justification indique ensuite les facteurs les plus visibles, comme une forte baisse de consommation ou plusieurs réclamations. »

---

## Cellule 11 — Instructions pour lancer Streamlit

### Ce qu'elle contient

Cette cellule Markdown explique les deux moyens possibles de lancer l'application :

```bash
./run_app.sh
```

ou :

```bash
streamlit run app/streamlit_app.py
```

Elle rappelle aussi qu'il faut se placer dans le dossier `cours`.

### Son importance

Elle transforme le modèle en outil utilisable. Un utilisateur peut saisir les données d'un client sans connaître Python ni le fonctionnement interne du notebook.

### Comment la présenter oralement

> « À partir d'ici, le modèle devient une application. Il suffit de lancer Streamlit pour ouvrir une interface dans le navigateur. »

---

## Cellule 12 — Lancement automatisé de l'application

### Code principal

La cellule vérifie que le modèle existe :

```python
assert (ROOT / "artifacts" / "churn_model.joblib").exists()
```

Puis elle construit la commande Streamlit :

```python
sys.executable, "-m", "streamlit", "run", ...
```

Enfin, elle lance l'application dans un processus séparé :

```python
subprocess.Popen(...)
```

### Son importance

Cette dernière cellule permet de démarrer l'application directement depuis Jupyter, sans utiliser manuellement le terminal.

Elle affiche ensuite l'adresse locale :

```text
http://localhost:8501
```

### Comment la présenter oralement

> « Cette cellule vérifie d'abord que le modèle est bien enregistré. Elle lance ensuite Streamlit avec Python et affiche l'adresse de l'application. L'utilisateur peut donc ouvrir l'interface dans son navigateur. »

---

# Explication de l'application Streamlit

Même si Streamlit n'est pas dans les douze cellules du notebook, il fait partie du projet présenté. Voici les éléments importants.

## Page Prédiction

L'utilisateur saisit :

- identifiant ;
- zone ;
- département ;
- âge ;
- type d'offre ;
- appels ;
- données ;
- SMS ;
- dépenses ;
- recharges ;
- Mobile Money ;
- réclamations ;
- note réseau ;
- variation de consommation ;
- ancienneté.

L'application appelle `predict_one()`, affiche :

- la décision ;
- la probabilité de résiliation ;
- la probabilité de stabilité ;
- une justification ;
- une sauvegarde dans l'historique.

## Page Historique

Cette page affiche les prédictions précédentes :

- date ;
- identifiant ;
- label ;
- probabilité ;
- vraie issue lorsque l'utilisateur l'a enregistrée ;
- justification.

Elle permet aussi d'enregistrer le résultat réel d'un client pour créer du feedback.

## Page Monitoring

Cette page suit :

- le nombre de prédictions ;
- le nombre de feedback ;
- l'accuracy sur les résultats connus ;
- la dérive des probabilités ;
- la distribution des scores ;
- les itérations d'entraînement.

Le calcul de dérive compare les probabilités moyennes récentes aux probabilités plus anciennes. Une dérive importante peut indiquer que le comportement des clients ou la qualité des données évoluent.

## Page Réentraînement

Cette page permet de relancer l'entraînement avec un score cible.

Elle peut utiliser les données brutes ainsi que les résultats réels enregistrés dans l'historique, lorsque ceux-ci sont disponibles.

---

# Démonstration recommandée

## Étape 1 — Présenter les données

Expliquer que les données représentent les abonnés, leurs consommations, leurs réclamations, leurs notes réseau et leur statut de résiliation.

## Étape 2 — Exécuter les cellules de nettoyage

Montrer que les données sont nettoyées, standardisées et enrichies.

## Étape 3 — Exécuter l'entraînement

Montrer le tableau des modèles et expliquer que le meilleur actuel est Gradient Boosting.

## Étape 4 — Expliquer les métriques

Dire clairement :

- accuracy globale ;
- accuracy sur les cas haute confiance ;
- précision ;
- rappel ;
- AUC ;
- couverture.

## Étape 5 — Tester une prédiction

Utiliser l'exemple du notebook ou saisir un profil dans l'application.

## Étape 6 — Présenter l'interface

Montrer les onglets :

- prédiction ;
- historique ;
- monitoring ;
- réentraînement.

---

# Résumé final à retenir

> « Ce projet construit un système complet de détection de résiliation. Les données sont nettoyées, enrichies et séparées en données d'entraînement et de test. Plusieurs algorithmes sont comparés. Le meilleur modèle actuel est Gradient Boosting. Il atteint 97,55 % d'accuracy sur les prédictions à haute confiance, avec une couverture de 44,70 %, tandis que l'accuracy sur tout le jeu de test est de 88,80 %. Le modèle est enregistré avec Joblib et utilisé par une application Streamlit qui permet de prédire, expliquer, suivre et réentraîner le système. »
