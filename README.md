# NeuroVision AI — Classification d’IRM cérébrales

> **Projet pédagogique de vision par ordinateur.** NeuroVision AI est un prototype d’aide à l’analyse d’images IRM, **non validé pour un usage clinique**. Il ne produit pas de diagnostic et ne remplace jamais un radiologue ou un médecin. Un résultat « notumor » ne permet pas d’exclure une pathologie.

## 1. Présentation

NeuroVision AI entraîne un réseau de neurones convolutif (CNN) avec **TensorFlow/Keras** afin de classer une image IRM en quatre catégories : `glioma`, `meningioma`, `pituitary`, `notumor`. Le chargement et le prétraitement reposent sur **OpenCV**, NumPy et `tf.data`. Une interface **Streamlit** permet de tester des images, de visualiser la classe et les scores du modèle, et éventuellement une carte **Grad-CAM**.

### Objectifs

- Auditer, explorer et préparer un ensemble d’images de manière robuste.
- Construire un pipeline rapide, traçable et sans fuite de données.
- Développer, entraîner et comparer des CNN multiclasses.
- Évaluer une seule fois sur le test après gel du modèle.
- Analyser les erreurs et expliquer prudemment les zones influentes via Grad-CAM.
- Fournir une démonstration interactive documentée et reproductible.

## 2. Données

Structure attendue (fichiers JPEG/JPG/PNG/BMP) :

```text
data/raw/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
```

Le dataset doit être obtenu séparément et placé dans `data/raw/` : il **n’est pas distribué dans le dépôt**. Vérifier l’origine, la licence, le consentement et les restrictions de redistribution. Ne jamais publier des identifiants patients, des IRM privées ni des exemples ré-identifiables.

**Point critique :** si plusieurs images proviennent du même patient ou de la même étude, réaliser une séparation **par patient/étude** pour éviter les fuites. Si ces identifiants n’existent pas, signaler explicitement le risque résiduel. Détecter également les doublons avant le découpage.

## 3. Structure cible proposée

```text
neurovision-ai/
├── data/
│   ├── raw/                    # hors Git
│   └── processed/              # manifests et partitions (hors Git si sensibles)
├── notebooks/
│   └── 01_eda.ipynb
├── src/neurovision/
│   ├── config.py               # paramètres et chemins
│   ├── data_loader.py          # audit et lecture OpenCV
│   ├── dataset.py              # splits et tf.data
│   ├── preprocessing.py        # resize, normalisation, augmentation
│   ├── model.py                # architecture CNN
│   ├── train.py                # entraînement et callbacks
│   ├── evaluate.py             # métriques et matrice de confusion
│   ├── explain.py              # Grad-CAM
│   └── predict.py              # predict_image
├── app/
│   └── streamlit_app.py
├── tests/
├── artifacts/                  # poids, figures et rapports (à filtrer)
├── docs/
│   ├── roadmap.md
│   └── evaluation.md
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

Cette arborescence est une **cible de développement**, pas l’affirmation que tous les fichiers de code existent déjà.

## 4. Installation (à finaliser avec les scripts)

**Pré-requis :** Python compatible avec la version retenue de TensorFlow, `uv` et un environnement avec ressources suffisantes. CPU possible, GPU facultatif. Vérifier la compatibilité Python/OS/GPU avant installation.

```bash
uv sync
```

Dépendances prévues : `tensorflow`, `opencv-python`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `streamlit`, `pillow`, `pytest`, et éventuellement `pydot`/Graphviz pour `plot_model`. Fixer les versions dans `pyproject.toml` et `uv.lock`.

**Commandes cibles** (disponibles après implémentation des modules indiqués) :

```bash
uv run python -m neurovision.train
uv run python -m neurovision.evaluate
uv run streamlit run app/streamlit_app.py
uv run pytest
```

Si la structure `src/` est utilisée, configurer le packaging pour que `neurovision` soit importable. Ne pas exécuter ces commandes avant que les scripts correspondants aient été créés.

## 5. Pipeline ML

1. **Audit :** extensions, détection des images corrompues, doublons, échantillons et distribution des classes.
2. **Découpage :** train, validation et test distincts, stratifiés quand possible et groupés par patient/étude si possible. Sauvegarder des manifests. Conserver le test inaccessible pendant les choix de modèle.
3. **Lecture :** chargement robuste par OpenCV, couleurs converties explicitement, `224 × 224`, `float32`, pixels dans `[0, 1]`; mapping de classes stable.
4. **Performance :** `tf.data` avec `batch`, `prefetch` et cache lorsque compatible avec la RAM. `shuffle` sur train uniquement.
5. **Augmentation :** petites transformations spatialement plausibles **seulement sur train** ; éviter les modifications anatomiques non justifiées (dont les flips si leur effet n’est pas validé).
6. **Modèle :** `Conv2D → MaxPooling2D`, `Dropout`, `Dense`, sortie `Dense(4, activation="softmax")` ; Adam et taux d’apprentissage explicite.
7. **Perte :** `sparse_categorical_crossentropy` pour labels entiers **ou** `categorical_crossentropy` pour labels one-hot ; ne pas mélanger.
8. **Entraînement :** `fit` avec validation, `ModelCheckpoint`, `EarlyStopping`, `BackupAndRestore`, mesure du temps et historisation des essais.
9. **Sélection :** analyser `accuracy/loss` sur train/validation et choisir la configuration sur validation uniquement.
10. **Évaluation finale :** charger le modèle gelé puis rapport par classe, macro-F1, matrice de confusion, exemples corrects/incorrects, analyse des erreurs et limites.
11. **Explicabilité :** Grad-CAM comme représentation heuristique de zones influentes ; ce n’est ni une segmentation médicale ni une preuve causale.
12. **Inférence :** réutiliser le même prétraitement que pendant l’entraînement, afficher scores et avertissement dans Streamlit.

## 6. Métriques et critères de validation

| Indicateur | Pourquoi |
|---|---|
| Accuracy | Vue globale, sensible au déséquilibre |
| Macro-F1 | Traite chaque classe de manière égale |
| Précision/rappel par classe | Révèle les défaillances par catégorie |
| Matrice de confusion | Identifie les classes confondues |
| Loss train/validation | Détecte sous- et surapprentissage |
| Temps d’entraînement | Suit coût et reproductibilité |

Ne pas annoncer de seuil clinique de performance sans protocole médical et validation externe. Les probabilités softmax ne sont **pas** automatiquement calibrées et ne constituent pas un niveau de certitude clinique.

## 7. Planification Jira — 15 jours ouvrés

- **Début proposé :** mardi 13 octobre 2026. **Fin :** lundi 2 novembre 2026, hors jours fériés et congés.
- **7 Epics**, **40 tickets** (Stories et Tasks), jalons avec dépendances, priorités, story points et critères d’acceptation.
- Export fourni : `jira_neurovision_ai.csv`.

| Phase | Jours | Résultat attendu |
|---|---|---|
| Fondations et gouvernance | J1–J2 | Projet, environnement et risques documentés |
| Audit et exploration | J2–J4 | Rapport d’intégrité des IRM et EDA |
| Pipeline de données | J4–J6 | Loader et splits fiables ; augmentation maîtrisée |
| Architecture CNN | J7–J8 | Modèle de référence compilé |
| Entraînement | J8–J10 | Meilleur modèle sauvegardé et expériences comparées |
| Évaluation et XAI | J11–J12 | Rapport final, erreurs et Grad-CAM |
| Interface et livraison | J13–J15 | Streamlit, tests, documentation et démonstration |

### Import Jira Cloud

1. Ouvrir l’import CSV Jira (disponible selon droits du projet).
2. Importer `jira_neurovision_ai.csv` avec encodage **UTF-8** (BOM inclus).
3. Mapper `Summary → Summary`, `Work Type → Issue Type`, `Description → Description`, `Priority → Priority`, `Labels → Labels`, `Due Date → Due Date`, `Status → Status` si l’import le permet. Les dates sont au format `dd/MMM/yy` ; adapter le format dans l’assistant Jira si nécessaire.
4. `Story Points`, `Start Date` et `Epic Name` sont **facultatifs** et leur disponibilité dépend du type de projet/configuration Jira. Les colonnes `Issue ID`, `Epic Group`, `Dependencies`, `Acceptance Criteria` sont des **références de planification** : les mapper à des champs texte personnalisés s’ils existent, sinon les ignorer (leur contenu essentiel est aussi dans Description).
5. Créer/importer d’abord les **Epics** puis les Stories/Tasks si Jira ne gère pas les relations lors d’un seul import. Relier ensuite les tickets à leur Epic via `Epic Group` (`E01` à `E07`) et créer les liens **blocks/is blocked by** d’après `Dependencies` (`NV-xx`). Les identifiants `NV-xx` sont internes au CSV, **pas** des clés Jira créées à l’avance. Ne pas mapper directement les `Dependencies` sur un champ de liens Jira sans conversion en clés réelles.
6. Vérifier quelques tickets après import (types, priorités, dates, accents, liens et descriptions). Si l’instance exige une valeur de priorité ou de statut différente, adapter les valeurs avant l’import.

## 8. Livrables et Definition of Done

- Code modulaire, testé et reproductible avec dépendances verrouillées.
- Pipeline de données robuste, partitionnement documenté et rapports d’EDA.
- CNN retenu et paramètres sauvegardés au format `.keras` ; ordre des classes sauvegardé.
- Historique de l’entraînement et figures ; rapport final de test ; visualisations d’erreurs et Grad-CAM.
- Interface Streamlit avec validation des fichiers et avertissement non clinique.
- README, journal des limitations, licence/provenance du dataset et tickets Jira renseignés.

**Definition of Done par ticket :** code ou document versionné, critères d’acceptation vérifiés, absence de données privées dans le dépôt, décisions expliquées, aucune erreur bloquante ouverte.

## 9. Limites et précautions

Le jeu de données peut ne pas représenter la diversité des appareils, centres, protocoles, patients et pathologies. La qualité d’une évaluation sur un corpus public n’établit pas la sûreté ni l’efficacité clinique. La validation externe, la gestion des biais, la sécurité des données, la supervision humaine et les obligations réglementaires seraient indispensables avant tout usage réel.

## 10. Documentation conseillée

- TensorFlow/Keras : https://www.tensorflow.org/tutorials/images/classification
- `tf.data` : https://www.tensorflow.org/guide/data
- Keras callbacks : https://keras.io/api/callbacks/
- OpenCV : https://docs.opencv.org/
- Scikit-learn : https://scikit-learn.org/stable/modules/model_evaluation.html
- Streamlit : https://docs.streamlit.io/
