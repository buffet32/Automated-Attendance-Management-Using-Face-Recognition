
markdown
# Gestion Automatisée de Présence par Reconnaissance Faciale
 
Système de reconnaissance faciale pour la gestion automatisée de la présence des étudiants. Utilise RetinaFace pour la détection et FaceNet pour la reconnaissance des visages.
 
## Fonctionnalités
 
- **Détection de visages** : Utilisation de RetinaFace pour une détection précise des visages
- **Reconnaissance faciale** : FaceNet (InceptionResnetV1) pour l'extraction et la comparaison des embeddings
- **Interface graphique** : Application Tkinter intuitive pour gérer le système
- **Base de données de visages** : Stockage des embeddings faciaux pour la reconnaissance
- **Journalisation des présences** : Sauvegarde automatique des résultats avec images annotées
- **Support multi-classe** : Gestion de plusieurs classes (3IIR, 4IIR, 5IIR)
- **Ajout dynamique de personnes** : Possibilité d'ajouter de nouvelles personnes à la base de données
 
## Prérequis
 
- Python 3.8+
- CUDA (optionnel, pour accélération GPU)
- Webcam (pour la capture de photos)
 
## Installation
 
1. Cloner le repository :
```bash
git clone https://github.com/buffet32/Automated-Attendance-Management-Using-Face-Recognition.git
cd Automated-Attendance-Management-Using-Face-Recognition
Installer les dépendances :



bash
pip install -r requirements.txt
Utilisation
Lancer l'application principale



bash
python facecount_recognition.py
Initialiser la base de données (optionnel)
Si vous avez des images dans le dossier attendance/ avec des noms formatés (ex: nom_personne_123.jpg), vous pouvez initialiser la base de données :




bash
python create_database.py
Guide de l'interface
Ouvrir une image : Charge une image depuis votre ordinateur
Prendre une photo : Capture une photo via la webcam
Détecter & Reconnaître : Détecte les visages et les reconnaît
Ajouter une personne : Ajoute un visage détecté à la base de données
Voir la base de données : Affiche les personnes enregistrées
Sauvegarder la présence : Enregistre les résultats de reconnaissance
Structure du projet


.
├── facecount_recognition.py      # Application principale avec GUI
├── create_database.py            # Script d'initialisation de la base de données
├── requirements.txt              # Dépendances Python
├── face_database.pkl            # Base de données des visages (généré automatiquement)
├── attendance/                  # Dossier des images de présence et logs
│   ├── attendance_log.txt       # Journal des présences
│   └── *.jpg                    # Images annotées des présences
└── README.md                     # Ce fichier
Dépendances principales
opencv-python
numpy
torch
facenet-pytorch
retina-face
scikit-learn
Pillow
tkinter
Configuration
Le système utilise les paramètres suivants par défaut :

Seuil de reconnaissance : 0.6 (cosine similarity)
Taille minimale du visage : 60 pixels
Device : CUDA si disponible, sinon CPU
Performance
Détection : RetinaFace pour une détection robuste
Reconnaissance : FaceNet pré-entraîné sur VGGFace2
Similarité : Cosine similarity pour la comparaison des embeddings
Notes
Les embeddings sont normalisés (L2 norm)
Le système supporte plusieurs échantillons par personne
Les résultats sont sauvegardés avec horodatage et classe
Licence
Ce projet est fourni à des fins éducatives.

Auteur
buffet32
