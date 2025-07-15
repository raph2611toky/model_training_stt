
### Utilisation des scripts
- **Entraînement initial** :
  ```bash
  python train.py
  ```
  Le modèle sera entraîné, évalué, et sauvegardé dans `models/` ou `models_inutilisables/` selon son WER.
- **Test d’un modèle** :
  ```bash
  python test.py --model_zip ./models/model_abcdEfGh_202507142054.zip
  ```
- **Entraînement supplémentaire** :
  ```bash
  python train.py --continue_from ./models/model_abcdEfGh_202507142054.zip
  ```
  Le modèle sera chargé depuis le zip, entraîné davantage, puis sauvegardé à nouveau.
