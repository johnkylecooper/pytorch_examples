import kaggle

kaggle.api.authenticate()

DATASET_NAME = 'uciml/pima-indians-diabetes-database'
kaggle.api.dataset_download_files(DATASET_NAME, path='.', unzip=True)

print(f'Dataset from {DATASET_NAME} retrieved.')