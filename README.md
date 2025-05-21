# AnimeImageClassifier_MobileNet_ViT

## Setting Up

### Navigate to your project folder

### Create a clean virtual environment

```
python -m venv venv
```

### Activate the environment

venv\Scripts\activate

### Upgrade pip inside the venv

```
python -m pip install --upgrade pip setuptools wheel
```

### Now install your dependencies

```
pip install tensorflow-cpu rich markdown-it-py
```

### Download the images in the whatsapp group INTO anime_dataset_raw

Do NOT place it manually in the anime_dataset &rarr; we have the next part for it

### Run the line below and your downloaded images will split randomly

```
python split_dataset.py
```
