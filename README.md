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

### Download the images in the whatsapp group

```
python split_dataset.py
```
