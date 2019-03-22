# The bad guys in AI - atacando sistemas de machine learning"

<img src="https://github.com/aliciapj/pycon18-attack/blob/master/figure/results.png?raw=true" height="500">

### Descripción
El código de este repositorio ha sido desarrollado por [Alicia Pérez](https://github.com/aliciapj), [Javier Ordoñez](https://github.com/fjordonez) y [Beatriz Gómez](https://github.com/gomezabeatriz) como demo para la charla
"The bad guys in AI - atacando sistemas de machine learning" en el marco de la PyConES de 2018 en Málaga y en el T3chfest 2019.

Las transparencias que acompañan al contenido de la PyConES se pueden encontrar [aquí](/slides/PyCon2018_The_bad_guys_in_AI.pdf). Las transparencias del T3chfest 2019 están en este [enlace](https://docs.google.com/presentation/d/1YouJcWetSEbdBBSrXJWGlG-UfErXKsVGDpBn1dXpTKM/edit#slide=id.g35f391192_00).

El vídeo de la charla de la PyConES está disponible en el siguiente [enlace](https://www.youtube.com/watch?v=D2m9Ejx6S9k), y el vídeo correspondiente a la charla del T3chfest 2019 [aquí](https://youtu.be/d-8DdW7MTxQ).

### Oráculo - Modelo discriminativo

#### Instalación
Instalar el fichero de dependencias de la carpeta `attack` con pip
```
pip install -r discriminative/requirements.txt
```

#### Principales dependencias
- Pillow 5.2
- Numpy 1.14
- Keras 2.2

#### Entrenamiento del modelo
```
python discriminative/model_manager.py --train /path/to/train/data
```

#### Clasificación
```
# img_str puede ser una url o un path a la imagen
python discriminative/model_manager.py --predict img_str
```

### Ataque de caja negra

#### Principales dependencias
- Pillow 5.2
- Numpy 1.14
- Keras 2.2
- Cleverhans

#### Instalación

Instalar el fichero de dependencias de la carpeta `attack` con pip
```
pip install -r attack/requirements.txt
```
