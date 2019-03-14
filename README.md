# The bad guys in AI - atacando sistemas de machine learning"

<img src="https://github.com/aliciapj/pycon18-attack/blob/master/figure/results.png?raw=true" height="500">

### Descripción
El código de este repositorio ha sido desarrollado por Alicia Pérez y Javier Ordoñez como demo para la charla
"The bad guys in AI - atacando sistemas de machine learning" en el marco de la PyConES de 2018 en Málaga.

Las transparencias que acompañan este contenido se pueden encontrar [aquí](/slides/PyCon2018_The_bad_guys_in_AI.pdf)

El vídeo de la charla será enlazado tan pronto como la organización lo haga público.

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

#### Slides
- [4:3](https://docs.google.com/presentation/d/1YouJcWetSEbdBBSrXJWGlG-UfErXKsVGDpBn1dXpTKM/edit?usp=sharing)
- [16:9](https://docs.google.com/presentation/d/1ZSzXtfAvod6XOdgCSIsbcpttZ_vZazv7hMEKswioeYE/edit?usp=sharing)

