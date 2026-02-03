# Detector_Proyecto

Nombre: Matias Holguin

Fecha:3/02/2026

# DETECTOR DE RAZAS DE PERROS 

Este proyecto utiliza la arquitectura YOLOv11 para la detección y clasificación de razas específicas de perros en imágenes, entrenado en un entorno de Google Colab.

Tecnologías Utilizadas:

Google Colab, YOLOv11, Ultralytics

Dataset: Roboflow (versión 2)

Lenguaje: Google colab, Python

 # Implementación del Proyecto
 
1. Instalación de Dependencias
2. Primero, preparamos el entorno instalando las librerías necesarias de visión artificial y gestión de datos.
```python
!pip install roboflow ultralytics
```
 # 2. Preparación del Dataset
Descargamos los datos directamente desde Roboflow. En este caso el dataset está configurado para reconocer razas como San Bernardo, Husky y Golden Retriever.

```python
from roboflow import Roboflow

rf = Roboflow(api_key="TU_API_KEY_AQUI")

project = rf.workspace("estructuras-de-datos").project("detector_razas_perros")

version = project.version(2)

dataset = version.download("yolov11")
```

 # 3. Entrenamiento del Modelo
Se utilizó el modelo base yolo11s.pt para realizar un Transfer Learning durante 15 épocas.

```python
from ultralytics import YOLO

model = YOLO('yolo11s.pt')

data_path = "/content/Detector_razas_perros-2/data.yaml"

results = model.train(data=data_path, 
                      epochs=15, 
                      imgsz=640)
```
# 4. Inferencia y Pruebas
Una vez obtenido el archivo best.pt (los mejores pesos entrenados), realizamos pruebas con el conjunto de test para validar la precisión.

# Cargamos el modelo ya entrenado 

 ``` python
 
custom_model = YOLO('/content/runs/detect/train/weights/best.pt')
```
# Ejecutamos la detección en la carpeta de test

```python
res = custom_model("/content/Detector_razas_perros-2/test/images")
```
# Ejemplo** (Imagen 10)

python
```
res[9].show()
```

<img width="225" height="225" alt="image" src="https://github.com/user-attachments/assets/f5b2a5c7-2152-4cd3-84db-bda0c2152c10" />


Resultados Obtenidos

<img width="912" height="467" alt="image" src="https://github.com/user-attachments/assets/effc7827-27a8-4a9c-a70e-02ab9292b98c" />


Épocas: 11

Confianza detectada: ~0.62 en las primeras pruebas de inferencia.

Métricas: El modelo muestra una curva de aprendizaje estable en la disminución de pérdida de caja (box_loss).

# Conclusion
En conclusión, la implementación de este detector de razas de perros demuestra la alta eficacia y 
versatilidad de la arquitectura YOLOv11 para resolver problemas complejos de visión artificial mediante 
el uso de aprendizaje profundo. A través de la integración de herramientas como Roboflow para la 
gestión de datos y Google Colab para el procesamiento acelerado por GPU, se logró desarrollar un 
modelo capaz de identificar con precisión ejemplares de San Bernardo y Golden Retriever. Los resultados 
obtenidos tras 15 épocas de entrenamiento muestran una convergencia saludable en las funciones de 
pérdida y un rendimiento sólido, alcanzando una precisión media (mAP50) cercana al 60% y niveles de 
confianza iniciales de 0.41. Si bien el sistema actual funciona correctamente como un prototipo técnico, 
su potencial puede escalarse significativamente mediante la expansión del dataset y el aumento de los 
ciclos de entrenamiento, consolidándose como una base tecnológica robusta para aplicaciones de 
monitoreo o clasificación automatizada en tiempo real.


# Anexos 

<img width="595" height="651" alt="image" src="https://github.com/user-attachments/assets/145b035f-574b-4f30-8958-23eddb094763" />

<img width="620" height="584" alt="image" src="https://github.com/user-attachments/assets/37da1493-f657-40bb-b691-a4a6f1fa2c70" />

<img width="444" height="676" alt="image" src="https://github.com/user-attachments/assets/de541d3e-23cb-46df-8e01-7d665194437f" />

                      
