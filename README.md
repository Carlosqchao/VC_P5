# VC_P5: Detector de caras y clasificador de emociones mediante CNN

## Autores
- Carlos Ruano Ramos  
- Juan Boissier García

Descripción general
-------------------
Este repositorio recoge el notebook utilizado para entrenar y desplegar un clasificador de emociones sobre FER2013 y un módulo de inferencia en tiempo real que detecta caras en la webcam y aplica representaciones visuales según la emoción detectada.

Resumen de tareas realizadas
---------------------------
A continuación se describen, por tareas, las acciones implementadas en el notebook.

Entrenamiento de la red neuronal,
esta parte se realizó en Kaggle: https://www.kaggle.com/code/carlosruanoramos/vc-p5
-------------------------------
1) Preparación y carga de datos
- Configuración de ImageDataGenerator para entrenamiento (rescale + augmentations: rotaciones, desplazamientos, shear, zoom, flip horizontal y fill_mode) y para evaluación (rescale).
- Creación de generadores con flow_from_directory para leer las imágenes en modo grayscale con target_size=(48,48) y class_mode='categorical'.
- Comprobaciones: impresión de class_indices y número de muestras en train y test. Nota sobre la estructura requerida TRAIN_DIR/<clase>/* y TEST_DIR/<clase>/*.

2) Definición de la arquitectura
- Implementación de create_emotion_cnn() que construye un modelo Keras con:
  - Bloques convolucionales con Conv2D y BatchNormalization, seguidos por MaxPooling2D y Dropout.
  - Capas densas finales con Flatten, Dense y Dropout, terminando en una capa softmax con tantas salidas como clases.
- Input shape: (48, 48, 1) — imágenes en escala de grises.

3) Compilación y callbacks
- Compilación con Adam(learning_rate=1e-4), loss categorical_crossentropy y métrica accuracy.
- Callbacks configurados:
  - EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
  - ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
  - ModelCheckpoint('best_emotion_model.h5', monitor='val_accuracy', save_best_only=True)

4) Entrenamiento y guardado
- Ejecución de model.fit() con train_generator y validation_data=test_generator; historial almacenado en `history`.
- Guardado final del modelo con model.save('final_emotion_model.h5').
- Generación de la figura de historial (accuracy y loss) guardada como `training_history.png`.

5) Evaluación y métricas
- Evaluación con model.evaluate(test_generator): impresión de test_loss y test_accuracy.
- <img width="1389" height="490" alt="image" src="https://github.com/user-attachments/assets/07e454fb-1c50-48b6-a4b1-e6417eb1ebfa" />

- Cálculo de predicciones sobre test: predictions = model.predict(test_generator), y_pred = argmax(predictions, axis=1), y_true = test_generator.classes.
- Generación de classification_report (precision, recall, f1) y matriz de confusión; guardado de `confusion_matrix.png`.
- <img width="931" height="790" alt="image" src="https://github.com/user-attachments/assets/da8bf5c5-a9fb-4643-909e-2e9ac62a740c" />

Filtros para inferencia en tiempo real
-------------------------------------
El módulo de inferencia en tiempo real aplica diferentes representaciones visuales sobre las caras detectadas. A continuación se desglosan los dos filtros principales implementados en el notebook.

Filtro A — Overlay PNG con canal alfa
- Qué hace:
  - Carga imágenes PNG (u otros assets) por emoción con cv2.IMREAD_UNCHANGED para conservar el canal alpha.
  - Redimensiona y centra el PNG sobre la región de la cara detectada y mezcla el PNG con el frame usando el canal alfa.
- Implementación:
  - Diccionario de rutas de archivos por emoción (ej.: angry.png, happy.png, ...).
  - Función overlay_png_on_frame(frame, png, x, y, w, h, scale=1.0) que ajusta tamaño y realiza la mezcla pixel a pixel gestionando recortes fuera del frame.
  - En cada frame, si existe el PNG correspondiente a la emoción detectada, se invoca esta función para superponer el asset.
- Recursos generados/esperados:
  - PNGs de overlays opcionales: angry.jpg, disgust.png, fear.png, happy.png, neutral.png, sadness.png, surprise.png.

Filtro B — Efectos gráficos (fallback)
- Qué hace:
  - Dibuja elementos gráficos directamente sobre el frame sin depender de archivos externos: rectángulos alrededor de la cara, banner con texto (emoción en español + confianza) y elementos decorativos (círculos/partículas).
  - Además, si se detecta la emoción "Cabreado" (angry), el sistema reproduce sonidos y aumenta un contador; al alcanzar un umbral predefinido el proceso cierra la cámara automáticamente.
- Implementación:
  - Función add_emotion_effects(frame, emotion_idx, x, y, w, h, confidence) que decide la representación a aplicar:
    - Si el PNG existe y se puede leer, llama a overlay_png_on_frame.
    - Si no, dibuja el rectángulo, el banner de texto con la emoción en español y otras marcas visuales definidas.
    - En el caso de "Cabreado", activa la reproducción de sonido y gestiona el contador que puede desencadenar el cierre de la captura de vídeo.
  - Mapeos de etiquetas e internacionalización:
    - emotions = ['angry','disgust','fear','happy','neutral','sad','surprise']
    - emotions_spanish = ['Cabreado','Asqueado','Asustado','Feliz','Neutral','Triste','Sorprendido']

Detalles comunes a la inferencia en tiempo real
- Detector de rostros:
  - Uso de Haar cascade de OpenCV: haarcascade_frontalface_default.xml (cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').
  - DetectMultiScale con parámetros usados en el notebook.
- Flujo por frame:
  - Detector localiza rostros; para cada rostro se extrae la ROI, se convierte a gris, se redimensiona a 48×48, se normaliza y se predice la emoción con la CNN.
  - Se aplica el filtro correspondiente (Overlay PNG o Efectos fallback) y se renderiza texto con la emoción y la confianza.
- Ejecución:
  - Captura de vídeo con cv2.VideoCapture(0) y procesamiento, con la posibilidad de ejecutar la inferencia en un hilo separado mediante threading.Thread.

Salidas y artefactos generados
------------------------------
- best_emotion_model.h5  (ModelCheckpoint)  
- final_emotion_model.h5 (modelo guardado al finalizar)  
- training_history.png  
- confusion_matrix.png  
- PNGs de overlays (opcionales): angry.jpg, disgust.png, fear.png, happy.png, neutral.png, sadness.png, surprise.png

Requisitos básicos sugeridos
----------------------------
- Python 3.x  
- numpy  
- matplotlib  
- seaborn  
- opencv-python  
- tensorflow (compatible con la API usada)  
- scikit-learn  
- pandas  
- ipykernel (si se ejecuta en Jupyter)

Ejemplos de uso
---------------
- Predicción en una imagen:
  ```python
  emotion, confidence = predict_emotion('imagen_de_prueba.jpg')
  print(f'Emoción detectada: {emotion} (Confianza: {confidence:.2%})')
  ```

- Arrancar la detección en tiempo real:
  - Asegurarse de tener `best_emotion_model.h5` o modificar el path para cargar `final_emotion_model.h5`.
  - Ejecutar la celda de webcam en un entorno con cámara; la ventana mostrará la detección y puede cerrarse con la tecla `q`.
