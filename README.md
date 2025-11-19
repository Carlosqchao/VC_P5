# VC_P5: Detector de caras y clasificador de emociones mediante CNN

## Autores

- Carlos Ruano Ramos
- Juan Boissier García

Estructura y flujo del notebook
-------------------------------

### Nota: El entrenamiento se realizó en Kaggle, el link del notebook del entrenamiento es: https://www.kaggle.com/code/carlosruanoramos/vc-p5

1) Imports y dependencias
   - Librerías principales importadas:
     - numpy, matplotlib, seaborn
     - sklearn.metrics: classification_report, confusion_matrix
     - tensorflow/keras: modelos, capas, ImageDataGenerator, callbacks, load_model
     - cv2 (OpenCV) para procesamiento de imágenes y webcam
   - Comentario: asegúrate de tener ipykernel si usas Jupyter (el notebook muestra un aviso sobre ipykernel en una ejecución).

2) Configuración de parámetros globales
   - Rutas por defecto (configuradas para Kaggle):
     - TRAIN_DIR = '/kaggle/input/fer2013/train'
     - TEST_DIR  = '/kaggle/input/fer2013/test'
   - Parámetros de entrenamiento:
     - IMG_SIZE = 48
     - BATCH_SIZE = 64
     - EPOCHS = 50
   - Clases/etiquetas:
     - emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
     - num_classes = 7

3) Preparación de datos
   - Se define train_datagen con ImageDataGenerator para data augmentation:
     - rescale=1./255, rotation_range=15, width/height shift 0.1, shear 0.1, zoom 0.1, horizontal_flip=True, fill_mode='nearest'
   - test_datagen solo con rescale=1./255.
   - Generadores con flow_from_directory:
     - color_mode='grayscale', target_size=(48,48), class_mode='categorical'
   - El notebook imprime:
     - class_indices (mapeo etiqueta → índice)
     - número de muestras de entrenamiento y prueba
   - Requisito: la estructura de carpetas debe ser TRAIN_DIR/<clase>/* y TEST_DIR/<clase>/*.

4) Definición de la arquitectura CNN
   - Función create_emotion_cnn() que devuelve un modelo keras.Sequential con:
     - 4 bloques convolucionales progresivos (64 → 128 → 256 → 512 filtros)
       - Cada bloque: Conv2D, BatchNormalization, Conv2D, BatchNormalization, MaxPooling2D, Dropout(0.25)
     - Capas densas (clasificación):
       - Flatten
       - Dense(512, relu) + BatchNormalization + Dropout(0.5)
       - Dense(256, relu) + BatchNormalization + Dropout(0.5)
       - Dense(num_classes, softmax)
   - Input shape: (48, 48, 1) (imágenes en escala de grises)

5) Compilación y callbacks
   - Compilación:
     - optimizer: Adam(learning_rate=1e-4)
     - loss: categorical_crossentropy
     - metrics: ['accuracy']
   - Callbacks configurados:
     - EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
     - ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
     - ModelCheckpoint('best_emotion_model.h5', monitor='val_accuracy', save_best_only=True)
   - Observación: EarlyStopping restaura pesos del mejor val_loss; ModelCheckpoint guarda según val_accuracy (pueden corresponder a distintas épocas).

6) Entrenamiento
   - model.fit(train_generator, epochs=EPOCHS, validation_data=test_generator, callbacks=callbacks)
   - El historial se almacena en `history`.
   - Al final se guarda el modelo completo con model.save('final_emotion_model.h5').

7) Visualización del historial
   - Se trazan y guardan (training_history.png):
     - accuracy (train vs val)
     - loss (train vs val)ç
     <img width="1600" height="562" alt="image" src="https://github.com/user-attachments/assets/3c340495-6153-40da-ae19-e256c5978519" />

8) Evaluación
   - Se evalúa con model.evaluate(test_generator) y se imprimen test_loss y test_accuracy.
   - Predicciones:
     - predictions = model.predict(test_generator, steps=len(test_generator))
     - y_pred = argmax(predictions, axis=1)
     - y_true = test_generator.classes
   - Se muestra classification_report con precision/recall/f1 por clase.
   - Se genera y guarda la matriz de confusión (confusion_matrix.png) con seaborn heatmap.
     <img width="1600" height="1354" alt="image" src="https://github.com/user-attachments/assets/b814a116-e0d8-4c22-8057-e979bb591a58" />


9) Función para predecir imágenes individuales
   - predict_emotion(image_path):
     - Carga imagen con keras.preprocessing.image.load_img(target_size=(48,48), color_mode='grayscale')
     - Convierte a array, normaliza (/255), expande dimensiones y predice.
     - Devuelve (etiqueta_emoción, confianza).

10) Módulo de inferencia en tiempo real (webcam)
    - Carga del modelo: load_model('best_emotion_model.h5') (asegúrate de que exista).
    - Mapeos:
      - emotions = ['angry','disgust','fear','happy','neutral','sad','surprise']
      - emotions_spanish = ['Cabreado','Asqueado','Asustado','Feliz','Neutral','Triste','Sorprendido']
    - PNGs para overlay:
      - Diccionario png_files mapeando índices a nombres de fichero (angry.jpg, disgust.png, etc.); el código intenta leerlos con cv2.IMREAD_UNCHANGED.
      - Si faltan PNGs, se usan efectos alternativos (rectángulos, texto, círculos).
    - Detector de rostros:
      - face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
      - detectMultiScale con scaleFactor=1.3, minNeighbors=5, minSize=(30,30)
    - Funciones principales:
      - overlay_png_on_frame(frame, png, x, y, w, h, scale=1.0)
        - Redimensiona, centra y mezcla PNG con canal alfa sobre el frame; gestiona recortes fuera del frame.
      - add_emotion_effects(frame, emotion_idx, x, y, w, h, confidence)
        - Si existe PNG para la emoción, la sobrepone y escribe texto con confianza.
        - Si no, dibuja rectángulo coloreado, banner, texto, círculos y partículas decorativas.
      - emotion_detection_thread()
        - Captura frames de la webcam (cv2.VideoCapture(0)), detecta rostros, para cada rostro realiza:
          - ROI grayscaling → resize a 48x48 → normalizar → expand dims → predict
          - Selecciona la emoción con mayor confianza por frame
          - Aplica efectos y muestra información principal en la parte superior
        - Muestra la ventana OpenCV 'Deteccion de Emociones' y permite salir con 'q'.
        - El notebook ejecuta este flujo en un hilo (threading.Thread) y espera con thread.join(timeout=300).
    - Observación: la detección en tiempo real requiere entorno con GUI y acceso a la cámara (no funciona en servidores headless sin X11/Display).

Archivos y salidas que genera el notebook
----------------------------------------
- best_emotion_model.h5  (guardado por ModelCheckpoint)
- final_emotion_model.h5 (guardado manual al finalizar entrenamiento)
- training_history.png
- confusion_matrix.png
- PNGs de overlays (opcional): angry.jpg, disgust.png, fear.png, happy.png, neutral.png, sadness.png, surprise.png

Problemas habituales y consejos para resolverlos
------------------------------------------------
- Rutas de datos: las rutas están puestas para Kaggle; ajústalas si ejecutas local.
- Estructura de carpetas: flow_from_directory requiere subcarpetas por clase.
- Faltan PNGs: el código imprime advertencias y usa fallback visual.
- Falta ipykernel: instala ipykernel en el entorno para evitar avisos en Jupyter.
- best_emotion_model.h5 puede no existir si el entrenamiento no produjo mejora; en ese caso carga final_emotion_model.h5 o revisa callbacks.
- Haar cascade es sensible; para mayor robustez usa MTCNN, detector DNN de OpenCV o dlib.
- Reproducibilidad: no se fijan seeds ni hay requirements.txt; añade ambos para reproducibilidad.

Ejemplos de uso
---------------
- Ejecutar una predicción en una imagen:
  ```python
  emotion, confidence = predict_emotion('imagen_de_prueba.jpg')
  print(f'Emoción detectada: {emotion} (Confianza: {confidence:.2%})')
  ```

- Ejecutar la detección en tiempo real:
  - Asegúrate de tener `best_emotion_model.h5` en el directorio actual (o modifica el path para cargar `final_emotion_model.h5`).
  - Ejecuta la celda correspondiente al módulo de webcam en un entorno con cámara. Presiona 'q' para salir.


```
