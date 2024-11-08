#CNN2
import numpy as np
import cv2
import os
from PIL import Image  # Importación necesaria para abrir y manipular imágenes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox

# Configuración de la ventana de Tkinter
root = tk.Tk()
root.title("Clasificador de Defectos")
root.geometry("400x200")

# Directorio base de los datos originales
dataset_dir = os.path.join(os.path.expanduser("~/Downloads"), "dataset")
# Directorio de salida para guardar las imágenes con Transformada de Fourier
fourier_dataset_dir = os.path.join(os.path.expanduser("~/Downloads"), "dataset_fourier")

# Crear las carpetas de salida si no existen
os.makedirs(fourier_dataset_dir, exist_ok=True)
os.makedirs(os.path.join(fourier_dataset_dir, "defectuoso"), exist_ok=True)
os.makedirs(os.path.join(fourier_dataset_dir, "no_defectuoso"), exist_ok=True)

# Función para aplicar Transformada de Fourier a una imagen
def apply_fourier_transform_and_save(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert('L')  # Convertimos a escala de grises
        img_array = np.array(img)
        
        # Aplicamos Transformada de Fourier
        f_transform = np.fft.fft2(img_array)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
        
        # Guardamos la imagen con Fourier
        fourier_img = Image.fromarray(magnitude_spectrum.astype(np.uint8))
        fourier_img.save(os.path.join(output_dir, filename))

# Aplicar Transformada de Fourier y guardar en las carpetas correspondientes
apply_fourier_transform_and_save(os.path.join(dataset_dir, "defectuoso"), os.path.join(fourier_dataset_dir, "defectuoso"))
apply_fourier_transform_and_save(os.path.join(dataset_dir, "no_defectuoso"), os.path.join(fourier_dataset_dir, "no_defectuoso"))

print("Imágenes con Transformada de Fourier guardadas en 'dataset_fourier'")

# Configuración de imagen y batch
img_height, img_width = 256, 256
batch_size = 32

# Función para aplicar Transformada de Fourier y normalizar la imagen
def apply_fourier_transform(img):
    if img.shape[-1] == 3:  # Convierte a escala de grises si es necesario
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude_spectrum.astype(np.uint8)

# Función para clasificar una imagen cargada
def classify_uploaded_image(file_path, model, apply_fourier=False):
    img = Image.open(file_path).convert('L')  # Convertir a escala de grises
    img = img.resize((img_width, img_height))  # Redimensionar la imagen al tamaño de entrada del modelo
    img_array = np.array(img)

    if apply_fourier:
        img_array = apply_fourier_transform(img_array)

    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión de batch
    img_array = np.expand_dims(img_array, axis=-1)  # Añadir canal si es escala de grises

    prediction = model.predict(img_array)
    return "No Defectuoso" if prediction[0] > 0.5 else "Defectuoso"

# Generador de datos para cargar imágenes sin Transformada de Fourier (en color)
datagen_original = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Generador de datos para aplicar Transformada de Fourier en tiempo real (en escala de grises)
class FourierImageDataGenerator(ImageDataGenerator):
    def flow_from_directory(self, *args, **kwargs): 
        batches = super().flow_from_directory(*args, **kwargs)
        while True:
            batch_x, batch_y = next(batches)
            batch_x = np.array([apply_fourier_transform(img_to_array(img)) for img in batch_x])
            batch_x = batch_x.reshape(-1, img_height, img_width, 1)  # Añadir canal de 1 para imágenes en escala de grises
            yield batch_x / 255.0, batch_y

datagen_fourier = FourierImageDataGenerator(validation_split=0.2)

# Cargar datos sin Transformada de Fourier (en color)
train_generator_orig = datagen_original.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    color_mode='rgb',
    subset='training'
)

validation_generator_orig = datagen_original.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    color_mode='rgb',
    subset='validation'
)

# Cargar datos con Transformada de Fourier (preprocesadas y guardadas en escala de grises)
train_generator_fourier = ImageDataGenerator(rescale=1./255, validation_split=0.2).flow_from_directory(
    fourier_dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='binary',
    subset='training'
)

validation_generator_fourier = ImageDataGenerator(rescale=1./255, validation_split=0.2).flow_from_directory(
    fourier_dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='binary',
    subset='validation'
)

# Función para construir el modelo CNN
def build_model(input_shape):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(256, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(512, kernel_size=(3, 3), activation='relu'),  # Capa adicional para más profundidad
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Entrenamiento y evaluación sin Transformada de Fourier (en color)
model = build_model((img_height, img_width, 3))
history_orig = model.fit(
    train_generator_orig,
    epochs=10,
    validation_data=validation_generator_orig
)
val_loss_orig, val_accuracy_orig = model.evaluate(validation_generator_orig)
print(f"Precisión sin Transformada de Fourier: {val_accuracy_orig:.2f}")

# Entrenamiento y evaluación con Transformada de Fourier (en escala de grises)
model = build_model((img_height, img_width, 1))
history_fourier = model.fit(
    train_generator_fourier,
    epochs=10,
    validation_data=validation_generator_fourier
)
val_loss_fourier, val_accuracy_fourier = model.evaluate(validation_generator_fourier)
print(f"Precisión con Transformada de Fourier: {val_accuracy_fourier:.2f}")

# Función para abrir un archivo de imagen y clasificarlo
def upload_and_classify():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
    )
    if not file_path:
        return  # Si no se selecciona archivo, no se hace nada

    apply_fourier = tk.messagebox.askyesno("Aplicar Transformada de Fourier", "¿Desea aplicar la Transformada de Fourier?")
    result = classify_uploaded_image(file_path, model, apply_fourier)
    tk.messagebox.showinfo("Resultado de Clasificación", f"Resultado: {result}")

# Botón para seleccionar y clasificar la imagen
btn_upload = tk.Button(root, text="Subir Imagen y Clasificar", command=upload_and_classify)
btn_upload.pack(pady=20)

# Iniciar la aplicación de Tkinter
root.mainloop()

# Comparación de las gráficas de entrenamiento y validación para ambos casos

# Gráficas de precisión
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_orig.history['accuracy'], label='Entrenamiento (sin Fourier)')
plt.plot(history_orig.history['val_accuracy'], label='Validación (sin Fourier)')
plt.plot(history_fourier.history['accuracy'], label='Entrenamiento (con Fourier)')
plt.plot(history_fourier.history['val_accuracy'], label='Validación (con Fourier)')
plt.title('Comparación de Precisión')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

# Gráficas de pérdida
plt.subplot(1, 2, 2)
plt.plot(history_orig.history['loss'], label='Entrenamiento (sin Fourier)')
plt.plot(history_orig.history['val_loss'], label='Validación (sin Fourier)')
plt.plot(history_fourier.history['loss'], label='Entrenamiento (con Fourier)')
plt.plot(history_fourier.history['val_loss'], label='Validación (con Fourier)')
plt.title('Comparación de Pérdida')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.show()

