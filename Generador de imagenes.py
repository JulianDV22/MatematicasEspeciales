import numpy as np
import cv2
from PIL import Image
import os
from pathlib import Path

# Obtener la ruta de la carpeta de Descargas del usuario
downloads_dir = str(Path.home() / "Downloads")

# Define el directorio de salida en Descargas
output_dir = os.path.join(downloads_dir, "dataset")
defectuoso_dir = os.path.join(output_dir, "defectuoso")
no_defectuoso_dir = os.path.join(output_dir, "no_defectuoso")

# Subdirectorios para texturas textiles
textile_defectuoso_dir = os.path.join(defectuoso_dir, "textil")
textile_no_defectuoso_dir = os.path.join(no_defectuoso_dir, "textil")

os.makedirs(textile_defectuoso_dir, exist_ok=True)
os.makedirs(textile_no_defectuoso_dir, exist_ok=True)

# Funciones para generar diferentes texturas textiles
def generate_weave_texture(size=(256, 256), line_spacing=10):
    texture = np.ones(size, dtype=np.uint8) * 255
    for i in range(0, size[0], line_spacing):
        cv2.line(texture, (i, 0), (i, size[1]), (150), 1)
        cv2.line(texture, (0, i), (size[0], i), (150), 1)
    return Image.fromarray(texture)

def generate_herringbone_texture(size=(256, 256), line_spacing=15):
    texture = np.ones(size, dtype=np.uint8) * 240
    for i in range(0, size[0], line_spacing * 2):
        for j in range(0, size[1], line_spacing * 2):
            cv2.line(texture, (i, j), (i + line_spacing, j + line_spacing), (150), 1)
            cv2.line(texture, (i + line_spacing, j), (i, j + line_spacing), (150), 1)
    return Image.fromarray(texture)

def generate_knit_texture(size=(256, 256), knot_spacing=20):
    texture = np.ones(size, dtype=np.uint8) * 240
    for y in range(0, size[1], knot_spacing):
        for x in range(0, size[0], knot_spacing):
            cv2.circle(texture, (x, y), knot_spacing // 4, (100), -1)
    return Image.fromarray(texture)

# Funciones para agregar defectos específicos de textiles
def add_fiber_loose(image, size=(10, 40)):
    defective_image = np.array(image)
    x, y = np.random.randint(0, image.size[0] - size[0]), np.random.randint(0, image.size[1] - size[1])
    cv2.line(defective_image, (x, y), (x + size[0], y + size[1]), (50), 2)
    return Image.fromarray(defective_image)

def add_fading_spot(image, spot_radius=30):
    faded_image = np.array(image)
    x, y = np.random.randint(spot_radius, image.size[0] - spot_radius), np.random.randint(spot_radius, image.size[1] - spot_radius)
    cv2.circle(faded_image, (x, y), spot_radius, (200), -1)
    return Image.fromarray(faded_image)

def add_thread_bump(image, bump_size=(20, 20)):
    bumped_image = np.array(image)
    x, y = np.random.randint(0, image.size[0] - bump_size[0]), np.random.randint(0, image.size[1] - bump_size[1])
    cv2.rectangle(bumped_image, (x, y), (x + bump_size[0], y + bump_size[1]), (220), -1)
    return Image.fromarray(bumped_image)

# Generar imágenes y guardarlas en los directorios correspondientes
def generate_textile_images(num_images=250):
    textile_textures = [generate_weave_texture, generate_herringbone_texture, generate_knit_texture]
    textile_defects = [add_fiber_loose, add_fading_spot, add_thread_bump]

    # Generar texturas textiles sin defectos
    for i in range(num_images):
        texture_func = np.random.choice(textile_textures)
        texture_image = texture_func()
        texture_image.save(os.path.join(textile_no_defectuoso_dir, f"textil_{i+1}.png"))

    # Generar texturas textiles con defectos
    for i in range(num_images):
        texture_func = np.random.choice(textile_textures)
        defect_func = np.random.choice(textile_defects)
        texture_image = texture_func()
        defective_image = defect_func(texture_image)
        defective_image.save(os.path.join(textile_defectuoso_dir, f"textil_defectuosa_{i+1}.png"))

# Generar 250 texturas textiles para cada categoría
generate_textile_images(num_images=250)
print("Generación de imágenes textiles completada.")
