import os
import random
from PIL import Image

def synthesize_images(dataset_path, output_path, num_images, canvas_size=(1024, 1024)):
    os.makedirs(output_path, exist_ok=True)
    image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    random.seed(42)

    for i in range(num_images):
        base_image = Image.new('RGB', canvas_size, (255, 255, 255))  # Base image in 'RGB'

        for _ in range(random.randint(2, 5)):
            overlay_file = random.choice(image_files)
            with Image.open(os.path.join(dataset_path, overlay_file)) as overlay_img:
                # Resize overlay image
                max_size = random.randint(10, 500)
                overlay_img = overlay_img.resize((max_size, max_size), Image.LANCZOS)

                # Convert overlay image to 'RGB' if it has an alpha channel
                if overlay_img.mode == 'RGBA':
                    overlay_img = overlay_img.convert('RGB')

                # Random position and angle
                x = random.randint(0, canvas_size[0] - max_size)
                y = random.randint(0, canvas_size[1] - max_size)
                angle = random.randint(0, 360)

                overlay_img = overlay_img.rotate(angle, expand=True)
                base_image.paste(overlay_img, (x, y), overlay_img if overlay_img.mode == 'RGBA' else None)

        output_file = os.path.join(output_path, f'synthesized_{i}.jpg')
        base_image.save(output_file)


def merge_synthesized_images(src_image_path, dest_image_path, dest_label_path, num_images):
    synthesized_image_files = os.listdir(src_image_path)
    print(f"Found {len(synthesized_image_files)} files in {src_image_path}")

    if num_images:
        synthesized_image_files = synthesized_image_files[:num_images]

    for img_file in synthesized_image_files:
        print(f"Processing {img_file}")

        # Copy the image file
        shutil.copy2(os.path.join(src_image_path, img_file), os.path.join(dest_image_path, img_file))
        print(f"Copied image to {dest_image_path}")

        # Create a corresponding empty label file
        label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(dest_label_path, label_file)
        open(label_path, 'a').close()
        print(f"Created label file {label_path}")

