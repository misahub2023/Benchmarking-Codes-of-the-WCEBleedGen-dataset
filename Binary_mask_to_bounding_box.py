import os
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

def create_bounding_box(mask):
    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0:
        return 0, 0, 0, 0
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return x_min, y_min, x_max, y_max


def save_txt_format(bbox, filename):
    with open(filename, 'w') as f:
        f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

def save_xml_format(bbox, filename, image_filename, image_size):
    annotation = ET.Element("annotation")
    folder = ET.SubElement(annotation, "folder").text = "XML"
    filename_tag = ET.SubElement(annotation, "filename").text = os.path.basename(image_filename)
    path = ET.SubElement(annotation, "path").text = image_filename

    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width").text = str(image_size[0])
    height = ET.SubElement(size, "height").text = str(image_size[1])
    depth = ET.SubElement(size, "depth").text = "3"  

    segmented = ET.SubElement(annotation, "segmented").text = "0"

    obj = ET.SubElement(annotation, "object")
    name = ET.SubElement(obj, "name").text = "object"  
    pose = ET.SubElement(obj, "pose").text = "Unspecified"
    truncated = ET.SubElement(obj, "truncated").text = "0"
    difficult = ET.SubElement(obj, "difficult").text = "0"

    bndbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(bbox[0])
    ET.SubElement(bndbox, "ymin").text = str(bbox[1])
    ET.SubElement(bndbox, "xmax").text = str(bbox[2])
    ET.SubElement(bndbox, "ymax").text = str(bbox[3])

    tree = ET.ElementTree(annotation)
    tree.write(filename)

def save_yolo_txt_format(bbox, filename, image_size):
    class_id = 0  
    dw = 1.0 / image_size[0]
    dh = 1.0 / image_size[1]
    x_center = (bbox[0] + bbox[2]) / 2.0 * dw
    y_center = (bbox[1] + bbox[3]) / 2.0 * dh
    width = (bbox[2] - bbox[0]) * dw
    height = (bbox[3] - bbox[1]) * dh
    
    with open(filename, 'w') as f:
        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def process_masks(mask_folder, save_root_folder):
    txt_folder = os.path.join(save_root_folder, "TXT")
    xml_folder = os.path.join(save_root_folder, "XML")
    yolo_txt_folder = os.path.join(save_root_folder, "YOLO_TXT")

    os.makedirs(txt_folder, exist_ok=True)
    os.makedirs(xml_folder, exist_ok=True)
    os.makedirs(yolo_txt_folder, exist_ok=True)

    for mask_file in os.listdir(mask_folder):
        if mask_file.endswith('.png'):
            mask_path = os.path.join(mask_folder, mask_file)
            mask = np.array(Image.open(mask_path).convert('L'))
            bbox = create_bounding_box(mask)
            image_size = mask.shape[::-1]

            txt_filename = os.path.join(txt_folder, mask_file.replace('.png', '.txt'))
            xml_filename = os.path.join(xml_folder, mask_file.replace('.png', '.xml'))
            yolo_txt_filename = os.path.join(yolo_txt_folder, mask_file.replace('.png', '.txt'))

            save_txt_format(bbox, txt_filename)
            save_xml_format(bbox, xml_filename, mask_path, image_size)
            save_yolo_txt_format(bbox, yolo_txt_filename, image_size)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python generate_bounding_boxes.py <path_to_binary_masks_folder> <save_root_folder>")
    else:
        mask_folder = sys.argv[1]
        save_root_folder = sys.argv[2]
        process_masks(mask_folder, save_root_folder)