import json
import sys
import glob


all_images = glob.glob('scan_jpgs/*')

def get_matching_image(scan):
    for image in all_images:
        if scan in image:
            return image
    return None
result = {}
with open(sys.argv[1], 'r') as f:
    old_labels = json.loads(f.read())
    for image in old_labels:
        matching_image = get_matching_image(image.split('/')[-1])
        result[matching_image.split('/')[-1]] = old_labels[image]
with open('image_jpg_labelled.json','w') as f:
    f.write(json.dumps(result,indent=4))
