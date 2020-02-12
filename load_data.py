import csv
import json
import SimpleITK as sitk
import numpy as np
import argparse
import sys
from pathlib import Path
from os import fspath

def load_results(csv_path):
    norm_cog = []
    demented = []
    with open(csv_path,'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            if row['NORMCOG'] == "1":
                norm_cog.append(row['UDS_D1DXDATA ID'])
            elif row['DEMENTED'] == "1":
                demented.append(row['UDS_D1DXDATA ID'])
    with open('norm_cog.txt', 'w') as f:
        f.write("\n".join(norm_cog))
    with open('demented.txt','w') as f:
        f.write("\n".join(demented))


def get_all_images(directory):
    paths = []
    for filename in Path(directory).rglob('*.nii'):
        paths.append(fspath(filename))
    return paths

def get_closest_scan(scan, demented_records,normcog_records):
    closest_scan = ""
    largest_time = 9999999
    scan_record = scan.split('/')[1]
    scan_record_split = scan_record.split('_')
    scan_record_patient_id = scan_record_split[0]
    scan_record_time = scan_record_split[2][1:]
    label = 0 #1 for normcog, 0 for demented
    for record in demented_records:
        record_split = record.split('_')
        patient_id = record_split[0]
        record_time = record_split[2][1:]
        if scan_record_patient_id == patient_id:
            time_between_scans = abs(int(scan_record_time) - int(record_time))
            if time_between_scans < largest_time:
                closest_scan = scan
                largest_time = time_between_scans
                label = 0 
    for record in normcog_records:
        record_split = record.split('_')
        patient_id = record_split[0]
        record_time = record_split[2][1:]
        if scan_record_patient_id == patient_id:
            time_between_scans = abs(int(scan_record_time) - int(record_time))
            if time_between_scans < largest_time:
                closest_scan = scan
                largest_time = time_between_scans
                label = 1 
    return closest_scan.strip(),label

def create_labelled_image_paths():
    with open('databases/demented.txt','r') as f:
        demented_records = f.readlines()
    with open('databases/norm_cog.txt','r') as f:
        normcog_records = f.readlines()
    with open('databases/images.txt','r') as f:
        all_images = f.readlines()

    result = {}
    for image in all_images:
        scan,label = get_closest_scan(image,demented_records,normcog_records)
        result[scan] = label

    with open('images_labelled.json','w') as f:
        f.write(json.dumps(result,indent=4))


if __name__ == "__main__":
    create_labelled_image_paths()
