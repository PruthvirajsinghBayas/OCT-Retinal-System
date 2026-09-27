"""
diagnose_preprocessing.py
--------------------------
Automatically finds the correct preprocessing for each TFLite model
by testing 6 different normalization schemes against the
Kermany OCT 2018 test dataset.

PATHS USED (Kermany 2018 dataset):
  ~/retinal_web/dataset/test/CNV/
  ~/retinal_web/dataset/test/DME/
  ~/retinal_web/dataset/test/DRUSEN/
  ~/retinal_web/dataset/test/NORMAL/

USAGE:
  python3 diagnose_preprocessing.py --models ResNet50 EfficientNetB0
  python3 diagnose_preprocessing.py
  python3 diagnose_preprocessing.py --samples 30
"""

import os
import argparse
import numpy as np
import cv2
import tensorflow as tf

BASE_DIR    = os.path.expanduser('~/retinal_web')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset', 'test')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')

CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

MODELS = {
    'ResNet50':       {'path': os.path.join(MODELS_DIR, 'ResNet50_5epochs.tflite'),        'input_size': (224, 224)},
    'EfficientNetB0': {'path': os.path.join(MODELS_DIR, 'EfficientNetB0_final.tflite'),    'input_size': (224, 224)},
    'InceptionV3':    {'path': os.path.join(MODELS_DIR, 'InceptionV3_retinal__2_.tflite'), 'input_size': (224, 224)},
    'VGG19':          {'path': os.path.join(MODELS_DIR, 'VGG19_retinal__1_.tflite'),       'input_size': (224, 224)},
}

def prep_torch_imagenet(img):
    img = img / 255.0
    return ((img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).astype(np.float32)

def prep_caffe(img):
    img = img[..., ::-1].copy()
    return (img - [103.939, 116.779, 123.68]).astype(np.float32)

def prep_caffe_rgb(img):
    return (img - [123.68, 116.779, 103.939]).astype(np.float32)

def prep_scale_0_1(img):
    return (img / 255.0).astype(np.float32)

def prep_scale_neg1_1(img):
    return ((img / 127.5) - 1.0).astype(np.float32)

def prep_raw(img):
    return img.astype(np.float32)

PREPROCESSING_VARIANTS = {
    'torch_imagenet  (/255, mean/std, RGB)': prep_torch_imagenet,
    'caffe           (mean-subtract, BGR) ': prep_caffe,
    'caffe_rgb       (mean-subtract, RGB) ': prep_caffe_rgb,
    'scale_0_1       (/255 only)          ': prep_scale_0_1,
    'scale_-1_to_1   (/127.5 - 1)        ': prep_scale_neg1_1,
    'raw             (0-255, no change)   ': prep_raw,
}

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def load_test_images(samples_per_class):
    data = []
    print(f"\n  Loading from: {DATASET_DIR}")
    for cls_idx, cls in enumerate(CLASS_NAMES):
        folder = os.path.join(DATASET_DIR, cls)
        if not os.path.isdir(folder):
            print(f"  WARNING: {folder} not found — run download_dataset.sh first")
            continue
        files = sorted([f for f in os.listdir(folder)
                        if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))])[:samples_per_class]
        for f in files:
            data.append((os.path.join(folder, f), cls_idx))
        print(f"  {cls:8s} -> {len(files)} images")
    return data

def run_variant(interp, in_det, out_det, size, prep_fn, test_set):
    correct = 0
    total = 0
    pc_c = [0]*4
    pc_t = [0]*4
    for fpath, true_idx in test_set:
        img = cv2.imread(fpath)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size).astype(np.float32)
        img = prep_fn(img)
        tensor = np.expand_dims(img, 0)
        interp.set_tensor(in_det[0]['index'], tensor)
        interp.invoke()
        raw = interp.get_tensor(out_det[0]['index'])[0]
        scores = softmax(raw.astype(np.float32))
        pred = int(np.argmax(scores))
        pc_t[true_idx] += 1
        if pred == true_idx:
            correct += 1
            pc_c[true_idx] += 1
        total += 1
    acc = (correct / total * 100) if total > 0 else 0
    return acc, pc_c, pc_t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=15)
    parser.add_argument('--models', type=str, nargs='+', default=None)
    args = parser.parse_args()

    models_to_test = args.models or list(MODELS.keys())

    print("\n" + "="*65)
    print("  PREPROCESSING DIAGNOSIS — Kermany OCT 2018")
    print(f"  Samples/class: {args.samples}  |  Models: {', '.join(models_to_test)}")
    print("="*65)

    test_set = load_test_images(args.samples)
    if not test_set:
        print("\n  No images found. Run download_dataset.sh first.")
        return

    final_summary = {}

    for model_name in models_to_test:
        if model_name not in MODELS:
            continue
        cfg  = MODELS[model_name]
        path = cfg['path']
        size = cfg['input_size']

        print(f"\n{'='*65}")
        print(f"  Testing: {model_name}  ({path})")
        print(f"{'='*65}")

        if not os.path.exists(path):
            print(f"  ERROR: Model file not found")
            continue

        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        in_det  = interp.get_input_details()
        out_det = interp.get_output_details()

        variant_results = []
        best_so_far = 0

        for v_name, prep_fn in PREPROCESSING_VARIANTS.items():
            acc, pc_c, pc_t = run_variant(interp, in_det, out_det, size, prep_fn, test_set)
            variant_results.append((v_name, acc, pc_c, pc_t))
            marker = " <-- BEST SO FAR" if acc > best_so_far else ""
            best_so_far = max(best_so_far, acc)
            print(f"    {v_name}  {acc:6.2f}%{marker}")

        variant_results.sort(key=lambda x: -x[1])
        best_name, best_acc, best_pc_c, best_pc_t = variant_results[0]

        print(f"\n  WINNER: {best_name.strip()}  ->  {best_acc:.2f}%")
        print(f"  Per-class breakdown:")
        for i, cls in enumerate(CLASS_NAMES):
            t = best_pc_t[i]
            c = best_pc_c[i]
            pct = (c/t*100) if t > 0 else 0
            bar = '#' * int(pct//5) + '-' * (20 - int(pct//5))
            print(f"    {cls:8s} [{bar}] {pct:5.1f}% ({c}/{t})")

        final_summary[model_name] = {'best': best_name.strip(), 'acc': best_acc}

    print("\n" + "="*65)
    print("  SUMMARY — Use these preprocessing methods in app.py:")
    print("="*65)
    for m, info in final_summary.items():
        print(f"  {m:<18}  {info['best']:<42} {info['acc']:.1f}%")
    print("="*65)

    print("\n  EXACT CODE TO PASTE INTO app.py preprocess() function:")
    print("-"*65)
    keys = list(final_summary.keys())
    for idx, (m, info) in enumerate(final_summary.items()):
        bp = info['best'].lower()
        kw = 'if' if idx == 0 else 'elif'
        print(f"    {kw} model_name == '{m}':")
        print(f"        # {info['best']}")
        if 'caffe' in bp and 'rgb' not in bp:
            print(f"        img = img[..., ::-1].copy()  # RGB to BGR")
            print(f"        img = img - np.array([103.939, 116.779, 123.68], dtype=np.float32)")
        elif 'caffe_rgb' in bp:
            print(f"        img = img - np.array([123.68, 116.779, 103.939], dtype=np.float32)")
        elif 'torch' in bp:
            print(f"        img /= 255.0")
            print(f"        img = (img - np.array([0.485,0.456,0.406])) / np.array([0.229,0.224,0.225])")
            print(f"        img = img.astype(np.float32)")
        elif 'scale_0_1' in bp:
            print(f"        img /= 255.0")
        elif '-1' in bp:
            print(f"        img = (img / 127.5) - 1.0")
        elif 'raw' in bp:
            print(f"        pass  # no normalization")
        print()
    print("    return np.expand_dims(img, axis=0)")
    print("-"*65 + "\n")

if __name__ == '__main__':
    main()
