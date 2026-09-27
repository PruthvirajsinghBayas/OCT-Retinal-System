import os
import numpy as np
import tensorflow as tf

MODELS = {
    'ResNet50':       ('models/ResNet50_5epochs.tflite',        (224, 224)),
    'EfficientNetB0': ('models/EfficientNetB0_final.tflite',    (224, 224)),
    'InceptionV3':    ('models/InceptionV3_retinal__2_.tflite', (299, 299)),
    'VGG19':          ('models/VGG19_retinal__1_.tflite',       (224, 224)),
}

print("\n" + "="*50)
print("  TESTING ALL 4 MODELS")
print("="*50 + "\n")

for name, (path, size) in MODELS.items():
    print(f"--- {name} ---")
    if not os.path.exists(path):
        print(f"  ❌ File not found: {path}\n")
        continue

    try:
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()

        in_det = interp.get_input_details()
        out_det = interp.get_output_details()

        # Create dummy input matching expected shape
        dummy = np.random.rand(*in_det[0]['shape']).astype(in_det[0]['dtype'])

        interp.set_tensor(in_det[0]['index'], dummy)
        interp.invoke()
        out = interp.get_tensor(out_det[0]['index'])

        print(f"  ✅ SUCCESS")
        print(f"  Input shape:  {in_det[0]['shape']}")
        print(f"  Output shape: {out.shape}")
        print(f"  Output sample: {out[0][:4]}")

    except Exception as e:
        print(f"  ❌ FAILED: {str(e)[:200]}")

    print()

print("="*50)
