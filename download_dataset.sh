#!/bin/bash
# =============================================================
#  download_dataset.sh
#  Downloads Kermany OCT 2018 dataset from Kaggle
#  and sets up the correct folder structure for this project.
#
#  USAGE:
#    chmod +x download_dataset.sh
#    ./download_dataset.sh
# =============================================================

set -e

echo ""
echo "============================================================="
echo "  OCT Retinal System — Kaggle Dataset Setup"
echo "  Dataset: Kermany OCT 2018 (paultimothymooney/kermany2018)"
echo "============================================================="
echo ""

PROJECT_DIR=~/retinal_web
DATASET_DIR=$PROJECT_DIR/dataset

# ── Step 1: Activate venv ─────────────────────
echo "[1/7] Activating virtual environment..."
source $PROJECT_DIR/venv311/bin/activate
echo "  ✔  venv311 activated"

# ── Step 2: Install Kaggle API ────────────────
echo ""
echo "[2/7] Installing Kaggle API..."
pip install kaggle --quiet
echo "  ✔  Kaggle API installed"

# ── Step 3: Check kaggle.json ─────────────────
echo ""
echo "[3/7] Checking Kaggle credentials..."
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo ""
    echo "  ❌ kaggle.json NOT found!"
    echo ""
    echo "  You need to create it. Follow these steps:"
    echo ""
    echo "  1. Go to https://www.kaggle.com"
    echo "  2. Click your profile photo (top right) → Settings"
    echo "  3. Scroll to 'API' section → Click 'Create New Token'"
    echo "  4. A file 'kaggle.json' will download to your PC"
    echo "  5. Copy it to your Pi using WinSCP:"
    echo "     Destination on Pi: /home/octpi/.kaggle/kaggle.json"
    echo "  6. Then run this script again"
    echo ""
    exit 1
fi
chmod 600 ~/.kaggle/kaggle.json
echo "  ✔  kaggle.json found"

# ── Step 4: Create folder structure ───────────
echo ""
echo "[4/7] Creating dataset folder structure..."
mkdir -p $DATASET_DIR/test/CNV
mkdir -p $DATASET_DIR/test/DME
mkdir -p $DATASET_DIR/test/DRUSEN
mkdir -p $DATASET_DIR/test/NORMAL
mkdir -p $DATASET_DIR/val/CNV
mkdir -p $DATASET_DIR/val/DME
mkdir -p $DATASET_DIR/val/DRUSEN
mkdir -p $DATASET_DIR/val/NORMAL
mkdir -p $PROJECT_DIR/results
echo "  ✔  Folders created:"
echo "     $DATASET_DIR/test/{CNV,DME,DRUSEN,NORMAL}"
echo "     $DATASET_DIR/val/{CNV,DME,DRUSEN,NORMAL}"

# ── Step 5: Download dataset ──────────────────
echo ""
echo "[5/7] Downloading Kermany OCT 2018 dataset from Kaggle..."
echo "  ⏳ This is a large file (~6GB) — may take 20-60 minutes on Pi Wi-Fi"
echo "  ⏳ Do NOT close the terminal during download"
echo ""

cd $DATASET_DIR
kaggle datasets download -d paultimothymooney/kermany2018 --force
echo "  ✔  Download complete"

# ── Step 6: Unzip ─────────────────────────────
echo ""
echo "[6/7] Extracting dataset..."
echo "  ⏳ Extraction takes 5-10 minutes..."
unzip -q kermany2018.zip -d raw_extracted
echo "  ✔  Extraction complete"

# ── Step 7: Organise into project structure ───
echo ""
echo "[7/7] Organising files into project structure..."

# The Kaggle zip extracts to: raw_extracted/OCT2017/test/CNV/*.jpeg etc.
# We copy test set and val set to our dataset/ folder

RAW=$DATASET_DIR/raw_extracted

# Copy test set
for CLASS in CNV DME DRUSEN NORMAL; do
    echo "  Copying test/$CLASS..."
    if [ -d "$RAW/OCT2017/test/$CLASS" ]; then
        cp "$RAW/OCT2017/test/$CLASS/"* "$DATASET_DIR/test/$CLASS/" 2>/dev/null || true
    fi
done

# Copy val set
for CLASS in CNV DME DRUSEN NORMAL; do
    echo "  Copying val/$CLASS..."
    if [ -d "$RAW/OCT2017/val/$CLASS" ]; then
        cp "$RAW/OCT2017/val/$CLASS/"* "$DATASET_DIR/val/$CLASS/" 2>/dev/null || true
    fi
done

echo ""
echo "  Dataset image counts:"
for CLASS in CNV DME DRUSEN NORMAL; do
    COUNT=$(ls "$DATASET_DIR/test/$CLASS/" 2>/dev/null | wc -l)
    echo "    test/$CLASS  → $COUNT images"
done
for CLASS in CNV DME DRUSEN NORMAL; do
    COUNT=$(ls "$DATASET_DIR/val/$CLASS/" 2>/dev/null | wc -l)
    echo "    val/$CLASS   → $COUNT images"
done

# ── Cleanup zip ───────────────────────────────
echo ""
echo "  Cleaning up zip file to save disk space..."
rm -f $DATASET_DIR/kermany2018.zip
echo "  ✔  Cleanup done"

echo ""
echo "============================================================="
echo "  ✔  DATASET SETUP COMPLETE!"
echo "============================================================="
echo ""
echo "  Test images: $DATASET_DIR/test/"
echo "  Val  images: $DATASET_DIR/val/"
echo ""
echo "  Next step — run the diagnosis script:"
echo "  python3 diagnose_preprocessing.py"
echo ""
echo "  Then run evaluation:"
echo "  python3 evaluate_models.py"
echo "============================================================="
