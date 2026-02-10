#!/bin/bash
#
# Download EMPIAR-10025 dataset (subset or full)
#
# Options:
#   subset  - Download 8 GB subset (20 movies) for testing
#   full    - Download full 2 TB dataset (requires streaming)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"

echo "========================================================================"
echo "  EMPIAR-10025 Download Script"
echo "========================================================================"
echo ""

# Parse arguments
MODE="${1:-subset}"

if [ "$MODE" != "subset" ] && [ "$MODE" != "full" ]; then
    echo "Usage: $0 [subset|full]"
    echo ""
    echo "  subset - Download 8 GB subset (20 movies) - RECOMMENDED"
    echo "  full   - Download full 2 TB dataset (requires Aspera/Globus)"
    echo ""
    exit 1
fi

echo "Mode: $MODE"
echo ""

#================================================================
# SUBSET DOWNLOAD (8 GB)
#================================================================

if [ "$MODE" == "subset" ]; then
    echo "Downloading EMPIAR-10025 subset (8 GB, 20 movies)..."
    echo ""
    echo "This subset is used in CryoSPARC tutorials and contains:"
    echo "  - 20 movie files (.mrc or .tiff format)"
    echo "  - Size: ~8 GB"
    echo "  - T20S Proteasome dataset"
    echo ""

    mkdir -p "$DATA_DIR/empiar_10025_subset"
    cd "$DATA_DIR/empiar_10025_subset"

    echo "========================================================================"
    echo "  Download Options"
    echo "========================================================================"
    echo ""
    echo "Option 1: CryoSPARC Download (if installed)"
    echo "  cryosparcm downloadtest"
    echo "  tar -xf empiar_10025_subset.tar"
    echo ""
    echo "Option 2: Direct FTP Download (recommended)"
    echo "  We'll download from EMPIAR FTP server"
    echo ""
    echo "Option 3: Manual Download"
    echo "  Visit: https://www.ebi.ac.uk/empiar/EMPIAR-10025/"
    echo "  Select: Browse FTP > Download subset"
    echo ""

    read -p "Use FTP download (recommended)? (y/n) " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Downloading via FTP..."
        echo ""

        # Download first 20 movies via FTP
        # Note: Need to find actual FTP paths - these are placeholders
        FTP_BASE="ftp://ftp.ebi.ac.uk/empiar/world_availability/10025/data"

        echo "Downloading movies 1-20..."
        for i in $(seq -f "%03g" 1 20); do
            FILENAME="FoilHole_${i}.mrc"
            if [ ! -f "$FILENAME" ]; then
                echo "  Downloading $FILENAME..."
                wget -q --show-progress "$FTP_BASE/$FILENAME" || {
                    echo "WARNING: Could not download $FILENAME"
                    echo "You may need to download manually from:"
                    echo "  https://www.ebi.ac.uk/empiar/EMPIAR-10025/"
                    exit 1
                }
            else
                echo "  $FILENAME already exists, skipping..."
            fi
        done

        echo ""
        echo "Download complete!"
        echo "Files saved to: $DATA_DIR/empiar_10025_subset/"
        echo ""

    else
        echo ""
        echo "========================================================================"
        echo "  Manual Download Instructions"
        echo "========================================================================"
        echo ""
        echo "1. Visit: https://www.ebi.ac.uk/empiar/EMPIAR-10025/"
        echo ""
        echo "2. Click 'Browse FTP' or use Aspera:"
        echo ""
        echo "   FTP Method:"
        echo "     - Navigate to the data/ directory"
        echo "     - Download first 20 .mrc or .tiff files"
        echo "     - Total size: ~8 GB"
        echo ""
        echo "   Aspera Method (faster):"
        echo "     ascp -QT -l 200m -P33001 -i ~/.aspera/cli/etc/asperaweb_id_dsa.openssh \\"
        echo "       emp_ext3@hx-fasp-1.ebi.ac.uk:/10025/data/*.mrc \\"
        echo "       $DATA_DIR/empiar_10025_subset/"
        echo ""
        echo "3. Save files to: $DATA_DIR/empiar_10025_subset/"
        echo ""
        echo "After manual download, verify with:"
        echo "  ls -lh $DATA_DIR/empiar_10025_subset/*.mrc"
        echo ""
    fi

#================================================================
# FULL DATASET DOWNLOAD (2 TB)
#================================================================

elif [ "$MODE" == "full" ]; then
    echo "WARNING: Full EMPIAR-10025 dataset is 2 TB!"
    echo ""
    echo "This will require:"
    echo "  - 2 TB free disk space"
    echo "  - Several hours/days to download"
    echo "  - Aspera or Globus (recommended for large downloads)"
    echo ""

    read -p "Are you sure you want to download 2 TB? (y/n) " -n 1 -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Download cancelled."
        echo ""
        echo "Consider using the subset instead:"
        echo "  $0 subset"
        exit 0
    fi

    mkdir -p "$DATA_DIR/empiar_10025_full"
    cd "$DATA_DIR/empiar_10025_full"

    echo ""
    echo "========================================================================"
    echo "  Full Dataset Download Methods"
    echo "========================================================================"
    echo ""
    echo "Option 1: Aspera (Recommended - fastest)"
    echo "----------"
    echo "Install Aspera CLI:"
    echo "  https://www.ibm.com/aspera/connect/"
    echo ""
    echo "Download command:"
    echo "  ascp -QT -l 200m -P33001 -i ~/.aspera/cli/etc/asperaweb_id_dsa.openssh \\"
    echo "    emp_ext3@hx-fasp-1.ebi.ac.uk:/10025/data/ \\"
    echo "    $DATA_DIR/empiar_10025_full/"
    echo ""
    echo "Option 2: Globus (Recommended - most reliable)"
    echo "----------"
    echo "1. Create Globus account: https://www.globus.org/"
    echo "2. Install Globus Connect Personal"
    echo "3. Visit: https://app.globus.org/"
    echo "4. Search for collection: 'EMPIAR'"
    echo "5. Navigate to EMPIAR-10025"
    echo "6. Transfer to your endpoint"
    echo ""
    echo "Option 3: rsync (Slower but resumable)"
    echo "----------"
    echo "  rsync -avz --progress \\"
    echo "    emp_ext3@hx-fasp-1.ebi.ac.uk:/10025/data/ \\"
    echo "    $DATA_DIR/empiar_10025_full/"
    echo ""
    echo "========================================================================"
    echo ""
    echo "We recommend using Aspera or Globus for this large download."
    echo "HTTP/FTP are not suitable for 2 TB datasets."
    echo ""
fi

echo "========================================================================"
echo "  Next Steps"
echo "========================================================================"
echo ""
if [ "$MODE" == "subset" ]; then
    echo "After download completes:"
    echo "  1. Verify files: ls -lh $DATA_DIR/empiar_10025_subset/*.mrc"
    echo "  2. Run preprocessing: python tools/preprocess_empiar.py --subset"
    echo "  3. Train model: ./cryo_train_unet --epochs 15 --save"
    echo ""
    echo "Dataset characteristics:"
    echo "  - 20 movies"
    echo "  - ~8 GB total"
    echo "  - Fits in RAM (managed memory mode)"
    echo "  - Fast iteration for development"
else
    echo "After download completes:"
    echo "  1. Verify files: ls -lh $DATA_DIR/empiar_10025_full/*.mrc"
    echo "  2. Run preprocessing: python tools/preprocess_empiar.py --full"
    echo "  3. Train model: ./cryo_train_unet --stream --epochs 15 --save"
    echo ""
    echo "Dataset characteristics:"
    echo "  - 196 movies"
    echo "  - ~2 TB total"
    echo "  - Requires streaming mode"
    echo "  - Production-scale training"
fi
echo ""

echo "Done!"
