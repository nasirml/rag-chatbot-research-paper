#!/bin/bash

# Setup script for PDF Research Paper Processing System
# This script installs dependencies and sets up the environment

echo "🔧 Setting up PDF Research Paper Processing System..."
echo "================================================"

# Check Python version
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+')
echo "📦 Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🏗️  Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements_pdf_processor.txt

# Download NLTK data
echo "📚 Downloading NLTK data..."
python3 -c "
import nltk
resources = ['punkt', 'punkt_tab']
for resource in resources:
    try:
        nltk.download(resource, quiet=True)
        print(f'✅ Downloaded {resource}')
    except Exception as e:
        print(f'⚠️  Could not download {resource}: {e}')
"

# Verify installation
echo "🧪 Verifying installation..."
python3 -c "
try:
    import PyPDF2
    import fitz
    import nltk
    from sentence_transformers import SentenceTransformer
    import numpy as np
    print('✅ All dependencies installed successfully!')
    
    # Test model loading
    print('🤖 Testing model loading...')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print('✅ Model loaded successfully!')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
except Exception as e:
    print(f'❌ Error: {e}')
    exit(1)
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📖 Usage:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run the example: python3 example_usage.py --demo"
echo "  3. Process your PDF: python3 example_usage.py --pdf your_paper.pdf"
echo ""
echo "💡 Tips:"
echo "  - Use --help to see all available options"
echo "  - The system will automatically download the embedding model on first use"
echo "  - Results are saved to the ./output directory by default"
