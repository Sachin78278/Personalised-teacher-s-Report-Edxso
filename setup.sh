#!/bin/bash
# Quick Setup Script for Teacher Assessment Analysis

echo "🚀 Teacher Assessment Analysis App - Setup"
echo "==========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if .streamlit directory exists
if [ ! -d ".streamlit" ]; then
    mkdir -p .streamlit
    echo "✅ Created .streamlit directory"
fi

# Create secrets.toml if it doesn't exist
if [ ! -f ".streamlit/secrets.toml" ]; then
    echo ""
    echo "⚠️  API Key Configuration Needed"
    echo "==============================="
    echo ""
    echo "1. Go to: https://aistudio.google.com/app/apikey"
    echo "2. Create a new API key"
    echo "3. Copy the key"
    echo "4. Edit .streamlit/secrets.toml and paste your API key"
    echo ""
    echo "Run this command to edit secrets:"
    echo "  nano .streamlit/secrets.toml"
else
    echo "✅ API key configuration found"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Start the app with:"
echo "  streamlit run app.py"
echo ""
echo "Sample CSVs available:"
echo "  - sample_pre_assessment.csv"
echo "  - sample_post_assessment.csv"
