#!/bin/bash

# Configuration
ENV_NAME=".venv_backward_thinning"
REQ_FILE="requirements.txt"

echo "------------------------------------------------"
echo "🛠️  Starting OpenML Project Environment Setup"
echo "------------------------------------------------"

# 1. Create Virtual Environment
if [ ! -d "$ENV_NAME" ]; then
    echo "Creating virtual environment: $ENV_NAME..."
    python3 -m venv $ENV_NAME
else
    echo "Virtual environment already exists."
fi

# 2. Activate Environment
echo "Activating environment..."
source $ENV_NAME/bin/activate

# 3. Upgrade Pip
echo "Upgrading pip..."
pip install --upgrade pip

# 4. Install Dependencies
if [ -f "$REQ_FILE" ]; then
    echo "Installing dependencies from $REQ_FILE..."
    pip install -r $REQ_FILE
else
    echo "Error: $REQ_FILE not found. Installing base packages (OpenML, Pandas, Scikit-learn)..."
    pip install openml pandas scikit-learn python-dotenv
fi

# 5. Create .env template if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Generating .env template..."
    echo "OPENML_API_KEY=your_key_here" > .env
    echo "Done. Please update your .env file with your actual OpenML API key."
fi

echo "------------------------------------------------"
echo "✅ Setup Complete!"
echo "To activate the environment, run: source $ENV_NAME/bin/activate"
echo "------------------------------------------------"