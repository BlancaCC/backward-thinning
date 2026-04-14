#!/bin/bash

# Configuration
ENV_NAME=".venv_backward_thinning"

echo "------------------------------------------------"
echo "🧪 Preparing to run OpenML Project Tests"
echo "------------------------------------------------"

# 1. Check if environment exists
if [ -d "$ENV_NAME" ]; then
    echo "🔌 Activating environment: $ENV_NAME..."
    source $ENV_NAME/bin/activate
else
    echo "❌ Error: Virtual environment '$ENV_NAME' not found."
    echo "Please run your setup script first."
    exit 1
fi

# 2. Execute tests
echo "🚀 Running: python -m unittest discover -s tests"
echo "------------------------------------------------"
python -m unittest discover -s tests

# 3. Capture exit code
RESULT=$?

echo "------------------------------------------------"
if [ $RESULT -eq 0 ]; then
    echo "✅ Tests passed successfully!"
else
    echo "❌ Tests failed. Please check the output above."
fi

