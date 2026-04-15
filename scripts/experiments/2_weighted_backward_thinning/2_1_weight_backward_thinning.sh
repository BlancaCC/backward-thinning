#!/bin/bash

# Configuration
ENV_NAME=".venv_backward_thinning"


# 1. Check if environment exists
if [ -d "$ENV_NAME" ]; then
    echo "🔌 Activating environment: $ENV_NAME..."
    source $ENV_NAME/bin/activate
else
    echo "❌ Error: Virtual environment '$ENV_NAME' not found."
    echo "Please run your setup script first."
    exit 1
fi

# 2. Execute

problem_types=("classification" "regression")
problem_type="classification"

task_id=3
experiment="2_1_weight_backward_thinning"
path_to_save="results/$experiment/"
version="test"
max_removals=0

args=(
    "--task_id" "$task_id"
    "--path_to_save" "$path_to_save"
    "--version" "$version"
    "--problem_type" "$problem_type"
    "--max_removals" "$max_removals"
)

# Note: Removed --experiment because your help output shows 
# that the script doesn't actually accept a --experiment flag.

python -m experiments.2_weighted_backward_thinning.2_1_initial_test "${args[@]}"