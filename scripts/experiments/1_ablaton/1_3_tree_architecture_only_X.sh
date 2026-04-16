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
experiment="1_3_tree_architecture_only_X"
path_to_save="results/$experiment/"
version="test_t10"
max_removals=1

args=(
    "--task_id" "$task_id"
    "--path_to_save" "$path_to_save"
    "--version" "$version"
    "--problem_type" "$problem_type"
    "--max_removals" "$max_removals"
    "--tree_depth" "10"
)

# Note: Removed --experiment because your help output shows 
# that the script doesn't actually accept a --experiment flag.

python -m experiments.1_ablation_multiple_removal.1_3_tree_architecture_only_X "${args[@]}"