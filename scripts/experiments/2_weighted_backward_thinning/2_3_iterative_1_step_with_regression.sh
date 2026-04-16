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
experiment="2_3_iterative_1_step_with_regression"
version="version_0"
path_to_save="results/$experiment/$version"

target_size_percentage=70

problem_types=("classification" "regression")
problem_types=("regression")  # Solo regresión para este experimento

for task_id in {1..10}; do
    for problem_type in "${problem_types[@]}"; do
        echo "🚀 Running task_id: $task_id, problem_type: $problem_type..."
        args=(
            "--task_id" "$task_id"
            "--path_to_save" "$path_to_save"
            "--version" "$version"
            "--problem_type" "$problem_type"
            "--target_size" "$target_size_percentage"
        )

        # Note: Removed --experiment because your help output shows 
        # that the script doesn't actually accept a --experiment flag.

        python -m experiments.2_weighted_backward_thinning.2_3_iterative_1_step_with_regression "${args[@]}" &
    done
done