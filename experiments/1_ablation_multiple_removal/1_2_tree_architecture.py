import os
import argparse
import numpy as np
from itertools import combinations
from datasets import get_data
from utils import build_joint_embedding, save_data_to_csv

def get_signatures(features, T):
    """
    Generates a bitstring signature for each sample based on T random projections.
    """
    n_samples, d = features.shape
    # Generate T random projection vectors
    projections = np.random.randn(d, T)
    # Compute dot products and get signs (True for +, False for -)
    bool_matrix = (features @ projections) > 0
    # Convert rows to tuples to use as dictionary keys
    return [tuple(row) for row in bool_matrix]

def get_opposite_signature(sig):
    """Returns the bitwise NOT of a signature tuple."""
    return tuple(not b for b in sig)

def main():
    parser = argparse.ArgumentParser(description="Ablation Study: Random Projection Tree")
    parser.add_argument('--problem_type', type=str, choices=['classification', 'regression'])
    parser.add_argument('--task_id', type=int)
    parser.add_argument('--path_to_save', type=str)
    parser.add_argument('--version', type=str, default='version_tree')
    parser.add_argument('--max_removals', type=int, default=2)
    parser.add_argument('--tree_depth', type=int, default=8, help='Number of random projections (T)')
    args = parser.parse_args()

    X, y, dataset_name, task_id = get_data(args.problem_type, args.task_id)
    
    # Compute the joint embedding components
    phi_X, psi_y = build_joint_embedding(X, y, problem=args.problem_type)
    lifted_X = phi_X(X)
    lifted_y = psi_y(y)
    n_samples = lifted_X.shape[0]

    # 1. Compute mu_p (The mean operator)
    mu_p = (lifted_X.T @ lifted_y) / n_samples

    # 2. Construct the Joint Features Phi(x,y) = phi(x) \otimes psi(y)
    # We flatten them to treat them as rows in a matrix for the projections
    print("Constructing joint features...")
    joint_features = np.zeros((n_samples, lifted_X.shape[1] * lifted_y.shape[1]))
    for i in range(n_samples):
        # Vectorized outer product
        joint_features[i] = np.outer(lifted_X[i], lifted_y[i]).flatten()

    # 3. Generate Signatures (The "Tree" partitioning)
    print(f"Generating signatures with depth T={args.tree_depth}...")
    signatures = get_signatures(joint_features, args.tree_depth)
    
    # Map signatures to indices
    sig_map = {}
    for idx, sig in enumerate(signatures):
        if sig not in sig_map:
            sig_map[sig] = []
        sig_map[sig].append(idx)

    # 4. Heuristic Selection: Match opposites for m=2
    # For m=1, we just take all. For m=2, we search for opposites.
    if args.max_removals >= 2:
        print("Searching for opposite combinations...")
        found_combinations = []
        
        for sig, indices in sig_map.items():
            opposite_sig = get_opposite_signature(sig)
            if opposite_sig in sig_map:
                # Pair every index in current bucket with every index in the opposite bucket
                for i in indices:
                    for j in sig_map[opposite_sig]:
                        if i < j: # Avoid duplicates
                            found_combinations.append((i, j))

        # Process the selected heuristic combinations
        print(f"Found {len(found_combinations)} opposite pairs vs {n_samples * (n_samples - 1) // 2}. Evaluating errors...")
        # Add the m=1 case this is (i) for i in range(n_samples)
        found_combinations.extend([(i,) for i in range(n_samples)])
        for indices in found_combinations:
            # We use the same joint_features we already computed
            subset_sum = np.sum(joint_features[list(indices)], axis=0).reshape(mu_p.shape)
            
            # Error = || m * mu_p - sum(Phi_selected) ||
            error = np.linalg.norm(len(indices) * mu_p - subset_sum)

            # Save the results
            save_path = os.path.join(args.path_to_save, args.version)
            save_data_to_csv(
                save_path,
                f"{args.problem_type[:3]}_{args.task_id}.csv",
                {   
                    "problem_type": args.problem_type,
                    "error": error,
                    "task_id": task_id,
                    "m": len(indices),
                    "dataset": dataset_name,
                    "removed_indices": indices,
                }
            )
            
    print(f"Done. Processed {len(found_combinations)} heuristic pairs.")

if __name__ == "__main__":
    main()