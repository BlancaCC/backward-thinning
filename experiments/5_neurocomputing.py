import argparse
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, r2_score

from subsampling_models import (
    KernelHerding,
    FlexibleKernelThinning,
    phiKBKH,
    make_gaussian_kernel
)

from utils import (
    HardwareProfiler,
    build_joint_embedding,
    save_data_to_csv,
    compute_gamma_scale
)

from datasets import get_data


_r2_score = lambda y_true, y_pred: max(0, r2_score(y_true, y_pred))

def evaluate_model(model, X_test, y_test, problem_type, model_type):
    """
    Evaluate model performance.

    GP classification uses ROC-AUC.
    SVM classification uses balanced accuracy.
    Regression uses R2 score.
    """

    if problem_type == "classification":

        # Gaussian Process classification
        if model_type == "gp":
            y_probs = model.predict_proba(X_test)

            if y_probs.shape[1] == 2:
                return roc_auc_score(y_test.ravel(), y_probs[:, 1])

            return roc_auc_score(
                y_test.ravel(),
                y_probs,
                multi_class="ovr"
            )

        # SVM classification
        y_pred = model.predict(X_test)
        return balanced_accuracy_score(y_test, y_pred)

    # Regression
    y_pred = model.predict(X_test)
    return _r2_score(y_test, y_pred)


def build_model(problem_type, model_type, gamma, random_state=42):
    """
    Create ML model.
    """

    # -------------------------
    # SVM
    # -------------------------
    if model_type == "svm":

        if problem_type == "classification":

            model = SVC(
                kernel="rbf",
                gamma=gamma,
                class_weight="balanced"
            )

        else:
            model = SVR(
                kernel="rbf",
                gamma=gamma
            )

    # -------------------------
    # Gaussian Process
    # -------------------------
    else:

        length_scale = 1.0 / np.sqrt(2 * gamma)
        kernel = C(1.0) * RBF(length_scale=length_scale)

        if problem_type == "classification":

            model = GaussianProcessClassifier(
                kernel=kernel,
                random_state=random_state
            )

        else:

            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                n_restarts_optimizer=5,
                random_state=random_state
            )

    return model


def main():

    parser = argparse.ArgumentParser(
        description="Compare coreset methods"
    )

    parser.add_argument(
        "--problem_type",
        type=str,
        choices=["classification", "regression"],
        default="classification"
    )

    parser.add_argument(
        "--model_type",
        type=str,
        choices=["svm", "gp"],
        default="svm"
    )

    parser.add_argument(
        "--task_id",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target_size",
        type=int,
        default=50,
        help="Coreset percentage"
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=1
    )

    parser.add_argument(
        "--path_to_save",
        type=str,
        default="./results"
    )

    args = parser.parse_args()

    # -------------------------
    # Load dataset
    # -------------------------
    X, y, dataset_name, task_id = get_data(
        args.problem_type,
        args.task_id
    )

    all_results = []

    # =========================================================
    # Repetitions
    # =========================================================
    for repetition in range(args.repetitions):

        print(f"\n========== REPETITION {repetition + 1} ==========")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.5,
            random_state=repetition
        )

        # Number of samples to keep
        target_n = int(args.target_size / 100 * len(X_train))

        # Feature maps
        phi_X, psi_y = build_joint_embedding(
            X_train,
            y_train,
            problem=args.problem_type,
            n_components=100
        )

        # Kernel scale
        gamma = compute_gamma_scale(X_train)

        # =====================================================
        # Kernel Herding
        # =====================================================
        with HardwareProfiler(
            label=f"Kernel Herding ({dataset_name})"
        ) as herd_prof:

            herding = KernelHerding(
                m=target_n,
                phi_x=phi_X,
                phi_y=psi_y
            )

            indices = herding.get_coreset(X_train, y_train)

            X_core = X_train[indices]
            y_core = y_train[indices]

            model = build_model(
                args.problem_type,
                args.model_type,
                gamma
            )

            model.fit(X_core, y_core.ravel())

        score_herding = evaluate_model(
            model,
            X_test,
            y_test,
            args.problem_type,
            args.model_type
        )

        print(f"Kernel Herding score: {score_herding:.4f}")

        # =====================================================
        # Flexible Kernel Thinning
        # =====================================================
        bw = 1 / np.sqrt(2 * gamma)

        k_rt = make_gaussian_kernel(bw / np.sqrt(2))
        k_star = make_gaussian_kernel(bw)

        if args.problem_type == "classification":
            k_y = lambda x, y: 1.0 if x == y else 0.0
        else:
            k_y = make_gaussian_kernel(bw)

        with HardwareProfiler(
            label=f"Flexible Kernel Thinning ({dataset_name})"
        ) as fkt_prof:

            thinning = FlexibleKernelThinning(
                k_rt=k_rt,
                phi_rt=phi_X,
                k_star=k_star,
                k_y=k_y,
                phi_rt_y=psi_y
            )

            indices = thinning.thin_fraction(
                X_train,
                y=y_train,
                p=args.target_size / 100
            )

            X_core = X_train[indices]
            y_core = y_train[indices]

            model = build_model(
                args.problem_type,
                args.model_type,
                gamma
            )

            model.fit(X_core, y_core.ravel())

        score_fkt = evaluate_model(
            model,
            X_test,
            y_test,
            args.problem_type,
            args.model_type
        )

        print(f"Flexible Kernel Thinning score: {score_fkt:.4f}")

        # =====================================================
        # phiKBKH
        # =====================================================
        with HardwareProfiler(
            label=f"phiKBKH ({dataset_name})"
        ) as pkbkh_prof:

            pkbkh = phiKBKH(
                phi_x=phi_X,
                phi_y=psi_y
            )

            num_to_remove = len(X_train) - target_n

            X_core, y_core = pkbkh._thin_feature_map(
                num_to_remove,
                X=X_train,
                Y=y_train
            )

            model = build_model(
                args.problem_type,
                args.model_type,
                gamma
            )

            model.fit(X_core, y_core.ravel())

        score_pkbkh = evaluate_model(
            model,
            X_test,
            y_test,
            args.problem_type,
            args.model_type
        )

        print(f"phiKBKH score: {score_pkbkh:.4f}")

        # =====================================================
        # Save results
        # =====================================================
        result = {
            "dataset": dataset_name,
            "dataset_size": len(X),
            "task_id": task_id,
            "repetition": repetition,
            "problem_type": args.problem_type,
            "model_type": args.model_type,
            "coreset_percent": args.target_size,

            # Scores
            "kernel_herding": score_herding,
            "flexible_kernel_thinning": score_fkt,
            "phiKBKH": score_pkbkh,

            # Times
            "time_kernel_herding": herd_prof.results["time_sec"],
            "time_flexible_kernel_thinning": fkt_prof.results["time_sec"],
            "time_phiKBKH": pkbkh_prof.results["time_sec"],

            # Memory
            "memory_kernel_herding": herd_prof.results["memory_peak_mb"],
            "memory_flexible_kernel_thinning": fkt_prof.results["memory_peak_mb"],
            "memory_phiKBKH": pkbkh_prof.results["memory_peak_mb"],

            # FLOPS
            "flops_kernel_herding": herd_prof.results["flops"],
            "flops_flexible_kernel_thinning": fkt_prof.results["flops"],
            "flops_phiKBKH": pkbkh_prof.results["flops"],

        }

        all_results.append(result)

    # =========================================================
    # Save all repetitions
    # =========================================================
    final_path = os.path.join(
        args.path_to_save,
        args.problem_type,
        args.model_type
    )

    os.makedirs(final_path, exist_ok=True)

    file_name = f"{args.task_id}_results.csv"

    for result in all_results:
        save_data_to_csv(
            directory_path=final_path,
            file_name=file_name,
            data=result
        )

    print(f"\nResults saved to {final_path}/{file_name}")


if __name__ == "__main__":
    main()