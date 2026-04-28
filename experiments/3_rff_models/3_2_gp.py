import argparse
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import roc_auc_score

from subsampling_models import FastBackwardThinning, KernelHerding, FlexibleKernelThinning, make_gaussian_kernel
from utils import HardwareProfiler

# Asumo que importas la nueva clase desde donde la hayas guardado
from utils import build_joint_embedding, save_data_to_csv, compute_gamma_scale
from datasets import get_data

def main():
    parser = argparse.ArgumentParser(description="Prueba de Coreset Iterativo")
    parser.add_argument('--problem_type', type=str, choices=['classification', 'regression'], default='classification')
    parser.add_argument('--task_id', type=int, default=1)
    parser.add_argument('--path_to_save', type=str, default='./results')
    parser.add_argument('--version', type=str, default='iterative_v1')
    # Nuevo argumento para el tamaño del coreset
    parser.add_argument('--target_size', type=int, default=50, help="Número final de puntos deseados")
    parser.add_argument('--alpha', type=float, default=1.0, help="Alpha para la redistribución de pesos")
  
    args = parser.parse_args()

    # Selección de modelo según tipo de problema
    if args.problem_type == 'classification':
        ml_model = GaussianProcessClassifier
        model_params = {
            'kernel': C(1.0) * RBF(1.0)
        }
    else:
        ml_model = GaussianProcessRegressor
        model_params = {
            'kernel': C(1.0) * RBF(1.0),
            'alpha': 1e-2, # Noise level
            'n_restarts_optimizer': 10,
            'random_state': 42
        }
    metric = 'accuracy' if args.problem_type == 'classification' else 'r2'
    # Carga de datos
    X, y, dataset_name, task_id = get_data(args.problem_type, args.task_id)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Ajuste de alpha para regresión basado en los datos
    if args.problem_type == 'regression':
        noise_std = 0.1 * np.std(y_train)
        model_params['alpha'] = noise_std
    
    # Construcción del embedding (mapeo de características)
    # Nota: phi_X suele ser una función o un objeto que proyecta los datos
    phi_X, psi_y = build_joint_embedding(X_train, y_train, problem=args.problem_type, n_components=100)
    
    #psi_y = None
    
    print(f"Calculando coreset iterativo para {dataset_name}...")
    with HardwareProfiler(label=f"Fast Backward Thinning ({dataset_name})") as fbt_prof:
        model = FastBackwardThinning(
            feature_map_X=phi_X,
            feature_map_y=psi_y
        )
        num_to_remove = len(X_train) - int(args.target_size*len(X_train) // 100)
        coreset_X,coreset_y = model.thin(num_to_remove, X=X_train, Y=y_train)
        gamma = compute_gamma_scale(coreset_X)
        init_length_scale = 1.0 / np.sqrt(2 * gamma)
        model_params['kernel'] = C(1.0) * RBF(length_scale=init_length_scale)
          # Convertimos porcentaje a número absoluto
        clf = ml_model(**model_params)
        clf.fit(coreset_X, coreset_y.ravel())  # Aseguramos que coreset_y sea un vector 1D
    
    if args.problem_type == 'classification':
        y_probs = clf.predict_proba(X_test)
        if y_probs.shape[1] == 2:
            acc_fast_backward = roc_auc_score(y_test.ravel(), y_probs[:, 1])
        else:
            # For multi-class One-vs-Rest
            acc_fast_backward = roc_auc_score(y_test.ravel(), y_probs, multi_class='ovr')
    else:
        acc_fast_backward = clf.score(X_test, y_test.ravel())

    time_coreset_fast_backward = fbt_prof.results['time_sec']
    print(f"{metric} fast backward thinning: {acc_fast_backward:.4f} en {time_coreset_fast_backward:.2f} segundos.")


    # Evaluación: Full Dataset (Baseline)
    clf_full = ml_model(**model_params)
    clf_full.fit(X_train, y_train.ravel())
    if args.problem_type == 'classification':
        y_probs = clf_full.predict_proba(X_test)
        if y_probs.shape[1] == 2:
            acc_full = roc_auc_score(y_test.ravel(), y_probs[:, 1])
        else:
            # For multi-class One-vs-Rest
            acc_full = roc_auc_score(y_test.ravel(), y_probs, multi_class='ovr')
    else:
        acc_full = clf_full.score(X_test, y_test.ravel())
    print(f"{metric} (Full data): {acc_full:.4f}")

    # Evaluación: Random Subset (Baseline)
    random_indices = np.random.choice(len(X_train), size=len(coreset_X), replace=False)
    clf_random = ml_model(**model_params)
    clf_random.fit(X_train[random_indices], y_train[random_indices].ravel())
    if args.problem_type == 'classification':
        y_probs = clf_random.predict_proba(X_test)
        if y_probs.shape[1] == 2:
            acc_random = roc_auc_score(y_test.ravel(), y_probs[:, 1])
        else:
            # For multi-class One-vs-Rest
            acc_random = roc_auc_score(y_test.ravel(), y_probs, multi_class='ovr')
    else:
        acc_random = clf_random.score(X_test, y_test.ravel())
    print(f"{metric} (Random subset): {acc_random:.4f}")


    # Evaluación de Kernel Herding (Baseline)
    bw = 1/np.sqrt(2*gamma)
    k_x = make_gaussian_kernel(bw)
    if args.problem_type == 'regression':
        bw_y = 1/np.sqrt(2*np.sqrt(model_params['alpha']))
        k_y = make_gaussian_kernel(bw_y)
    else:
        k_y = lambda x, y: 1.0 if x == y else 0.0  # Kernel delta para clasificación == k_rt
    with HardwareProfiler(label=f"Kernel Herding ({dataset_name})") as herding_prof:
        kernel_herding = KernelHerding(m=len(coreset_X), phi_x=phi_X, phi_y=psi_y)
        index = kernel_herding.get_coreset(X_train, y_train)
        coreset_X_herding = X_train[index]
        coreset_y_herding = y_train[index]
        clf_herding = ml_model(**model_params)
        clf_herding.fit(coreset_X_herding, coreset_y_herding.ravel())

    time_herding = herding_prof.results['time_sec']
    if args.problem_type == 'classification':
        y_probs = clf_herding.predict_proba(X_test)
        if y_probs.shape[1] == 2:
            acc_herding = roc_auc_score(y_test.ravel(), y_probs[:, 1])
        else:
            # For multi-class One-vs-Rest
            acc_herding = roc_auc_score(y_test.ravel(), y_probs, multi_class='ovr')
    else:
        acc_herding = clf_herding.score(X_test, y_test.ravel())
    print(f"{metric} (Kernel Herding): {acc_herding:.4f} en {time_herding:.2f} segundos.")

    # Flexible Kernel Thinning (Baseline)
    bw = 1/np.sqrt(2*gamma)
    k_rt = make_gaussian_kernel(bw/np.sqrt(2))
    k_star = make_gaussian_kernel(bw )
    if args.problem_type == 'regression':
        bw_y = 1/np.sqrt(2*np.sqrt(model_params['alpha']))
        k_y = make_gaussian_kernel(bw_y)
    else:
        k_y = lambda x, y: 1.0 if x == y else 0.0  # Kernel delta para clasificación == k_rt

    with HardwareProfiler(label=f"Flexible Kernel Thinning RFF ({dataset_name})") as kt_prof:
        kt_rff = FlexibleKernelThinning(
            k_rt=k_rt,
            phi_rt=phi_X,   
            k_star=k_star,
            k_y=k_y,
            phi_rt_y=psi_y   
        )

        coreset_flexible_indices_thinning = kt_rff.thin_fraction(X_train, y=y_train, p=args.target_size/100)
        X_flexible_thinning_rff = X_train[coreset_flexible_indices_thinning]
        y_flexible_thinning_rff = y_train[coreset_flexible_indices_thinning] 
        clf_thinning_rff = ml_model(**model_params)
        clf_thinning_rff.fit(X_flexible_thinning_rff, y=y_flexible_thinning_rff.ravel())

    time_flexible_thinning_rff = kt_prof.results['time_sec']
    if args.problem_type == 'classification':
        y_probs = clf_thinning_rff.predict_proba(X_test)
        if y_probs.shape[1] == 2:
            acc_flexible_thinning_rff = roc_auc_score(y_test.ravel(), y_probs[:, 1])
        else:
            # For multi-class One-vs-Rest
            acc_flexible_thinning_rff = roc_auc_score(y_test.ravel(), y_probs, multi_class='ovr')
    else:
        acc_flexible_thinning_rff = clf_thinning_rff.score(X_test, y_test.ravel())

    print(f"{metric} (FLEXIBLE Kernel Thinning con RFF): {acc_flexible_thinning_rff:.4f} en {time_flexible_thinning_rff:.2f} segundos.")
    print(f'Flexible Thinning RFF seleccionó {len(coreset_flexible_indices_thinning)} muestras, objetivo era {args.target_size}% ({int(args.target_size/100*len(X_train))} muestras)')


    # Guardar Resultados
    results = {
        'dataset': dataset_name,
        'task_id': task_id,
        'coreset_size': len(coreset_X),
        'fast_backward': acc_fast_backward,
        'full': acc_full,
        'random': acc_random,
        'herding': acc_herding,
        'flexible_thinning_rff': acc_flexible_thinning_rff,
        'time_herding': time_herding,
        'time_flexible_thinning_rff': time_flexible_thinning_rff,
        'time_coreset_fast_backward': time_coreset_fast_backward,
        'herding_flops': herding_prof.results['flops'],
        'flexible_thinning_rff_flops': kt_prof.results['flops'],
        'fast_backward_flops': fbt_prof.results['flops'],
        'herding_memory_peak_mb': herding_prof.results['memory_peak_mb'],
        'flexible_thinning_rff_memory_peak_mb': kt_prof.results['memory_peak_mb'],
        'fast_backward_memory_peak_mb': fbt_prof.results['memory_peak_mb']
    }
    
    final_path = os.path.join(args.path_to_save, args.problem_type)
    save_path = os.path.join(final_path, 'gp')
    os.makedirs(final_path, exist_ok=True)
    file_name = f"{args.task_id}_results.csv"
    
    save_data_to_csv(directory_path=save_path, file_name=file_name, data=results)
    print(f"Resultados guardados en {save_path}/gp/{file_name}")

if __name__ == "__main__":
    main()