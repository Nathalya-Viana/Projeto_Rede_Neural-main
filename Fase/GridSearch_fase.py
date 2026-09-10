import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor

# ---------------------------------------------------------
# 1. CRIAÇÃO DE PASTAS
# ---------------------------------------------------------
pasta_base = "resultados_fase_minmax"

os.makedirs(os.path.join(pasta_base, "graficos"), exist_ok=True)
os.makedirs(os.path.join(pasta_base, "stats"), exist_ok=True)
os.makedirs(os.path.join(pasta_base, "residuos"), exist_ok=True)

# ---------------------------------------------------------
# 2. CARREGAMENTO E MESCLA DOS DADOS
# ---------------------------------------------------------
paths_data = [
    r'database\data_occ0_2.csv',
    r'database\data_occ0_5.csv',
    r'database\data_occ0_7.csv',
    r'database\data_occ0_9.csv'
]

paths_phase = [
    r'database\original-database\occupancy_0.2.CSV',
    r'database\original-database\occupancy_0.5.CSV',
    r'database\original-database\occupancy_0.7.CSV',
    r'database\original-database\occupancy_0.9.CSV'
]

occ_names = ['0.2', '0.5', '0.7', '0.9']

data = []
for p_data, p_phase in zip(paths_data, paths_phase):
    df = pd.read_csv(p_data)
    # Extrai a coluna 'phases' do CSV original
    phase_series = pd.read_csv(p_phase, index_col=False)['phases'][3:69997]
    df['phases'] = phase_series.values
    data.append(df)

colunas_X = ['sample(0)', 'sample(1)', 'sample(2)', 'sample(3)', 'sample(4)', 'sample(5)', 'sample(6)']
coluna_Y = 'phases'  # Alvo único: FASE

# Concatenar para o GridSearch Global
df_total = pd.concat(data, ignore_index=True)

X_total = df_total[colunas_X].values
y_total = df_total[coluna_Y].values

# ---------------------------------------------------------
# 3. NORMALIZAÇÃO GLOBAL E GRID SEARCH
# ---------------------------------------------------------
# Normalização 
fase_min_global = y_total.min()
fase_max_global = y_total.max()

y_total_scaled = (y_total - fase_min_global) / (fase_max_global - fase_min_global)

base_model = MLPRegressor(
    solver='adam',
    max_iter=2000,
    random_state=42
)

param_grid = {
    'hidden_layer_sizes': [(20,), (30,), (20, 10), (30, 15), (20, 10, 5)],
    'activation': ['relu', 'tanh'],
    'alpha': [1e-5, 1e-4, 1e-3],
    'learning_rate_init': [0.001, 0.01]
}

print("Executando GridSearch Global para Fase (Normalização Min-Max)...")

X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(
    X_total, y_total_scaled, test_size=0.2, random_state=42
)

grid = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=10,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_global, y_train_global)

print("\nMelhores Hiperparâmetros para Fase:")
for k, v in grid.best_params_.items():
    print(f"{k}: {v}")

best_params = grid.best_params_

# ---------------------------------------------------------
# 4. LOOP POR OCUPAÇÃO
# ---------------------------------------------------------
results = []
residuals_phase_dict = {}

for occ, df_occ in zip(occ_names, data):

    print(f"\n{'='*50}")
    print(f" Treinando Fase - OCC = {occ}")
    print(f"{'='*50}")

    X = df_occ[colunas_X].values
    y = df_occ[coluna_Y].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------------------------------------------------
    # NORMALIZAÇÃO LOCAL
    # ---------------------------------------------------------
    fase_min_local = y_train.min()
    fase_max_local = y_train.max()

    # Mapeando treino e teste para [0, 1]
    y_train_scaled = (y_train - fase_min_local) / (fase_max_local - fase_min_local)
    y_test_scaled = (y_test - fase_min_local) / (fase_max_local - fase_min_local)

    model = MLPRegressor(
        hidden_layer_sizes=best_params['hidden_layer_sizes'],
        activation=best_params['activation'],
        alpha=best_params['alpha'],
        learning_rate_init=best_params['learning_rate_init'],
        solver='adam',
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train_scaled)

    # Predição e desnormalização de volta para Nanosegundos (ns)
    y_test_pred_scaled = model.predict(X_test)
    y_test_pred = y_test_pred_scaled * (fase_max_local - fase_min_local) + fase_min_local

    # Cálculo dos Resíduos
    residuos_fase = y_test - y_test_pred
    mae_fase = np.mean(np.abs(residuos_fase))
    std_fase = np.std(residuos_fase)

    residuals_phase_dict[occ] = residuos_fase

    # ---------------------------------------------------------
    # PLOTS DE AVALIAÇÃO
    # ---------------------------------------------------------
    # 1. Curva de Aprendizado
    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_curve_)
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.title(f'Curva de Aprendizado (Fase Min-Max) - OCC {occ}')
    plt.grid(True)
    plt.savefig(os.path.join(pasta_base, "graficos", f"curva_aprendizado_fase_OCC_{occ}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Real vs Predito
    a, b = np.polyfit(y_test, y_test_pred, 1)
    x_fit = np.linspace(y_test.min(), y_test.max(), 100)
    y_fit = a * x_fit + b

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.2, s=15, label='Predições')
    plt.plot(x_fit, x_fit, '--', color='black', linewidth=2, label='Ideal')
    plt.plot(x_fit, y_fit, '-', color='red', linewidth=2, label=f'Fit: y={a:.3f}x + {b:.3f}')
    plt.xlabel('Fase Real (ns)')
    plt.ylabel('Fase Predita (ns)')
    plt.title(f'Fase Real vs Predita (Min-Max) - OCC {occ}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(pasta_base, "graficos", f"real_vs_pred_fase_OCC_{occ}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Registro de Métricas
    results.append({
        'Occupancy': float(occ),
        'Fase_MAE': mae_fase,
        'Fase_STD': std_fase,
        'Epochs': model.n_iter_,
        'Final_Loss': model.loss_
    })

    print(f"Fase -> MAE: {mae_fase:.6f} ns | STD: {std_fase:.6f} ns")

# ---------------------------------------------------------
# 5. SALVAR STATS E RESÍDUOS
# ---------------------------------------------------------
pd.DataFrame(results).to_csv(os.path.join(pasta_base, "stats", "rn_stats_fase.csv"), index=False)
pd.DataFrame(residuals_phase_dict).to_csv(os.path.join(pasta_base, "residuos", "rn_residuals_fase.csv"), index=False)

