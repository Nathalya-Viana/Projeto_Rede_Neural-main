import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# Garante a criação da pasta de estatísticas
os.makedirs("resultados_2_camadas/stats", exist_ok=True)
os.makedirs("resultados_2_camadas/residuos", exist_ok=True)

# Arquivos de entrada
data = {
    '0.2': r"C:\Users\Nat\Desktop\Projeto_Rede_Neural\database\data_occ0_2.csv",
    '0.5': r"C:\Users\Nat\Desktop\Projeto_Rede_Neural\database\data_occ0_5.csv",
    '0.7': r"C:\Users\Nat\Desktop\Projeto_Rede_Neural\database\data_occ0_7.csv",
    '0.9': r"C:\Users\Nat\Desktop\Projeto_Rede_Neural\database\data_occ0_9.csv"
}

results = []
residuals_dict = {}

print("Iniciando treinamento rápido para a arquitetura (20, 10)...")

# Loop por Ocupação (OCC)
for occ, path in data.items():
    print(f"Processando OCC = {occ}...")

    df = pd.read_csv(path)

    X = df[['sample(0)', 'sample(1)', 'sample(2)',
            'sample(3)', 'sample(4)', 'sample(5)',
            'sample(6)']].values

    y = df['AmplitudeSample(4)'].values

    # Divisão 80/20 com semente fixa
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Modelo fixo com 2 camadas e hiperparâmetros predefinidos
    model = MLPRegressor(
        hidden_layer_sizes=(20, 10),
        activation='relu',
        alpha=0.0001,
        learning_rate_init=0.001,
        solver='adam',
        max_iter=500,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    residuals = y_test - y_test_pred

    test_mae = np.mean(np.abs(residuals))
    test_std = np.std(residuals)

    residuals_dict[occ] = residuals

    # Guardar estatísticas da execução
    results.append({
        'Occupancy': float(occ),
        'Hidden_Layers': '(20, 10)',
        'Test_MAE': test_mae,
        'Test_STD': test_std,
        'Epochs': model.n_iter_,
        'Final_Loss': model.loss_
    })

    print(f"  -> Test MAE: {test_mae:.6f} | Test STD: {test_std:.6f}")

# Salvar CSV de estatísticas
stats_df = pd.DataFrame(results)
stats_df.to_csv("resultados_2_camadas/stats/rn_stats_2camadas.csv", index=False)

# Salvar CSV de resíduos
rn_residuals = pd.DataFrame(residuals_dict)
rn_residuals.to_csv("resultados_2_camadas/residuos/rn_residuals_2camadas.csv", index=False)

print("\nConcluído! Os resultados foram salvos em 'resultados_2_camadas/stats/rn_stats_2camadas.csv'.")