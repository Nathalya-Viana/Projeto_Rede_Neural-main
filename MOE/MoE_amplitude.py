import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Garantir reproducibilidade
torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------
# 1. CRIAÇÃO DE PASTAS
# ---------------------------------------------------------
pasta_base = "ResultadoporFaixaMoE"

os.makedirs(os.path.join(pasta_base, "resultados", "graficos"), exist_ok=True)
os.makedirs(os.path.join(pasta_base, "resultados", "stats"), exist_ok=True)
os.makedirs(os.path.join(pasta_base, "resultados", "residuos"), exist_ok=True)


# ---------------------------------------------------------
# 2. ARQUITETURA MOE EM PYTORCH
# ---------------------------------------------------------
class Expert(nn.Module):
    def __init__(self, input_dim=7):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

    def forward(self, x):
        return self.net(x)


class Router(nn.Module):
    """Gating Network: Determina a contribuição de cada especialista"""
    def __init__(self, input_dim=7, num_experts=6):
        super(Router, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.gate(x)


class MixtureOfExperts(nn.Module):
    """Modelo MoE Unificado"""
    def __init__(self, num_experts=6, input_dim=7):
        super(MixtureOfExperts, self).__init__()
        self.router = Router(input_dim=input_dim, num_experts=num_experts)
        self.experts = nn.ModuleList([Expert(input_dim=input_dim) for _ in range(num_experts)])

    def forward(self, x):
        weights = self.router(x)  # (batch_size, num_experts)
        expert_outputs = torch.stack([expert(x).squeeze(-1) for expert in self.experts], dim=1)  # (batch_size, num_experts)
        final_output = torch.sum(weights * expert_outputs, dim=1, keepdim=True)
        return final_output, weights


# ---------------------------------------------------------
# 3. FUNÇÕES DE TREINAMENTO E PREDIÇÃO (OTIMIZADAS)
# ---------------------------------------------------------
def train_moe_model(model, X_train_np, y_train_np, epochs=50, lr=0.001, batch_size=256):
    X_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(-1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)
            pred, _ = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
    return model


def predict_moe_model(model, X_test_np):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test_np, dtype=torch.float32)
        pred, weights = model(X_tensor)
    return pred.numpy().squeeze(), weights.numpy()


# ---------------------------------------------------------
# 4. LEITURA DOS DADOS
# ---------------------------------------------------------
datasets = {
    '0.2': pd.read_csv('ResultadoporFaixaNormalizado/data_occ0_2.csv'),
    '0.5': pd.read_csv('ResultadoporFaixaNormalizado/data_occ0_5.csv'),
    '0.7': pd.read_csv('ResultadoporFaixaNormalizado/data_occ0_7.csv'),
    '0.9': pd.read_csv('ResultadoporFaixaNormalizado/data_occ0_9.csv')
}

results = []
residuals_dict = {}
heatmap_data = {}

colunas = ['sample(0)', 'sample(1)', 'sample(2)', 'sample(3)', 'sample(4)', 'sample(5)', 'sample(6)']
bins = [0, 50, 100, 150, 200, 300, np.inf]
labels = ['0-50', '50-100', '100-150', '150-200', '200-300', '>300']


# ---------------------------------------------------------
# 5. LOOP DE OCUPAÇÃO
# ---------------------------------------------------------
for occ, df_occ in datasets.items():

    print(f"\n===== OCC {occ} (Treinando MoE) =====")

    X = df_occ[colunas].values
    y = df_occ['AmplitudeSample(4)'].values

    # Holdout
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # K-Fold Cross-Validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_mae = []
    fold_std = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp)):
        X_train, X_val = X_temp[train_idx], X_temp[val_idx]
        y_train, y_val = y_temp[train_idx], y_temp[val_idx]

        # Modelo do Fold
        moe_fold = MixtureOfExperts(num_experts=6, input_dim=7)
        moe_fold = train_moe_model(moe_fold, X_train, y_train, epochs=30, batch_size=256)

        y_val_pred, _ = predict_moe_model(moe_fold, X_val)
        residuals = y_val - y_val_pred

        fold_mae.append(np.mean(np.abs(residuals)))
        fold_std.append(np.std(residuals))

    # Treino Final com o bloco de desenvolvimento (32.000 amostras)
    final_moe = MixtureOfExperts(num_experts=6, input_dim=7)
    final_moe = train_moe_model(final_moe, X_temp, y_temp, epochs=100, batch_size=256)

    # Teste no conjunto mantido isolado (8.000 amostras)
    y_test_pred, test_weights = predict_moe_model(final_moe, X_test)
    test_residuals = y_test - y_test_pred

    test_mae = np.mean(np.abs(test_residuals))
    test_std = np.std(test_residuals)
    residuals_dict[occ] = test_residuals

    # Armazenando Métricas Globais
    results.append({
        'Occupancy': float(occ),
        'KFold_MAE_mean': np.mean(fold_mae),
        'KFold_MAE_std': np.std(fold_mae),
        'KFold_STD_mean': np.mean(fold_std),
        'Test_MAE': test_mae,
        'Test_STD': test_std
    })

    # ---------------------------------------------------------
    # 6. ANÁLISE POR FAIXA DE ENERGIA E PLOTS
    # ---------------------------------------------------------
    df_analysis = pd.DataFrame({'Amplitude_Real': y_test, 'Amplitude_Predita': y_test_pred})
    df_analysis['Erro'] = df_analysis['Amplitude_Real'] - df_analysis['Amplitude_Predita']
    df_analysis['Erro_Abs'] = np.abs(df_analysis['Erro'])
    df_analysis['Faixa'] = pd.cut(df_analysis['Amplitude_Real'], bins=bins, labels=labels)

    mae_faixa = df_analysis.groupby('Faixa', observed=False)['Erro_Abs'].mean()
    heatmap_data[occ] = mae_faixa

    # Boxplot dos Resíduos
    plt.figure(figsize=(10, 6))
    df_analysis.boxplot(column='Erro', by='Faixa')
    plt.title(f'Resíduos por Faixa de Energia (MoE) - OCC {occ}')
    plt.suptitle('')
    plt.xlabel('Faixa de Energia')
    plt.ylabel('Resíduo')
    plt.grid(True)
    
    caminho_boxplot = os.path.join(pasta_base, "resultados", "graficos", f"boxplot_residuos_OCC_{occ}.png")
    plt.savefig(caminho_boxplot, dpi=300, bbox_inches='tight')
    plt.close()

    # Resíduo vs Energia
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, test_residuals, alpha=0.2, s=10)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Amplitude Real')
    plt.ylabel('Resíduo')
    plt.title(f'Resíduo vs Energia (MoE) - OCC {occ}')
    plt.grid(True)
    
    caminho_residuo = os.path.join(pasta_base, "resultados", "graficos", f"residuo_vs_energia_OCC_{occ}.png")
    plt.savefig(caminho_residuo, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Test MAE (MoE) = {test_mae:.4f}")


# ---------------------------------------------------------
# 7. GERAÇÃO DOS RESULTADOS GLOBAIS
# ---------------------------------------------------------
heatmap_df = pd.DataFrame(heatmap_data).T

# Heatmap MAE
plt.figure(figsize=(10, 6))
im = plt.imshow(heatmap_df, aspect='auto')
plt.colorbar(im, label='MAE')
plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns)
plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)
plt.xlabel('Faixa de Energia')
plt.ylabel('Ocupação')
plt.title('MAE por Faixa de Energia e Ocupação (MoE)')

caminho_heatmap = os.path.join(pasta_base, "resultados", "graficos", "heatmap_mae_energia.png")
plt.savefig(caminho_heatmap, dpi=300, bbox_inches='tight')
plt.close()

# STD por Ocupação
results_df = pd.DataFrame(results)
plt.figure(figsize=(8, 6))
plt.plot(results_df['Occupancy'], results_df['Test_STD'], marker='o')
plt.xticks([0.2, 0.5, 0.7, 0.9])
plt.xlabel('Ocupação')
plt.ylabel('STD dos Resíduos')
plt.title('Desvio Padrão dos Resíduos por Ocupação (MoE)')
plt.grid(True)

caminho_std = os.path.join(pasta_base, "resultados", "graficos", "std_por_ocupacao.png")
plt.savefig(caminho_std, dpi=300, bbox_inches='tight')
plt.close()

# Salvando os arquivos CSV
caminho_csv_stats = os.path.join(pasta_base, "resultados", "stats", "rn_stats.csv")
results_df.to_csv(caminho_csv_stats, index=False)

caminho_csv_residuos = os.path.join(pasta_base, "resultados", "residuos", "rn_residuals.csv")
pd.DataFrame(residuals_dict).to_csv(caminho_csv_residuos, index=False)

caminho_csv_heatmap = os.path.join(pasta_base, "resultados", "stats", "mae_por_faixa_energia.csv")
heatmap_df.to_csv(caminho_csv_heatmap)

print("\nProcessamento concluído! Todos os gráficos e CSVs foram salvos com sucesso.")