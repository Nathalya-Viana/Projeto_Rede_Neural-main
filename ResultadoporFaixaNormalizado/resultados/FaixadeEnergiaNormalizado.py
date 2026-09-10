import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone

import os


# CRIA PASTAS

os.makedirs("ResultadoporFaixaNormalizado/resultados/graficos", exist_ok=True)
os.makedirs("ResultadoporFaixaNormalizado/resultados/stats", exist_ok=True)
os.makedirs("ResultadoporFaixaNormalizado/resultados/residuos", exist_ok=True)


# LEITURA DOS DADOS


datasets = {
    '0.2': pd.read_csv('ResultadoporFaixaNormalizado/data_occ0_2.csv'),
    '0.5': pd.read_csv('ResultadoporFaixaNormalizado/data_occ0_5.csv'),
    '0.7': pd.read_csv('ResultadoporFaixaNormalizado/data_occ0_7.csv'),
    '0.9': pd.read_csv('ResultadoporFaixaNormalizado/data_occ0_9.csv')
}

# REDE NEURAL


base_model = MLPRegressor(
    hidden_layer_sizes=(20,10,5),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)


# ARMAZENAMENTO


results = []
residuals_dict = {}

# dados para heatmap
heatmap_data = {}


# LOOP DAS OCUPAÇÕES


for occ, df_occ in datasets.items():

    print(f"\n===== OCC {occ} =====")

    
    # Entradas
    

    colunas = [
        'sample(0)',
        'sample(1)',
        'sample(2)',
        'sample(3)',
        'sample(4)',
        'sample(5)',
        'sample(6)'
    ]


    # Cria faixas de energia 

    
    bins = [
        0,
        50,
        100,
        150,
        200,
        300,
        np.inf
    ]

    labels = [
        '0-50',
        '50-100',
        '100-150',
        '150-200',
        '200-300',
        '>300'
    ]

    df_occ['Faixa'] = pd.cut(df_occ['AmplitudeSample(4)'],bins=bins,labels=labels
)
    # Picos de cada faixa
    picos_faixa = (df_occ.groupby('Faixa', observed=True)[colunas].max().max(axis=1))
    
    # Normalização
    for faixa in picos_faixa.index:

        idx = df_occ['Faixa'] == faixa

        df_occ.loc[idx, colunas] = (df_occ.loc[idx, colunas] / picos_faixa[faixa])

    # Entradas
    
    X = df_occ[colunas].values

    # Saída
    

    y = df_occ['AmplitudeSample(4)'].values

    
    # Holdout
    

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    
    # K-Fold
    

    kf = KFold(n_splits=10,shuffle=True,random_state=42)

    fold_mae = []
    fold_std = []

    for train_idx, val_idx in kf.split(X_temp):

        X_train = X_temp[train_idx]
        X_val = X_temp[val_idx]

        y_train = y_temp[train_idx]
        y_val = y_temp[val_idx]

        model = clone(base_model)

        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)

        residuals = y_val - y_val_pred

        fold_mae.append(np.mean(np.abs(residuals)))

        fold_std.append(np.std(residuals))

    
    # Treino Final
    

    final_model = clone(base_model)

    final_model.fit(X_temp, y_temp)

    
    # Teste
    

    y_test_pred = final_model.predict(X_test)

    test_residuals = y_test - y_test_pred

    test_mae = np.mean(np.abs(test_residuals))

    test_std = np.std(test_residuals)

    residuals_dict[occ] = test_residuals

    
    # Métricas
    

    results.append({
        'Occupancy': float(occ),

        'KFold_MAE_mean':
            np.mean(fold_mae),

        'KFold_MAE_std':
            np.std(fold_mae),

        'KFold_STD_mean':
            np.mean(fold_std),

        'Test_MAE':
            test_mae,

        'Test_STD':
            test_std
    })
    
    
    # ANÁLISE POR FAIXA DE ENERGIA
    

    df_analysis = pd.DataFrame({'Amplitude_Real': y_test,'Amplitude_Predita': y_test_pred})

    df_analysis['Erro'] = (df_analysis['Amplitude_Real'] - df_analysis['Amplitude_Predita'])

    df_analysis['Erro_Abs'] = np.abs(df_analysis['Erro'])

    df_analysis['Faixa'] = pd.cut(df_analysis['Amplitude_Real'],bins=bins,labels=labels)

    
    # MAE por faixa
    

    mae_faixa = (
            df_analysis
            .groupby('Faixa')['Erro_Abs']
            .mean()
        )

    heatmap_data[occ] = mae_faixa

        
    # Boxplot
    

    plt.figure(figsize=(10,6))

    df_analysis.boxplot(column='Erro',by='Faixa')

    plt.title(f'Resíduos por Faixa de Energia - OCC {occ}')

    plt.suptitle('')

    plt.xlabel('Faixa de Energia')
    plt.ylabel('Resíduo')

    plt.grid(True)

    plt.savefig(f"ResultadoporFaixaNormalizado/resultados/graficos/boxplot_residuos_OCC_{occ}.png",dpi=300,bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"Test MAE = {test_mae:.4f}")


    # HEATMAP


    heatmap_df = pd.DataFrame(heatmap_data).T

    plt.figure(figsize=(10,6))

    im = plt.imshow(heatmap_df,aspect='auto')

    plt.colorbar(im,label='MAE')

    plt.xticks(range(len(heatmap_df.columns)),heatmap_df.columns)

    plt.yticks(range(len(heatmap_df.index)),heatmap_df.index)

    plt.xlabel('Faixa de Energia')
    plt.ylabel('Ocupação')

    plt.title('MAE por Faixa de Energia e Ocupação')

    plt.savefig("ResultadoporFaixaNormalizado/resultados/graficos/heatmap_mae_energia.png",dpi=300,bbox_inches='tight')
    plt.show()
    plt.close()

    # OCUPAÇÃO X DESVIO PADRÃO

    results_df = pd.DataFrame(results)

    plt.figure(figsize=(8,6))

    plt.plot(results_df['Occupancy'],results_df['Test_STD'],marker='o')

    plt.xticks([0.2,0.5,0.7,0.9])

    plt.xlabel('Ocupação')
    plt.ylabel('STD dos Resíduos')

    plt.title('Desvio Padrão dos Resíduos por Ocupação')

    plt.grid(True)

    plt.savefig('ResultadoporFaixaNormalizado/resultados/graficos/std_por_ocupacao.png',dpi=300,bbox_inches='tight')

    plt.show()
    plt.close()
  # Resíduo x Energia
    

    plt.figure(figsize=(8,6))

    plt.scatter(
            y_test,
            test_residuals,
            alpha=0.2,
            s=10
        )

    plt.axhline(
            0,
            color='red',
            linestyle='--'
        )

    plt.xlabel('Amplitude Real')
    plt.ylabel('Resíduo')

    plt.title(
            f'Resíduo vs Energia - OCC {occ}'
        )

    plt.grid(True)

    plt.savefig(
            f"ResultadoporFaixaNormalizado/resultados/graficos/residuo_vs_energia_OCC_{occ}.png",
            dpi=300,
            bbox_inches='tight'
        )
    plt.show()
    plt.close()

# SALVAR CSVs


stats_df = pd.DataFrame(results)

stats_df.to_csv("ResultadoporFaixaNormalizado/resultados/stats/rn_stats.csv",index=False)

rn_residuals = pd.DataFrame(residuals_dict)

rn_residuals.to_csv("ResultadoporFaixaNormalizado/resultados/residuos/rn_residuals.csv", index=False)

heatmap_df.to_csv("ResultadoporFaixaNormalizado/resultados/stats/mae_por_faixa_energia.csv")

