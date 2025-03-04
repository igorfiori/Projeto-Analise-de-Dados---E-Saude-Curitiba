import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configurações globais
pd.options.display.float_format = '{:,.2f}'.format  # Formatação de 2 casas decimais
pd.set_option('display.max_columns', 100)  # Mostrar todas as colunas
sns.set()

# Criar diretório de saída se não existir
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

def carregar_dados(caminho):
    """Carrega o dataset e remove espaços extras nas colunas."""
    df = pd.read_csv(caminho, sep=";", encoding="ISO-8859-1")
    df.columns = df.columns.str.strip()  # Remover espaços extras nos nomes das colunas
    print(f"Dataset carregado com sucesso! {df.shape[0]} registros e {df.shape[1]} colunas.")
    return df

def corrigir_colunas(df):
    """Corrige nomes de colunas erradas."""
    df.rename(columns={'Municício': 'Município'}, inplace=True)

def converter_datas(df):
    """Converte colunas de data para formato datetime."""
    colunas_data = ['Data do Atendimento', 'Data de Nascimento', 'Data do Internamento']
    for col in colunas_data:
        df[col] = pd.to_datetime(df[col].str.strip(), errors='coerce', dayfirst=True)

def criar_colunas_adicionais(df):
    """Cria colunas de idade, classificação etária, dia da semana e turno do atendimento."""
    df['Idade'] = (df['Data do Atendimento'] - df['Data de Nascimento']).dt.days // 365
    
    def classificar_idade(idade):
        if idade < 12:
            return 'Criança'
        elif idade < 18:
            return 'Adolescente'
        elif idade < 60:
            return 'Adulto'
        return 'Idoso'
    
    df['Classificação Etária'] = df['Idade'].apply(classificar_idade)
    
    dias_traducao = {
        "Monday": "Segunda-feira",
        "Tuesday": "Terça-feira",
        "Wednesday": "Quarta-feira",
        "Thursday": "Quinta-feira",
        "Friday": "Sexta-feira",
        "Saturday": "Sábado",
        "Sunday": "Domingo"
    }
    df['Dia da Semana'] = df['Data do Atendimento'].dt.day_name().map(dias_traducao)
    
    def turno_do_atendimento(hora):
        if hora < 6:
            return 'Madrugada'
        elif hora < 19:
            return 'Dia'
        return 'Noite'
    
    df['Turno do Atendimento'] = df['Data do Atendimento'].dt.hour.apply(turno_do_atendimento)
    df['Fim de Semana'] = df['Dia da Semana'].isin(['Sábado', 'Domingo'])

def remover_colunas_e_linhas(df):
    """Remove colunas desnecessárias e linhas com dados ausentes."""
    colunas_remover = [
        'Código do Tipo de Unidade', 'Código da Unidade', 'Código do Procedimento',
        'Descrição do Procedimento', 'Código do CBO', 'Descrição do CBO',
        'Descrição do CID', 'Qtde Prescrita Farmácia Curitibana',
        'Qtde Dispensada Farmácia Curitibana', 'Qtde de Medicamento Não Padronizado',
        'Área de Atuação'
    ]
    df.drop(columns=colunas_remover, inplace=True, errors='ignore')
    df.dropna(subset=['Código do CID'], inplace=True)

def verificar_tipos_de_dados(df):
    """Exibe informações sobre os tipos de dados das colunas."""
    print("\nTipos de dados das colunas:")
    print(df.dtypes)

def analise_exploratoria(df):
    """Realiza análises exploratórias dos dados e salva os gráficos."""
    print("\nResumo estatístico das colunas numéricas:")
    resumo = df.describe(include='all')
    resumo.to_csv(os.path.join(output_dir, "estatisticas_gerais.csv"))
    print(resumo)
    
    print("\nDistribuição de Idades:")
    plt.figure(figsize=(8,5))
    sns.histplot(df['Idade'], bins=30, kde=True)
    plt.xlabel("Idade")
    plt.ylabel("Frequência")
    plt.title("Distribuição de Idades dos Pacientes")
    plt.savefig(os.path.join(output_dir, "distribuicao_idades.png"))
    plt.close()
    
    print("\nDistribuição de Atendimentos por Dia da Semana:")
    plt.figure(figsize=(8,5))
    sns.countplot(x=df['Dia da Semana'], order=["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira", "Sexta-feira", "Sábado", "Domingo"])
    plt.xlabel("Dia da Semana")
    plt.ylabel("Quantidade de Atendimentos")
    plt.title("Distribuição de Atendimentos por Dia da Semana")
    plt.savefig(os.path.join(output_dir, "atendimentos_dia_semana.png"))
    plt.close()
    
    print("\nDistribuição por Turno de Atendimento:")
    plt.figure(figsize=(8,5))
    sns.countplot(x=df['Turno do Atendimento'], order=["Madrugada", "Dia", "Noite"])
    plt.xlabel("Turno")
    plt.ylabel("Quantidade de Atendimentos")
    plt.title("Distribuição de Atendimentos por Turno")
    plt.savefig(os.path.join(output_dir, "atendimentos_turno.png"))
    plt.close()
    
    print("\nDistribuição de Atendimentos por Tipo de Unidade:")
    plt.figure(figsize=(10,5))
    sns.countplot(y=df['Tipo de Unidade'], order=df['Tipo de Unidade'].value_counts().index)
    plt.xlabel("Quantidade de Atendimentos")
    plt.ylabel("Tipo de Unidade")
    plt.title("Distribuição de Atendimentos por Tipo de Unidade")
    plt.savefig(os.path.join(output_dir, "atendimentos_tipo_unidade.png"))
    plt.close()
    
    print("\nMunicípios que mais enviam pacientes para Curitiba:")
    municipios = df['Município'].value_counts().head(10)
    plt.figure(figsize=(17,5))
    sns.barplot(x=municipios.values, y=municipios.index)
    plt.xlabel("Quantidade de Atendimentos")
    plt.ylabel("Município")
    plt.title("Municípios que mais enviam pacientes para Curitiba")
    plt.savefig(os.path.join(output_dir, "municipios_atendimentos.png"))
    plt.close()

def proporcao_encaminhados_especialistas_e_solicitacao_exames(df):
    """Calcula e exibe as proporções de atendimentos encaminhados para especialistas e com solicitação de exames."""
    # Garantir que as colunas existam e estejam com valores válidos
    if 'Encaminhado para Especialista' in df.columns and 'Solicitação de Exames' in df.columns:
        # Corrigir os valores para binários (0 ou 1) se necessário
        df['Encaminhado para Especialista'] = df['Encaminhado para Especialista'].replace({'Nao': 0, 'Sim': 1})
        df['Solicitação de Exames'] = df['Solicitação de Exames'].replace({'Nao': 0, 'Sim': 1})
        
        # Calcular a proporção de atendimentos encaminhados para especialistas
        total_atendimentos = len(df)
        atendimentos_especialista = df['Encaminhado para Especialista'].sum()
        proporcao_especialista = atendimentos_especialista / total_atendimentos
        
        # Calcular a proporção de atendimentos com solicitação de exames
        atendimentos_exames = df['Solicitação de Exames'].sum()
        proporcao_exames = atendimentos_exames / total_atendimentos
        
        # Exibir os resultados
        print(f"Total de Atendimentos: {total_atendimentos}")
        print(f"Atendimentos encaminhados para especialistas: {atendimentos_especialista} ({proporcao_especialista:.2%})")
        print(f"Atendimentos com solicitação de exames: {atendimentos_exames} ({proporcao_exames:.2%})")
        
        # Gráfico de barras para proporção de encaminhados para especialistas
        plt.figure(figsize=(12, 8))
        proporcao_especialista_data = [proporcao_especialista, 1 - proporcao_especialista]
        labels_especialista = ['Encaminhados para Especialista', 'Não Encaminhados']
        plt.bar(labels_especialista, proporcao_especialista_data, color=['#FF6347', '#4682B4'], width=0.5)
        plt.title("Proporção de Atendimentos Encaminhados para Especialistas", fontsize=22, pad=20)
        plt.ylabel("Proporção", fontsize=18)
        plt.xlabel("Categorias", fontsize=18)
        plt.ylim(0, 1)
        for i, v in enumerate(proporcao_especialista_data):
            plt.text(i, v + 0.03, f'{v:.2%}', ha='center', fontsize=18, color='black')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "proporcao_encaminhados_especialista.png"))
        plt.close()

        # Gráfico de barras para proporção de atendimentos com solicitação de exames
        plt.figure(figsize=(12, 8))
        proporcao_exames_data = [proporcao_exames, 1 - proporcao_exames]
        labels_exames = ['Com Solicitação de Exames', 'Sem Solicitação']
        plt.bar(labels_exames, proporcao_exames_data, color=['#32CD32', '#FFD700'], width=0.5)
        plt.title("Proporção de Atendimentos com Solicitação de Exames", fontsize=22, pad=20)
        plt.ylabel("Proporção", fontsize=18)
        plt.xlabel("Categorias", fontsize=18)
        plt.ylim(0, 1)
        for i, v in enumerate(proporcao_exames_data):
            plt.text(i, v + 0.03, f'{v:.2%}', ha='center', fontsize=18, color='black')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "proporcao_solicitacao_exames.png"))
        plt.close()

    else:
        print("As colunas necessárias ('Encaminhado para Especialista' e 'Solicitação de Exames') não estão presentes no dataset.")

def proporcao_internacao(df):
    """Calcula e exibe a proporção de atendimentos que desencadeiam internação e gera gráfico de barras.""" 
    if 'Desencadeou Internamento' in df.columns:
        df['Desencadeou Internamento'] = df['Desencadeou Internamento'].replace({'Nao': 0, 'Sim': 1})
        df['Desencadeou Internamento'] = pd.to_numeric(df['Desencadeou Internamento'], errors='coerce').fillna(0).astype(int)
        
        total_atendimentos = len(df)
        atendimentos_internacao = df['Desencadeou Internamento'].sum()
        proporcao = atendimentos_internacao / total_atendimentos
        
        print(f"Total de Atendimentos: {total_atendimentos}")
        print(f"Atendimentos que desencadearam internação: {atendimentos_internacao}")
        print(f"Proporção de Atendimentos que desencadeiam Internação: {proporcao:.2%}")
        
        plt.figure(figsize=(12, 8))
        proporcao_data = [proporcao, 1 - proporcao]
        labels = ['Não', 'Sim']
        plt.bar(labels, proporcao_data, color=['#FF6347', '#4682B4'], width=0.5)
        plt.title("Proporção de Atendimentos que Desencadeiam Internação", fontsize=22, pad=20)
        plt.ylabel("Proporção", fontsize=18)
        plt.xlabel("Categorias", fontsize=18)
        plt.ylim(0, 1)
        for i, v in enumerate(proporcao_data):
            plt.text(i, v + 0.03, f'{v:.2%}', ha='center', fontsize=18, color='black')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "proporcao_internacao.png"))
        plt.close()

    else:
        print("A coluna 'Desencadeou Internamento' não foi encontrada no dataset.")

# Carregar o arquivo CSV
df = carregar_dados('data/2018-08-13_Sistema_E-Saude_Medicos_-_Base_de_Dados.csv')

# Corrigir os dados
corrigir_colunas(df)
converter_datas(df)
criar_colunas_adicionais(df)
remover_colunas_e_linhas(df)

# Realizar a análise exploratória
analise_exploratoria(df)

# Calcular a proporção de atendimentos encaminhados para especialistas e com solicitação de exames
proporcao_encaminhados_especialistas_e_solicitacao_exames(df)

# Calcular a proporção de internações
proporcao_internacao(df)
