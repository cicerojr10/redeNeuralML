# Relatório do Projeto: Classificador de Imagens com Redes Neurais Artificiais 🤖🖼️

**Aluno:** [Cícero Junior/cicerojr10]
**Data:** [Data da Conclusão: 31/05/2025 | Postagem: 01/06/2025]

## 1. Resumo do Projeto 🎯

Este projeto teve como objetivo desenvolver um classificador de imagens utilizando Redes Neurais Convolucionais (CNNs). O processo envolveu a preparação de um dataset customizado, a construção de um modelo de Deep Learning, o treinamento iterativo deste modelo e a aplicação de técnicas para otimizar seu desempenho e combater o overfitting. O ambiente de desenvolvimento principal foi o Google Colab, com dados hospedados no Google Drive.

## 2. Fases do Projeto 🛠️

O desenvolvimento do projeto pode ser dividido nas seguintes fases principais:

### 2.1. Preparação do Dataset e Ambiente Inicial
* **Definição do Dataset:** O projeto iniciou com a necessidade de um dataset com pelo menos duas classes de objetos, cada uma contendo no mínimo 100 imagens com resolução mínima de 400x400 pixels.
* **Configuração do Ambiente:** Foi escolhido o Google Colab como plataforma de desenvolvimento, aproveitando seus recursos de GPU e ambiente Python pré-configurado.
* **Gerenciamento de Dados:** As imagens foram armazenadas no Google Drive. Desenvolvemos um script em Python para automatizar o download e a descompactação do dataset (arquivos `.zip`) diretamente no ambiente do Colab. Esta etapa envolveu a resolução de desafios como o manuseio de arquivos grandes e a interação com o Google Drive via código (utilizando a biblioteca `gdown`).

### 2.2. Carregamento e Pré-processamento dos Dados
* **Carregamento:** Utilizamos a função `tf.keras.utils.image_dataset_from_directory` do TensorFlow/Keras para carregar as imagens de forma eficiente, inferindo os rótulos a partir da estrutura de diretórios e dividindo os dados em conjuntos de treino e validação.
* **Redimensionamento:** As imagens foram redimensionadas para um tamanho padrão (ex: 150x150 pixels) para entrada na rede neural.
* **Normalização:** Os valores dos pixels foram reescalados do intervalo [0, 255] para [0, 1] usando uma camada `layers.Rescaling` para facilitar o treinamento.
* **Data Augmentation:** Para aumentar a robustez do modelo e a variabilidade dos dados de treino, foi implementada uma camada de `data_augmentation` com transformações como `RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomBrightness` e `RandomContrast`. A intensidade dessas transformações foi ajustada iterativamente.

### 2.3. Construção do Modelo de Rede Neural Convolucional (CNN)
* Foi construída uma CNN do zero utilizando a API Sequencial do Keras.
* A arquitetura base consistiu em:
    * Camada de Data Augmentation.
    * Camada de Rescaling.
    * Múltiplos blocos convolucionais (geralmente `Conv2D` + `MaxPooling2D`). Experimentamos com diferentes números de filtros (32, 64, 128).
    * Camadas de `Dropout` adicionadas estrategicamente para regularização.
    * Camada `Flatten`.
    * Camadas `Dense` (totalmente conectadas). Experimento com diferentes números de neurônios (ex: 128, 256).
    * Camada `Dense` de saída com ativação `softmax`.

### 2.4. Compilação, Treinamento e Otimização Iterativa
* **Compilação:** O modelo foi compilado usando o otimizador `adam` (com ajuste na taxa de aprendizado), a função de perda `categorical_crossentropy` e a métrica `accuracy`.
* **Treinamento:** O modelo foi treinado com o método `model.fit()`.
* **Callbacks:**
    * `EarlyStopping`: Implementado para monitorar `val_loss` e, posteriormente, `val_accuracy`, interrompendo o treinamento otimamente e restaurando os melhores pesos.
    * `ReduceLROnPlateau`: Adicionado para reduzir dinamicamente a taxa de aprendizado quando a `val_loss` estagnava.
* **Análise e Iteração:** Após cada rodada, gráficos de acurácia e perda foram analisados para guiar os ajustes no modelo e nos hiperparâmetros.

## 3. Tecnologias e Bibliotecas Utilizadas 💻

* **Linguagem de Programação:** Python
* **Ambiente Principal:** Google Colaboratory (Colab)
* **Framework de Deep Learning:** TensorFlow com a API Keras
* **Manipulação de Dados e Arquivos:**
    * `gdown`
    * `zipfile`
    * `os`, `shutil`, `pathlib`
* **Visualização:** Matplotlib
* **Computação Numérica:** NumPy (usado implicitamente pelo TensorFlow)

## 4. Áreas de Conhecimento Aplicadas 🧠

Este projeto abrangeu diversas áreas fundamentais, incluindo:

* **Inteligência Artificial (IA):** O campo mais amplo.
* **Machine Learning (ML):** Especificamente, aprendizado supervisionado para classificação de imagens.
* **Deep Learning:** Utilização de Redes Neurais Artificiais Profundas (CNNs).
* **Ciência de Dados (Data Science):**
    * Preparação e gerenciamento de datasets.
    * Pré-processamento e aumento de dados (Data Augmentation).
    * Modelagem preditiva e avaliação de modelos.
    * Processo iterativo e experimental.
* **Visão Computacional (Computer Vision):** Classificação de imagens como tarefa central.
* **Redes Neurais Artificiais (ANNs):** Foco em Redes Neurais Convolucionais (CNNs).
* **Programação em Python:** Implementação de todas as etapas.
* **Boas Práticas de Desenvolvimento (Iniciais):** Uso de ambiente interativo, organização, tratamento de erros e intenção de versionamento.
* **Cloud Computing:** Uso do Google Colab e Google Drive.

## 5. Resultados e Conclusões (do Ponto Atual) 📈

Ao longo de várias iterações de treinamento e ajuste de hiperparâmetros (arquitetura da rede, taxas de dropout, estratégias de data augmentation, taxa de aprendizado, callbacks), o modelo alcançou uma **acurácia de validação máxima de 85%**.

O processo demonstrou a importância:
* Da qualidade e preparação dos dados.
* De técnicas de regularização (Dropout, Data Augmentation) para combater o overfitting.
* Do uso de callbacks como `EarlyStopping` e `ReduceLROnPlateau` para otimizar o processo de treinamento.
* Da análise iterativa dos resultados para guiar as próximas etapas de otimização.

Embora o objetivo inicial mais ambicioso de 90% de acurácia não tenha sido atingido com a arquitetura e estratégias atuais (sem Transfer Learning), o projeto serviu como uma excelente plataforma para aplicar e entender os conceitos fundamentais de construção e treinamento de Redes Neurais Convolucionais. Próximos passos para buscar maior acurácia poderiam incluir o uso de Transfer Learning, aumento do dataset ou um ajuste ainda mais fino e sistemático dos hiperparâmetros.

---
