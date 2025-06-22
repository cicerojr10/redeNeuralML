import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# --- Configuração da Página ---
st.set_page_config(
    page_title="Demo de Classificação de Imagens",
    page_icon="🧠",
    layout="wide"
)

# --- Funções de Lógica e Carregamento ---

# Usa cache para carregar o modelo de ML apenas uma vez, otimizando a performance
@st.cache_resource
def carregar_modelo_ml():
    """Carrega o modelo Keras treinado a partir de um arquivo .h5"""
    try:
        model = keras.models.load_model('meu_modelo.h5')
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

def preprocessar_imagem(imagem_carregada):
    """Prepara a imagem enviada pelo usuário para o formato que o modelo espera."""
    # Abre a imagem usando a biblioteca PIL
    img = Image.open(imagem_carregada)
    # Redimensiona para o tamanho que o modelo foi treinado (150x150)
    img = img.resize((150, 150))
    # Converte a imagem para um array numpy
    img_array = np.array(img)
    # Expande as dimensões para criar um "batch" de 1 imagem: (1, 150, 150, 3)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return img_array_expanded

# Nomes das classes na mesma ordem que o TensorFlow as encontrou
CLASS_NAMES = ['Gato', 'Cachorro'] # Baseado em ['datasetCat', 'datasetDog'], mas mais amigável

# --- Construção da Interface do App ---

st.title("🧠 Classificador de Imagens: Gato ou Cachorro?")
st.write("""
Bem-vindo à demonstração do meu projeto de Rede Neural Convolucional (CNN). 
Esta aplicação utiliza um modelo treinado para classificar se uma imagem contém um gato ou um cachorro.
""")

# Dividir a tela em abas
tab1, tab2 = st.tabs(["🤖 Fazer uma Predição", "📈 Performance do Modelo"])

with tab1:
    st.header("Envie sua imagem")
    uploaded_file = st.file_uploader("Escolha uma imagem de um gato ou cachorro...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Mostra a imagem enviada
        st.image(uploaded_file, caption='Imagem Carregada.', use_column_width=True, width=250)
        
        # Botão para iniciar a classificação
        if st.button('Classificar Imagem'):
            with st.spinner('Analisando a imagem...'):
                # Carrega o modelo
                model = carregar_modelo_ml()
                
                if model:
                    # Prepara a imagem
                    imagem_processada = preprocessar_imagem(uploaded_file)
                    
                    # Faz a predição
                    prediction = model.predict(imagem_processada)
                    
                    # Interpreta o resultado
                    score = tf.nn.softmax(prediction[0])
                    predicted_class = CLASS_NAMES[np.argmax(score)]
                    confidence = 100 * np.max(score)

                    # Exibe o resultado
                    st.success(f"**Resultado:** A imagem é um **{predicted_class}** com **{confidence:.2f}%** de confiança.")

# SUBSTITUA O BLOCO 'with tab2:' INTEIRO POR ESTE:

with tab2:
    st.header("Análise de Performance do Modelo")
    st.write("Resultados visuais obtidos durante a fase de treinamento e validação do modelo.")
    
    st.markdown("---") # Adiciona uma linha divisória para organizar

    try:
        # Imagem 1: Acurácia e Perda
        st.subheader("Acurácia e Perda (Loss) vs. Épocas")
        st.image(
            'images/imagem1.png', 
            caption='Este gráfico mostra como a acurácia do modelo aumentou e o erro (perda) diminuiu a cada época de treinamento.', 
            use_column_width=True
        )

        # Imagem 2: Matriz de Confusão
        st.subheader("Matriz de Confusão")
        st.image(
            'images/imagem2.png', 
            caption='A matriz de confusão demonstra o desempenho do modelo para classificar corretamente cada classe nos dados de teste.', 
            use_column_width=True
        )

        # Imagem 3: Taxa de Aprendizado
        st.subheader("Variação da Taxa de Aprendizado (Learning Rate)")
        st.image(
            'images/imagem3.png', 
            caption='Este gráfico mostra como a taxa de aprendizado foi ajustada dinamicamente durante o treinamento para otimizar a convergência.', 
            use_column_width=True
        )

    except FileNotFoundError as e:
        st.error(f"Erro ao carregar imagem: {e}. Verifique se os arquivos 'imagem1.png', 'imagem2.png' e 'imagem3.png' estão dentro da pasta 'images/'.")

st.sidebar.header("Sobre o Projeto")
st.sidebar.info("""
Modelo de Deep Learning (CNN) treinado para diferenciar imagens de gatos e cachorros.
**Tecnologias:** Python, TensorFlow, Keras, Streamlit.
""")
st.sidebar.markdown("[Ver o código no GitHub](https://github.com/cicerojr10/redeNeuralML)")