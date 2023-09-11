"""
Módulo VModel: Proporciona una clase para trabajar con embeddings de texto y buscar coincidencias en una base de datos de Preguntas y Respuestas.

@author: Martinez, Nicolas Agustin

@credits: Text embeddings & semantic search from https://HuggingFace.co
@credits: https://www.youtube.com/watch?v=OATCgQtNX2o

<a href="https://www.freepik.com/free-vector/abstract-vector-mesh-background-chaotically-connected-points-polygons-flying-space-flying-debris-futuristic-technology-style-card-lines-points-circles-planes-futuristic-design_1283683.htm#from_view=detail_alsolike">Image by GarryKillian</a> on Freepik
"""

import json
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset

class VModel:
    def __init__(self, json_path):
        """
        Initialize the VModel instance.
        
        Args:
        - paragraphs (list): List of paragraphs extracted from the PDF.
        """

        self.json_path = json_path

        # Initialize the transformer model and tokenizer
        model_ckpt = 'all-MiniLM-L6-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt)        

        # Preprocess the data
        self.__data_preproccesing()

        # Load the model and prepare the FAISS index
        self.__load_model()

    def __data_preproccesing(self):

        # Cargar datos de tags desde el archivo JSON
        with open(self.json_path, "r", encoding='utf-8') as json_file:
            self.JSON_data = json.load(json_file)
            json_file.flush()

        # Extraer todos los tags y preparar el contexto
        all_tags = [tags for tags in self.JSON_data['questions']]
        self.context = {'context': all_tags}

        self.context_dataset = Dataset.from_dict(self.context)

    def __load_model(self):
        """
        Vectoriza la base de datos de tags y agrega un índice FAISS para búsquedas rápidas.
        """

        self.context_dataset = self.context_dataset.map(lambda x: {"embeddings": self.__get_embeddings(x["context"]).cpu().numpy()[0]})
        self.context_dataset.add_faiss_index(column="embeddings")

    def __mean_pooling(self, model_output, attention_mask):
        """
        Realiza mean pooling en los embeddings para obtener una representación consolidada.
        
        Args:
        - model_output (torch.Tensor): Salida del modelo.
        - attention_mask (torch.Tensor): Máscara de atención para el input.
        
        Returns:
        - torch.Tensor: Embedding consolidado.
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def __get_embeddings(self, text_list):
        """
        Obtiene embeddings para una lista de textos utilizando el modelo y el tokenizador.
        
        Args:
        - text_list (list): Lista de textos para los cuales obtener embeddings.
        
        Returns:
        - torch.Tensor: Embeddings de los textos.
        """
        encoded_input = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        return self.__mean_pooling(model_output, encoded_input["attention_mask"])

    def __find_best_matches(self, user_input, k=10):
        """
        Busca las k mejores coincidencias para el input del usuario en el dataset de preguntas.
        
        Args:
        - user_input (str): Texto de entrada del usuario.
        - k (int): Número de coincidencias a retornar.
        
        Returns:
        - list: Lista de las k mejores coincidencias.
        """

        user_embedding = self.__get_embeddings([user_input]).cpu().detach().numpy()
        scores, samples = self.context_dataset.get_nearest_examples("embeddings", user_embedding, k=k)
        return samples

    def ask_question_to_JSON(self, sentence):
        """
        Obtiene tags similares para una oración dada.
        
        Args:
        - sentence (str): Oración para la cual buscar tags similares.
        
        Returns:
        - list: Lista de tags similares.
        """
        responses = self.__find_best_matches(sentence)['context'][0]

        index = self.JSON_data['questions'].index(responses)

        return {
            'pregunta asociada' : responses,
            'respuesta' : self.JSON_data['answers'][index]
        }


if __name__ == '__main__':

    json_path = 'QA.json'
    model = VModel(json_path=json_path)

    while True:
        question = input('insert your query: ')
        answer = model.ask_question_to_JSON(question)

        print(answer['pregunta asociada'])
        print(answer['respuesta'])

