"""
VModel Module: Provides a class for working with text embeddings and searching for matches in a Question-Answer database.
@author: Martinez, Nicolas Agustin
@credits: Text embeddings & semantic search from https://HuggingFace.co
Inspired on the Youtube video: https://www.youtube.com/watch?v=OATCgQtNX2o
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
        Vectorize the QA database and add a FAISS index for fast searches.
        """

        self.context_dataset = self.context_dataset.map(lambda x: {"embeddings": self.__get_embeddings(x["context"]).cpu().numpy()[0]})
        self.context_dataset.add_faiss_index(column="embeddings")

    def __mean_pooling(self, model_output, attention_mask):
        """
        Performs mean pooling on the embeddings to obtain a consolidated representation.
        
        Args:
        - model_output (torch.Tensor): Model output.
        - attention_mask (torch.Tensor): Attention mask for the input.
        
        Returns:
        - torch.Tensor: Consolidated embedding.
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def __get_embeddings(self, text_list):
        """
        Obtains embeddings for a list of texts using the model and tokenizer.
        
        Args:
        - text_list (list): List of texts to obtain embeddings for.
        
        Returns:
        - torch.Tensor: Embeddings of the texts.
        """
        
        encoded_input = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        return self.__mean_pooling(model_output, encoded_input["attention_mask"])

    def __find_best_matches(self, user_input, k=3):
        """
        Searches for the top k matches for user input in the question QA JSON (could be any source of data).
        
        Args:
        - user_input (str): User's input text.
        - k (int): Number of matches to return.
        
        Returns:
        - list: List of the top k matches.
        """

        user_embedding = self.__get_embeddings([user_input]).cpu().detach().numpy()
        scores, samples = self.context_dataset.get_nearest_examples("embeddings", user_embedding, k=k)
        return samples

    def ask_question_to_JSON(self, sentence):
        """
        Retrieves a list of similar questions for a given sentence.
        
        Args:
        - sentence (str): Sentence for which to find similar questions.
        
        Returns:
        - list: List of similar questions.
        """
        responses = self.__find_best_matches(sentence)['context'][0]
    
        index = self.JSON_data['questions'].index(responses)
    
        return {
            'associated question': responses,
            'answer': self.JSON_data['answers'][index]
        }



if __name__ == '__main__':

    json_path = 'QA.json'
    model = VModel(json_path=json_path)

    while True:
        question = input('\nUSER INPUT >> ')
        answer = model.ask_question_to_JSON(question)
        print('Q: ',answer['associated question'])
        print('A: ', answer['answer'])

