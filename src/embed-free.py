# **************************************************************************************************************************************
# *  embed-free.py                                                                                                                     *
# **************************************************************************************************************************************
# *                                                                                                                                    *
# *  Project: Create Text and Image Embeddings for Database Engines with Python GenAI Free Libraries.                                  *
# *                                                                                                                                    *
# *  Copyright © 2024. MongoExpUser.  All Rights Reserved.                                                                             *
# *                                                                                                                                    *
# *  License: MIT - https://github.com/MongoExpUser/Popular-Nodejs-Libraries-for-Data-Driven-AI-Applications/blob/main/LICENSE         *
# *                                                                                                                                    *
# *                                                                                                                                    *
# *   This script implements a class for generating text and image embeddings                                                          *
# *                                                                                                                                    *
# *   The following 3rd party libraries are used: (1) sentence-transformers (2) imgbeddings and (3) Pillows (PIL)                      *
# *                                                                                                                                    *
# **************************************************************************************************************************************



from PIL import Image
from os import system
from pprint import pprint
from os.path import basename
from urllib.parse import urlparse
from imgbeddings import imgbeddings
from sentence_transformers import SentenceTransformer


class CreateEmbedding():

    def __init__(self):

        """
            Create text and image embeddings, which are a numerical representation of text or images, encoded into vector representation.
            The created text and image vectors can be inserted into the vector columms of database engines that support vector column.
            The vector dimension (D) can also be processed & re-sized from the output dimension of 768 to lower values (say, 32 64, 128, etc.) before inserting into table.
        """
        print("Creating Embedding")

    def show_result(self, embedding=None, show=None):
        if show:
            pprint( { "embedding" : embedding } )

    def create_text_embedding(self, text_to_embed=None, model_name=None, show=None):
        if model_name == "clip-ViT-B-32":
            model = SentenceTransformer(model_name)     
            embedding = model.encode([text_to_embed]) 
            self.show_result(embedding=embedding, show=show)
            return  embedding 
            
    def create_image_embedding(self, local_image_path=None, remote_image_url=None, model_name=None, show=None):
        image = None
        embedding = None
        
        if local_image_path:
            image = local_image_path
        elif remote_image_url:
            print("Downloading Image")
            system(f"sudo wget {remote_image_url}")
            url = urlparse(remote_image_url)
            image_path = url.path
            image = basename(image_path)
       
        pprint({ "Image-File-Name" : image } )
        print("Creating Embedding")

        if model_name == "clip-ViT-B-32":
            model = SentenceTransformer(model_name) 
            embedding = model.encode(Image.open(image)) 
        elif model_name == "imgbeddings":
            model = imgbeddings()
            embedding = model.to_embeddings(image)

        self.show_result(embedding=embedding, show=show)
        return  embedding 

def main():
    cemd = CreateEmbedding()
    test_text_embedding = False  # or True
    test_image_embedding = True  # or False
    show = True                  # or False
    image_option = 1             # or 2

    if test_text_embedding:
        model_name = "clip-ViT-B-32"
        text_to_embed = "How can I create vector embeddings using the pgvector extension for PostgreSQL?"
        cemd.create_text_embedding(text_to_embed=text_to_embed, model_name=model_name, show=show)

    if test_image_embedding:
        model_name = None
        if image_option == 1:
            model_name = "clip-ViT-B-32" 
        elif image_option == 2:
            model_name = "imgbeddings" 
        local_image_path = None # or "person_right.png" 
        remote_image_url = "https://codeskulptor-demos.commondatastorage.googleapis.com/descent/person_right.png" # or None
        cemd.create_image_embedding(local_image_path=local_image_path, remote_image_url=remote_image_url, model_name=model_name, show=show)


if __name__ in ["__main__"]:
    main()

