from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from PyPDF2 import PdfReader

word_embeddings = KeyedVectors.load_word2vec_format('/Users/rohanshah/Downloads/GoogleNews-vectors-negative300.bin', binary=True)

class Word2Vec:

    def __init__(self):
        load_dotenv()

    def similarity(self,document1,document2):
        # Tokenize the documents
        tokens1 = document1.lower().split()
        tokens2 = document2.lower().split()

        # Calculate document embeddings by averaging word embeddings
        document_embedding1 = np.mean([word_embeddings[word] for word in tokens1 if word in word_embeddings], axis=0)
        document_embedding2 = np.mean([word_embeddings[word] for word in tokens2 if word in word_embeddings], axis=0)

        # Calculate cosine similarity between the document embeddings
        similarity_score = cosine_similarity([document_embedding1], [document_embedding2])[0][0]
        return similarity_score
    
    def get_similarity_scores(self,pdf_docs,summary):
        similarity_scores = []
        for i,pdf in enumerate(pdf_docs):
            text = ''
            pdf_reader = PdfReader(pdf)
            for pages in pdf_reader.pages:
                text += pages.extract_text()
            similarity_scores.append(self.similarity(text,summary[i]))

        return similarity_scores