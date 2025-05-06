import numpy as np  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity  
from sentence_transformers import SentenceTransformer  
import pymongo  
import json  
import os  
  
class VectorRecommender:  
    """  
    Recommends CVs based on their similarity to a job offer  
    using vectorization and similarity calculation techniques.  
    """  
      
    def __init__(self, mongodb_uri=None, vectorizer_type="tfidf", model_name=None):  
        """  
        Initialize the vector recommender.  
          
        Args:  
            mongodb_uri: MongoDB connection URI  
            vectorizer_type: Type of vectorization ('tfidf', 'sentence_transformer')  
            model_name: Name of the Sentence Transformer model (if applicable)  
        """  
        self.mongodb_uri = mongodb_uri or os.environ.get("MONGODB_URI")  
        self.vectorizer_type = vectorizer_type  
          
        # Initialize vectorizer  
        if vectorizer_type == "tfidf":  
            self.vectorizer = TfidfVectorizer(stop_words='english')  
        elif vectorizer_type == "sentence_transformer":  
            model_name = model_name or "all-MiniLM-L6-v2"  
            self.vectorizer = SentenceTransformer(model_name)  
        else:  
            raise ValueError(f"Unsupported vectorization type: {vectorizer_type}")  
      
    def preprocess_job_offer(self, job_offer):  
        """  
        Preprocess a job offer to extract relevant information.  
          
        Args:  
            job_offer: Dictionary containing job offer information  
              
        Returns:  
            text: Preprocessed job offer text  
        """  
        # Extract relevant fields  
        title = job_offer.get("titre", "")  
        skills = ", ".join(job_offer.get("competences_requises", []))  
        experience = job_offer.get("experience", "")  
        education = job_offer.get("diplome", "")  
        description = job_offer.get("description", "")  
          
        # Concatenate fields into a single text  
        text = f"{title}. {description}. Required skills: {skills}. Experience: {experience}. Education: {education}"  
        return text  
      
    def preprocess_cv(self, cv):  
        """  
        Preprocess a CV to extract relevant information.  
          
        Args:  
            cv: Dictionary containing CV information  
              
        Returns:  
            text: Preprocessed CV text  
        """  
        # Extract skills  
        skills = []  
        competences = cv.get("competences", {})  
        for category, skill_list in competences.items():  
            if isinstance(skill_list, list):  
                skills.extend(skill_list)  
            elif isinstance(skill_list, str):  
                skills.append(skill_list)  
          
        # Extract education  
        education = []  
        for edu in cv.get("education", []):  
            diplome = edu.get("diplome", "")  
            etablissement = edu.get("etablissement", "")  
            filiere = edu.get("filiere", "")  
            education.append(f"{diplome} {etablissement} {filiere}")  
          
        # Extract projects  
        projects = []  
        for project in cv.get("projets", []):  
            nom = project.get("nom", "")  
            techs = ", ".join(project.get("techs", []))  
            detail = project.get("detail", "")  
            projects.append(f"{nom}: {techs}. {detail}")  
          
        # Concatenate information into a single text  
        text = f"Skills: {', '.join(skills)}. "  
        text += f"Education: {'. '.join(education)}. "  
        text += f"Projects: {'. '.join(projects)}"  
          
        return text  
      
    def vectorize_text(self, text):  
        """  
        Vectorize a text using the chosen method.  
          
        Args:  
            text: Text to vectorize  
              
        Returns:  
            vector: Numerical vector representing the text  
        """  
        if self.vectorizer_type == "tfidf":  
            # For TF-IDF, we need to fit the vectorizer on the text first  
            vector_matrix = self.vectorizer.fit_transform([text])  
            return vector_matrix.toarray()[0]  
        elif self.vectorizer_type == "sentence_transformer":  
            # For Sentence Transformer, we can directly encode the text  
            return self.vectorizer.encode(text)  
      
    def vectorize_texts(self, texts):  
        """  
        Vectorize a list of texts using the chosen method.  
          
        Args:  
            texts: List of texts to vectorize  
              
        Returns:  
            vectors: Matrix of numerical vectors representing the texts  
        """  
        if self.vectorizer_type == "tfidf":  
            # For TF-IDF, we need to fit the vectorizer on all texts  
            return self.vectorizer.fit_transform(texts).toarray()  
        elif self.vectorizer_type == "sentence_transformer":  
            # For Sentence Transformer, we can directly encode the texts  
            return self.vectorizer.encode(texts)  
      
    def calculate_similarity(self, vector1, vector2):  
        """  
        Calculate cosine similarity between two vectors.  
          
        Args:  
            vector1: First vector  
            vector2: Second vector  
              
        Returns:  
            similarity: Similarity score between 0 and 1  
        """  
        # Reshape vectors for similarity calculation  
        v1 = vector1.reshape(1, -1)  
        v2 = vector2.reshape(1, -1)  
          
        # Calculate cosine similarity  
        return cosine_similarity(v1, v2)[0][0]  
      
    def recommend_cvs(self, job_offer, top_n=5):  
        """  
        Recommend the most relevant CVs for a job offer.  
          
        Args:  
            job_offer: Dictionary containing job offer information  
            top_n: Number of CVs to recommend  
              
        Returns:  
            recommendations: List of recommended CVs with their similarity score  
        """  
        if not self.mongodb_uri:  
            print("WARNING: No MongoDB URI provided. Unable to load CVs.")  
            return []  
          
        try:  
            # Connect to MongoDB  
            client = pymongo.MongoClient(self.mongodb_uri)  
            db = client.get_default_database()  
            collection = db.cvs  
              
            # Retrieve all CVs  
            cvs = list(collection.find())  
              
            # Preprocess job offer  
            job_offer_text = self.preprocess_job_offer(job_offer)  
              
            # Preprocess CVs  
            cv_texts = [self.preprocess_cv(cv) for cv in cvs]  
              
            # Vectorize job offer and CVs  
            if self.vectorizer_type == "tfidf":  
                # For TF-IDF, we need to vectorize all texts together  
                all_texts = [job_offer_text] + cv_texts  
                all_vectors = self.vectorize_texts(all_texts)  
                job_offer_vector = all_vectors[0]  
                cv_vectors = all_vectors[1:]  
            else:  
                # For other methods, we can vectorize separately  
                job_offer_vector = self.vectorize_text(job_offer_text)  
                cv_vectors = [self.vectorize_text(cv_text) for cv_text in cv_texts]  
              
            # Calculate similarity scores  
            similarities = []  
            for i, cv_vector in enumerate(cv_vectors):  
                similarity = self.calculate_similarity(job_offer_vector, cv_vector)  
                similarities.append((cvs[i], similarity))  
              
            # Sort CVs by similarity score in descending order  
            similarities.sort(key=lambda x: x[1], reverse=True)  
              
            # Select top_n CVs  
            top_cvs = similarities[:top_n]  
              
            # Format recommendations  
            recommendations = []  
            for cv, score in top_cvs:  
                recommendation = {  
                    "cv": cv,  
                    "score": float(score),  # Convert to float for JSON serialization  
                    "raison": f"Similarity score: {score:.2f}"  
                }  
                recommendations.append(recommendation)  
              
            return recommendations  
              
        except Exception as e:  
            print(f"Error recommending CVs: {e}")  
            return []  
      
    def save_recommendations_to_file(self, recommendations, output_file):  
        """  
        Save recommendations to a JSON file.  
          
        Args:  
            recommendations: List of recommended CVs  
            output_file: Path where to save the recommendations  
              
        Returns:  
            success: True if save was successful, False otherwise  
        """  
        try:  
            with open(output_file, 'w', encoding='utf-8') as f:  
                json.dump(recommendations, f, ensure_ascii=False, indent=2)  
            print(f"Recommendations saved to {output_file}")  
            return True  
        except Exception as e:  
            print(f"Error saving recommendations: {e}")  
            return False