from extractor import extract_text_from_pdf  
from llm_structurer import LLMStructurer  
from llm_recommender import LLMRecommender  
from vector_recommender import VectorRecommender  
from job_offer import JobOffer  
import json  
import os

def process_cv_and_recommend(  
    pdf_path,  
    output_txt_path,  
    output_json_path,  
    output_recommendations_path,  
    profiles_file,  
    api_key=None,  
    mongodb_uri=None  
):  
    """  
    Process a CV and recommend similar profiles.  
      
    Args:  
        pdf_path: Path to the CV PDF file  
        output_txt_path: Path where to save the extracted raw text  
        output_json_path: Path where to save the structured CV  
        output_recommendations_path: Path where to save the recommendations  
        profiles_file: Path to the JSON file containing IT profiles  
        api_key: Groq API key (optional)  
        mongodb_uri: MongoDB connection URI (optional)  
    """  
    # 1. Extract raw text from CV  
    raw_text = extract_text_from_pdf(pdf_path, output_txt_path)  
      
    # 2. Structure CV via LLM  
    structurer = LLMStructurer(api_key=api_key)  
    cv_json = structurer.structure_cv(raw_text)  
      
    if cv_json:  
        # Save structured CV  
        with open(output_json_path, "w", encoding="utf-8") as f:  
            json.dump(cv_json, f, ensure_ascii=False, indent=2)  
        print(f"Structured CV saved to {output_json_path}")  
          
        # Save to MongoDB if URI provided  
        if mongodb_uri:  
            try:  
                client = MongoClient(mongodb_uri)  
                inserted_id = structurer.save_to_mongodb(cv_json, client)  
                if inserted_id:  
                    print(f"Data saved to MongoDB with ID: {inserted_id}")  
                client.close()  
            except Exception as e:  
                print(f"Error connecting to MongoDB: {e}")  
          
        # 3. Recommend similar profiles  
        recommender = LLMRecommender(api_key=api_key, mongodb_uri=mongodb_uri)  
        recommendations = recommender.recommend_profiles(  
            cv_json=cv_json,  
            profiles_file=profiles_file  
        )  
          
        if recommendations:  
            # Save recommendations  
            recommender.save_recommendations_to_file(  
                recommendations,  
                output_recommendations_path  
            )  
              
            # Display recommendations  
            print("\n--- Recommended Profiles ---\n")  
            for i, profile in enumerate(recommendations.get("profils_recommandés", [])):  
                print(f"{i+1}. {profile.get('nom')} (Score: {profile.get('score_similarité')})")  
                print(f"   Reason: {profile.get('raisons')}\n")  
                  
            return recommendations  
        else:  
            print("No recommendations could be generated.")  
    else:  
        print("CV structuring failed.")  
      
    return None  
  
def match_job_offer_with_cvs(  
    job_offer_data,  
    output_recommendations_path,  
    top_n=5,  
    vectorizer_type="tfidf",  
    model_name=None,  
    mongodb_uri=None  
):  
    """  
    Compare a job offer with CVs stored in MongoDB.  
      
    Args:  
        job_offer_data: Job offer data (dictionary)  
        output_recommendations_path: Path where to save the recommendations  
        top_n: Number of CVs to recommend  
        vectorizer_type: Type of vectorization ('tfidf', 'sentence_transformer')  
        model_name: Name of the Sentence Transformer model (if applicable)  
        mongodb_uri: MongoDB connection URI (optional)  
    """  
    # Create job offer  
    job_offer = JobOffer.from_dict(job_offer_data)  
      
    # Display job offer information  
    print("\n--- Job Offer ---\n")  
    print(f"Title: {job_offer.title}")  
    print(f"Required skills: {', '.join(job_offer.required_skills)}")  
    print(f"Experience: {job_offer.experience}")  
    print(f"Education: {job_offer.education}")  
    print(f"Description: {job_offer.description}")  
      
    # Initialize vector recommender  
    mongodb_uri = mongodb_uri or os.environ.get("MONGODB_URI")  
    recommender = VectorRecommender(  
        mongodb_uri=mongodb_uri,  
        vectorizer_type=vectorizer_type,  
        model_name=model_name  
    )  
      
    # Recommend CVs  
    recommendations = recommender.recommend_cvs(job_offer.to_dict(), top_n=top_n)  
      
    if recommendations:  
        # Save recommendations  
        recommender.save_recommendations_to_file(  
            recommendations,  
            output_recommendations_path  
        )  
          
        # Display recommendations  
        print("\n--- Recommended CVs ---\n")  
        for i, recommendation in enumerate(recommendations):  
            cv = recommendation["cv"]  
            score = recommendation["score"]  
            reason = recommendation["raison"]  
              
            nom = cv.get("informations_personnelles", {}).get("nom", "")  
            prenom = cv.get("informations_personnelles", {}).get("prenom", "")  
              
            print(f"{i+1}. {prenom} {nom} (Score: {score:.2f})")  
            print(f"   Reason: {reason}\n")  
          
        return recommendations  
    else:  
        print("No recommendations could be generated.")  
      
    return None  
  
# Example usage  
if __name__ == "__main__":  
    # File paths  
    pdf_path = "data/cvs/CV.pdf"  
    output_txt_path = "data/outputs/output.txt"  
    output_json_path = "data/outputs/structured_cv.json"  
    output_recommendations_path = "data/outputs/recommendations.json"  
    profiles_file = "data/profiles/it_profiles.json"  
      
    # Groq API key  
    api_key = "your_groq_api_key"  
      
    # Set environment variable (alternative)  
    os.environ["GROQ_API_KEY"] = "your_groq_api_key"  
      
    # MongoDB URI (optional)  
    mongodb_uri = "mongodb://localhost:27017/cv_database"  
    os.environ["MONGODB_URI"] = mongodb_uri  
      
    # Operation mode (1: CV processing, 2: Job offer matching)  
    mode = int(input("Mode (1: CV processing, 2: Job offer matching): "))  
      
    if mode == 1:  
        # Process CV and recommend profiles  
        recommendations = process_cv_and_recommend(  
            pdf_path,  
            output_txt_path,  
            output_json_path,  
            output_recommendations_path,  
            profiles_file,  
            api_key,  
            mongodb_uri  
        )  
    elif mode == 2:  
        # Example job offer  
        job_offer_data = {  
            "titre": "Full Stack Developer",  
            "competences_requises": ["JavaScript", "React", "Node.js", "MongoDB", "Express"],  
            "experience": "2 years minimum",  
            "diplome": "Bachelor's degree in Computer Science",  
            "description": "We are looking for a Full Stack Developer to join our team and participate in the development of our web applications."  
        }  
          
        # Path for job offer-based recommendations  
        job_recommendations_path = "data/outputs/job_recommendations.json"  
          
        # Match job offer with CVs  
        recommendations = match_job_offer_with_cvs(  
            job_offer_data,  
            job_recommendations_path,  
            top_n=5,  
            vectorizer_type="tfidf",  
            mongodb_uri=mongodb_uri  
        )  
    else:  
        print("Unrecognized mode.")