import os  
import json  
import logging  
from flask import Flask, request, jsonify, render_template  
from pymongo import MongoClient  
from bson import ObjectId  # Import ObjectId for proper handling  
from services.vector_recommender import VectorRecommender  
from services.job_offer import JobOffer  
from services.extractor import extract_text_from_pdf  
from services.llm_structurer import LLMStructurer  
  
# Create Flask application  
app = Flask(__name__)  
logging.basicConfig(level=logging.DEBUG)  
logger = logging.getLogger(__name__)  
  
# Function to convert ObjectId to string in nested dictionaries and lists  
def convert_objectid_to_str(obj):  
    if isinstance(obj, ObjectId):  
        return str(obj)  
    elif isinstance(obj, dict):  
        return {k: convert_objectid_to_str(v) for k, v in obj.items()}  
    elif isinstance(obj, list):  
        return [convert_objectid_to_str(item) for item in obj]  
    return obj  
  
# MongoDB connection  
def get_mongodb_client():  
    mongodb_uri = os.environ.get("MONGODB_URI", "mongodb://mongo:27017/cv_database")  
    logger.debug(f"Connecting to MongoDB with URI: {mongodb_uri}")  
    client = MongoClient(mongodb_uri)  
    # Test the connection  
    try:  
        client.admin.command('ping')  
        logger.debug("MongoDB connection successful")  
        return client  
    except Exception as e:  
        logger.error(f"MongoDB connection failed: {str(e)}")  
        raise  
  
@app.route('/')  
def index():  
    return render_template('index.html')  
  
@app.route('/api/recommander-cv', methods=['POST'])  
def recommander_cv():  
    # Get form data  
    data = request.form  
      
    # Create job offer object  
    offre = {  
        "titre": data["titre_poste"],  
        "competences_requises": [comp.strip() for comp in data["competences"].split(",") if comp.strip()],  
        "experience": data["experience"],  
        "diplome": data["diplome"],  
        "description": data.get("description", "")  
    }  
      
    # Initialize vector recommender  
    mongodb_uri = os.environ.get("MONGODB_URI", "mongodb://mongo:27017/cv_database")  
    recommender = VectorRecommender(  
        mongodb_uri=mongodb_uri,  
        vectorizer_type="tfidf"  
    )  
      
    # Recommend CVs  
    job_offer = JobOffer.from_dict(offre)  
    recommendations = recommender.recommend_cvs(job_offer.to_dict(), top_n=5)  
      
    # Format results for display  
    results = []  
    if recommendations:  
        for recommendation in recommendations:  
            cv = recommendation["cv"]  
            score = recommendation["score"]  
              
            # Extract relevant CV information  
            nom = cv.get("informations_personnelles", {}).get("nom", "")  
            prenom = cv.get("informations_personnelles", {}).get("prenom", "")  
            email = cv.get("informations_personnelles", {}).get("email", "")  
              
            # Create dictionary with information to display  
            result = {  
                "nom_complet": f"{prenom} {nom}",  
                "email": email,  
                "score": round(score * 100, 2),  
                "competences": cv.get("competences", {})  
            }  
              
            results.append(result)  
      
    # Convert any ObjectId to string before JSON serialization  
    results = convert_objectid_to_str(results)  
    return jsonify(results)  
  
@app.route('/upload-cv', methods=['POST'])  
def upload_cv():  
    logger.debug("Début de la fonction upload_cv")  
      
    if 'cv_file' not in request.files:  
        logger.error("Erreur: Pas de fichier fourni")  
        return jsonify({"error": "No file provided"}), 400  
      
    file = request.files['cv_file']  
    logger.debug(f"Fichier reçu: {file.filename}")

    if file.filename == '':  
        logger.error("Erreur: Nom de fichier vide")  
        return jsonify({"error": "No file selected"}), 400  
      
    # Save uploaded file  
    upload_folder = "data/cvs"  
    os.makedirs(upload_folder, exist_ok=True)  
    file_path = os.path.join(upload_folder, file.filename)  
    file.save(file_path)  
      
    # Process CV  
    output_folder = "data/outputs"  
    os.makedirs(output_folder, exist_ok=True)  
    output_txt_path = os.path.join(output_folder, "output.txt")  
    output_json_path = os.path.join(output_folder, "structured_cv.json")  
      
    # Extract and structure CV  
    try:  
        raw_text = extract_text_from_pdf(file_path, output_txt_path)  
        logger.debug("Extraction du texte terminée")  
          
        # Structure CV via LLM  
        logger.debug("Début de la structuration via LLM")  
        api_key = os.environ.get("GROQ_API_KEY")  
        if not api_key:  
            logger.error("Clé API GROQ non définie")  
            return jsonify({"success": False, "error": "GROQ API key not set"}), 500  
              
        structurer = LLMStructurer(api_key=api_key)  
        cv_json = structurer.structure_cv(raw_text)  
          
        if cv_json:  
            logger.debug("Structuration réussie, sauvegarde du JSON")  
            # Log the CV JSON content for debugging  
            logger.debug(f"CV JSON content preview: {json.dumps(cv_json, ensure_ascii=False)[:200]}...")  
              
            # Save structured CV  
            with open(output_json_path, "w", encoding="utf-8") as f:  
                json.dump(cv_json, f, ensure_ascii=False, indent=2)  
              
            # Save to MongoDB  
            try:  
                logger.debug("Tentative de connexion à MongoDB")  
                client = get_mongodb_client()  
                logger.debug("Connexion à MongoDB réussie")  
                  
                # Ensure we're using the right database - use explicit database name  
                db = client["cv_database"]  # Use explicit database name instead of get_database()  
                logger.debug(f"Using database: {db.name}")  
                  
                # Make sure the collection exists  
                collection_name = "cvs"  
                if collection_name not in db.list_collection_names():  
                    logger.debug(f"Creating collection: {collection_name}")  
                    db.create_collection(collection_name)  
                  
                # Save to MongoDB and get the ID as a string  
                inserted_id = structurer.save_to_mongodb(cv_json, client)  
                logger.debug(f"Données sauvegardées avec ID: {inserted_id}")  
                client.close()  
                  
                # Make sure inserted_id is a string  
                if inserted_id is not None:  
                    inserted_id_str = str(inserted_id)  
                else:  
                    inserted_id_str = None  
                  
                # Create response data  
                response_data = {  
                    "success": True,  
                    "message": "CV processed and saved successfully",  
                    "cv_data": cv_json,  
                    "id": inserted_id_str  
                }  
                  
                # Convert any ObjectId objects to strings in the entire response  
                response_data = convert_objectid_to_str(response_data)  
                  
                # Return JSON response with string ID  
                return jsonify(response_data)  
            except Exception as e:  
                logger.error(f"Erreur lors de la sauvegarde dans MongoDB: {str(e)}")  
                logger.debug(traceback.format_exc())  # Add stack trace for better debugging  
                return jsonify({  
                    "success": False,  
                    "error": f"Error saving to MongoDB: {str(e)}"  
                }), 500  
        else:  
            logger.error("Échec de la structuration du CV")  
            return jsonify({  
                "success": False,  
                "error": "CV structuring failed"  
            }), 500  
    except Exception as e:  
        logger.error(f"Erreur lors du traitement du CV: {str(e)}")  
        logger.debug(traceback.format_exc())  # Add stack trace for better debugging  
        return jsonify({  
            "success": False,  
            "error": f"Error processing CV: {str(e)}"  
        }), 500  
  
if __name__ == '__main__':  
    app.run(debug=True, host='0.0.0.0', port=5000)
   