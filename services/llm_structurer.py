import json  
import os  
import re  
import logging  
import traceback  
from groq import Groq  
from bson import ObjectId  # Import ObjectId for proper handling  
  
# Initialize logger  
logger = logging.getLogger(__name__)  
  
class LLMStructurer:  
    """  
    Transforms raw text extracted from a CV into structured JSON format  
    using LLaMA3 via the Groq API.  
    """  
  
    def __init__(self, api_key=None):  
        """  
        Initialize the Groq client with the API key.  
        """  
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")  
  
        if not self.api_key:  
            logger.warning("No Groq API key provided. LLM structuring will not be available.")  
            self.client = None  
        else:  
            self.client = Groq(api_key=self.api_key)  
  
        self.model = "llama3-70b-8192"  # LLaMA3 model via Groq  
  
    def create_prompt(self, raw_text):  
        """  
        Create a suitable prompt for structuring the CV.  
        """  
        return f"""  
        Analyze the following CV and extract structured information in JSON format.  
  
        Organize the data according to the following categories:  
        - Personal information (name, first name, phone, email)  
        - Education (degrees, institutions, dates, fields)  
        - Skills (technical, languages, frameworks, tools)  
        - Professional experiences (positions, companies, dates, descriptions)  
        - Projects (names, technologies, descriptions)  
        - Languages (spoken languages and levels)  
        - Extracurricular activities  
  
        IMPORTANT:    
        1. Return ONLY a valid JSON object without any additional text.  
        2. Make sure the JSON is properly formatted with balanced braces and brackets.  
        3. Use French keys without accents to facilitate processing.  
        4. NEVER use duplicate keys in the same object - use numbered keys like description1, description2 instead.  
        5. For arrays of objects, ensure each object has unique keys and proper comma separators.  
  
        CV to analyze:  
        {raw_text}  
        """  
  
    def structure_cv(self, raw_text):  
        """  
        Send the raw CV text to the LLM and retrieve the structured JSON response.  
        """  
        if not self.client:  
            logger.error("Unable to structure CV: no Groq API key available.")  
            return None  
              
        prompt = self.create_prompt(raw_text)  
          
        try:  
            # Call to the Groq API with LLaMA3  
            logger.debug("Sending request to Groq API")  
            response = self.client.chat.completions.create(  
                model=self.model,  
                messages=[  
                    {"role": "system", "content": "You are an assistant specialized in CV analysis. Extract relevant information and return it ONLY as valid JSON without any additional text. Make sure the JSON is properly formatted with no duplicate keys. Use numbered keys (description1, description2) for multiple descriptions."},  
                    {"role": "user", "content": prompt}  
                ],  
                temperature=0.05,  # Very low value for more deterministic responses  
                max_tokens=4000  
            )  
              
            # Extract JSON from the response  
            json_response = response.choices[0].message.content  
              
            # Display raw response for debugging  
            logger.debug("Raw API response received")  
            print("Raw API response:")  
            print(json_response)  
            logger.debug(f"Response length: {len(json_response)}")  
              
            # Clean the response to ensure it contains only valid JSON  
            logger.debug("Cleaning JSON response")  
            json_response = json_response.strip()  
            if json_response.startswith("```json"):  
                json_response = json_response.replace("```json", "", 1)  
            if json_response.endswith("```"):  
                json_response = json_response.rsplit("```", 1)[0]  
            json_response = json_response.strip()  
              
            # Display cleaned response for debugging  
            print("Cleaned response:")  
            print(json_response)  
            logger.debug(f"Cleaned response length: {len(json_response)}")  
              
            # Validate JSON  
            try:  
                logger.debug("Parsing JSON")  
                structured_data = json.loads(json_response)  
                logger.debug("JSON parsed successfully")  
                return structured_data  
            except json.JSONDecodeError as e:  
                logger.error(f"Error parsing JSON: {e}")  
                logger.debug(f"First 100 chars of JSON: {json_response[:100]}...")  
                logger.debug(f"Last 100 chars of JSON: {json_response[-100:]}...")  
                logger.debug("Attempting advanced JSON correction...")  
                  
                # Attempt to correct common syntax errors  
                corrected_json = json_response  
                  
                # Fix duplicate keys by renaming the second one  
                corrected_json = re.sub(r'"description":\s*"([^"]+)",?\s*"description":', '"description": "\\1", "description2":', corrected_json)  
                  
                # Fix missing commas between duplicate keys  
                corrected_json = re.sub(r'"([^"]+)":\s*"([^"]+)"\s*"([^"]+)":', '"\\1": "\\2", "\\3":', corrected_json)  
                  
                # Fix misplaced closing brackets  
                corrected_json = re.sub(r'"\]', '"', corrected_json)  
                  
                # Fix misplaced closing braces  
                corrected_json = re.sub(r'\}]', '}', corrected_json)  
                  
                # Fix missing commas between objects  
                corrected_json = re.sub(r'\}\s*\{', '},{', corrected_json)  
                  
                # Fix missing commas after closing braces  
                corrected_json = re.sub(r'}\s*"', '},"', corrected_json)  
                  
                # Fix extra closing braces at the end  
                if corrected_json.count('{') < corrected_json.count('}'):  
                    corrected_json = corrected_json.rsplit('}', 1)[0] + '}'  
                  
                # Fix specific issue with activites_extra_scolaires section  
                if '"activites_extra_scolaires"' in corrected_json:  
                    # Find the activites_extra_scolaires section  
                    pattern = r'"activites_extra_scolaires"\s*:\s*\[\s*\{\s*"description"\s*:\s*"([^"]*)"\s*,?\s*"description"\s*:\s*"([^"]*)"\s*\}\s*\]'  
                    match = re.search(pattern, corrected_json)  
                    if match:  
                        desc1 = match.group(1)  
                        desc2 = match.group(2)  
                        replacement = f'"activites_extra_scolaires": [{"description": "{desc1}", "description2": "{desc2}"}]'  
                        corrected_json = re.sub(pattern, replacement, corrected_json)  
                  
                logger.debug("Corrected JSON:")  
                logger.debug(corrected_json)  
                  
                try:  
                    structured_data = json.loads(corrected_json)  
                    logger.debug("JSON successfully corrected!")  
                    return structured_data  
                except json.JSONDecodeError as e:  
                    logger.error(f"Failed to correct JSON: {e}")  
                      
                    # Last resort: manual reconstruction of the JSON  
                    try:  
                        # Extract all the valid sections we can  
                        info_personnelles_match = re.search(r'"informations_personnelles"\s*:\s*\{[^}]*\}', corrected_json)  
                        education_match = re.search(r'"education"\s*:\s*\[[^\]]*\]', corrected_json)  
                        competences_match = re.search(r'"competences"\s*:\s*\{[^}]*\}', corrected_json)  
                        projets_match = re.search(r'"projets"\s*:\s*\[[^\]]*\]', corrected_json)  
                        langues_match = re.search(r'"langues"\s*:\s*\{[^}]*\}', corrected_json)  
                          
                        # Build a minimal valid JSON  
                        minimal_json = "{"  
                        if info_personnelles_match:  
                            minimal_json += info_personnelles_match.group(0) + ","  
                        if education_match:  
                            minimal_json += education_match.group(0) + ","  
                        if competences_match:  
                            minimal_json += competences_match.group(0) + ","  
                        if projets_match:  
                            minimal_json += projets_match.group(0) + ","  
                        if langues_match:  
                            minimal_json += langues_match.group(0) + ","  
                          
                        # Add a simple activites_extra_scolaires section  
                        minimal_json += '"activites_extra_scolaires": [{"description": "Activités parascolaires"}]'  
                        minimal_json += "}"  
                          
                        # Try to parse this minimal JSON  
                        structured_data = json.loads(minimal_json)  
                        logger.debug("Minimal JSON reconstruction successful!")  
                        return structured_data  
                    except Exception as e:  
                        logger.error(f"Failed to reconstruct JSON: {e}")  
                        return None  
        except Exception as e:  
            logger.error(f"Error calling Groq API: {e}")  
            logger.debug(traceback.format_exc())  
            return None  
  
    def save_to_mongodb(self, structured_data, mongodb_client, collection_name="cvs"):  
        """  
        Save the structured CV data to MongoDB.  
          
        Args:  
            structured_data: The structured CV data as a dictionary  
            mongodb_client: MongoDB client instance  
            collection_name: Name of the collection to save to  
              
        Returns:  
            str: The ID of the inserted document as a string (not ObjectId)  
        """  
        try:  
            # Get the database - use explicit database name instead of get_database()  
            db = mongodb_client["cv_database"]  
            logger.debug(f"Using database: {db.name}")  
              
            # Log the structured data before insertion  
            logger.debug(f"Structured data to save: {json.dumps(structured_data, ensure_ascii=False)[:200]}...")  
              
            # Make sure the collection exists  
            if collection_name not in db.list_collection_names():  
                logger.info(f"Creating collection: {collection_name}")  
                db.create_collection(collection_name)  
              
            # Get the collection  
            collection = db[collection_name]  
            logger.debug(f"Using collection: {collection_name}")  
              
            # Insert the document  
            result = collection.insert_one(structured_data)  
            inserted_id = result.inserted_id  
            logger.info(f"Data saved to MongoDB with ID: {inserted_id}")  
              
            # Return the ID as a string to avoid JSON serialization issues  
            return str(inserted_id)  
        except Exception as e:  
            logger.error(f"Error saving to MongoDB: {e}")  
            logger.debug(traceback.format_exc())  
            return None