import json
import os
import re
from groq import Groq
from pymongo import MongoClient


class LLMRecommender:
    """
    Compare a structured CV to a database of IT profiles and return the most similar profiles
    using LLaMA3 via the Groq API.
    """

    def __init__(self, api_key=None, mongodb_uri=None):
        """Initialize the Groq client with the API key and MongoDB connection."""
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.mongodb_uri = mongodb_uri or os.environ.get("MONGODB_URI")

        if not self.api_key:
            print("WARNING: No Groq API key provided. LLM recommendation will not be available.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)

        self.model = "llama3-70b-8192"  # LLaMA3 model via Groq

    def load_profiles_from_mongodb(self, collection_name="profiles", limit=50):
        """Load IT profiles from MongoDB."""
        if not self.mongodb_uri:
            print("WARNING: No MongoDB URI provided. Unable to load profiles.")
            return []

        try:
            client = MongoClient(self.mongodb_uri)
            db = client.get_default_database()
            collection = db[collection_name]

            profiles = list(collection.find().limit(limit))

            for profile in profiles:
                if '_id' in profile:
                    profile['_id'] = str(profile['_id'])

            client.close()
            return profiles

        except Exception as e:
            print(f"Error loading profiles from MongoDB: {e}")
            return []

    def load_profiles_from_file(self, file_path):
        """Load IT profiles from a JSON file."""
        try:
            print(f"Attempting to load profiles from {file_path}")
            if not os.path.exists(file_path):
                print(f"ERROR: File {file_path} does not exist!")
                return []

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"File content: {content[:100]}...")

                if not content.strip():
                    print("ERROR: File is empty!")
                    return []

                try:
                    profiles = json.loads(content)
                    print(f"Profiles successfully loaded: {len(profiles)} profiles found")
                    return profiles
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                    print("Attempting to correct JSON...")

                    if content.strip().startswith('[') and content.strip().endswith(']'):
                        strings = re.findall(r'"([^"]*)"', content)
                        if strings:
                            print(f"Extracted {len(strings)} strings from file")
                            return strings

                    return []

        except Exception as e:
            print(f"Error loading profiles from file: {e}")
            print(f"Problematic content: {content if 'content' in locals() else 'Not available'}")
            return []

    def create_prompt(self, cv_json, profiles):
        """Create a suitable prompt to compare the CV with profiles."""
        return f"""
        Here is a structured CV of a candidate:
        {json.dumps(cv_json, ensure_ascii=False, indent=2)}

        And here is a list of available profiles:
        {json.dumps(profiles, ensure_ascii=False, indent=2)}

        What are the 3 most similar profiles to this CV?
        Analyze technical skills, experience, projects, and education.
        Respond ONLY in JSON format with profile names and an explanation for each match.
        Expected format:
        {{
            "profils_recommandés": [
                {{
                    "nom": "profile_name",
                    "score_similarité": <score between 0 and 1>,
                    "raisons": "Explanation of the reasons for this recommendation"
                }},
                ...
            ]
        }}

        IMPORTANT: Make sure your response is a valid and well-formatted JSON, without any additional text.
        """

    def recommend_profiles(self, cv_json, profiles=None, profiles_file=None, collection_name="profiles"):
        """
        Compare a structured CV to a list of IT profiles and return the 3 most similar profiles.
        """
        if not self.client:
            print("Unable to make recommendations: no Groq API key available.")
            return None

        if not profiles:
            if profiles_file:
                profiles = self.load_profiles_from_file(profiles_file)
            else:
                profiles = self.load_profiles_from_mongodb(collection_name)

        if not profiles:
            print("No profiles available for comparison.")
            return None

        prompt = self.create_prompt(cv_json, profiles)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant specialized in CV analysis and IT profile recommendation. You must compare a CV to a list of profiles and identify the 3 most similar profiles. Your response must be ONLY a valid and well-formatted JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )

            json_response = response.choices[0].message.content
            print("Raw API response:")
            print(json_response)

            json_response = json_response.strip()
            if json_response.startswith("```json"):
                json_response = json_response.replace("```json", "", 1)
            if json_response.endswith("```"):
                json_response = json_response.rsplit("```", 1)[0]
            json_response = json_response.strip()

            print("Cleaned response:")
            print(json_response)

            try:
                recommendations = json.loads(json_response)
                return recommendations
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                print("Attempting to correct JSON...")

                corrected_json = json_response
                corrected_json = re.sub(r'"\]', '"', corrected_json)
                corrected_json = re.sub(r'\}]', '}', corrected_json)
                corrected_json = re.sub(r'\}\s*\{', '},{', corrected_json)

                try:
                    recommendations = json.loads(corrected_json)
                    print("JSON successfully corrected!")
                    return recommendations
                except json.JSONDecodeError:
                    print("Failed to correct JSON.")
                    return None

        except Exception as e:
            print(f"Error recommending profiles: {e}")
            return None

    def save_recommendations_to_file(self, recommendations, output_file):
        """
        Save recommendations to a JSON file.

        Args:
            recommendations: Profile recommendations in JSON format
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
