class JobOffer:  
    """  
    Manages job offers and their structuring.  
    """  
      
    def __init__(self, title=None, required_skills=None, experience=None, education=None, description=None):  
        """  
        Initialize a job offer.  
          
        Args:  
            title: Job title  
            required_skills: List of required skills  
            experience: Required experience  
            education: Required education  
            description: Job description  
        """  
        self.title = title or ""  
        self.required_skills = required_skills or []  
        self.experience = experience or ""  
        self.education = education or ""  
        self.description = description or ""  
      
    def to_dict(self):  
        """  
        Convert job offer to dictionary.  
          
        Returns:  
            dict: Dictionary representing the job offer  
        """  
        return {  
            "titre": self.title,  
            "competences_requises": self.required_skills,  
            "experience": self.experience,  
            "diplome": self.education,  
            "description": self.description  
        }  
      
    @classmethod  
    def from_dict(cls, data):  
        """  
        Create a job offer from a dictionary.  
          
        Args:  
            data: Dictionary containing job offer information  
              
        Returns:  
            JobOffer: JobOffer instance  
        """  
        return cls(  
            title=data.get("titre"),  
            required_skills=data.get("competences_requises"),  
            experience=data.get("experience"),  
            education=data.get("diplome"),  
            description=data.get("description")  
        )  
      
    @classmethod  
    def from_form(cls, form_data):  
        """  
        Create a job offer from form data.  
          
        Args:  
            form_data: Dictionary containing form data  
              
        Returns:  
            JobOffer: JobOffer instance  
        """  
        # Convert skills to list (if provided as string)  
        skills = form_data.get("competences_requises", "")  
        if isinstance(skills, str):  
            skills = [skill.strip() for skill in skills.split(",") if skill.strip()]  
          
        return cls(  
            title=form_data.get("titre_poste"),  
            required_skills=skills,  
            experience=form_data.get("experience"),  
            education=form_data.get("diplome"),  
            description=form_data.get("description", "")  
        )