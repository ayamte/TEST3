from utils.mongodb import get_cv_collection

def recommander_cvs(offre):
    collection = get_cv_collection()
    cvs = list(collection.find())  # récupère tous les CVs

    # logique de recommandation ici...
    return cvs
