import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class RecommendationSystem:
    def __init__(self, items):
        """
        items: DataFrame con al menos una columna 'description'
        """
        self.items = items
        self.vectorizer = TfidfVectorizer(stop_words='spanish')
        self.tfidf_matrix = self.vectorizer.fit_transform(items['description'])

    def recommend(self, query, top_n=5):
        """
        query: texto de búsqueda del usuario
        top_n: número de recomendaciones a retornar
        """
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_n:][::-1]
        return self.items.iloc[top_indices]

# Ejemplo de uso:
if __name__ == "__main__":
    # Datos de ejemplo
    data = {
        'title': ['Libro A', 'Libro B', 'Libro C'],
        'description': [
            'Un libro sobre inteligencia artificial y aprendizaje automático.',
            'Una novela de ciencia ficción ambientada en el futuro.',
            'Guía práctica de programación en Python para principiantes.'
        ]
    }
    df = pd.DataFrame(data)
    recommender = RecommendationSystem(df)
    resultados = recommender.recommend("Quiero aprender sobre inteligencia artificial", top_n=2)
    print(resultados)