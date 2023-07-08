from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

app = FastAPI()


df = pd.read_csv('../individualProjectMLOPI/clean_data.csv')

@app.get("/")
async def root():
    return {"message": "Welcome to my project", 
            "Instruction 1": "Double click on the addresss bar",
            " Instruction 2": "Add at the end of the URL /docs and enter."}



@app.get('/peliculas_idioma/{idioma}')
def peliculas_idioma(idioma: str):
    '''
    This function retrieves the number of movies produced in a certain language
    
    Parameters:
    idioma (str): It refers to an element of the original_language column.
    
    Returns:
    A dictionary with key: language and  value: number of movies produced in the language 
    
    '''
    idioma_lower = idioma.lower()  # it converts the input 'idioma' to lower case for comparison.
    cantidad = sum(df['original_language'].str.lower() == idioma_lower)  # it counts the movies of a given idioma.
    
    return {'idioma': idioma, 'cantidad': cantidad}

@app.get('/peliculas_duracion/{pelicula}')
def peliculas_duracion(pelicula: str):
    '''
    This function uses as input the name of a movie (película) and returns runtime (duracion) and released year(anio).
    
    Parameters:
    pelicula (str): It refers to an element of the title column.
    
    Returns:
    A dictionary with keys 'pelicula', 'duracion' and 'anio' and its values the title of the movie entered(pelicula),
    runtime (duracion) and released year (anio).
    '''
    pelicula_lower = pelicula.lower()  # it converts the input 'idioma' to lower case for comparison.
    
    pelicula_filtrada = df.loc[df['title'].str.lower() == pelicula_lower]  # It filters the rows that match the input.
    
    if len(pelicula_filtrada) == 0: # If the input does not match any element of the column, it returns the input and 'not found'(no encontrado)
        return {'pelicula': pelicula, 'duracion': 'No encontrada', 'anio': 'No encontrada'}
    
    duracion = pelicula_filtrada['runtime'].values[0]  # it gets the runtime of the movie.
    anio = pelicula_filtrada['release_year'].values[0]  # it gets the released year.
    
    return {'pelicula': pelicula, 'duracion': duracion, 'anio': anio}

@app.get('/franquicia/{franquicia}')
def franquicia(franquicia: str):
    '''
    This function needs as input "nombre de la franquicia" and returns a number of movies, a total revenue
    and an revenue's average of the movies that belong to the franchise.
    
    Parameters:
    franquicia (str): It refers to an element of the column belongs_to_collection_name
    or a franchise you would like to get information(returns).
    
    Returns:
    A dictionary with keys 'franquicia', 'cantidad', 'ganancia_total' y 'ganancia_promedio'.
    'franquicia' contains a franchise entered as input, 'cantidad' gets the amount of franchise's movies,
    'ganancia_total' retrieves the total revenue of the franchise's movies and
    'ganancia_promedio' return a revenue's average of the movies that belong to the franchise.
    '''
    franquicia_lower = franquicia.lower()  # it converts the input 'idioma' to lower case for comparison.
    
    franquicia_filtrada = df.loc[df['belongs_to_collection_name'].str.lower() == franquicia_lower]  # It filters the rows that match the input.
    
    cantidad = len(franquicia_filtrada)  # it gets the amount of movies that belong to the input.
    
    if cantidad == 0:
        return {'franquicia': franquicia, 'cantidad': 'No encontrada', 'ganancia_total': 'No encontrada', 'ganancia_promedio': 'No encontrada'}
    
    ganancia_total = franquicia_filtrada['revenue'].sum()  # it gets the total revenue.
    ganancia_promedio = franquicia_filtrada['revenue'].mean()  # it retrieves revenue's average of the movies that belong to the franchise.
    
    return {'franquicia': franquicia, 'cantidad': cantidad, 'ganancia_total': ganancia_total, 'ganancia_promedio': ganancia_promedio}

@app.get('/peliculas_pais/{pais}')
def peliculas_pais(pais: str):
    '''
    This function retrieves all of the movies produced in a country.
    
    Parameters:
    pais (str): The name of the country where the movies were produced.
    
    Returns:
    A dictionary with key-value 'pais' and 'cantidad'. 
    '''
    pais_lower = pais.lower()  # it converts the input 'idioma' to lower case for comparison.
    cantidad = sum(df['production_countries_name'].str.lower() == pais_lower)  # it counts the countries where the movies were produced.
    
    return {'pais': pais, 'cantidad': cantidad}

@app.get('/productoras_exitosas/{productora}')
def productoras_exitosas(productora: str):
    '''
    It uses as input "nombre de la productora" and retrieves its total revenue 
    and all of the movies produced by the company.
    
    Parameters:
    productora (str): it refers to an element of the production companies name column.
    
    Returns:
    A dictionary with keys 'productora', 'revenue_total' and 'cantidad'. 'productora' refers to a company,
    'revenue_total' all revenues produced by the released of its movies
    and 'cantidad' refers to all of the movies produced by the company.
    '''
    productora_lower = productora.lower()   # it converts the input 'idioma' to lower case for comparison.
 
    productora_filtrada = df[df['production_companies_name'].apply(lambda x: isinstance(x, str) 
                                                    and productora_lower in x.lower())]  # it filters all of the rows that match with the input.
    # we apply lambda function because there are more than one element in each cell of the column.
    
    cantidad = len(productora_filtrada)  # it counts the movies produced by the company.
    
    if cantidad == 0:
        return {'productora': productora, 'revenue_total': 'No encontrada', 'cantidad': 'No encontrada'}
    
    revenue_total = productora_filtrada['revenue'].sum()  # it gets the total revenue of the produced movies.
    
    return {'productora': productora, 'revenue_total': revenue_total, 'cantidad': cantidad}

@app.get('/get_director/{nombre_director}')
def get_director(nombre_director: str):
    '''
    This function has as input nombre de un director on the current dataset and shows how successful the director was according to the return of all produced movies, 
    it retrieves the nombre de cada of each movie with its released date, its return(only of the movie) , budget and its revenue in list format.
    
    Parameters:
    nombre_director (str): It refers to an element of the director column.
    
    Returns:
    A dictionary with keys 'director', 'retorno_total_director', 'peliculas', 'anio', 'retorno_pelicula', 'budget_pelicula' and 'revenue_pelicula'.
    'director' retrieves the name of a director, 
    'retorno_total_director' measures the success of a director as the sum of all produced movies' revenues,
    'peliculas' is a list of all movies directed, 
    'fecha' list of movies' released dates, 
    'retorno_pelicula' list of revenue by movie, 
    'budget_pelicula' list of a budget  by movie and
    'revenue_pelicula' list of revenue by movie.
    '''
    nombre_director_lower = nombre_director.lower()  
    
    director_films = df[df['director'].str.lower() == nombre_director_lower]  
    
    if director_films.empty:
        return {'director': nombre_director, 'retorno_total_director': 'No encontrado', 'peliculas': [], 'fecha': [], 'retorno_pelicula': [], 'budget_pelicula': [], 'revenue_pelicula': []}
    
    retorno_total_director = director_films['return'].sum()  # it figures out the director's return.
    
    peliculas = director_films['title'].tolist()  # it gets a list of movies' titles
    anio = director_films['release_date'].tolist()  # it gets a list of released dates 
    retorno_pelicula = director_films['return'].tolist()  # it retrieves a list of returns by movie
    budget_pelicula = director_films['budget'].tolist()  # it retrieves a list of budgets
    revenue_pelicula = director_films['revenue'].tolist()  # it gets a list of revenues by movie
    
    return {'director': nombre_director, 'retorno_total_director': retorno_total_director, 'peliculas': peliculas, 'fecha': anio, 'retorno_pelicula': retorno_pelicula, 'budget_pelicula': budget_pelicula, 'revenue_pelicula': revenue_pelicula}


# ML
@app.get('/recomendacion/{titulo}')
def recomendacion(titulo: str):
    '''
    it enters a movie name and retrieves a list of recommended movies.
    
    Parameters:
    titulo (str): It refers to an element of the title column and it is a reference to recommend other movies.
    
    Returns:
    A dictionary with a key 'lista_recomendada' that contains a list of movie names. The movies are similar according to
    a punctuation order of alikeness.
    '''
    
    titulo = titulo.lower()
    
    # it gets the index of a given movie (by tittle)
    index = df[df['title'].str.lower() == titulo].index[0]
    
    # it gets the characteristic vector TF-IDF for the movie synopsis
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'].fillna(''))
    
    # it calculates a punctuation of alikeness(cosine_similarity) between a given movie and other movies.
    similarity_scores = cosine_similarity(tfidf_matrix[index], tfidf_matrix).flatten()
    
    # it gets the index of more similar movies excluding the given movie
    similar_indices = similarity_scores.argsort()[::-1][1:6]
    
    # it gets the titles of the more similar movies
    peliculas_recomendadas = df.iloc[similar_indices]['title'].tolist()
    
    return {'lista_recomendada': peliculas_recomendadas}


