# PROYECTO INDIVIDUAL 1 MACHINE LEARNING OPERATIONS (MLOPs): CLASSIFICATION MODEL FOR A MOVIE RECOMMENDER SYSTEM

The moviedataset is a collection of around 45000 entries between the years 1980 and 2017 with information of movies on cast, crew, overview, budget, revenue, release dates, languages and more. 

Moviedatasets :
    * Movies_dataset: it includes features like title, budget, revenue, languages, release dates, languages, production countries         and companies.
    * Credits: it shows information on cast and crew of all the movies.

Main tasks:
    * Build APIs to request information.
    * Build a recommender system.

$ I. Data Engineering

I.1 ETL

You can see all of the procedures done for extraction, transformation and load of the moviedataset.
The main problems found during ETL to accomplish the main tasks were:

    A. Nested dictionaries or list of dictionaries columns: 
          1) belongs_to_collection
          2) genres 
          3) production_companies
          4) production_countries
          5) spoken_languages 
    B. These columns were transformed as well (missing values, adequate format, column creation and remove of capital letters):
          1) revenue, budget, release_dates.
          2) release_year and return(revenue/budget).
          3) video,imdb_id,adult,original_title,poster_path and homepage.
          
This can be seen in data_wranglingETL.ipynb.

I.2 EDA

I am going to focus on the following aspects: 
 
          1. Values of central tendency (mean, median, mode).
          2. Values of Variability (minimum, maximum, percentile, variance, correlation).
          3. Shape (if values are symmetrically or assymetrically distributed when it is possible).
          4. Outliers if there are values that represent adnormalities in the data. 
          5. Visualization (a combination of statistics and visualization).

Note: 
   * I will "not delete outliers" because we will use a Cosine Similarity and TFIDF model where variability is not important, but 
     how similar our inputs are.
   * There will be a wordcloud showing the most frecuent words on title and overview.
   * I will choose vote_count over popularity according to the pairplot and correlation matrix. 

This can be seen in data_wranglingEDA.ipynb.

I.3 API development

I will create functions to be deployed using a FASTAPI framework (RENDER) which means I will consider critical means for the API to be easily and successfully consumed. 

This can be seen in main.py, requirements.txt, .gitignore and Procfile.

II. Data Science

I will develop a TFIDF model (based on Cosine Similarity) because it adjust better the similarities of the variable that i chose 
to run the model.

I will use just three variables, not the best approach but still meaningful:
    * Title (as an indexer).
    * Overview (from a view of the movie plot: corpus or characters to be compared).
    * vote_count (from a view of the watcher sentiment: corpus).


Tools:

🐍Python: Lenguaje de programación principal utilizado en el desarrollo del proyecto.
💻Numpy: Utilizado para realizar operaciones numéricas y manipulación de datos.
🐼Pandas: Utilizado para la manipulación y análisis de datos estructurados.
📈Matplotlib: Utilizado para la visualización de datos y generación de gráficos.
📈Seaborn: Utilizado para la visualización de datos y generación de gráficos.
📊 Scikit Learn: Utilizado para vectorizar, tokenizar y calcular la similitud coseno.
📳FastAPI: Utilizado para crear la interfaz de la aplicación y procesar los parámetros de funciones.
🦄Uvicorn: Servidor ASGI utilizado para ejecutar la aplicación FastAPI.
🌐Render: Plataforma utilizada para el despliegue del modelo y la aplicación.


# Links

https://first-project-deploy11.onrender.com

# Autor

Max Jeffer Cuellar Vizcarra

Correo electrónico: max_83_14@hotmail.com

LinkedIn: https://www.linkedin.com/in/max-jeffer-cuellar-vizcarra-25197433/

          






