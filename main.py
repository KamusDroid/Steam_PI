from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from typing import Dict, List, Union
import logging
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def inicio():
    template = """
    <!DOCTYPE html>
<html>

<head>
    <title>Proyecto Individual - Henry</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            padding: 20px;
            background-color: black;
            color: #FFD700; /* Amarillo */
        }

        h1 {
            color: #FF4500; /* Rojo oscuro */
            text-align: center;
        }

        p {
            color: #FF6347; /* Tomate */
            text-align: center;
            font-size: 18px;
            margin-top: 20px;
        }

        button {
            background-color: #FF8C00; /* Naranja oscuro */
            color: #000; /* Negro */
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            cursor: pointer;
            margin-top: 20px;
        }

        button:hover {
            background-color: #FFA500; /* Naranja */
        }
    </style>
</head>

<body>
    <h1>PROYECTO INDIVIDUAL NÚMERO 1 - HENRY - MLOPs</h1>
    <p>En este proyecto a partir de un DataSet proporcionado de la Empresa STEAM debemos realizar diferentes analisis y operaciones con el finde cumplimentar los end points que encontrará al presionar el BOTON Docs, que lo lleva hacia la pagina de comprobación de funcionamiento de Endpoints</p>

    <!-- Botón para ir a la carpeta "docs" -->
    <button onclick="window.location.href='docs'">Ir a la carpeta "docs"</button>

    <!-- Mensaje de autorrealización nerd en el fondo de la página -->
    <div style="position: fixed; bottom: 10px; right: 10px; font-size: 11px;">
        ¡Autorealización nerd alcanzada!
    </div>
</body>

</html>

    """
    return HTMLResponse(content=template)



sgames = pd.read_parquet("./steam_games.parquet")
userforgenre = pd.read_parquet("./userforgenre.parquet")
ureviews = pd.read_parquet("./user_reviews.parquet")
uitems = pd.read_parquet("./user_items.parquet")
modelo = pd.read_parquet("./modelo1.parquet")
games_reviews = pd.read_parquet("./games_reviews.parquet")

uitems['item_id'] = uitems['item_id'].astype(str)
sgames['id'] = sgames['id'].astype(str)


@app.get("/playtimegenre/{genero}")
async def PlayTimeGenre(genero: str):
   
    # Filtro juegos por género
    sgames2 = sgames[sgames['genres'].str.contains(fr'\b{genero}\b', case=False, na=False)]

    if not sgames2.empty:
        result = {"Para el género {} el año que más horas fue jugado fue el: {}".format(
        genero.capitalize(),
        pd.merge(uitems.astype({'item_id': str}), sgames2[['id', 'anio']].astype({'id': str}), left_on='item_id', right_on='id')
        .groupby('anio')['playtime_forever'].sum().idxmax()
    )}
    else:
        result = {"Para el género {}: {}".format(genero.capitalize(), "No encontramos nada...Intentalo de nuevo")}


    return result


userforgenre['genres']= userforgenre['genres']

@app.get("/userforgenre/{genres}")
async def UserForGenre(genres: str) -> Dict[str, Union[str, List[Dict[str, Union[str, int]]]]]:
    # Filtro el DataFrame por el género proporcionado
    df_filtered = userforgenre[userforgenre['genres'] == genres]

    print("DataFrame filtrado por género:")
    print(df_filtered)

    if not df_filtered.empty:
        # Busco el usuario que acumula más horas jugadas
        user_with_most_playtime = df_filtered.groupby('user_id')['playtime_forever'].sum().idxmax()

        print("Usuario con más horas jugadas:")
        print(user_with_most_playtime)

        # Creo una lista de la acumulación de horas jugadas por año
        playtime_by_year = df_filtered.groupby('anio')['playtime_forever'].sum().reset_index()
        playtime_by_year = playtime_by_year.rename(columns={'anio': 'Año', 'playtime_forever': 'Horas'})

        # Convierto las horas a enteros
        playtime_by_year['Horas'] = playtime_by_year['Horas'].astype(int)

        playtime_list = playtime_by_year.to_dict(orient='records')


        print("Horas jugadas por año:")
        print(playtime_by_year)

        result = {
            "Usuario con más horas jugadas para Género {}:".format(genres.capitalize()): user_with_most_playtime,
            "Horas jugadas por año": [{"Año": str(row['Año']), "Horas": row['Horas']} for _, row in playtime_by_year.iterrows()]
        }
    else:
        result = {"Género no encontrado": "Intentalo otra vez"}

    return result



@app.get("/userrecommend/{year}")
async def user_recommend(year: int) -> list:
    # Filtrar por Año
    filtered_df = games_reviews[games_reviews['anio_post'] == year]

    if filtered_df.empty:
        return ["No hay datos para el año especificado"]  # Mensaje de datos que no estan en el dataframe

    # Filtro por comentarios recomendados y sentiment_analysis positivo/neutral
    filtered_df = filtered_df[(filtered_df['recommend'] == True) & (filtered_df['sentiments'].isin([-1, 0, 1]))]

    # Obtengo el top 3 de juegos recomendados
    top_games = filtered_df['app_name'].value_counts().head(3).reset_index()
    top_games = top_games.rename(columns={'index': 'Puesto 1', 'app_name': 'Juego'})

    # Adapto el resultado a la salida que espero
    result = [{"Puesto {}".format(i + 1): juego, 'recomendaciones': count} for i, (juego, count) in enumerate(zip(top_games['Juego'], top_games['count']))]

    return result


@app.get("/worstdevelopers/{year}")
def worstDevelopers(year: int) -> list:
    # Filtro por Año
    filtered_df = games_reviews[games_reviews['anio_post'] == year]

    if filtered_df.empty:
        return ["No hay datos para el año especificado"]  # No hay datos para el año especificado

    # Filtro por comentarios no recomendados
    filtered_df = filtered_df[filtered_df['recommend'] == False]

    # Obtengo el top 3 de desarrolladores menos recomendados
    worst_developers = filtered_df['developer'].value_counts().head(3)

    # Adapto el Resultado
    result = [{"Desarrollador": developer, 'no_recomendaciones': count} for developer, count in worst_developers.items()]

    return result



games_reviews['developer'] = games_reviews['developer']


@app.get("/sentiment/{developer}", response_model=dict)
async def get_sentiment(developer: str):
    if games_reviews.empty:
        return {"error": "NO CARGAAAAAAA AAAAAA"}

    if 'sentiments' not in games_reviews.columns:
        return {"error": "No leo la columna que esta cargada correctamente en el dataframe."}

    developer_reviews = games_reviews[games_reviews['developer'] == developer]

    if developer_reviews.empty:
        return {"error": f"No hay datos para el desarrollador {developer}."}

    # Agrupo por análisis de sentimiento y cuento registros
    grouped_df = developer_reviews.groupby('sentiments').size().reset_index(name='count')

    # Creo un diccionario para almacenar los resultados
    result_dict = {'Negativo': 0, 'Neutral': 0, 'Positivo': 0}

    # Itero sobre los grupos y actualizo el diccionario
    for index, row in grouped_df.iterrows():
        sentiment = row['sentiments']
        count = row['count']

        if sentiment == -1:
            result_dict['Negativo'] = int(count)
        elif sentiment == 0:
            result_dict['Neutral'] = int(count)
        elif sentiment == 1:
            result_dict['Positivo'] = int(count)

    return {developer: result_dict}





@app.get("/gamerecomendation/{id}")

async def game_recommendation(id: int):
    id = int(id)
    
    similarity = cosine_similarity(modelo.iloc[:,3:])

    if id not in modelo['id'].values:
        return "El juego con el ID especificado no existe en la base de datos."
    
    # Obtener el índice del juego seleccionado
    indice_juego = modelo[modelo['id'] == id].index[0]
    
    # Imprimir el nombre del juego y el género
    nombre_juego = modelo.at[indice_juego, 'app_name']
    print(f"Juego seleccionado: {nombre_juego}")
    
    # Obtener los géneros del juego seleccionado
    generos_juego = [columna for columna in modelo.columns[3:] if modelo.at[indice_juego, columna] == 1]
    print(f"Géneros: {', '.join(generos_juego)}")
    
    # Calculamos la similitud del juego que se ingresa con otros juegos del dataframe
    similarity_scores = similarity[indice_juego]
    
    # Calculamos los índices de los juegos más similares (excluyendo el juego de entrada)
    similarity_games = similarity_scores.argsort()[::-1][1:6]
    
    # Obtenemos los nombres de los juegos 5 recomendados
    recommend = modelo.iloc[similarity_games]['app_name'].values.tolist()
    
    return recommend