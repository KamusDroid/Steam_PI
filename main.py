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
            <title>API Steam</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    padding: 20px;
                }
                h1 {
                    color: #333;
                    text-align: center;
                }
                p {
                    color: #666;
                    text-align: center;
                    font-size: 18px;
                    margin-top: 20px;
                }
            </style>
        </head>
        <body>
            <h1>API de consultas sobre juegos de la plataforma Steam</h1>
            <p>Bienvenido a la API de Steam, su fuente confiable para consultas especializadas sobre la plataforma de videojuegos.</p>
        </body>
    </html>
    """
    return HTMLResponse(content=template)



sgames = pd.read_parquet("./steam_games.parquet")
userforgenre = pd.read_parquet("./userforgenre.parquet")
ureviews = pd.read_parquet("./user_reviews.parquet")
uitems = pd.read_parquet("./user_items.parquet")
modelo = pd.read_parquet("./modelo1.parquet")

uitems['item_id'] = uitems['item_id'].astype(str)
sgames['id'] = sgames['id'].astype(str)


@app.get("/playtimegenre/{genero}")
async def PlayTimeGenre(genero: str):
    genero = genero.lower()
    
    # Filtrar juegos por género
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
    # Filtrar el DataFrame por el género proporcionado
    df_filtered = userforgenre[userforgenre['genres'] == genres]

    print("DataFrame filtrado por género:")
    print(df_filtered)

    if not df_filtered.empty:
        # Encontrar el usuario que acumula más horas jugadas
        user_with_most_playtime = df_filtered.groupby('user_id')['playtime_forever'].sum().idxmax()

        print("Usuario con más horas jugadas:")
        print(user_with_most_playtime)

        # Crear una lista de la acumulación de horas jugadas por año
        playtime_by_year = df_filtered.groupby('anio')['playtime_forever'].sum().reset_index()
        playtime_by_year = playtime_by_year.rename(columns={'anio': 'Año', 'playtime_forever': 'Horas'})

        # Convertir las horas a enteros
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