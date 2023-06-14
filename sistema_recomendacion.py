# Importamos pandas
import pandas as pd
import numpy as np
# Importamos la librería Surprise y sus métodos
from surprise import NMF, SVD, SVDpp, KNNBasic, KNNWithMeans, KNNWithZScore, CoClustering
from surprise.model_selection import cross_validate
from surprise import Reader, Dataset

# Tenemos los datos divididos en varios archivos. Nosotros valomos a utilizar en principio los archivos "u.data"
# con la información del usuario, el id de la película y la valoración, y el archivo "u.item" del que únicamente
# rescataremos los datos de id de la película y el título.

colum_usu = ['usuario_id', 'pelicula_id', 'valoracion']
valora_usu = pd.read_csv(
    r'C:\Users\lizet\OneDrive - Universidad Tecnologica del Peru\Documents\CICLO 10-MARZO 2023\Inteligencia\ml-100k\u.data',
    sep='\t', names=colum_usu, usecols=range(3), encoding="ISO-8859-1")

colum_pelis = ['pelicula_id', 'titulo']
peliculas = pd.read_csv(
    r'C:\Users\lizet\OneDrive - Universidad Tecnologica del Peru\Documents\CICLO 10-MARZO 2023\Inteligencia\ml-100k\u.item',
    sep='|', names=colum_pelis, usecols=range(2), encoding="ISO-8859-1")
# Combinamos ambos datasets ...
valoraciones = pd.merge(peliculas, valora_usu)

# Utilizamos nuevamente las películas con más de 50 valoraciones
valoraciones['n_votaciones'] = valoraciones.groupby(['titulo'])['valoracion'].transform('count')
valoraciones= valoraciones[valoraciones.n_votaciones>50][['usuario_id', 'titulo', 'valoracion']]
algo = Reader(rating_scale=(1, 5))
datos = Dataset.load_from_df(valoraciones, algo)
# Obtenemos la lista de películas
listaPeliculas = valoraciones['titulo'].unique()
# Las películas votadas
misVotaciones = valoraciones.loc[valoraciones['usuario_id']==999, 'titulo']
# Eliminamos nuestras películas
peliculas_predecir = np.setdiff1d(listaPeliculas,misVotaciones)

nmf = NMF()
nmf.fit(datos.build_full_trainset())
my_recs = []
for iid in peliculas_predecir:
    my_recs.append((iid, nmf.predict(uid=8, iid=iid).est))

nmf=pd.DataFrame(my_recs, columns=['titulo', 'prediccion']).sort_values('prediccion', ascending=False).head(10)
print('Recomendador con NMF')
print(nmf)

svd = SVD()
svd.fit(datos.build_full_trainset())
my_recs = []
for iid in peliculas_predecir:
    my_recs.append((iid, svd.predict(uid=8, iid=iid).est))

svd=pd.DataFrame(my_recs, columns=['titulo', 'prediccion']).sort_values('prediccion', ascending=False).head(10)

print('Recomendador con SVD')
print(svd)

svdpp = SVDpp()
svdpp.fit(datos.build_full_trainset())
my_recs = []
for iid in peliculas_predecir:
    my_recs.append((iid, svdpp.predict(uid=8, iid=iid).est))

svdpp=pd.DataFrame(my_recs, columns=['titulo', 'prediccion']).sort_values('prediccion', ascending=False).head(10)
print('Recomendador con SVD++')
print(svdpp)

KNN = KNNWithZScore()
KNN.fit(datos.build_full_trainset())
my_recs = []
for iid in peliculas_predecir:
    my_recs.append((iid, KNN.predict(uid=8, iid=iid).est))

KNN=pd.DataFrame(my_recs, columns=['titulo', 'prediccion']).sort_values('prediccion', ascending=False).head(10)
print('Recomendador con el algoritmo KNN with Z-Score')
print(KNN)

clust = CoClustering()
clust.fit(datos.build_full_trainset())
my_recs = []
for iid in peliculas_predecir:
    my_recs.append((iid, clust.predict(uid=8, iid=iid).est))

clust=pd.DataFrame(my_recs, columns=['titulo', 'prediccion']).sort_values('prediccion', ascending=False).head(10)
print('Recomendador con el algoritmo Co-Clustering')
print(clust)

cv = []
# Iteramos sobre cada algoritmo
for recsys in [NMF(), SVD(), SVDpp(), KNNWithZScore(), CoClustering()]:
    # Utilizamos cross-validation
    tmp = cross_validate(recsys, datos, measures=['RMSE'], cv=3, verbose=False)
    cv.append((str(recsys).split(' ')[0].split('.')[-1], tmp['test_rmse'].mean()))
evaluacion=pd.DataFrame(cv, columns=['Algoritmo', 'RMSE'])
print('Evaluación de los algoritmos 1')
print(evaluacion)
cv2 = []
# Iteramos sobre cada algoritmo
for recsys in [NMF(), SVD(), SVDpp(), KNNWithZScore(), CoClustering()]:
    # Utilizamos cross-validation
    tmp2 = cross_validate(recsys, datos, measures=['MAE'], cv=3, verbose=False)
    cv2.append((str(recsys).split(' ')[0].split('.')[-1], tmp2['test_mae'].mean()))
evaluacion2=pd.DataFrame(cv2, columns=['Algoritmo', 'MAE'])
print('Evaluación de los algoritmos 2')
print(evaluacion2)
