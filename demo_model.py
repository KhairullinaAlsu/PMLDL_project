import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Чтение датасета с данными о музыке
df = pd.read_csv('spotify_music_dataset.csv')

# Удаление пропущенных значений и сброс индексов
df = df.dropna()
df = df.reset_index(drop=True)

# Удаление дубликатов по track_id
df = df.drop_duplicates(subset='track_id', keep='first')

# Выбор аудио признаков, которые будут использоваться в рекомендательной системе
audio_features = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]
df_features = df[audio_features].copy()

# Нормализация признаков
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_features)

# Преобразование нормализованных данных обратно в DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=audio_features)

# Вычисление матрицы косинусного сходства между треками
similarity_matrix = cosine_similarity(df_scaled)


def get_song_recommendations(song_name, artist_name=None, n_recommendations=5):
    """
    Возвращает список рекомендаций песен на основе заданной песни.

    Параметры:
    song_name (str): Название песни, для которой требуется найти рекомендации.
    artist_name (str, optional): Имя артиста (необязательно для точного поиска).
    n_recommendations (int): Количество рекомендованных песен (по умолчанию 5).

    Возвращает:
    pd.DataFrame: DataFrame с рекомендованными песнями и их характеристиками.
    """

    # Приведение названий песен и артистов к нижнему регистру для корректного сравнения
    df['track_name_lower'] = df['track_name'].str.lower()
    df['artist_name_lower'] = df['artist_name_x'].str.lower()

    # Поиск индекса песни в датасете по названию и (опционально) имени артиста
    if artist_name:
        idx = df[(df['track_name_lower'] == song_name.lower()) &
                 (df['artist_name_lower'] == artist_name.lower())].index
    else:
        idx = df[df['track_name_lower'] == song_name.lower()].index

    if len(idx) == 0:
        print("Песня не найдена в датасете.")
        return None

    idx = idx[0]

    # Получение списка сходства для выбранной песни
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Получение индексов наиболее похожих песен
    similar_songs_indices = [i[0] for i in similarity_scores[1:n_recommendations + 1]]

    # Возвращение информации о рекомендуемых песнях
    recommended_songs = df.iloc[similar_songs_indices][['track_name', 'artist_name_x', 'genres']]

    # Сброс индексов и переименование колонок для удобства
    recommended_songs = recommended_songs.reset_index(drop=True)
    recommended_songs.columns = ['Track Name', 'Artist', 'Genres']

    return recommended_songs


# Пример использования функции

song_to_recommend = input('song_to_recommend: ')  # "Die With A Smile"
artist_of_song = input('artist_of_song: ')  # "Lady Gaga"
number = int(input('number of recommendations: '))

recommendations = get_song_recommendations(song_to_recommend, artist_of_song, n_recommendations=number)

if recommendations is not None:
    print("Рекомендуемые песни:")
    print(recommendations)
