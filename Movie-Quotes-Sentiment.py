# Movie Quotes Sentiment with TextBlob
# Tutorial from John Watson Rooney YouTube channel

# https://textblob.readthedocs.io/en/dev/
# https://imdbpy.readthedocs.io/en/latest/

from imdb import IMDb
from textblob import TextBlob
import pandas as pd

# create an instance of the IMDb class
ia = IMDb()

movie_quotes = []

movie = ia.search_movie(input('Enter in a movie title: '))
movie_id = movie[0].movieID
my_movie = ia.get_movie(movie_id)

ia.update(my_movie, 'quotes', 'rating')

for quote in my_movie['quotes']:
    blob = TextBlob(str(quote))

    score = {
        'movie': str(my_movie),
        'rating': my_movie['rating'], 
        'quote': str(quote),
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity,
    }

    movie_quotes.append(score)

df = pd.DataFrame(movie_quotes)
df = df[df.polarity != 0.0]
total = pd.pivot_table(df, index=['movie'])
df.to_csv('Movie-Quotes-Sentiment.csv')
#print(df.head())
print(total.head())
print('Saved items to CSV file.')


# NOTES - TO-DO: 
# Use round() function for rounding numbers. 
# f-String for CSV file based on movie name


