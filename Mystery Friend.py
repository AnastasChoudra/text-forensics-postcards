from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import import_ipynb
from goldman_emma_raw import goldman_docs
from henson_matthew_raw import henson_docs
from wu_tingfang_raw import wu_docs

# Create bow_vectorizer:
bow_vectorizer = CountVectorizer()

friends_docs = goldman_docs + henson_docs + wu_docs

# Define friends_vectors:
friends_vectors = bow_vectorizer.fit_transform(friends_docs)

mystery_postcard = """
My friend,
From the 10th of July to the 13th, a fierce storm raged, clouds of
freeing spray broke over the ship, incasing her in a coat of icy mail,
and the tempest forced all of the ice out of the lower end of the
channel and beyond as far as the eye could see, but the _Roosevelt_
still remained surrounded by ice.
Hope to see you soon.
"""

# Define mystery_vector:
mystery_vector = bow_vectorizer.transform([mystery_postcard])

# Print out a document from each friend:
print("Goldman's Letter Sample: ")
print(goldman_docs[49],'\n')
print("Henson's Letter Sample: ")
print(henson_docs[49],'\n')
print("Wu's Letter Sample: ")
print(wu_docs[49],'\n')

# Define friends_classifier:
friends_classifier = MultinomialNB()

friends_labels = ["Emma"] * 154 + ["Matthew"] * 141 + ["Tingfang"] * 166

# Train the classifier:
friends_classifier.fit(friends_vectors, friends_labels)

predictions = friends_classifier.predict(mystery_vector)

mystery_friend = predictions[0] if predictions[0] else "someone else"

print("Prediction: The postcard was from {}!".format(mystery_friend))

