import random

# Pick a random test image
random_index = random.randint(0, len(test_generator.filenames)-1)
img_path = os.path.join(test_dir, test_generator.filenames[random_index])

# Load and preprocess the image
def predict_emotion(img_path, model):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48,48))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)

    prediction = model.predict(img)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    predicted_emotion = emotion_labels[np.argmax(prediction)]
    
    plt.imshow(cv2.imread(img_path))
    plt.title(f"Predicted Emotion: {predicted_emotion}")
    plt.axis("off")
    plt.show()

predict_emotion(img_path, model)

def map_emotions_to_mood_scores(emotion_probabilities, emotion_labels, emotion_to_mood_map):
    """
    Maps emotion probabilities to mood scores.

    Args:
        emotion_probabilities (np.array): Probabilities for each emotion class.
        emotion_labels (list): List of emotion labels corresponding to the probabilities.
        emotion_to_mood_map (dict): Mapping from emotion labels to mood scores.

    Returns:
        dict: Normalized mood scores.
    """
    mood_scores = {
        "happy": 0.0,
        "sad": 0.0,
        "energetic": 0.0,
        "relaxed": 0.0,
        "romantic": 0.0
    }

    for i, prob in enumerate(emotion_probabilities):
        emotion_label = emotion_labels[i]
        if emotion_label in emotion_to_mood_map:
            for mood, weight in emotion_to_mood_map[emotion_label].items():
                mood_scores[mood] += prob * weight

    # Normalize mood scores to sum to 100
    total_score = sum(mood_scores.values())
    if total_score > 0:
        mood_scores = {mood: (score / total_score) * 100 for mood, score in mood_scores.items()}
    return mood_scores

# --- Prediction and Recommendation Flow ---

# Load the trained model
try:
    model = load_model("fer2013_cnn_improved.h5")
except Exception as e:
    print(f"Error loading model: {e}. Please ensure 'fer2013_cnn_improved.h5' exists after training.")
    exit()

# Get emotion labels from the train generator
emotion_labels = list(train_generator.class_indices.keys())

# Example: Predict emotion for a dummy image (replace with actual image input)
# In a real application, you would load an image, preprocess it, and then predict
dummy_image_path = os.path.join(test_dir, 'happy', os.listdir(os.path.join(test_dir, 'happy'))[0]) # Example: pick a happy image
img = cv2.imread(dummy_image_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (48, 48))
img = np.expand_dims(img, axis=-1)
img = np.expand_dims(img, axis=0)
img = img / 255.0

emotion_probabilities = model.predict(img)[0]
print(f"Emotion Probabilities: {emotion_probabilities}")

# Map predicted emotions to mood scores
face_mood_scores = map_emotions_to_mood_scores(emotion_probabilities, emotion_labels, emotion_to_mood_mapping)
print(f"Face Mood Scores: {face_mood_scores}")

# For simplicity, let's assume dummy values for bg_mood and text_mood.
# In a full system, these would come from other analysis modules.
bg_mood_dummy = {
    "happy": 0.1,
    "sad": 0.1,
    "energetic": 0.3,
    "relaxed": 0.4,
    "romantic": 0.1
}
text_mood_dummy = {
    "happy": 0.2,
    "sad": 0.1,
    "energetic": 0.4,
    "relaxed": 0.2,
    "romantic": 0.1
}

# Combine moods
combined_mood_scores = final_mood(face_mood_scores, bg_mood_dummy, text_mood_dummy)
print(f"Combined Mood Scores: {combined_mood_scores}")

# Generate the final mood vector
final_mood_vector = mapping_values(mood_feature_mapping_test=mood_feature_mapping, mood_scores_test=combined_mood_scores)
print(f"Final Mood Vector: {final_mood_vector}")

# Load the song dataset
df = pd.read_csv("/kaggle/input/moodify/updated_moodify_dataset.csv")

features = ["valence", "energy", "danceability", "loudness", "speechiness",
            "acousticness", "instrumentalness", "tempo"]

song_vectors = df[features].values

# Calculate cosine similarities
similarities = cosine_similarity([final_mood_vector], song_vectors)

# Get top 5 recommended songs
top_indices = np.argsort(similarities[0])[::-1][:5]
recommended_songs = df.iloc[top_indices]
print("\nRecommended Songs:")
print(recommended_songs[["uri","song_name"]])