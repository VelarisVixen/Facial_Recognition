# Define optimizer with weight decay
optimizer = AdamW(learning_rate=3e-4, weight_decay=1e-4)

# Compile model with label smoothing
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),  
    optimizer=optimizer,
    metrics=["accuracy"]
)

# Show model summary
model.summary()

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=30,
    class_weight=class_weights_dict,
    callbacks=[early_stopping, lr_scheduler]
)

def final_mood(face_mood, bg_mood, text_mood):
    """
    Combines mood scores from face, background, and text.
    """
    combined = {}
    for mood in face_mood:
        combined[mood] = 0.5 * face_mood[mood] + 0.3 * text_mood[mood] + 0.2 * bg_mood[mood]
    total = sum(combined.values())
    return {mood: (score / total) * 100 for mood, score in combined.items()}


def mapping_values(mood_scores_test, mood_feature_mapping_test):    
    """
    Maps mood scores to a final mood vector.
    """
    scaler = MinMaxScaler()
    mood_feature_matrix = np.array(list(mood_feature_mapping_test.values()))
    mood_feature_matrix = scaler.fit_transform(mood_feature_matrix)

    for i, mood in enumerate(mood_feature_mapping_test.keys()):
        mood_feature_mapping_test[mood] = mood_feature_matrix[i].tolist()

    num_features = 8 
    final_mood_vector = np.zeros(num_features)

    for mood, weight in mood_scores_test.items():
        mood_vector = np.array(mood_feature_mapping_test[mood])
        final_mood_vector += weight * mood_vector 

    final_mood_vector /= sum(mood_scores_test.values()) 
    return final_mood_vector


mood_feature_mapping = {
    "happy":         [0.8, 0.7, 0.8, -5, 0.1, 0.1, 0.0, 140],
    "sad":           [0.2, 0.2, 0.3, -20, 0.05, 0.8, 0.3, 80],
    "energetic":     [0.6, 0.9, 0.9, -3, 0.2, 0.1, 0.0, 160],
    "relaxed":       [0.5, 0.4, 0.4, -15, 0.05, 0.7, 0.5, 90],
    "romantic":      [0.7, 0.5, 0.6, -8, 0.1, 0.5, 0.4, 100],
}