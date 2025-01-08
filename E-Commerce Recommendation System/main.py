import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

class RecommendationSystem:
    def __init__(self):
        self.ratings_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.mean_user_rating = None
        self.ratings_diff = None

    def load_and_prepare_data(self, file_path):
        """
        Load and prepare the MovieLens dataset
        """
        # Read the ratings file
        df = pd.read_csv(file_path)
        
        # Create user-item matrix
        self.ratings_matrix = df.pivot(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)
        
        return df
    
    def calculate_similarity_matrices(self):
        """
        Calculate user-user and item-item similarity matrices
        """
        # User similarity matrix
        self.user_similarity = cosine_similarity(self.ratings_matrix)
        
        # Item similarity matrix
        self.item_similarity = cosine_similarity(self.ratings_matrix.T)
        
        return self.user_similarity, self.item_similarity
    
    def get_user_based_recommendations(self, user_id, n_recommendations=5):
        """
        Generate user-based collaborative filtering recommendations
        """
        if user_id not in self.ratings_matrix.index:
            return "User not found in the dataset"
        
        # Get user's ratings
        user_ratings = self.ratings_matrix.loc[user_id]
        
        # Get similarity scores
        user_sim_scores = pd.Series(
            self.user_similarity[self.ratings_matrix.index.get_loc(user_id)],
            index=self.ratings_matrix.index
        )
        
        # Find similar users
        similar_users = user_sim_scores.sort_values(ascending=False)[1:11]
        
        # Get items user hasn't rated
        unrated_items = user_ratings[user_ratings == 0].index
        
        # Calculate predicted ratings
        recommendations = {}
        for item in unrated_items:
            item_ratings = self.ratings_matrix[item]
            weighted_sum = sum(similar_users[user] * item_ratings[similar_users.index[i]]
                             for i, user in enumerate(similar_users.index))
            recommendations[item] = weighted_sum / sum(similar_users)
        
        # Sort and return top N recommendations
        recommendations = pd.Series(recommendations).sort_values(ascending=False)
        return recommendations.head(n_recommendations)
    
    def get_item_based_recommendations(self, user_id, n_recommendations=5):
        """
        Generate item-based collaborative filtering recommendations
        """
        if user_id not in self.ratings_matrix.index:
            return "User not found in the dataset"
        
        # Get user's ratings
        user_ratings = self.ratings_matrix.loc[user_id]
        
        # Get items user hasn't rated
        unrated_items = user_ratings[user_ratings == 0].index
        
        # Calculate predicted ratings
        recommendations = {}
        for item in unrated_items:
            item_sim_scores = pd.Series(
                self.item_similarity[self.ratings_matrix.columns.get_loc(item)],
                index=self.ratings_matrix.columns
            )
            # Get similar items that user has rated
            similar_items = item_sim_scores[user_ratings > 0]
            
            if len(similar_items) > 0:
                weighted_sum = sum(similar_items[i] * user_ratings[i] 
                                 for i in similar_items.index)
                recommendations[item] = weighted_sum / sum(similar_items)
        
        # Sort and return top N recommendations
        recommendations = pd.Series(recommendations).sort_values(ascending=False)
        return recommendations.head(n_recommendations)
    
    def evaluate_recommendations(self, test_size=0.2):
        """
        Evaluate the recommendation system using RMSE
        """
        # Create train-test split
        ratings = self.ratings_matrix.values
        mask = ratings > 0
        ratings_train, ratings_test = train_test_split(
            ratings[mask],
            test_size=test_size,
            random_state=42
        )
        
        # Calculate RMSE
        predictions = []
        actuals = []
        
        for user_id in range(len(self.ratings_matrix)):
            user_predictions = self.get_user_based_recommendations(
                self.ratings_matrix.index[user_id],
                n_recommendations=10
            )
            
            if isinstance(user_predictions, pd.Series):
                for item_id in user_predictions.index:
                    if self.ratings_matrix.iloc[user_id][item_id] > 0:
                        predictions.append(user_predictions[item_id])
                        actuals.append(self.ratings_matrix.iloc[user_id][item_id])
        
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        return rmse
    
    def visualize_ratings_distribution(self, original_df):
        """
        Visualize the distribution of ratings
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(data=original_df, x='rating', bins=10)
        plt.title('Distribution of Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.show()
        
    def visualize_user_activity(self, original_df):
        """
        Visualize user rating activity
        """
        user_activity = original_df['userId'].value_counts()
        plt.figure(figsize=(12, 6))
        sns.histplot(data=user_activity, bins=50)
        plt.title('Distribution of User Activity')
        plt.xlabel('Number of Ratings per User')
        plt.ylabel('Count')
        plt.show()

# Example usage
def main():
    # Initialize the recommendation system
    rec_system = RecommendationSystem()
    
    # Load and prepare data with your specific file path
    df = rec_system.load_and_prepare_data(r"C:\Users\Ayaan\OneDrive\Desktop\E-Commerce Recommendation System\ratings.csv")
    
    # Calculate similarity matrices
    rec_system.calculate_similarity_matrices()
    
    # Get recommendations for a specific user
    user_id = 1
    print("\nUser-based recommendations for user", user_id)
    print(rec_system.get_user_based_recommendations(user_id))
    
    print("\nItem-based recommendations for user", user_id)
    print(rec_system.get_item_based_recommendations(user_id))
    
    # Evaluate the system
    rmse = rec_system.evaluate_recommendations()
    print("\nSystem RMSE:", rmse)
    
    # Visualize the data
    rec_system.visualize_ratings_distribution(df)
    rec_system.visualize_user_activity(df)

if __name__ == "__main__":
    main()