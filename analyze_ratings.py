import collections

def analyze_user_ratings(ratings_file_path, top_n=10):
    """
    Analyzes the ratings.dat file to find users with the most positive ratings.

    Args:
        ratings_file_path (str): The path to the ratings.dat file.
        top_n (int): The number of top users to display.

    Returns:
        list: A list of tuples, where each tuple contains (user_id, positive_rating_count),
              sorted by positive_rating_count in descending order.
    """
    user_positive_ratings = collections.defaultdict(int)
    positive_rating_threshold = 4

    try:
        with open(ratings_file_path, 'r', encoding='latin-1') as f: # MovieLens files often use latin-1
            for line in f:
                parts = line.strip().split('::')
                if len(parts) == 4:
                    user_id, _, rating, _ = parts
                    try:
                        rating = int(rating)
                        if rating >= positive_rating_threshold:
                            user_positive_ratings[user_id] += 1
                    except ValueError:
                        print(f"Skipping malformed rating: {rating} in line: {line.strip()}")
                        continue
    except FileNotFoundError:
        print(f"Error: The file {ratings_file_path} was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

    # Sort users by the number of positive ratings in descending order
    sorted_users = sorted(user_positive_ratings.items(), key=lambda item: item[1], reverse=True)

    return sorted_users[:top_n]

if __name__ == "__main__":
    # Assuming the script is in the root and data/raw/ratings.dat is the path
    ratings_path = "data/raw/ratings.dat"
    
    print(f"Analyzing ratings from: {ratings_path}")
    top_users = analyze_user_ratings(ratings_path)

    if top_users:
        print(f"\nTop {len(top_users)} users with the most positive ratings (>= 4):")
        for user_id, count in top_users:
            print(f"User ID: {user_id}, Positive Ratings: {count}")
    else:
        print("No user data to display.") 