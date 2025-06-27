import json
import csv

def process_recommendations(input_filename, output_prefix):
    with open(input_filename, 'r') as f:
        data = json.load(f)

    recommendations = data.get("user_aggregated_recommendations", {})
    
    recommendation_types = [
        "precision_recommendations",
        "coverage_recommendations",
        "aggregated_recommendations",
    ]

    file_row_counts = {}
    users_with_less_than_20_recs = {rec_type.replace("_recommendations", ""): [] for rec_type in recommendation_types}

    for rec_type in recommendation_types:
        output_filename = f"{rec_type}.csv"
        rows_written = 0
        
        with open(output_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["user_id", "item_id", "probabilita"])

            for user_id in range(1, 6041):
                user_id_str = str(user_id)
                if user_id_str in recommendations:
                    user_data = recommendations[user_id_str]
                    
                    if rec_type in user_data and user_data[rec_type]:
                        rec_list = user_data[rec_type]
                        
                        # Remove duplicates by keeping the last occurrence, preserving order
                        unique_list_rev = []
                        seen = set()
                        for item in reversed(rec_list):
                            if item not in seen:
                                unique_list_rev.append(item)
                                seen.add(item)
                        unique_list = unique_list_rev[::-1]


                        if len(unique_list) < 20:
                            users_with_less_than_20_recs[rec_type.replace("_recommendations", "")].append(user_id)

                        # The list is already ordered from most to least probable.
                        # We need to assign probabilities based on rank.
                        n = len(unique_list)
                        for i, item_id in enumerate(unique_list):
                            # i is 0 for the most probable (rank 1), n-1 for the least probable (rank n).
                            # probability should be 1 for rank 1, and scaled down for other ranks.
                            
                            # The request is to have probabilities from 1 down to a minimum.
                            # "posizione 1 -> 20/20 = 1"
                            # "posizione 20 -> 0"
                            # This implies a scale. Let's use a linear scale based on rank.
                            # The prompt says "ordinala in ordine inverso rispetto al JSON", and then defines probabilities
                            # from pos 1 (max) to pos 20 (min). The list from JSON is already from max to min.
                            # So we just need to calculate probability.
                            
                            # Let's map index `i` (from 0 to n-1) to a probability.
                            # i=0 is rank 1, i=n-1 is rank n.
                            # prob(i) = (n - 1 - i) / (n - 1) for n > 1
                            if n > 1:
                                probability = (n - 1 - i) / (n - 1)
                            else:
                                probability = 1.0 # Only one item, max probability

                            writer.writerow([user_id, item_id, f"{probability:.4f}"])
                            rows_written += 1
        
        file_row_counts[output_filename] = rows_written

    print("Riepilogo:")
    for filename, count in file_row_counts.items():
        print(f"- File '{filename}' generato con {count} righe.")

    print("\nUtenti con meno di 20 raccomandazioni:")
    for rec_type, user_ids in users_with_less_than_20_recs.items():
        if user_ids:
            print(f"- {rec_type}: {len(user_ids)} utenti")


process_recommendations('llm_aggregated_recommendations.json', '')
