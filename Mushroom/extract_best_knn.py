import pandas as pd


def select_top_model(input_csv, output_csv, model_rank=1):
    """
    Select the top N performing models based on average accuracy.

    Parameters:
    - input_csv: Path to the input CSV file
    - output_csv: Path to save the output CSV file
    - model_rank: Rank of the model to select (1 = best, 2 = second best, etc.)

    Returns:
    - DataFrame with entries for the selected model
    """
    # Read the input CSV
    df = pd.read_csv(input_csv)

    # Group by Model and calculate mean accuracy
    model_avg_accuracy = df.groupby('Model')['Accuracy'].mean().reset_index()

    # Sort models by accuracy in descending order
    sorted_models = model_avg_accuracy.sort_values('Accuracy', ascending=False)

    # Validate the requested model rank
    if model_rank < 1 or model_rank > len(sorted_models):
        raise ValueError(f"Invalid model rank. Please choose a rank between 1 and {len(sorted_models)}")

    # Select the model at the specified rank (adjusting for 0-based indexing)
    selected_model = sorted_models.iloc[model_rank - 1]['Model']

    # Filter the original dataframe to keep only entries of the selected model
    selected_model_entries = df[df['Model'] == selected_model]

    # Save the filtered entries to a new CSV
    selected_model_entries.to_csv(output_csv, index=False)

    print(f"Selected model (rank {model_rank}): {selected_model}")
    print(f"Average accuracy: {sorted_models.iloc[model_rank - 1]['Accuracy']:.4f}")

    return selected_model_entries

# Example usage
select_top_model('knn_base_results.csv', 'top_knn_results.csv', model_rank=4)