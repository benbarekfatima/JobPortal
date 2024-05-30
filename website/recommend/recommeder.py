
import torch


def recommend_top_k(user_id, data, model, k=10):
    # Find the index of the user in the 'user' node index tensor
    user_index = (data['user'].index == user_id).nonzero(as_tuple=True)[0]
    if user_index.numel() == 0:
        raise ValueError(f"userID {user_id} not found in user nodes")

    user_index = user_index.item()

    # Encode user and job features using the model
    encoded_data = model.encoder(data.x_dict, data.edge_index_dict)

    # Find jobs that the user has interacted with
    user_interacted_jobs = data['user', 'applies', 'job'].edge_index[1][data['user', 'applies', 'job'].edge_index[0] == user_index]

    # Get all job indices
    all_job_indices = torch.arange(data['job'].num_nodes)

    # Remove jobs that the user has interacted with
    candidate_job_indices = all_job_indices[~torch.isin(all_job_indices, user_interacted_jobs)]

    # Adjust k if it exceeds the number of candidate jobs
    k = min(k, len(candidate_job_indices))

    # Create a tensor with the same length as candidate_job_indices, filled with user_index
    user_index_tensor = torch.full((len(candidate_job_indices),), user_index, dtype=torch.long)

    # Calculate recommendation scores using model's decoder
    recommendation_scores = model.decoder(encoded_data, (user_index_tensor, candidate_job_indices))

    # Extract top-k job indices with highest recommendation scores
    top_k_values, top_k_indices = torch.topk(recommendation_scores, k, largest=True, sorted=True)
    top_k_job_indices = candidate_job_indices[top_k_indices]

    # Map job indices to their original IDs in data['job'].index
    top_k_job_ids = data['job'].index[top_k_job_indices].tolist()

    return top_k_job_ids
