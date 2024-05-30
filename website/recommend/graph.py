# import
import torch
from transformers import BertTokenizer, BertModel


def get_bert_embedding(text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)

    encoding = tokenizer.encode_plus(
        text,
        padding=True,
        truncation=True,
        return_tensors='pt',
        add_special_tokens=True
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        word_embeddings = outputs.last_hidden_state
        sentence_embedding = word_embeddings.mean(dim=1)

    return sentence_embedding.squeeze().cpu()



def add_node_user(data, userID, text, topic):
    # Add user node
    new_index = data['user'].num_nodes
    data['user'].num_nodes += 1
    new_embedding = get_bert_embedding(text)
    if data['user'].x is None:
        data['user'].x = new_embedding.unsqueeze(0)
    else:
        data['user'].x = torch.cat([data['user'].x, new_embedding.unsqueeze(0)], dim=0)
    data['user'].topic = torch.cat([data['user'].topic, torch.tensor([topic], dtype=torch.long)], dim=0)
    data['user'].index = torch.cat([data['user'].index, torch.tensor([userID], dtype=torch.long)], dim=0)

    # Create edges with similar users
    user_indices = (data['user'].topic == topic).nonzero(as_tuple=True)[0]
    new_edges = []
    for idx in user_indices:
        if idx != new_index:
            new_edges.append([new_index, idx.item()])
            new_edges.append([idx.item(), new_index])

    if new_edges:
        new_edges = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
        if data['user', 'similar_U', 'user'].edge_index is None:
            data['user', 'similar_U', 'user'].edge_index = new_edges
        else:
            data['user', 'similar_U', 'user'].edge_index = torch.cat([data['user', 'similar_U', 'user'].edge_index, new_edges], dim=1)
    return data 

def add_node_job(data, jobID, text, topic):
    # Add job node
    new_index = data['job'].num_nodes
    data['job'].num_nodes += 1
    new_embedding = get_bert_embedding(text)
    if data['job'].x is None:
        data['job'].x = new_embedding.unsqueeze(0)
    else:
        data['job'].x = torch.cat([data['job'].x, new_embedding.unsqueeze(0)], dim=0)
    data['job'].topic = torch.cat([data['job'].topic, torch.tensor([topic], dtype=torch.long)], dim=0)
    data['job'].index = torch.cat([data['job'].index, torch.tensor([jobID], dtype=torch.long)], dim=0)

    # Create edges with similar jobs
    job_indices = (data['job'].topic == topic).nonzero(as_tuple=True)[0]
    new_edges = []
    for idx in job_indices:
        if idx != new_index:
            new_edges.append([new_index, idx.item()])
            new_edges.append([idx.item(), new_index])

    if new_edges:
        new_edges = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
        if data['job', 'similar_J', 'job'].edge_index is None:
            data['job', 'similar_J', 'job'].edge_index = new_edges
        else:
            data['job', 'similar_J', 'job'].edge_index = torch.cat([data['job', 'similar_J', 'job'].edge_index, new_edges], dim=1)
    return data

def delete_node_user(data, userID):
    # Find the index of the userID in the 'user' node index tensor
    idx = (data['user'].index == userID).nonzero(as_tuple=True)[0]
    if idx.numel() > 0:
        idx = idx.item()  # Convert tensor to a scalar index

        # Update user attributes by removing the node
        data['user'].x = torch.cat([data['user'].x[:idx], data['user'].x[idx+1:]], dim=0)
        data['user'].topic = torch.cat([data['user'].topic[:idx], data['user'].topic[idx+1:]], dim=0)
        data['user'].index = torch.cat([data['user'].index[:idx], data['user'].index[idx+1:]], dim=0)
        data['user'].num_nodes -= 1

        # Update edges for 'user_applies_job' relationship
        edges_to_remove = (data['user', 'applies', 'job'].edge_index[0] == idx).nonzero(as_tuple=True)[0]
        keep_edges = torch.ones(data['user', 'applies', 'job'].edge_index.size(1), dtype=torch.bool)
        keep_edges[edges_to_remove] = False
        data['user', 'applies', 'job'].edge_index = data['user', 'applies', 'job'].edge_index[:, keep_edges]
        data['user', 'applies', 'job'].edge_label = data['user', 'applies', 'job'].edge_label[keep_edges]

        # Decrement indices in 'user_applies_job'
        data['user', 'applies', 'job'].edge_index[0] -= (data['user', 'applies', 'job'].edge_index[0] > idx).int()

        # Update edges for 'job_rev_applies_user' relationship
        edges_to_remove = (data['job', 'rev_applies', 'user'].edge_index[1] == idx).nonzero(as_tuple=True)[0]
        keep_edges = torch.ones(data['job', 'rev_applies', 'user'].edge_index.size(1), dtype=torch.bool)
        keep_edges[edges_to_remove] = False
        data['job', 'rev_applies', 'user'].edge_index = data['job', 'rev_applies', 'user'].edge_index[:, keep_edges]
        data['job', 'rev_applies', 'user'].edge_label = data['job', 'rev_applies', 'user'].edge_label[keep_edges]

        # Decrement indices in 'job_rev_applies_user'
        data['job', 'rev_applies', 'user'].edge_index[1] -= (data['job', 'rev_applies', 'user'].edge_index[1] > idx).int()

        # Update edges for 'user_similar_user' relationship
        edges_to_remove = ((data['user', 'similar_U', 'user'].edge_index[0] == idx) | (data['user', 'similar_U', 'user'].edge_index[1] == idx)).nonzero(as_tuple=True)[0]
        keep_edges = torch.ones(data['user', 'similar_U', 'user'].edge_index.size(1), dtype=torch.bool)
        keep_edges[edges_to_remove] = False
        data['user', 'similar_U', 'user'].edge_index = data['user', 'similar_U', 'user'].edge_index[:, keep_edges]

        # Decrement indices in 'user_similar_user'
        data['user', 'similar_U', 'user'].edge_index[0] -= (data['user', 'similar_U', 'user'].edge_index[0] > idx).int()
        data['user', 'similar_U', 'user'].edge_index[1] -= (data['user', 'similar_U', 'user'].edge_index[1] > idx).int()
    return data

def delete_node_job(data, jobID):
    # Find the index of the jobID in the 'job' node index tensor
    idx = (data['job'].index == jobID).nonzero(as_tuple=True)[0]
    if idx.numel() > 0:
        idx = idx.item()  # Convert tensor to a scalar index

        # Update job attributes by removing the node
        data['job'].x = torch.cat([data['job'].x[:idx], data['job'].x[idx+1:]], dim=0)
        data['job'].topic = torch.cat([data['job'].topic[:idx], data['job'].topic[idx+1:]], dim=0)
        data['job'].index = torch.cat([data['job'].index[:idx], data['job'].index[idx+1:]], dim=0)
        data['job'].num_nodes -= 1

        # Update edges for 'user_applies_job' relationship
        edges_to_remove = (data['user', 'applies', 'job'].edge_index[1] == idx).nonzero(as_tuple=True)[0]
        keep_edges = torch.ones(data['user', 'applies', 'job'].edge_index.size(1), dtype=torch.bool)
        keep_edges[edges_to_remove] = False
        data['user', 'applies', 'job'].edge_index = data['user', 'applies', 'job'].edge_index[:, keep_edges]
        data['user', 'applies', 'job'].edge_label = data['user', 'applies', 'job'].edge_label[keep_edges]

        # Decrement indices in 'user_applies_job'
        data['user', 'applies', 'job'].edge_index[1] -= (data['user', 'applies', 'job'].edge_index[1] > idx).int()

        # Update edges for 'job_rev_applies_user' relationship
        edges_to_remove = (data['job', 'rev_applies', 'user'].edge_index[0] == idx).nonzero(as_tuple=True)[0]
        keep_edges = torch.ones(data['job', 'rev_applies', 'user'].edge_index.size(1), dtype=torch.bool)
        keep_edges[edges_to_remove] = False
        data['job', 'rev_applies', 'user'].edge_index = data['job', 'rev_applies', 'user'].edge_index[:, keep_edges]
        data['job', 'rev_applies', 'user'].edge_label = data['job', 'rev_applies', 'user'].edge_label[keep_edges]

        # Decrement indices in 'job_rev_applies_user'
        data['job', 'rev_applies', 'user'].edge_index[0] -= (data['job', 'rev_applies', 'user'].edge_index[0] > idx).int()

        # Update edges for 'job_similar_job' relationship
        edges_to_remove = ((data['job', 'similar_J', 'job'].edge_index[0] == idx) | (data['job', 'similar_J', 'job'].edge_index[1] == idx)).nonzero(as_tuple=True)[0]
        keep_edges = torch.ones(data['job', 'similar_J', 'job'].edge_index.size(1), dtype=torch.bool)
        keep_edges[edges_to_remove] = False
        data['job', 'similar_J', 'job'].edge_index = data['job', 'similar_J', 'job'].edge_index[:, keep_edges]

        # Decrement indices in 'job_similar_job'
        data['job', 'similar_J', 'job'].edge_index[0] -= (data['job', 'similar_J', 'job'].edge_index[0] > idx).int()
        data['job', 'similar_J', 'job'].edge_index[1] -= (data['job', 'similar_J', 'job'].edge_index[1] > idx).int()
    return data

def modify_node_user(data, userID, new_text, new_topic):
    # Find the index of the userID in the 'user' node index tensor
    idx = (data['user'].index == userID).nonzero(as_tuple=True)[0]
    if idx.numel() > 0:
        idx = idx.item()  # Convert tensor to a scalar index

        # Update user attributes
        new_embedding = get_bert_embedding(new_text)
        data['user'].x[idx] = new_embedding
        data['user'].topic[idx] = new_topic

        # Update edges for 'user_similar_user' relationship
        # Remove old edges
        edges_to_remove = ((data['user', 'similar_U', 'user'].edge_index[0] == idx) | 
                           (data['user', 'similar_U', 'user'].edge_index[1] == idx)).nonzero(as_tuple=True)[0]
        keep_edges = torch.ones(data['user', 'similar_U', 'user'].edge_index.size(1), dtype=torch.bool)
        keep_edges[edges_to_remove] = False
        data['user', 'similar_U', 'user'].edge_index = data['user', 'similar_U', 'user'].edge_index[:, keep_edges]

        # Create new edges based on the new topic
        user_indices = (data['user'].topic == new_topic).nonzero(as_tuple=True)[0]
        new_edges = []
        for user_idx in user_indices:
            if user_idx.item() != idx:
                new_edges.append([idx, user_idx.item()])
                new_edges.append([user_idx.item(), idx])

        if new_edges:
            new_edges = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
            if data['user', 'similar_U', 'user'].edge_index is None:
                data['user', 'similar_U', 'user'].edge_index = new_edges
            else:
                data['user', 'similar_U', 'user'].edge_index = torch.cat([data['user', 'similar_U', 'user'].edge_index, new_edges], dim=1)
    return data

def modify_node_job(data, jobID, new_text, new_topic):
    # Find the index of the jobID in the 'job' node index tensor
    idx = (data['job'].index == jobID).nonzero(as_tuple=True)
    if idx[0].numel() > 0:
        idx = idx[0].item()  # Convert tensor to a scalar index

        # Update job attributes
        new_embedding = get_bert_embedding(new_text)
        data['job'].x[idx] = new_embedding
        data['job'].topic[idx] = new_topic

        # Update edges for 'job_similar_job' relationship
        # Remove old edges
        edges_to_remove = ((data['job', 'similar_J', 'job'].edge_index[0] == idx) | 
                           (data['job', 'similar_J', 'job'].edge_index[1] == idx)).nonzero(as_tuple=True)[0]
        keep_edges = torch.ones(data['job', 'similar_J', 'job'].edge_index.size(1), dtype=torch.bool)
        keep_edges[edges_to_remove] = False
        data['job', 'similar_J', 'job'].edge_index = data['job', 'similar_J', 'job'].edge_index[:, keep_edges]

        # Create new edges based on the new topic
        job_indices = (data['job'].topic == new_topic).nonzero(as_tuple=True)[0]
        new_edges = []
        for job_idx in job_indices:
            if job_idx.item() != idx:
                new_edges.append([idx, job_idx.item()])
                new_edges.append([job_idx.item(), idx])

        if new_edges:
            new_edges = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
            if data['job', 'similar_J', 'job'].edge_index is None:
                data['job', 'similar_J', 'job'].edge_index = new_edges
            else:
                data['job', 'similar_J', 'job'].edge_index = torch.cat([data['job', 'similar_J', 'job'].edge_index, new_edges], dim=1)

    return data

def add_edge_app(data, userID, jobID):
    # Find the index positions of userID and jobID in their respective node tensors
    user_index = (data['user'].index == userID).nonzero(as_tuple=True)
    job_index = (data['job'].index == jobID).nonzero(as_tuple=True)

    if user_index[0].numel() > 0 and job_index[0].numel() > 0:
        user_index = user_index[0].item()
        job_index = job_index[0].item()

        # Add edge between user and job with edge label 1
        data['user', 'applies', 'job'].edge_index = torch.cat([data['user', 'applies', 'job'].edge_index, torch.tensor([[user_index], [job_index]], dtype=torch.long)], dim=1)
        data['user', 'applies', 'job'].edge_label = torch.cat([data['user', 'applies', 'job'].edge_label, torch.tensor([1], dtype=torch.long)], dim=0)

        # Add reverse edge between job and user (rev_applies) with edge label 1
        data['job', 'rev_applies', 'user'].edge_index = torch.cat([data['job', 'rev_applies', 'user'].edge_index, torch.tensor([[job_index], [user_index]], dtype=torch.long)], dim=1)
        data['job', 'rev_applies', 'user'].edge_label = torch.cat([data['job', 'rev_applies', 'user'].edge_label, torch.tensor([1], dtype=torch.long)], dim=0)

    return data

def delete_app_edge(data, userID, jobID):
    # Find the index positions of userID and jobID in their respective node tensors
    user_index = (data['user'].index == userID).nonzero(as_tuple=True)
    job_index = (data['job'].index == jobID).nonzero(as_tuple=True)

    if user_index[0].numel() > 0 and job_index[0].numel() > 0:
        user_index = user_index[0].item()
        job_index = job_index[0].item()

        # Remove edge between user and job
        user_applies_job_edges = (data['user', 'applies', 'job'].edge_index[0] == user_index) & (data['user', 'applies', 'job'].edge_index[1] == job_index)
        user_applies_job_edges = user_applies_job_edges.nonzero(as_tuple=True)[0]

        if user_applies_job_edges.numel() > 0:
            user_applies_job_edges = user_applies_job_edges.item()

            data['user', 'applies', 'job'].edge_index = torch.cat([data['user', 'applies', 'job'].edge_index[:, :user_applies_job_edges], data['user', 'applies', 'job'].edge_index[:, user_applies_job_edges+1:]], dim=1)
            data['user', 'applies', 'job'].edge_label = torch.cat([data['user', 'applies', 'job'].edge_label[:user_applies_job_edges], data['user', 'applies', 'job'].edge_label[user_applies_job_edges+1:]], dim=0)

        # Remove reverse edge between job and user (rev_applies)
        job_rev_applies_user_edges = (data['job', 'rev_applies', 'user'].edge_index[0] == job_index) & (data['job', 'rev_applies', 'user'].edge_index[1] == user_index)
        job_rev_applies_user_edges = job_rev_applies_user_edges.nonzero(as_tuple=True)[0]

        if job_rev_applies_user_edges.numel() > 0:
            job_rev_applies_user_edges = job_rev_applies_user_edges.item()

            data['job', 'rev_applies', 'user'].edge_index = torch.cat([data['job', 'rev_applies', 'user'].edge_index[:, :job_rev_applies_user_edges], data['job', 'rev_applies', 'user'].edge_index[:, job_rev_applies_user_edges+1:]], dim=1)
            data['job', 'rev_applies', 'user'].edge_label = torch.cat([data['job', 'rev_applies', 'user'].edge_label[:job_rev_applies_user_edges], data['job', 'rev_applies', 'user'].edge_label[job_rev_applies_user_edges+1:]], dim=0)
    return data
