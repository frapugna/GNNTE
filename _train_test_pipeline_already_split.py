from GNNTE import *

def train_test_pipeline_split(train_file: str, test_file: str, valid_file: str, graph_file: str, model_file: str, hidden_channels: int, num_layers: int,
                        batch_size: int=64, lr: float=0.01, dropout: float=0, initial_embedding_method: str='fasttext',
                        num_epochs: int=100, weight_decay: float=0.0001, act: str='relu', log_wandb: bool=False,
                        step_size: int=15, gamma: float=0.1, gnn_type: str='GIN', compute_bins_stats: bool=False, relu: bool=False, loss_type: str='MSE') -> GNNTE:
    """This function performs the full train-validate-test pipeline

    Args:
        train_file (str): path to the train triple file
        test_file (str): path to the test triple file
        valid_file (str): path to the validation triple file
        graph_file (str): path to the graph file
        model_file (str): path to the backup file for the model
        test_predictions_file (str): path to the directory containing the logs of the predictions
        hidden_channels (int): size of the generated embeddings
        num_layers (int): number of layers of the network, every embedding will be generated using using his neighbours at distance num_layers
        ttv_ratio (set, optional): a triple that tells the function how to split the dataset (TRAIN, TEST, VALIDATE). Defaults to (0.8,0.1,0.1).
        batch_size (int, optional): number of elements to put in the training batches. Defaults to 64.
        lr (float, optional): learning rate. Defaults to 0.01.
        dropout (float, optional): dropout probability. Defaults to 0.
        num_epochs (int, optional): number of training epochs. Defaults to 100.
        weight_decay (float, optional): NA. Defaults to 0.
        act (str, optional): the activation function used between the layers. Defaults to 'relu'.
        log_wandb (bool, optional): if True all the outputs of the experiments will be logged to wandb, an open session is necessary to avoid errors. Defaults to False.
        step_size (int, optional): number of epochs to wait to update the learning rate. Defaults to 5.
        gamma (float, optional): reduction factor of the learning rate. Defaults to 0.1
        gnn_type (str): the gnn to use. Defaults to 'GIN'
        compute_bins_stats (bool): set to true to compute stats about intervals of table overlaps. Default to False
        relu (bool, optional): if set to Tre a relu layer will be added at the end of the network, it will prevent negative cosine similarities between the embeddings. Defaults to False.

    Returns:
        GNNTE: the trained network
    """
    set_seed()
    # Load datasets
    print('Loading datasets, it may take some time')
    train_triples = pd.read_csv(train_file)[['r_id','s_id','a%']]
    test_triples = pd.read_csv(test_file)[['r_id','s_id','a%']]
    valid_triples = pd.read_csv(valid_file)[['r_id','s_id','a%']]

    with open(graph_file, 'rb') as f:
        graphs = pickle.load(f)

    train_dataset = GraphTriplesDataset(train_triples, graphs)
    test_dataset = GraphTriplesDataset(test_triples, graphs)
    valid_dataset = GraphTriplesDataset(valid_triples, graphs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GNNTE(hidden_channels, num_layers, dropout=dropout, act=act, gnn_type=gnn_type, relu=relu, initial_embedding_method=initial_embedding_method)
    start = time.time()

    print('Training starts')

    model = train(model, train_dataset, valid_dataset, batch_size, lr, num_epochs, device, model_file, 
                  weight_decay=weight_decay, log_wandb=log_wandb, step_size=step_size, gamma=gamma, loss_type=loss_type)
    end = time.time()
    t_train=end-start
    print(f'T_train: {t_train}s')

    start = time.time()
    execution_insights_test = test(model, test_dataset, batch_size) 
    mse = execution_insights_test['mse'] 
    end = time.time()
    t_test = end-start
    print(f'T_test: {t_test}s')
    print(f'MSE: {mse}')

    print('Generating tests for bags')

    execution_insights = {'test':execution_insights_test}
    
    if compute_bins_stats:
        execution_insights_bins = test_bins(model, test_dataset, batch_size)
        execution_insights['bins'] = execution_insights_bins
    if log_wandb:
        wandb.run.summary["T_train"] = t_train 
        wandb.run.summary["T_test"] = t_test
        wandb.run.summary["MSE"] = mse
        wandb.run.summary["insights"] = execution_insights

    execution_insights['test']['model'] = model

    return execution_insights

def run_GNNTE_experiment_split(project_name: str, train_file: str, test_file: str, valid_file: str, graph_file: str, checkpoint: str, lr: float, batch_size: int,
                         num_epochs: int, out_channels: int, n_layers: int, dropout: float, weight_decay: float, step_size: int, gamma: float, 
                         gnn_type: str, initial_embedding_method: str='fasttext', log_wandb=False, relu: bool=False, loss_type: str='MSE') -> None:
    """Utility function to run experiments that will be logged in wandb

    Args:
        project_name (str): name of the project in wandb
        dataset (str): directory containing the stuff necessary to build the dataset
        lr (float): learning rate
        batch_size (int): size of the training batches
        num_epochs (int): number of training epochs
        out_channels (int): size of the embeddings
        n_layers (int): number of layers
        dropout (float): dropout probability
        weight_decay (float): an L2 penalty
        step_size (int): number of epochs to wait to update the learning rate
        gamma (float): reduction factor of the learning rate
        gnn_type (str): the gnn to use, accepted 'GIN' and 'GAT'
        relu (bool, optional): if set to Tre a relu layer will be added at the end of the network, it will prevent negative cosine similarities between the embeddings

    """
    #name = f"SPLIT_128_{gnn_type}_{batch_size}_{lr}_{num_epochs}_{out_channels}_{n_layers}_{dropout}_{weight_decay}_{step_size}_{gamma}"
    name = checkpoint
    if relu:
        name += "_relu"
    else:
        name += "_no_relu"
    if log_wandb:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=project_name,
            name=name,
            # track hyperparameters and run metadata
            config={
                "gnn_type":gnn_type,
                #"dataset": dataset,
                "batch_size": batch_size,
                "learning_rate": lr,
                "num_epochs": num_epochs,
                "out_channels": out_channels,
                "n_layers": n_layers,
                "dropout": dropout,
                "step_size": step_size,
                "gamma": gamma,
                "relu" : relu
            }
        )
    #checkpoint = dataset+f"/{name}.pth"
    print(f'Starting training with {num_epochs} epochs')
    train_test_pipeline_split(train_file=train_file, test_file=test_file, valid_file=valid_file, graph_file=graph_file, model_file=checkpoint, hidden_channels=out_channels, num_layers=n_layers, 
                        num_epochs=num_epochs, batch_size=batch_size, lr=lr, dropout=dropout, log_wandb=log_wandb,
                        weight_decay=weight_decay, step_size=step_size, gamma=gamma, gnn_type=gnn_type, compute_bins_stats=True,
                        relu=relu, initial_embedding_method=initial_embedding_method, loss_type=loss_type)
        
    wandb.finish()

if __name__ == "__main__":
    name = 'GNNTE'
    # train_file = '/home/francesco.pugnaloni/GNNTE/Datasets/CoreEvaluationDatasets/1M_wikitables_disjointed/819716_13583_12918/train.csv'
    # test_file = '/home/francesco.pugnaloni/GNNTE/Datasets/CoreEvaluationDatasets/1M_wikitables_disjointed/819716_13583_12918/test.csv'
    # valid_file = '/home/francesco.pugnaloni/GNNTE/Datasets/CoreEvaluationDatasets/1M_wikitables_disjointed/819716_13583_12918/valid.csv'
    #train_file = '/home/francesco.pugnaloni/GNNTE/Datasets/wikipedia_datasets/1000_samples/train.csv'
    #test_file = '/home/francesco.pugnaloni/GNNTE/Datasets/wikipedia_datasets/1000_samples/test.csv'
    #valid_file = '/home/francesco.pugnaloni/GNNTE/Datasets/wikipedia_datasets/1000_samples/valid.csv'
    # train_file = '/home/francesco.pugnaloni/GNNTE/Datasets/2_WikiTables/1M_wikitables_disjointed/train_test_val_datasets/train.csv'
    # test_file = '/home/francesco.pugnaloni/GNNTE/Datasets/2_WikiTables/1M_wikitables_disjointed/train_test_val_datasets/test.csv'
    # valid_file = '/home/francesco.pugnaloni/GNNTE/Datasets/2_WikiTables/1M_wikitables_disjointed/train_test_val_datasets/valid.csv'

    train_file = '/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/balanced_datasets/train.csv'
    test_file = '/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/balanced_datasets/test.csv'
    valid_file = '/home/francesco.pugnaloni/GNNTE/Datasets/1_Gittables/balanced_datasets/valid.csv'

    graph_file = '/home/francesco.pugnaloni/GNNTE/Datasets/2_WikiTables/1M_wikitables_disjointed/graphs_sha256_null_not_0_no_merge_nodes.pkl'
    #graph_file = '/home/francesco.pugnaloni/GNNTE/Datasets/wikipedia_datasets/1000_samples/graphs.pkl'

    
    #checkpoint = '/home/francesco.pugnaloni/GNNTE/tmp/model_test_1k.pth'
    lr = 0.001
    batch_size = 128
    num_epochs = 50
    #num_epochs = 10
    out_channels = 300
    n_layers = 3
    dropout_prob = 0
    weight_decay = 0.0001
    step_size = 15
    gamma = 0.1
    GNN_type = 'GraphSAGE'
    checkpoint = f'/home/francesco.pugnaloni/GNNTE/models/gittables/wikidata_19-03-24_{GNN_type}_50_ep_max_1000_tokens_init_emb_sha256_3_layers_null_not_0_no_merge_nodes.pth'
    log_wandb = True
    initial_embedding_method = 'sha256'
    #dataset = "/home/francesco.pugnaloni/GNNTE/Datasets/wikipedia_datasets/1000_samples"

    #graphs_path = dataset+"/graphs.pkl"
    #checkpoint = dataset+f"/{name}.pth"
    

    run_GNNTE_experiment_split(project_name=name, train_file=train_file, test_file=test_file, valid_file=valid_file, graph_file=graph_file, 
                               checkpoint=checkpoint, lr=lr, batch_size=batch_size, num_epochs=num_epochs, out_channels=out_channels, n_layers=n_layers, 
                               dropout=dropout_prob, weight_decay=weight_decay, step_size=step_size, gamma=gamma, gnn_type=GNN_type,
                               log_wandb=log_wandb, initial_embedding_method=initial_embedding_method
                               )

    print('Finish')