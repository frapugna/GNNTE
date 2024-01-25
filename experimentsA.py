from GNNTE import run_GNNTE_experiment

if __name__ == "__main__":
    t1 = {  'project_name' : 'GNNTE',
            'dataset' :"/home/francesco.pugnaloni/wikipedia_tables/training_data/100000_samples",
            'lr' : 0.001,
            'batch_size' : 128,
            'num_epochs' : 50,
            'out_channels' : 300,
            'n_layers' : 3,
            'dropout' : 0,
            'n_sample':'100k',
            'weight_decay': 0,
            'step_size': 10,
            'gamma': 0.75,
            'gnn_type': 'GIN',
            'relu':True
    }

    # l = [t1]
    # for i in range(len(l)):
    #     print(f'Test number: {i+1} / {len(l)}')
    #     run_GNNTE_experiment(**l[i])

    for i in range(5):
        print(f'Test number: {i+1} / 5')
        t1['dataset'] = "/home/francesco.pugnaloni/wikipedia_tables/dataset_test_lukas/"+str(i)
        t1['relu'] = True
        run_GNNTE_experiment(**t1)

        t1['relu'] = False
        run_GNNTE_experiment(**t1)