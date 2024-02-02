from GNNTE import *

if __name__ == '__main__':
    set_seed()
    df = pd.read_csv("/home/francesco.pugnaloni/GNNTE/Datasets/wikipedia_datasets/1MR/samples.csv")
    #df = pd.read_csv("/home/francesco.pugnaloni/GNNTE/wikipedia_datasets/1000_samples/samples.csv")
    train_data, test_data, valid_data = train_test_valid_split(df)
 
    tables_list = set(list(test_data['s_id']) + list(test_data['r_id']))

    print(f'number of tables: {len(tables_list)}\nNumber of samples: {test_data.shape[0]}')

    with open("/home/francesco.pugnaloni/GNNTE/Datasets/wikipedia_datasets/1MR/full_table_dict_with_id.pkl", "rb") as f:
        table_dict_old = pickle.load(f) 
    # new_table_dict = {}
    # for i in range(test_data.shape[0]):
    #     new_table_dict[test_data.iloc[0][0]] = table_dict_old[test_data.iloc[0][0]]
    #     new_table_dict[test_data.iloc[0][1]] = table_dict_old[test_data.iloc[0][1]]

    new_table_dict = {str(k):table_dict_old[str(k)] for k in tables_list}

    with open("/home/francesco.pugnaloni/GNNTE/Datasets/CoreEvaluationDatasets/100k_valid_wikitables/100k_tables.pkl", "wb") as f1:
        pickle.dump(new_table_dict, f1)

    test_data.to_csv('/home/francesco.pugnaloni/GNNTE/Datasets/CoreEvaluationDatasets/100k_valid_wikitables/100k_samples.csv', index = False)

    test_data