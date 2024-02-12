import pickle
from train_test_valid_split_disjointed import train_test_split
from _script_embed_compare_triple_dataset import *
from itertools import chain
from graph import get_order_of_magnitude

class collection_explorer:
    def __init__(self, key_list: list, embedding_dictionary: dict) -> None:
        """Init method

        Args:
            key_list (list): list of the identifiers of the elements in the set
            embedding_dictionary (dict): dictionary containing the keys and the associated embeddings
        """
        self.key_list = key_list
        self.embedding_dictionary = embedding_dictionary
        self.starting_i = 0
        self.starting_j = 1
        self.i = 0
        self.j = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __call__(self) -> set:
        """Method to obtain a couple from the collection explorer

        Raises:
            StopIteration: thrown when all of the possible couples are explored

        Returns:
            set: couple of keys and their rpedicted overlap ratio
        """
        try:
            left = self.key_list[self.i]
            right = self.key_list[self.j]
        except:
            self.starting_j += 1
            if self.starting_j >= len(self.key_list):
                raise StopIteration
            self.i = self.starting_i
            self.j = self.starting_j
            left = self.key_list[self.i]
            right = self.key_list[self.j]
        
        emb_l = self.embedding_dictionary[left].to(self.device)
        emb_r = self.embedding_dictionary[right].to(self.device)

        try:
            overlap_ratio = max(0, float(F.cosine_similarity(emb_l, emb_r, dim=0)))
        except:
            overlap_ratio = max(0, float(F.cosine_similarity(emb_l, emb_r, dim=1)))
        self.i+=1
        self.j+=1
        return [left, right, overlap_ratio]


def save_checkpoint(outdir: str, train_triples: list, test_triples: list, valid_triples: list) -> None:
    """Save a checkpoint of the train/test/val dataset generated so far

    Args:
        outdir (str): directory where to save the outputs
        train_triples (list): dataframe containing the training data
        test_triples (list): dataframe containing the testing data
        valid_triples (list): dataframe containing the validation data
    """
    train = list(chain.from_iterable(train_triples))
    test = list(chain.from_iterable(test_triples))
    valid = list(chain.from_iterable(valid_triples))

    columns = ['l_id', 'r_id', 'table_overlap']

    pd.DataFrame(train, columns=columns).to_csv(outdir+'/train.csv', index=False)
    pd.DataFrame(test, columns=columns).to_csv(outdir+'/test.csv', index=False)
    pd.DataFrame(valid, columns=columns).to_csv(outdir+'/valid.csv', index=False)

def get_bucket(n: float) -> int:
    """return the bucket which a sample belong to

    Args:
        n (float): number to put into a bucket

    Returns:
        int: the bucket 
    """
    return int(n*100//10)

def generate_triples(embedding_dictionary: str | dict, out_dir: str, train_ratio: float=0.6, 
                     validation_ratio: float=0.2, test_ratio: float=0.2, train_target: int=800000, 
                     test_target: int=100, valid_target: int=100000) -> None:
    """method to generate the candidate train/test/val datasets

    Args:
        embedding_dictionary (str | dict): dictionary containing the embeddings of the tables
        out_dir (str): directory where to save the generated datasets
        train_ratio (float, optional): percentage of tables to put into training set. Defaults to 0.6.
        validation_ratio (float, optional): percentage of tables to put into validation set. Defaults to 0.2.
        test_ratio (float, optional): percentage of tables to put into test set. Defaults to 0.2.
        train_target (int, optional): n of triples desired for the training dataset. Defaults to 800000.
        test_target (int, optional): n of triples desired for the testing dataset. Defaults to 100.
        valid_target (int, optional): n of triples desired for the validation dataset. Defaults to 100000.
    """
    if isinstance(embedding_dictionary, str):
        with open(embedding_dictionary, 'rb') as f:
            embedding_dictionary = pickle.load(f)
    
    train_keys, test_keys, valid_keys = train_test_valid_split(list(embedding_dictionary.keys()))
    
    get_train_triple = collection_explorer(list(train_keys), embedding_dictionary)
    get_test_triple = collection_explorer(list(test_keys), embedding_dictionary)
    get_valid_triple = collection_explorer(list(valid_keys), embedding_dictionary)

    train_triples = [[] for _ in range(11)]
    test_triples = [[] for _ in range(11)]
    valid_triples = [[] for _ in range(11)]

    try:
        checkpoints = 0
        i = 0
        while True:
            if (i % 10000) == 0:
                checkpoints+=1
                print(f'Print saving checkpoint number {checkpoints}')
                save_checkpoint(out_dir, train_triples, test_triples, valid_triples)
            
            print(i)
            train_tuple = get_train_triple()
            if len(train_triples[get_bucket(train_tuple[2])]) < 100000:
                train_triples[get_bucket(train_tuple[2])].append(train_tuple)

            test_tuple = get_test_triple()
            if len(test_triples[get_bucket(test_tuple[2])]) < 100000:
                test_triples[get_bucket(test_tuple[2])].append(test_tuple)

            valid_tuple = get_valid_triple()
            if len(valid_triples[get_bucket(valid_tuple[2])]) < 100000:
                valid_triples[get_bucket(valid_tuple[2])].append(valid_tuple)
            
            i+=1
    except StopIteration:
        print('Finish')
        return


if __name__ == '__main__':
    print('Start')
    generate_triples(embedding_dictionary="/home/francesco.pugnaloni/GNNTE/Datasets/gittables_datasets/embeddings_partial.pkl", 
                     out_dir='')