import argparse
from data import *
from compressors import *
from experiments import *
from utils import *
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
import time
from torchtext.datasets import IMDB, AG_NEWS, SogouNews, DBpedia, YelpReviewPolarity, YahooAnswers, AmazonReviewPolarity
import pandas as pd
from functools import reduce

def ngram_hash_comparison(test_data, test_label, train_data, train_label):
    """Will be called when compressor == "hashed_ngrams"."""
    def custom_hash(ngram, n_integers=10e8):
        """Hashs to n-integer hash code."""
        return hash(ngram) % int(n_integers)

    def to_ngrams(text, min_ngram=5, max_ngram=50):
        """Converts the data to a set of its hashed n-grams."""
        return {
            custom_hash(text[i:i+n]) 
            for n in range(min_ngram, max_ngram)
            for i in range(len(text)-n+1)
        }
    
    def to_df(data, label):
        return pd.DataFrame(
            {
                "data": data,
                "label": label,
            }
        )

    def union_all(series):
        return reduce(lambda x, y: x.union(y), series)

    start = time.time()

    # Construct training set
    train_df = to_df(train_data, train_label)
    hashed = train_df["data"].apply(to_ngrams)
    class_groups = hashed.groupby(train_df["label"]).agg(union_all)

    # Construct test set
    test_df = to_df(test_data, test_label)
    test_ngrams = test_df["data"].apply(to_ngrams)

    prediction = test_ngrams.apply(
        lambda x: pd.Series(
            [len(x.intersection(class_ngram_set)) for _, class_ngram_set in class_groups.items()],
            index=class_groups.index
        ).idxmax()
    )

    # Report performance
    accuracy = (prediction == test_df["label"]).mean()
    time_spent = time.time() - start
    print(f"hashed_ngrams,{accuracy},{time_spent}")



def non_neural_knn_exp(compressor_name, test_data, test_label, train_data, train_label, agg_func, dis_func, k, para=True):
    cp = DefaultCompressor(compressor_name)
    knn_exp_ins = KnnExpText(agg_func, cp, dis_func)
    start = time.time()
    if para:
        with Pool(5) as p:
            pred_correct_pair = p.map(partial(knn_exp_ins.combine_dis_acc_single, k, train_data, train_label), test_data, test_label)
        print('accuracy:{}'.format(np.average(np.array(pred_correct_pair, dtype=np.int32)[:,1])))
        # print('accuracy:{}'.format(np.average(np.array(pred_correct_pair, dtype=np.object_)[:, 1])))
    else:
        knn_exp_ins.calc_dis(test_data, train_data=train_data)
        accuracy = knn_exp_ins.calc_acc(k, test_label, train_label=train_label)

    time_spent = time.time() - start

    print(f"{compressor_name},{accuracy},{time_spent}")

def record_distance(compressor_name, test_data, test_portion_name, train_data, agg_func, dis_func, out_dir, para=True):
    print("compressor={}".format(compressor_name))
    numpy_dir = os.path.join(out_dir, compressor_name)
    if not os.path.exists(numpy_dir):
        os.makedirs(numpy_dir)
    out_fn = os.path.join(numpy_dir, test_portion_name)
    cp = DefaultCompressor(compressor_name)
    knn_exp = KnnExpText(agg_func, cp, dis_func)
    start = time.time()
    if para:
        with Pool(6) as p:
            distance_for_selected_test = p.map(partial(knn_exp.calc_dis_single_multi, train_data), test_data)
        np.save(out_fn, np.array(distance_for_selected_test))
        del distance_for_selected_test
    else:
        knn_exp.calc_dis(test_data, train_data=train_data)
        np.save(out_fn, np.array(knn_exp.dis_matrix))
    print("spent: {}".format(time.time() - start))


def non_neurl_knn_exp_given_dis(dis_matrix, k, test_label, train_label):
    knn_exp = KnnExpText(None, None, None)
    _, correct = knn_exp.calc_acc(k, test_label, train_label=train_label, provided_distance_matrix=dis_matrix)
    return correct


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--dataset', default='AG_NEWS')
    parser.add_argument('--num_test', type=int, default=100)
    parser.add_argument('--num_train', type=int, default=100)
    parser.add_argument('--compressor', default='gzip')
    parser.add_argument('--all_test', action='store_true', default=False)
    parser.add_argument('--all_train', action='store_true', default=False)
    parser.add_argument('--para', action='store_true', default=False)
    parser.add_argument('--record', action='store_true', default=False, help='if record the distance into numpy')
    parser.add_argument('--output_dir', default='text_exp_output')
    parser.add_argument('--test_idx_fn', default=None)
    parser.add_argument('--test_idx_start', type=int, default=None)
    parser.add_argument('--test_idx_end', type=int, default=None)
    parser.add_argument('--distance_fn', default=None)
    parser.add_argument('--score', action='store_true', default=False)
    parser.add_argument('--k', default=2, type=int)
    parser.add_argument('--seed', default=1, type=int)
    args = parser.parse_args()

    # Print the args in csv-like format
    args_str = [
        f"{key}={val}"
        for key, val
        in args.__dict__.items()
    ]
    print(",".join(args_str)+",", end="")

    np.random.seed(args.seed)
    # create output dir
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    train_idx_fn = os.path.join(args.output_dir, '{}_train_indicies_{}per_class'.format(args.dataset, args.num_train))
    test_idx_fn = os.path.join(args.output_dir, '{}_test_indicies_{}per_class'.format(args.dataset, args.num_test))
    # all dataset class number
    ds2classes = {'AG_NEWS': 4, 'SogouNews': 5, 'DBpedia': 14, 'YahooAnswers': 10,
                  '20News': 20, 'Ohsumed': 23, 'Ohsumed_single': 23, 'R8': 8, 'R52': 52,
                  'kinnews': 14, 'swahili': 6, 'filipino': 5,  'kirnews': 14}
    # load dataset
    data_dir = os.path.join(args.data_dir, args.dataset)
    if args.dataset not in ['20News', 'Ohsumed', 'Ohsumed_single', 'R8', 'R52','kinnews', 'swahili', 'filipino', 'kirnews']:
        dataset_pair = eval(args.dataset)(root=args.data_dir)
    else:
        if args.dataset == '20News':
            dataset_pair = load_20news()
        elif args.dataset == 'Ohsumed':
            dataset_pair = load_ohsumed(args.data_dir)
        elif args.dataset == 'Ohsumed_single':
            dataset_pair = load_ohsumed_single(args.data_dir)
        elif args.dataset == 'R8' or args.dataset == 'R52':
            dataset_pair = load_r8(args.data_dir)
        elif args.dataset == 'kinnews':
            dataset_pair = load_kinnews()
        elif args.dataset == 'kirnews':
            dataset_pair = load_kirnews()
        elif args.dataset == 'swahili':
            dataset_pair = load_swahili()
        elif args.dataset == 'filipino':
            dataset_pair = load_filipino()
    num_classes = ds2classes[args.dataset]
    # choose indices
    if not args.all_test:
        # pick certain number per class
        if args.test_idx_fn is not None:
            try:
                test_idx = np.load(args.test_idx_fn)
                test_data, test_labels = read_torch_text_labels(dataset_pair[1], test_idx)
            except FileNotFoundError:
                print("No generated indices file for test set provided")
        elif args.test_idx_start is not None:
            test_idx = list(range(args.test_idx_start, args.test_idx_end))
            test_data, test_labels = read_torch_text_labels(dataset_pair[1], test_idx)
        else:
            test_data, test_labels = pick_n_sample_from_each_class_given_dataset(dataset_pair[1], args.num_test, test_idx_fn)
    else:
        train_pair, test_pair = dataset_pair[0], dataset_pair[1]
        test_data, test_labels = read_torch_text_labels(test_pair, range(len(test_pair)))
    if not args.all_train:
        if args.test_idx_fn is not None or args.test_idx_start is not None:
            train_idx = np.load(train_idx_fn+'.npy')
            train_data, train_labels = read_torch_text_labels(dataset_pair[0], train_idx)
        else:
            train_data, train_labels = pick_n_sample_from_each_class_given_dataset(dataset_pair[0], args.num_train,
                                                                               train_idx_fn)
    else:
        train_pair, test_pair = dataset_pair[0], dataset_pair[1]
        train_data, train_labels = read_torch_text_labels(train_pair, range(len(train_pair)))
    if not args.record:
        if args.compressor == "hashed_ngram":
            ngram_hash_comparison(test_data, test_labels, train_data, train_labels)
        else:
            non_neural_knn_exp(args.compressor, test_data, test_labels, train_data, train_labels, agg_by_concat_space, NCD, args.k, para=args.para)
    else:
        if args.test_idx_fn is None:
            output_rel_fn = 'test_dis_idx_from_{}_to_{}'.format(args.test_idx_start, args.test_idx_end)
        else:
            output_rel_fn = args.test_idx_fn.split('/')[-1]
        if not args.score:
            for i in range(0, len(test_data), 100):
                print('from {} to {}'.format(args.test_idx_start+i, args.test_idx_start+i+100))
                output_rel_fn = 'test_dis_idx_from_{}_to_{}'.format(args.test_idx_start+i, args.test_idx_start+i+100)
                output_dir = os.path.join(args.output_dir, os.path.join('distance', args.dataset))
                record_distance(args.compressor, np.array(test_data)[i:i+100], output_rel_fn, train_data, agg_by_concat_space, NCD, output_dir, para=args.para)
        else:
            if os.path.isdir(args.distance_fn):
                all_correct = 0
                total_num = 0
                for fn in tqdm(os.listdir(args.distance_fn)):
                    if fn.endswith('.npy'):
                        dis_matrix = np.load(os.path.join(args.distance_fn, fn))
                        start_idx, end_idx = int(fn.split('.')[0].split('_')[-3]), int(fn.split('.')[0].split('_')[-1])
                        sub_test_labels = test_labels[start_idx:end_idx] #assume all_test=True, all_train=True
                        correct = non_neurl_knn_exp_given_dis(dis_matrix, args.k, sub_test_labels, train_labels)
                        all_correct += sum(correct)
                        total_num += len(correct)
                        del dis_matrix
                print("Altogether Accuracy is: {}".format(all_correct/total_num))
            else:
                dis_matrix = np.load(args.distance_fn)
                non_neurl_knn_exp_given_dis(dis_matrix, 3, test_labels, train_labels)



