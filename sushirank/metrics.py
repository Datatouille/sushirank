import numpy as np

def reciprocal_rank(y_true,y_score,k=10):
    '''
    Reciprocal rank at k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        up to k-th item
    Returns
    -------
    reciprocal rank at k
    '''
    order_score = np.argsort(y_score)[::-1].reset_index(drop=True)
    order_true = np.argsort(y_true)[::-1][:k].reset_index(drop=True)
    for i in range(len(order_score)):
        if order_score[i] in order_true:
            return 1/(i+1)

#snippet adapted from @witchapong
#https://gist.github.com/witchapong/fdfdcaf39fee9bfa85311489c72923c1?fbclid=IwAR0Sd7HgLmpJhQnft28O1YKSYpawBLFRhED5RcvKayq5e-bCJJsOIonBmrU
def hit_at_k(y_true,y_score,k_true=10,k_score=10):
    '''
    Hit at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k_true : int
        up to k-th item for ground truth
    k_score : int
        up to k-th item for prediction
    Returns
    -------
    1 if there is hit 0 if there is no hit
    '''
    order_score = np.argsort(y_score)[::-1][:k_score]
    order_true = np.argsort(y_true)[::-1][:k_true]
    return 1 if set(order_score).intersection(set(order_true)) else 0

#snippets for average_precision_score and ndcg_score 
#from https://gist.github.com/mblondel/7337391
def average_precision_score(y_true, y_score, k=10):
    '''
    Average precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        up to k-th item
    Returns
    -------
    average precision @k : float
    '''
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1][:min(n_pos, k)]
    y_true = np.asarray(y_true)[order]

    score = 0
    for i in range(len(y_true)):
        if y_true[i] == pos_label:
            # Compute precision up to document i
            # i.e, percentage of relevant documents up to document i.
            prec = 0
            for j in range(0, i + 1):
                if y_true[j] == pos_label:
                    prec += 1.0
            prec /= (i + 1.0)
            score += prec

    if n_pos == 0:
        return 0

    return score / n_pos

def dcg_score(y_true:np.array, y_score:np.array, k:int=10, gains:str="exponential"):
    '''
    Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        up to k-th item
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    '''
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    '''
    Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        up to k-th item
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    '''
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best

def get_metrics(test_features, k_true=10):
    test_features['value_1'] = test_features.value+1
    test_features['value_bin'] = test_features.value.map(lambda x: 1 if x>0 else 0)
    acc_k = []
    mrr_k = []
    map_k = []
    ndcg_k = []
    for k in tqdm(range(5,21,5)):
        acc = []
        mrr = []
        map = []
        ndcg = []
        for user_id in list(test_features.user_id.unique()):
            d = test_features[test_features.user_id==user_id]
            acc.append(hit_at_k(d['value'],d['pred'],k_true,k))
            mrr.append(reciprocal_rank(d['value'],d['pred'],k))
            map.append(average_precision_score(d['value_bin'],d['pred'],k))
            ndcg.append(ndcg_score(d['value_1'],d['pred'],k))
        acc_k.append(np.mean(acc))
        mrr_k.append(np.mean(mrr))
        map_k.append(np.mean(map))
        ndcg_k.append(np.mean(ndcg))

    print(f'''
    acc@k {[np.round(i,4) for i in acc_k]}
    MRR@k {[np.round(i,4) for i in mrr_k]}
    MAP@k {[np.round(i,4) for i in map_k]}
    nDCG@k {[np.round(i,4) for i in ndcg_k]}
    ''')