import numpy as np
import heapq as hq
import pickle

def minimun_index_list(list, k):
    return hq.nsmallest(k, range(len(list)), list.__getitem__)

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def main():
    k=10
    # /--- Test ---
    Array = np.random.randint(0, 255, 50)
    list = Array.tolist()
    print(list)
    # --- Test ---/

    actual_apk = [26]                               # GroundTruth
    predicted_apk = minimun_index_list(list, k)     # 10 minimum indexes
    print(predicted_apk)
    score_apk = apk(actual_apk, predicted_apk)
    print(score_apk)

    gt = open("gt_corresps.pkl","rb")
    actual_mapk = pickle.load(gt)                    # GT pickle
    print(actual_mapk)
    predicted_mapk=[[1,1],[10,10]]                   # List of lists with 10 minimum indexes by image
    score_mapk = mapk(actual_mapk,predicted_mapk)
    print(score_mapk)

main()