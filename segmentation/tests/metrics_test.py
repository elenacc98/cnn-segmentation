from segmentation.metrics import count_tp

def test_tp():
    true_label = [[1,1,1],[1,0,0],[1,0,2]]
    pred_label = [[1,0,1],[2,1,1],[1,1,1]]
    tp = count_tp(0,true_label, pred_label)
    assert tp == 2