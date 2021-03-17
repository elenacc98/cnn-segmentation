from nose.tools import *
import segmentation.metrics

def test_per_class_IoU():
    m = segmentation.metrics.PerClassIoU(num_classes=3, class_to_return=0)
    m.update_state([0,1,1],[0,1,0.99])
    assert_equal(m.result().numpy(),0)