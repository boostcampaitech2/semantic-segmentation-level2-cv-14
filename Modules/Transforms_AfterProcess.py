import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

def DCRF_SubRoutine(parameters):
    img, probs, args = parameters
    c, h, w = probs.shape
    img = np.ascontiguousarray(img)

    U = utils.unary_from_softmax(probs)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=args['test_data_transform_dcrf_gau_sxy'], compat=args['test_data_transform_dcrf_gau_compat'])
    d.addPairwiseBilateral(sxy=args['test_data_transform_dcrf_bi_sxy'], srgb=args['test_data_transform_dcrf_bi_srgb'], rgbim=img, compat=args['test_data_transform_dcrf_bi_compat'])

    Q = d.inference(args['test_data_transform_dcrf_iter'])
    Q = np.array(Q).reshape((c, h, w))
    return Q