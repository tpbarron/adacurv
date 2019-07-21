import torch_version
import tf_version
import arguments

from adacurv.utils import logger

if __name__ == '__main__':
    args = arguments.get_args()
    torch_version.run(args)
    tf_version.run(args)

    logpath1 = "/tmp/adacurv/torch/adam_adaptive_ngd_cg/log.pkl"
    logpath2 = "/tmp/adacurv/tf/adam_adaptive_ngd_cg/log.pkl"
    logger.find_where_not_equal(logpath1, logpath2)
