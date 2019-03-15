import torch.optim as optim

def make_optimizer(args, model):
    if args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optim == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == "amsgrad":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    elif args.optim == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    else:
        import adacurv.torch.optim as fisher_optim

        common_kwargs = dict(lr=args.lr,
                             curv_type=args.curv_type,
                             cg_iters=args.cg_iters,
                             cg_residual_tol=args.cg_residual_tol,
                             cg_prev_init_coef=args.cg_prev_init_coef,
                             cg_precondition_empirical=args.cg_precondition_empirical,
                             cg_precondition_regu_coef=args.cg_precondition_regu_coef,
                             cg_precondition_exp=args.cg_precondition_exp,
                             shrinkage_method=args.shrinkage_method,
                             lanczos_amortization=args.lanczos_amortization,
                             lanczos_iters=args.lanczos_iters,
                             batch_size=args.batch_size)
        if args.optim == 'ngd_bd':
            # raise NotImplementedError
            block_diag_params = []
            mods = model.children()
            for m in mods:
                print (m)
                block_diag_params.append({'params': m.parameters()})
            print (block_diag_params)
            optimizer = fisher_optim.NGD_BD(block_diag_params,
                                             **common_kwargs)
        elif args.optim == 'ngd':
            optimizer = fisher_optim.NGD(model.parameters(), **common_kwargs)
        elif args.optim == 'natural_adam':
            optimizer = fisher_optim.NaturalAdam(model.parameters(),
                                                 **common_kwargs,
                                                 betas=(args.beta1, args.beta2),
                                                 assume_locally_linear=args.approx_adaptive)
        elif args.optim == 'natural_adam_bd':
            block_diag_params = []
            mods = model.children()
            for m in mods:
                print (m)
                block_diag_params.append({'params': m.parameters()})
            print (block_diag_params)
            optimizer = fisher_optim.NaturalAdam_BD(block_diag_params,
                                                        **common_kwargs,
                                                        betas=(args.beta1, args.beta2),
                                                        assume_locally_linear=args.approx_adaptive)
        elif args.optim == 'natural_amsgrad':
            optimizer = fisher_optim.NaturalAmsgrad(model.parameters(),
                                                    **common_kwargs,
                                                    betas=(args.beta1, args.beta2),
                                                    assume_locally_linear=args.approx_adaptive)
        elif args.optim == 'natural_adagrad':
            optimizer = fisher_optim.NaturalAdagrad(model.parameters(),
                                                    **common_kwargs,
                                                    assume_locally_linear=args.approx_adaptive)
        else:
            raise NotImplementedError

    print (optimizer)
    return optimizer
