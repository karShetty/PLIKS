import argparse

cfg_MeshReg = {
'use_mean_shape': True,
'lambda_factor': 2,
'sparse' : False,
'run_iters' : 1,
'run_pliks' : True
}

cfg_MeshRegSparse = {
'use_mean_shape': False,
'lambda_factor': 1,
'sparse' : True,
'run_iters' : 3,
'run_pliks' : True
}


class Options(object):
    """Object that handles command line options."""
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        io = self.parser.add_argument_group('io')
        io.add_argument('--checkpoint', default=None, help='Path to checkpoint')
        io.add_argument('--batch_size', type=int, default=1, help='Batch size')
        io.add_argument('--num_workers', type=int, default=0, help='Number of processes used for data loading')
        io.add_argument('--img_dir', default=r"demo/input/*", help='Image directory with extention')
        io.add_argument('--out_dir', default=r"demo/output/", help='Image directory to save output')

        arch = self.parser.add_argument_group('Architecture')
        arch.add_argument('--model', default='MeshRegSparse', choices=['MeshRegSparse', 'MeshReg'])
        arch.add_argument('--img_res', type=int, default=224, help='Image resolution fed to the network')

    def parse_args(self):
        """Parse input arguments."""
        args = self.parser.parse_args()
        if args.model == "MeshRegSparse":
            cfg_model = cfg_MeshRegSparse
        elif args.model == "MeshReg":
            cfg_model = cfg_MeshReg
        else:
            raise ValueError
        args = vars(args)
        args.update(cfg_model)
        return argparse.Namespace(**args)


