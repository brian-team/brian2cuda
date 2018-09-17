import argparse


def parse_arguments(N, single_precision, with_monitors, num_blocks, use_atomics,
                    bundle_mode, name):

    parser = argparse.ArgumentParser(description='Run brian2cuda example')

    parser.add_argument('--num_neurons', '-N', nargs=1, type=str, default=None,
                        help=("Number of neurons to run the network with"))

    parser.add_argument('--single-precision', '-p', nargs=1, type=bool,
                        choices=[True, False], default=None,
                        help=("Use single precision floating point numbers."))

    parser.add_argument('--with-monitors', '-m', nargs=1, type=bool,
                        choices=[True, False], default=None,
                        help=("Use brian2 Monitors to record activity"))

    parser.add_argument('--use_atomics', '-a', nargs=1, type=str, default=None,
                        choices=['True', 'False'],
                        help=("Use atomic operations for parallelisatoin."))

    parser.add_argument('--bundle-mode', '-bu', nargs=1, type=bool,
                        default=None, choices=[True, False],
                        help=("Push synapse bundle for synaptic event "
                              "propagation."))

    parser.add_argument('--num_blocks', '-bl', nargs=1, type=str, default=None,
                        help=("Number of post blocks in connectivity matrix "
                              "structure."))

    parser.add_argument('--suffix', '-s', nargs=1, type=str, default=None,
                        help=("Name suffix for results."))

    args = parser.parse_args()

    if args.num_neurons is not None:
        try:
            new_N = int(args.num_neurons[0])
        except ValueError:
            # exponential, e.g. 1e5
            new_N = int(float(args.num_neurons[0]))
        if N != new_N:
            N = new_N
            print "Setting num_neurons from command line to", N

    if args.single_precision is not None:
        if single_precision != args.single_precision[0]:
            single_precision = args.single_precision[0]
            print "Setting single_precision from command line to", \
                    single_precision

    if args.with_monitors is not None:
        if with_monitors != args.with_monitors[0]:
            with_monitors = args.with_monitors[0]
            print "Setting with_monitors from command line to", with_monitors

    if args.use_atomics is not None:
        if use_atomics != args.use_atomics[0]:
            use_atomics = args.use_atomics[0]
            print "Setting use_atomics from command line to", use_atomics

    if args.bundle_mode is not None:
        if bundle_mode != args.bundle_mode[0]:
            bundle_mode = args.bundle_mode[0]
            print "Setting bundle_mode from command line to", bundle_mode

    if args.num_blocks is not None:
        new_num_blocks = int(args.num_blocks[0])
        if num_blocks != new_num_blocks:
            num_blocks = new_num_blocks
            print "Setting num_blocks from command line to", num_blocks

    name = name + '_single-precision' if single_precision else name
    name = name + '_no-monitors' if not with_monitors else name
    name = name + '_no-atomics' if not use_atomics else name
    name = name + '_no-bundles' if not bundle_mode else name
    name = name + '_num-blocks-{}'.format(num_blocks) if num_blocks is not None else name
    name += '_N-' + str(N)
    if args.suffix is not None:
        name += '_' + args.suffix[0]

    return (N, single_precision, with_monitors, num_blocks, use_atomics,
            bundle_mode, name)
