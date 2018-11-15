import argparse
import logging

from carla.driving_benchmark import run_driving_benchmark
from carla.driving_benchmark.experiment_suites import LongitudinalControl2018

from agents.imitation.imitation_learning import ImitationLearning

try:
    from carla import carla_server_pb2 as carla_protocol
except ImportError:
    raise RuntimeError(
        'cannot import "carla_server_pb2.py", run the protobuf compiler to generate this file')

if (__name__ == '__main__'):
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-c', '--city-name',
        metavar='C',
        default='Town01',
        help='The town that is going to be used on benchmark'
             + '(needs to match active town in server, options: Town01 or Town02)')
    argparser.add_argument(
        '-n', '--log_name',
        metavar='T',
        default='test',
        help='The name of the log file to be created by the scripts'
    )

    argparser.add_argument(
        '--avoid-stopping',
        default=True,
        action='store_false',
        help=' Uses the speed prediction branch to avoid unwanted agent stops'
    )
    argparser.add_argument(
        '--continue-experiment',
        action='store_true',
        help='If you want to continue the experiment with the given log name'
    )

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    agent = ImitationLearning(args.city_name, args.avoid_stopping)

    corl = LongitudinalControl2018(args.city_name)

    # Now actually run the driving_benchmark
    run_driving_benchmark(agent, corl, args.city_name,
                          args.log_name, args.continue_experiment,
                          args.host, args.port)
