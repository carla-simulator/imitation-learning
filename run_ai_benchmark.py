
import argparse
import logging
import sys

sys.path.append('carla/PythonClient')

from carla.benchmarks.corl import CoRL
from carla.benchmarks.agent import Agent

from carla.tcp import TCPConnectionError
from carla.client import make_carla_client
from agents.imitation.imitation_learning import ImitationLearning
import time

try:
    from carla import carla_server_pb2 as carla_protocol
except ImportError:
    raise RuntimeError('cannot import "carla_server_pb2.py", run the protobuf compiler to generate this file')




if(__name__ == '__main__'):


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

	args = argparser.parse_args()

	log_level = logging.DEBUG if args.debug else logging.INFO
	logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

	logging.info('listening to server %s:%s', args.host, args.port)

	agent = ImitationLearning('Town02')

	while True:
		try:

			with make_carla_client(args.host, args.port) as client:
				corl= CoRL('Town02','TestAI2')

				results = corl.benchmark_agent(agent,client)
				corl.plot_summary_test()
				corl.plot_summary_train()

				break

		except TCPConnectionError as error:
			logging.error(error)
			time.sleep(1)
		except Exception as exception:
			logging.exception(exception)
			sys.exit(1)



	


# DETECT AN ERROR AND WRITE THE COMPLETE SUMMARY ALREADY
