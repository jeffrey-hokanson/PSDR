#! /usr/bin/env python

""" A multiprocessing like Pool using Redis as a backend to communicate with (remote) workers

Although celery and http://flask.pocoo.org/snippets/73/ provide similar functionality,
the HPC context requires a different approach.  Celery and other similar task queues
think of a queue of jobs that get distributed to anonyous consumers.  Here, instead,
we view information about the workers as important for scheduling. The main example
is using existing function values to provide guidance on where to sample when a new worker
becomes avalible.


WARNING
-------
This approach for running functions is powerful: it can run across multiple machines
but is also the most fragile.


FAQ
---
* If you see issues related to _dill, make sure that all computers are running the same version 
  of dill.


"""

from redis import Redis
import time
import uuid
import logging
from multiprocessing import Process, Queue
from copy import deepcopy
from dill import loads, dumps
import logging, sys, os
logging.basicConfig(level = logging.INFO)
import signal
import thread, threading
from threading import Thread, Event
import base64, hashlib
import contextlib

__all__ = ['RedisPool', 'RedisJob', 'RedisWorker']

# For better automatic names we optionally use petname
try:
	import petname
	PETNAME = True
except ImportError:
	PETNAME = False


class WatchDog(Thread):
	"""Class used with an event to create a repeating timer

	Example:
		
		def fun(*args, **kwargs):
			print "I run repeatedly"

		event = Event()
		t = WatchDog(heart, fun, args = (), kwargs = {}
		t.start()
		time.sleep(20)
		event.set()	# This stops the repeating timer part

		
	Based on an example from stackoverflow:
		https://stackoverflow.com/questions/12435211/python-threading-timer-repeat-function-every-n-seconds
	"""
	def __init__(self, event, fun, args = (), kwargs = {}, timestep = 0.1):
		Thread.__init__(self)
		self.stopped = event
		self.fun = fun
		self.args = args
		self.kwargs = {}
		self.timestep = timestep

	def run(self):
		while not self.stopped.wait(self.timestep):
			self.fun(*self.args, **self.kwargs)


class RedisJob(object):
	""" A job based on information in redis

	This class saves and loads functions to redis with the help of dill.


	Parameters
	----------
	redis_pool: RedisPool
		Object referencing the pool this job is attached to
	job_name: string or None
		Name of job
	f: function
		Function to call
	args: list
		Arguments to f
	kwargs: dict
		Keyword arguments for f	

	"""
	def __init__(self, redis_pool, job_name = None, f = None, args = None, kwargs = None, keep_data = True):
		self._owner = False
		self.redis_pool = redis_pool

		self.redis = redis_pool.redis
		self.pool_name = redis_pool.name
		self.job_name = job_name
		self.keep_data = keep_data

		if f is not None:
			# If we are the copy to which f has actually been provided, this is the owner
			self._owner = True

			# Provide no arguments by default
			if args is None: args = ()
			if kwargs is None: kwargs = {}
			self.f = f
			self.args = args
			self.kwargs = kwargs

			# Push job onto list of jobs
			self._init_job(f, args, kwargs)
		else:
			assert self.job_name is not None, "cannot load job"
			# Otherwise this is a copy which will not trigger a delete when it disappears
			self._owner = False
			self.f = None	
			self.args = None
			self.kwargs = None

	def _load_input(self):
		self.f, self.args, self.kwargs = loads(self.input)

	def _load_input(self):
		self.f, self.args, self.kwargs = loads(self.input)

	def _init_job(self, f, args, kwargs):

		job_input = dumps( (f, args, kwargs))
		if self.job_name is None:
			# Generate name based on function and arguments
			job_name = base64.b64encode(hashlib.sha1(job_input).digest())

		self.redis_pool.add_job(self.job_name)

		# Copy over the state
		self.input = job_input

		# Clear the remaining value to be safe
		self.output = None
		self.heartbeat = None
		self.start_time = None
		self.stop_time = None
		self.state = 'loaded'

	def enqueue(self):
		""" Queue the job
		"""
		
		# Push onto the queue
		self.redis_pool.push_job(self.job_name)
		
		# Update state to reflect this
		self.state = 'enqueued'	
	
	def run(self):
		""" Run the job
		"""	
		# Setup the heartbeat 
		def heartbeat():
			self.heartbeat = time.time()	
		# Setup the heartbeat
		heart = Event()
		t = WatchDog(heart, heartbeat)
		t.daemon = True
		t.start()
		
		self.start_time = time.time()
		f, args, kwargs = loads(self.input)
		self.state = 'running'
		try:
			self.output = f(*args, **kwargs)
			self.state = 'done'
		except KeyboardInterrupt:
			self.state = 'killed'
			self.output = None
		except Exception as e:
			self.output = e
			self.state = 'failed'		
		
		self.stop_time = time.time()
		heart.set()

	def __del__(self):
		if self._owner and not self.keep_data:
			self.redis.delete('pool_%s_job_%s' % (self.pool_name, self.job_name))
			self.redis_pool.del_job(self.job_name)
	
	@property
	def state(self):
		return self.redis.hget('pool_%s_job_%s' % (self.pool_name, self.job_name), 'state')
	
	@state.setter
	def state(self, value):
		if value is None: self.redis.hdel('pool_%s_job_%s' % (self.pool_name, self.job_name), 'state') 
		self.redis.hset('pool_%s_job_%s' % (self.pool_name, self.job_name), 'state', value)
	
	@property
	def input(self):
		return self.redis.hget('pool_%s_job_%s' % (self.pool_name, self.job_name), 'input')
	
	@input.setter
	def input(self, value):
		if value is None: self.redis.hdel('pool_%s_job_%s' % (self.pool_name, self.job_name), 'input') 
		else: self.redis.hset('pool_%s_job_%s' % (self.pool_name, self.job_name), 'input', value)

	@property
	def output(self):
		rv = self.redis.hget('pool_%s_job_%s' % (self.pool_name, self.job_name), 'output')
		if rv is None:
			return None
		return loads(rv)

	@output.setter
	def output(self, value):
		if value is None: self.redis.hdel('pool_%s_job_%s' % (self.pool_name, self.job_name), 'output') 
		else: self.redis.hset('pool_%s_job_%s' % (self.pool_name, self.job_name), 'output', dumps(value))
	
	@property
	def start_time(self):
		t = self.redis.hget('pool_%s_job_%s' % (self.pool_name, self.job_name), 'start_time')
		try: return float(t)
		except TypeError: return t

	@start_time.setter
	def start_time(self, value):	
		if value is None: self.redis.hdel('pool_%s_job_%s' % (self.pool_name, self.job_name), 'start_time') 
		else: self.redis.hset('pool_%s_job_%s' % (self.pool_name, self.job_name), 'start_time', "%.6f" % (value, ))
	
	@property
	def stop_time(self):
		t = self.redis.hget('pool_%s_job_%s' % (self.pool_name, self.job_name), 'stop_time')
		try: return float(t)
		except TypeError: return t
	
	@stop_time.setter
	def stop_time(self, value):
		if value is None: self.redis.hdel('pool_%s_job_%s' % (self.pool_name, self.job_name), 'stop_time') 
		else: self.redis.hset('pool_%s_job_%s' % (self.pool_name, self.job_name), 'stop_time', "%.6f" % (value,) )
	
	@property
	def heartbeat(self):	
		t = self.redis.hget('pool_%s_job_%s' % (self.pool_name, self.job_name), 'heartbeat')
		try: return float(t)
		except TypeError: return t

	@heartbeat.setter
	def heartbeat(self, value):
		if value is None: self.redis.hdel('pool_%s_job_%s' % (self.pool_name, self.job_name), 'heartbeat')
		else: self.redis.hset('pool_%s_job_%s' % (self.pool_name, self.job_name), 'heartbeat', '%.6f' % (value,) )
	
#	@property
#	def keep_data(self):
#		value = self.redis.hget('pool_%s_job_%s' % (self.pool_name, self.job_name), 'keep_data'))
#		return value == 'True'
#
#	@keep_data.setter
#	def keep_data(self, value):
#		assert value in [True, False]
#		if value: value = 'True'
#		else: value = 'False'
#		self.redis.hset('pool_%s_job_%s' % (self.pool_name, self.job_name), 'keep_data', value)
		

	@property
	def elapsed_time(self):
		start_time = self.start_time
		stop_time = self.stop_time
		heartbeat = self.heartbeat	
		
		if start_time is None: return 0
		elif heartbeat is None: return 0
		elif stop_time is None:	return heartbeat - start_time
		else: return stop_time - start_time

	def get(self):
		while True:
			output = self.output
			if output is not None:
				break
			time.sleep(0.01)
		return output	


	def __str__(self):
		return "job:%s, %s" % (self.job_name, self.state)
	
	def ready(self):
		if self.state in ['done', 'failed']:
			return True
		else:
			return False

	def successful(self):
		if self.state is 'done':
			return True
		elif self.state is 'failed':
			return False
		else:
			raise AssertionError

@contextlib.contextmanager
def runnable_worker(*args, **kwargs):
	worker = RedisWorker(*args, **kwargs)
	worker._init_worker()
	try:
		yield worker
	finally:
		worker._del_worker()
	return

class RedisWorker(object):
	def __init__(self, redis_pool, worker_name = None, max_jobs = float('inf')):
		if worker_name is None:
			worker_name = uuid.uuid4()

		self.redis_pool = redis_pool
		self.redis = redis_pool.redis
		self.pool_name = redis_pool.name
		self.worker_name = worker_name
		self.max_jobs = max_jobs

		# If the instance mentioned called the run function
		self._runnable = False

	def _init_worker(self):
		""" Register the worker with the pool
		"""
		self.redis_pool.add_worker(self.worker_name)
		self.state = 'idle'
		self.kill = False
		self.start_time = time.time()
		self._runnable = True

	def _del_worker(self):
		""" Remove the worker from the pool
		"""
		#logging.info('trying to kill worker %s' % (self.worker_name,))
		#self.kill = True
		#time.sleep(0.1)
		self.redis_pool.del_worker(self.worker_name)
		self._runnable = False


	@property
	def state(self):
		return self.redis.hget('pool_%s_worker_%s' % (self.pool_name, self.worker_name), 'state')
	
	@state.setter
	def state(self, value):
		self.redis.hset('pool_%s_worker_%s' % (self.pool_name, self.worker_name), 'state', value)
		logging.debug('worker:%s state set to "%s"' % (self.worker_name, value))

	@property
	def job(self):
		return self.redis.hget('pool_%s_worker_%s' % (self.pool_name, self.worker_name), 'job')

	@job.setter
	def job(self, value):
		if value is None: self.redis.hdel('pool_%s_worker_%s' % (self.pool_name, self.worker_name), 'job')	
		else: self.redis.hset('pool_%s_worker_%s' % (self.pool_name, self.worker_name), 'job', value)	


	@property
	def heartbeat(self):
		value = self.redis.hget('pool_%s_worker_%s' % (self.pool_name, self.worker_name), 'heartbeat')
		try: return float(value)
		except: return value

	@heartbeat.setter
	def heartbeat(self,value):
		self.redis.hset('pool_%s_worker_%s' % (self.pool_name, self.worker_name),
			 'heartbeat', '%.3f' %(value,))
	
	@property
	def start_time(self):
		t = self.redis.hget('pool_%s_worker_%s' % (self.pool_name, self.worker_name), 'start_time')
		try: return float(t)
		except TypeError: return t

	@start_time.setter
	def start_time(self, value):	
		if value is None: self.redis.hdel('pool_%s_worker_%s' % (self.pool_name, self.worker_name), 'start_time') 
		else: self.redis.hset('pool_%s_worker_%s' % (self.pool_name, self.worker_name), 'start_time', "%.6f" % (value, ))
	
	@property
	def run_time(self):
		try: return self.heartbeat - self.start_time
		except: return 0

	@property
	def kill(self):
		value = self.redis.hget('pool_%s_worker_%s' % (self.pool_name, self.worker_name), 'kill')
		logging.debug('pool %s worker %s read kill %s' % (self.pool_name, self.worker_name, value) )
		return value == 'True'
		
	@kill.setter
	def kill(self, value):
		assert value in [True, False], "Kill must be either True or False"
		if value: value = 'True'
		else: value = 'False'
		logging.debug('setting kill %s' % value)
		self.redis.hset('pool_%s_worker_%s' % (self.pool_name, self.worker_name),
			 'kill', value)


	def alive(self):
		return self.redis.exists('pool_%s_worker_%s' % (self.pool_name, self.worker_name))

	def run(self):
		assert self._runnable == True, "Class must be started using context manager"
		logging.debug('starting worker')
		self.start_time = time.time()
		njobs = 0

		def callback(pid):
			#logging.debug('run worker watchdog')
			self.heartbeat = time.time()
			if self.kill:
				logging.debug('detected kill=True sent SIGINT to parent')
				os.kill(pid, signal.SIGINT)
				# THis doesn't seem to work
				#thread.interrupt_main()
				thread.exit()

		stop_dog = Event() 
		t = WatchDog(stop_dog, callback, args = (os.getpid(),) )
		t.daemon = True
		t.start()	

		while njobs < self.max_jobs:
			self.state = 'idle'

			# Try to grab a new job	
			job_name = self.redis_pool.pop_job()
			logging.debug('worker:%s got job %s' % (self.worker_name, job_name))

			job = RedisJob(self.redis_pool, job_name)
			self.job = job_name 
			self.state = 'active'
			job.run()
			self.job = None
			njobs += 1	

		# Remove the worker from the list of active workers
		stop_dog.set()

	def __str__(self):
		return "worker:%s, %s" % (self.worker_name, self.state)

	
class RedisPool(object):
	def __init__(self, host = 'localhost', port = 6379, password = None, name = None):
		if name is None:
			name = uuid.uuid4()
		self.name = name
		self.redis = Redis(host = host, port = port, password = password)

	def add_job(self, job_name):
		assert self.redis.hget('pool_%s_jobs' % (self.name,) , job_name) is None, "Job already exists"
		self.redis.hset('pool_%s_jobs' % (self.name,) , job_name, 'exists')

	def del_job(self, job_name):
		self.redis.hdel('pool_%s_jobs' % (self.name, ) , job_name)

	def add_worker(self, worker_name):
		assert self.redis.hget('pool_%s_workers' % (self.name,), worker_name) is None, "Worker already exists"
		self.redis.hset('pool_%s_workers' % (self.name,), worker_name, 'exists')

	def del_worker(self, worker_name):
		self.redis.hdel('pool_%s_workers' % (self.name,), worker_name)

	def clear(self):
		self.redis.delete('pool_%s_queue' % (self.name,))
		self.redis.delete('pool_%s_workers' % (self.name, ))
		self.redis.delete('pool_%s_jobs' % (self.name, ))

	def push_job(self, job_name, side = 'right'):
		if side is 'right':
			self.redis.rpush('pool_%s_queue' % (self.name), job_name)
		else:
			self.redis.lpush('pool_%s_queue' % (self.name), job_name)

	def pop_job(self, timeout = None):
		try:
			variable, item = self.redis.blpop('pool_%s_queue' % (self.name), timeout)
			return item
		except:
			return None

	@property
	def workers(self):
		worker_names = self.redis.hgetall('pool_%s_workers' % self.name).keys()
		return [RedisWorker(self, worker_name) for worker_name in worker_names]

	@property
	def jobs(self):
		job_names = self.redis.hgetall('pool_%s_jobs' % self.name ).keys()
		return [RedisJob(self, job_name) for job_name in job_names]

	def apply_async(self, f, args = None, kwargs = None , callback = None, job_name = None):
		if args is None:
			args = ()
		if kwargs is None:
			kwargs = {}

		job = RedisJob(self, job_name, f, args = args, kwargs = kwargs)
		job.enqueue()

		return job

	def close(self):
		"""Close the queue to taking on new members
		"""
		pass

	def terminate(self):
		""" Immediately end all working jobs
		"""
		pass

	def join(self):
		"""Wait for all processes to exit
		"""
		pass
		
if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description = 'Start a redis pool worker',
		usage = '''./redis_pool <command> [<args>]

Commands
	flush : clear the current database
	worker <pool_name> [worker_name] : start a worker on the specified pool
	info <pool_name>: print info about pool 
''')


	parser.add_argument('command', help='subcommand to run')


	# inspired by https://chase-seibert.github.io/blog/2014/03/21/python-multilevel-argparse.html
	args = parser.parse_args(sys.argv[1:2])
	
	parser.add_argument('--host', help='host for redis connection', default = 'localhost')
	parser.add_argument('--port', help='port for redis connection', default= 6379, type = int)
	parser.add_argument('--password', help='password for redis connection', default = None)
	
	if args.command == 'flush':
		args = parser.parse_args()
		r = Redis(host = args.host, port = args.port, password = args.password)
		r.flushall()
		r.flushdb()

	elif args.command == 'worker':
		parser.add_argument('pool_name', help = 'name of pool')
		parser.add_argument('worker_name', help = 'name of worker', default = None, nargs = '?')
		parser.add_argument('--max_jobs', help = 'maximum number of jobs for the worker to consume',
			default = float('inf'), type = int) 
		args = parser.parse_args()

		pool = RedisPool(name = args.pool_name, host = args.host, port = args.port, password = args.password)
		if args.worker_name is None and PETNAME:
			args.worker_name = petname.Generate(3)
		with runnable_worker(pool, worker_name = args.worker_name, max_jobs = args.max_jobs) as worker:
			worker.run()
		print "worker done"

	elif args.command == 'info':
		parser.add_argument('pool_name', help = 'name of pool')
		args = parser.parse_args()

		pool = RedisPool(name = args.pool_name, host = args.host, port = args.port, password = args.password)
		print "Pool: %s" % (args.pool_name,)
		
		print "------Workers-------"
		for worker in pool.workers:
			print "\t %20s %10s %04d" % (worker.worker_name, worker.state,  worker.run_time)	

		print "------Jobs----------"
		for job in pool.jobs:
			print "\t %20s %10s %5d" % (job.job_name, job.state, job.elapsed_time)

	elif args.command == 'kill':
		parser.add_argument('pool_name', help = 'name of pool')
		parser.add_argument('worker_name', help = 'name of worker')
		args = parser.parse_args()
		pool = RedisPool(name = args.pool_name, host = args.host, port = args.port, password = args.password)

		worker = RedisWorker(pool, args.worker_name)
		worker.kill = True


