

class Job(object):
	pass

class Pool(object):
	pass


class SequentialJob(Job):
	def __init__(self, f, args = None, kwargs = None):
		self.f = f
		if args is None: args = ()
		if kwargs is None:	kwargs = {}
		self.args = args
		self.kwargs = kwargs
		self.output = f(*args, **kwargs)

		
	def ready(self):
		return True


class SequentialPool(Pool):
	def apply(self, f, args = None, kwargs = None):
		if args is None: args = ()
		if kwargs is None:	kwargs = {}
		return SequentialJob(f, args, kwargs)

	def avail_workers(self):
		return 1
	
	def join(self):
		""" wait until all tasks done
		"""
		# Since the SequentialPool runs all jobs immediately, simply stop
		pass
