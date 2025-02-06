import time
from functools import wraps

PROFILE_TIMES = {}
PROFILE_NCALLS = {}

#########################################################################
# NOTE: This will not work with multiprocessing due to independent processes
#       not talking to each other
#
# This module defines a profiler that captures elapsed times of functions
#
# User should wrap all relevant functions with the @profile_time decorator
# which will populate PROFILE_TIMES and PROFILE_NCALLS dictionaries that 
# can then be imported into another module and dumped
#
# @profile_time(arg_idx=i) will track elapsed time of a function call
# with respect to the i-th argument. This is useful when broadcasting requests
# to daemon processes based on some flag. 
#
# User should take care in analyzing the times. Depending on what functions
# you decorate, the elapsed times can overlap.
#
# Stores class name along with the function name if its a class method
#########################################################################

def profile_time(arg_idx=None):

    def decorator(func):
        # Use a closure to keep track of total time
        total_time_spent = 0
        ncalls = 0

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal total_time_spent
            nonlocal ncalls
            start_time = time.time()  # Record the start time

            # Call the original function
            result = func(*args, **kwargs)
            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time  # Calculate elapsed time
            total_time_spent += elapsed_time  # Add to total time spent
            ncalls += 1

            # Attach class name if its a class method (instance or static)
            if args and hasattr(args[0], '__class__'):
                # Instance method
                class_name = args[0].__class__.__name__
                base_func_name = f"{class_name}.{func.__name__}"
            elif hasattr(func, '__self__') and func.__self__ is not None:
                # Static method
                class_name = func.__self__.__name__
                base_func_name = f"{class_name}.{func.__name__}"
            else:
                # Static method or standalone function
                base_func_name = func.__name__

            # Include arguments in the function name if tracking unique args
            if arg_idx is not None:
                arg_repr = repr(args[arg_idx])
                func_name = f"{base_func_name}({arg_repr})"
            else:
                func_name = base_func_name

            PROFILE_TIMES[func_name] = total_time_spent
            PROFILE_NCALLS[func_name] = ncalls

            return result
        return wrapper
    return decorator