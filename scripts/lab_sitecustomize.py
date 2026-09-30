"""Study Python startup: prevent OS crash handlers from writing to the SSD."""
import ctypes
import os
import resource
if os.name=='posix':
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    if ctypes.CDLL(None,use_errno=True).prctl(4,0,0,0,0)!=0:
        os._exit(78)
