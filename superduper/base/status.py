from datetime import datetime

# List of Job's stages
JOB_PHASE_UNINITIALIZED = "uninitialized"
JOB_PHASE_PENDING = "pending"
JOB_PHASE_RUNNING = "running"
JOB_PHASE_FAILED = "failed"
JOB_PHASE_SUCCESS = "success"


def init_status():
    """Initialize the status of the component."""
    return {
        'phase': JOB_PHASE_UNINITIALIZED,
        'reason': 'waiting for initialization',
        'message': None,
        'last_change_time': str(datetime.now()),
        'children': {},
    }


def running_status():
    """The status of the component as ready."""
    return {
        'phase': JOB_PHASE_RUNNING,
        'reason': 'the component is ready to use',
        'message': None,
        'last_change_time': str(datetime.now()),
        'children': {},
    }
