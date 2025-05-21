from datetime import datetime

# List of Job's stages
STATUS_UNINITIALIZED = "submitted"
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DELETING = "deletion"
STATUS_FAILED = "failed"
STATUS_SUCCESS = "success"


def init_status():
    """Initialize the status of the component."""
    return STATUS_UNINITIALIZED, {
        'reason': 'waiting for initialization',
        'message': None,
        'last_change_time': str(datetime.now()),
        'failed_children': {},
    }


def running_status():
    """The status of the component as ready."""
    return STATUS_RUNNING, {
        'reason': 'the component is ready to use',
        'message': None,
        'last_change_time': str(datetime.now()),
        'failed_children': {},
    }


def pending_status():
    """The status of the component as ready."""
    return STATUS_PENDING, {
        'reason': 'the component is being initialized by jobs',
        'message': None,
        'last_change_time': str(datetime.now()),
        'failed_children': {},
    }
