{
    'phase': 'running',
    'reason': 'Exception msg',
    'msg': 'Traceback msg',
    'last_change_time': '2023-10-01T12:00:00Z',
}

# component level

# assumimg this is the Listener level

{
    'status': {
        'phase': 'running',
        'reason': 'Exception msg',
        'msg': 'Traceback msg',
        'last_change_time': '2023-10-01T12:00:00Z',
    },
    'children_statuses': {
        'Listener.my_listener.my_listener.list': {
            'phase': 'running',
            'reason': 'Exception msg',
            'msg': 'Traceback msg',
            'last_change_time': '2023-10-01T12:00:00Z',
        },
    }
}


# job level

{
    'status': 'failed',
    'msg': 'error message',
    'traceback': 'traceback message',
}