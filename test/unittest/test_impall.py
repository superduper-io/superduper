try:
    import torch
except ImportError:
    torch = None
import impall


class ImpAllTest(impall.ImpAllTest):
    CLEAR_SYS_MODULES = False
    WARNINGS_ACTION = 'ignore'
    EXCLUDE = (torch is None) * ['superduperdb/ext/torch/**']
    PATHS = 'superduperdb', 'test', 'apps/question-the-docs'
