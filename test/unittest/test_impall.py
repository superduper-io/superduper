from test.torch import torch

import impall


class ImpAllTest(impall.ImpAllTest):
    CLEAR_SYS_MODULES = False
    WARNINGS_ACTION = 'ignore'
    EXCLUDE = (torch is None) * ['superduperdb/ext/torch/**']


assert ImpAllTest.EXCLUDE
