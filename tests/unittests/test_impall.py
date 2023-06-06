import impall


class ImpAllTest(impall.ImpAllTest):
    # Do not reload modules for each file tested
    CLEAR_SYS_MODULES = False
