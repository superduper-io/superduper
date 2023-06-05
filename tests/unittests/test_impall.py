import impall


class ImpAllTest(impall.ImpAllTest):
    # Do not reload modules for each file tested
    CLEAR_SYS_MODULES = False

    EXCLUDE = (
        'superduperdb/apis/openai/wrapper',
        'superduperdb/cluster/ray/predict',
        'superduperdb/cluster/ray/predict_one',
    )
