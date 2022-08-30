import importlib
import pickle


def load(r):
    if r['type'] == 'import':
        module = importlib.import_module('.'.join(r['args']['path'].split('.')[:-1]))
        object_ = getattr(module, r['args']['path'].split('.')[-1])
        return object_(**r['args']['kwargs'])

    elif r['type'] == 'pickle':
        with open(r['args']['path'], 'rb') as f:
            return pickle.load(f)

    elif r['type'] == 'dill':
        import dill
        with open(r['args']['path'], 'rb') as f:
            return dill.load(f)

    elif r['type'] == 'aijson':
        raise NotImplementedError

    elif r['type'] == 'padl':
        raise NotImplementedError

    else:
        raise NotImplementedError