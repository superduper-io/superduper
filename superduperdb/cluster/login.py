from superduperdb import CFG


def maybe_login_required(auth, service):
    """
    Require login depending on the contents of the config file.:w

    :param auth: basic auth instance
    :param service: collection of the service on question
    """

    def decorator(f):
        if getattr(CFG, service).username:
            return auth.login_required(f)
        return f

    return decorator
