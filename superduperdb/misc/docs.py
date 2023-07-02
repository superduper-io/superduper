def api(type="experimental"):
    def decorator(cls):
        if type == "experimental":
            cls.__doc__ = f"""{cls.__doc__}
            --

            Note: This feature is {type}, it is not production ready yet!
            """
        elif type == "alpha":
            cls.__doc__ = f"""{cls.__doc__}
            --

            Note: This feature is {type}, it is not production ready yet!
            """
        else:
            cls.__doc__ = f"""{cls.__doc__}
            --

            Note: This is a {type} feature!
            """
        return cls

    return decorator
