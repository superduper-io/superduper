# TODO add more exceptions
# for example ArtifactNotFoundError
from superduper import logging
from http import HTTPStatus

class AppException(Exception):
    """Generic exception for application-specific errors.
    Should not be used directly. Only though the other classes.
    """

    def __init__(self, code, reason, message="", details=""):
        self.code = code
        self.reason = reason
        self.message = message
        self.details = details


class BadRequest(AppException):
    """
    BadRequest means that the request itself was invalid, because the request
	doesn't make any sense, for example deleting a read-only object.

    :param msg: details about the exception
    """
    def __init__(self, msg: str):
        AppException.__init__(self,
            code=HTTPStatus.BAD_REQUEST,
            reason=HTTPStatus.BAD_REQUEST.phrase,
            message=msg,
        )

class NotFound(AppException):
    def __init__(self, obj_type: str, obj_id: str):
        """
        NotFound means one or more resources required for this operation could not be found.

        :param obj_type: the type of the missing resource (e.g Table, Job)
        :param obj_id: the identifier of the missing resource  (e.g MyTable)
        """
        AppException.__init__(self,
            code=HTTPStatus.NOT_FOUND,
            reason=HTTPStatus.NOT_FOUND.phrase,
            message=f"{obj_type} {obj_id} not found",
        )


class AlreadyExists(AppException):
    """
    AlreadyExists means the resource you are creating already exists.

    :param obj_type: the type of the conflict resource (e.g Table, Job)
    :param obj_id: the identifier of conflicting resource (e.g MyTable)
    """
    def __init__(self, obj_type: str, obj_id: str):
        AppException.__init__(self,
            code=HTTPStatus.CONFLICT,
            reason=HTTPStatus.CONFLICT.phrase,
            message=f"{obj_type} {obj_id} already exists",
        )


class Unauthorized(AppException):
    """
    Unauthorized means the server can be reached and understood the request, but requires
	the user to present appropriate authorization credentials.
	If the user has specified credentials on the request, the server considers them insufficient.

    :param msg: details about the exception
    """
    def __init__(self, msg: str):
        AppException.__init__(
            code=HTTPStatus.UNAUTHORIZED,
            reason=HTTPStatus.UNAUTHORIZED.phrase,
            message=msg
        )



class Forbidden(AppException):
    """
    Forbidden means the server can be reached and understood the request, but refuses
	to take any further action.  It is the result of the server being configured to deny access for some reason
	to the requested resource by the client.

    :param msg: details about the exception
    """
    def __init__(self, msg: str):
        AppException.__init__(self,
            code=HTTPStatus.FORBIDDEN,
            reason=HTTPStatus.FORBIDDEN.phrase,
            message=msg,
        )


class Conflict(AppException):
    """
    Conflict indicates that the request could not be completed due to a conflict with
    the current state of the target resource. This code is used in situations where
    the user might be able to resolve the conflict and resubmit the request.

    :param obj_type: the type of the conflict resource (e.g Table, Job)
    :param obj_id: the identifier of conflicting resource (e.g MyTable)
    :param details: additional information that indicate the nature of the conflict.
    """
    def __init__(self, obj_type: str, obj_id: str, details: str):
        AppException.__init__(self,
            code=HTTPStatus.CONFLICT,
            reason=HTTPStatus.CONFLICT.phrase,
            message=f"Operation cannot be fulfilled on {obj_type} {obj_id}: {details}",
        )



class InternalServerError(AppException):
    """
    InternalError indicates that an internal error occurred, it is unexpected and the outcome of the call is unknown.

    :param cause: the original exception
    """
    def __init__(self, cause: Exception):
        AppException.__init__(self,
            code=HTTPStatus.INTERNAL_SERVER_ERROR,
            reason=HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
            message=str(cause),
        )


class UnprocessableContent(AppException):
    """
    UnprocessableContent indicates that the server understands the content type of the request content,
	but it was unable to process the contained instructions. For example, ask for resources
	that exceed the current capacity.

    :param msg: details about the exception
    """
    def __init__(self, msg: str):
        AppException.__init__(self,
            code=HTTPStatus.UNPROCESSABLE_CONTENT,
            reason=HTTPStatus.UNPROCESSABLE_CONTENT.phrase,
            message=msg,
        )


class ComponentLifecycleError(Exception):
    """Exception raised when a component lifecycle error occurs.

    :param args: *args for Exception
    :param kwargs: **kwargs for Exception
    """




#
# __known_exception_classes = [NotFound, AlreadyExists, BadRequest, UnprocessableContent]
#
# def is_known_exception(exc: Exception) -> bool:
#     """
#     Check whether exc is an instance of a known exception class.
#     """
#     return any(isinstance(exc, exc_class) for exc_class in __known_exception_classes)
