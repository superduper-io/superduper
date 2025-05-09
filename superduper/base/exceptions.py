from http import HTTPStatus


class AppException(Exception):
    """Generic exception for application-specific errors.

    Should not be used directly. Only though the other classes.

    :param code: the HTTP code of the respective error
    :param reason: The human-readable representation of the code.
    :param message: information about the error
    """

    def __init__(self, code, reason, message=""):
        self.code = code
        self.reason = reason
        self.message = message


class BadRequest(AppException):
    """
    BadRequest means that the request itself was invalid, because the request doesn't make any sense.

    For example deleting a read-only object.

    :param msg: details about the exception
    """

    def __init__(self, msg: str):
        AppException.__init__(
            self,
            code=HTTPStatus.BAD_REQUEST,
            reason=HTTPStatus.BAD_REQUEST.phrase,
            message=msg,
        )


class NotFound(AppException):
    """
    NotFound means one or more resources required for this operation could not be found.

    :param obj_type: the type of the missing resource (e.g Table, Job)
    :param obj_id: the identifier of the missing resource  (e.g MyTable)
    """

    def __init__(self, obj_type: str, obj_id: str):
        AppException.__init__(
            self,
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
        AppException.__init__(
            self,
            code=HTTPStatus.CONFLICT,
            reason=HTTPStatus.CONFLICT.phrase,
            message=f"{obj_type} {obj_id} already exists",
        )


class Unauthorized(AppException):
    """
    The request has not been applied because it lacks valid authentication credentials for the target resource.

    :param msg: details about the exception
    """

    def __init__(self, msg: str):
        AppException.__init__(
            code=HTTPStatus.UNAUTHORIZED,
            reason=HTTPStatus.UNAUTHORIZED.phrase,
            message=msg,
        )


class Forbidden(AppException):
    """
    The server understood the request but refuses to fulfill it.

    If authentication credentials were provided in the request, the server considers them insufficient to grant access.

    :param msg: details about the exception
    """

    def __init__(self, msg: str):
        AppException.__init__(
            self,
            code=HTTPStatus.FORBIDDEN,
            reason=HTTPStatus.FORBIDDEN.phrase,
            message=msg,
        )


class Conflict(AppException):
    """
    The request could not be completed due to a conflict with the current state of the target resource.

    This code is used in situations where the user might be able to resolve the conflict and resubmit the request.

    :param obj_type: the type of the conflict resource (e.g Table, Job)
    :param obj_id: the identifier of conflicting resource (e.g MyTable)
    :param details: additional information that indicate the nature of the conflict.
    """

    def __init__(self, obj_type: str, obj_id: str, details: str):
        AppException.__init__(
            self,
            code=HTTPStatus.CONFLICT,
            reason=HTTPStatus.CONFLICT.phrase,
            message=f"Operation cannot be fulfilled on {obj_type} {obj_id}: {details}",
        )


class InternalServerError(AppException):
    """
    The server encountered an unexpected condition that prevented it from fulfilling the request.

    :param cause: the original exception
    :param details: details about the situation
    """

    def __init__(self, cause: Exception, details: str):
        AppException.__init__(
            self,
            code=HTTPStatus.INTERNAL_SERVER_ERROR,
            reason=HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
            message=f"Cause:{str(cause)} Details:{details} ",
        )


class UnprocessableContent(AppException):
    """
    The request is valid, but the server was unable to process the contained instructions.

    For example, to read the payload from an empty file

    :param msg: details about the exception
    """

    def __init__(self, msg: str):
        AppException.__init__(
            self,
            code=HTTPStatus.UNPROCESSABLE_CONTENT,
            reason=HTTPStatus.UNPROCESSABLE_CONTENT.phrase,
            message=msg,
        )


class GenericServerResponse(AppException):
    """
    Raised for server responses that do not match any specific known error type.

    It should only be used by REST clients that need to raise a local exception if they
    receive a non 200 (ok) request.

    :param code: the http code we are interested to wrap.
    :param server_message: the payload of the http response
    """

    def __init__(self, code: int, server_message: str):
        # Default values
        reason = "Unknown"
        message = f"the server responded with the status code {code} but did not return more information"

        # Reason/message mapping
        if code == HTTPStatus.BAD_REQUEST:
            reason = HTTPStatus.BAD_REQUEST.phrase
            message = "the server rejected our request for an unknown reason"
        elif code == HTTPStatus.NOT_FOUND:
            reason = HTTPStatus.NOT_FOUND.phrase
            message = "the server could not find the requested resource"
        elif code == HTTPStatus.CONFLICT:
            reason = HTTPStatus.CONFLICT.phrase
            message = "the server reported a conflict"
        elif code == HTTPStatus.UNAUTHORIZED:
            reason = HTTPStatus.UNAUTHORIZED.phrase
            message = "the server has asked for the client to provide credentials"
        elif code == HTTPStatus.FORBIDDEN:
            reason = HTTPStatus.FORBIDDEN.phrase
            message = server_message
        elif code == HTTPStatus.UNPROCESSABLE_CONTENT:
            reason = HTTPStatus.UNPROCESSABLE_CONTENT.phrase
            message = "the server rejected our request due to an error in our request"

        elif code >= 500:
            reason = "InternalError"
            message = f'an error on the server ("{server_message}") has prevented the request from succeeding'

        AppException.__init__(self, code, reason, message)


#################################################################################
#################################################################################
#
#   DO NOT BE TEMPTED TO ADD MORE EXCEPTIONS.
#
#   If you think that a specific exception family is missing, raise an issue.
#
#   Before you do so, read the following:
#   https://datatracker.ietf.org/doc/html/rfc9110.html#status.422
#   https://github.com/kubernetes/apimachinery/blob/master/pkg/api/errors/errors.go
#   https://github.com/kubernetes/apimachinery/blob/f7c43800319c674eecce7c80a6ac7521a9b50aa8/pkg/apis/meta/v1/types.go#L857C1-L1015C68
#################################################################################
#################################################################################


class ComponentLifecycleError(Exception):
    """Exception raised when a component lifecycle error occurs.

    :param args: *args for Exception
    :param kwargs: **kwargs for Exception
    """
