import sys
from enum import Enum
from http import HTTPStatus

from superduper import logging


class StatusReason(str, Enum):
    """
    StatusReason is an enumeration of possible failure causes.

    Each StatusReason must map to a single HTTP status code, but multiple reasons may map to the same code.

    noqa:
    """

    UNKNOWN = ""  # Status code 500
    """The server has declined to indicate a specific reason. The details field may contain other information about this error."""

    UNAUTHORIZED = "Unauthorized"  # Status code 401
    """The server requires the user to present appropriate authorization credentials."""

    FORBIDDEN = "Forbidden"  # Status code 403
    """The server refuses to take further action due to access denial."""

    NOT_FOUND = "NotFound"  # Status code 404
    """Required resources for the operation could not be found."""

    ALREADY_EXISTS = "AlreadyExists"  # Status code 409
    """The resource being created already exists."""

    CONFLICT = "Conflict"  # Status code 409
    """The requested operation cannot be completed due to a conflict."""

    GONE = "Gone"  # Status code 410
    """The item is no longer available and no forwarding address is known."""

    INVALID = "Invalid"  # Status code 422
    """Create or update operation failed due to invalid data."""

    SERVER_TIMEOUT = "ServerTimeout"  # Status code 500
    """The server cannot complete the action in a reasonable time. Retry is suggested."""

    STORE_READ_ERROR = "StorageReadError"  # Status code 500
    """Error retrieving resources from the backend object store."""

    TIMEOUT = "Timeout"  # Status code 504
    """Request could not be completed in the given time."""

    TOO_MANY_REQUESTS = "TooManyRequests"  # Status code 429
    """Client must wait due to too many requests."""

    BAD_REQUEST = "BadRequest"  # Status code 400
    """The request itself was invalid and makes no sense."""

    METHOD_NOT_ALLOWED = "MethodNotAllowed"  # Status code 405
    """The action attempted is not supported for the resource."""

    NOT_ACCEPTABLE = "NotAcceptable"  # Status code 406
    """The accept types indicated by the client were not acceptable."""

    REQUEST_ENTITY_TOO_LARGE = "RequestEntityTooLarge"  # Status code 413
    """The request entity is too large."""

    UNSUPPORTED_MEDIA_TYPE = "UnsupportedMediaType"  # Status code 415
    """The content type sent by the client is not supported."""

    INTERNAL_ERROR = "InternalError"  # Status code 500
    """An unexpected internal error occurred."""

    EXPIRED = "Expired"  # Status code 410
    """The requested content has expired and is no longer available."""

    SERVICE_UNAVAILABLE = "ServiceUnavailable"  # Status code 503
    """The requested service is unavailable at this time. Retry may succeed."""


# Static mapping of status reasons to HTTP status codes
# StatusReason_HTTP_CODE: Dict[StatusReason, int] = {
#     StatusReason.UNKNOWN: HTTPStatus.INTERNAL_SERVER_ERROR,
#     StatusReason.UNAUTHORIZED: HTTPStatus.UNAUTHORIZED,
#     StatusReason.FORBIDDEN: HTTPStatus.FORBIDDEN,
#     StatusReason.NOT_FOUND: HTTPStatus.NOT_FOUND,
#     StatusReason.ALREADY_EXISTS: HTTPStatus.CONFLICT,
#     StatusReason.CONFLICT: HTTPStatus.CONFLICT,
#     StatusReason.GONE: HTTPStatus.GONE,
#     StatusReason.INVALID: HTTPStatus.UNPROCESSABLE_ENTITY,
#     StatusReason.SERVER_TIMEOUT: HTTPStatus.INTERNAL_SERVER_ERROR,
#     StatusReason.STORE_READ_ERROR: HTTPStatus.INTERNAL_SERVER_ERROR,
#     StatusReason.TIMEOUT: HTTPStatus.GATEWAY_TIMEOUT,
#     StatusReason.TOO_MANY_REQUESTS: HTTPStatus.TOO_MANY_REQUESTS,
#     StatusReason.BAD_REQUEST: HTTPStatus.BAD_REQUEST,
#     StatusReason.METHOD_NOT_ALLOWED: HTTPStatus.METHOD_NOT_ALLOWED,
#     StatusReason.NOT_ACCEPTABLE: HTTPStatus.NOT_ACCEPTABLE,
#     StatusReason.REQUEST_ENTITY_TOO_LARGE: HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
#     StatusReason.UNSUPPORTED_MEDIA_TYPE: HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
#     StatusReason.INTERNAL_ERROR: HTTPStatus.INTERNAL_SERVER_ERROR,
#     StatusReason.EXPIRED: HTTPStatus.GONE,
#     StatusReason.SERVICE_UNAVAILABLE: HTTPStatus.SERVICE_UNAVAILABLE,
# }


class AppException(Exception):
    """
    Generic exception for application-specific errors.

    Should not be used directly. Only through the other classes.

    :param code: the HTTP status code of the error (e.g 500)
    :param reason: the specific failure cause. (e.g Timeout, InternalError)
    :param message: details about the reason (e.g read file XXX).
    """

    def __init__(self, code: int, reason: StatusReason, message: str):
        self.code = code
        self.reason = reason
        self.message = message


class BadRequest(AppException):
    """
    BadRequest means the request itself was invalid and makes no sense.

    For example, deleting a read-only object.

    :param message: details about the exception
    """

    def __init__(self, message: str):
        super().__init__(
            code=HTTPStatus.BAD_REQUEST,
            reason=StatusReason.BAD_REQUEST,
            message=message,
        )


class NotFound(AppException):
    """
    NotFound means one or more resources required for this operation could not be found.

    :param obj_type: the type of the missing resource (e.g Table, Job)
    :param obj_id: the identifier of the missing resource  (e.g MyTable)
    """

    def __init__(self, obj_type: str, obj_id: str):
        super().__init__(
            code=HTTPStatus.NOT_FOUND,
            reason=StatusReason.NOT_FOUND,
            message=f"{obj_type} {obj_id} not found",
        )


class AlreadyExists(AppException):
    """
    AlreadyExists means the resource you are creating already exists.

    :param obj_type: the type of the conflict resource (e.g Table, Job)
    :param obj_id: the identifier of conflicting resource (e.g MyTable)
    """

    def __init__(self, obj_type: str, obj_id: str):
        super().__init__(
            code=HTTPStatus.CONFLICT,
            reason=StatusReason.ALREADY_EXISTS,
            message=f"{obj_type} {obj_id} already exists",
        )


class Unauthorized(AppException):
    """
    The request has not been applied because it lacks valid authentication credentials for the target resource.

    :param message: details about the exception
    """

    def __init__(self, message: str):
        super().__init__(
            code=HTTPStatus.UNAUTHORIZED,
            reason=StatusReason.UNAUTHORIZED,
            message=message,
        )


class Forbidden(AppException):
    """
    The server understood the request but refuses to fulfill it.

    If authentication credentials were provided in the request, the server considers them insufficient to grant access.

    :param message: details about the exception
    """

    def __init__(self, message: str):
        super().__init__(
            code=HTTPStatus.FORBIDDEN,
            reason=StatusReason.FORBIDDEN,
            message=message,
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
        super().__init__(
            code=HTTPStatus.CONFLICT,
            reason=StatusReason.CONFLICT,
            message=f"Operation cannot be fulfilled on {obj_type} {obj_id}: {details}",
        )


class InternalError(AppException):
    """
    The server encountered an unexpected condition that prevented it from fulfilling the request.

    :param message: details about the situation
    :param cause: the original exception (optional)
    """

    def __init__(self, message: str, cause: Exception | None):
        # Prevent cyclic exception nesting
        if isinstance(cause, AppException):
            logging.error(
                f"Cyclic AppException detected. Original message: {message}. "
                f"Cause is another AppException: {cause}"
            )
            sys.exit("Fatal error: recursive AppException encountered. Exiting.")

        # Create a new exception
        detailed_message = f"Details: {message}"
        if cause:
            detailed_message += f" | Cause: {str(cause)}"

        super().__init__(
            code=HTTPStatus.INTERNAL_SERVER_ERROR,
            reason=StatusReason.INTERNAL_ERROR,
            message=detailed_message,
        )

        # Log the exception clearly
        logging.error(f"InternalServer Raised: {detailed_message}")

        # TODO: Implement persistence logic here (e.g., send to monitoring, queue, or DB)
        # self._persist_exception(cause, message)


class TimeoutError(AppException):
    """
    Timeout occurred before the request could be completed.

    This is may be due to temporary server load or a transient communication issue with
        another server. Clients may retry, but the operation may still complete.

    :param message: details about the exception
    """

    def __init__(self, message: str):
        super().__init__(
            code=HTTPStatus.GATEWAY_TIMEOUT,
            reason=StatusReason.TIMEOUT,
            message=message,
        )


class InvalidResource(AppException):
    """
    The request is valid, but the server was unable to process the contained instructions for the resource.

    For example, to read the payload from an empty file.

    :param message: details about the exception
    """

    def __init__(self, message: str):
        super().__init__(
            code=HTTPStatus.UNPROCESSABLE_ENTITY,
            reason=StatusReason.INVALID,
            message=message,
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
        reason = StatusReason.UNKNOWN
        message = f"the server responded with the status code {code} but did not return more information"

        # Reason/message mapping
        if code == HTTPStatus.BAD_REQUEST:
            reason = StatusReason.BAD_REQUEST
            message = "the server rejected our request for an unknown reason"
        elif code == HTTPStatus.NOT_FOUND:
            reason = StatusReason.NOT_FOUND
            message = "the server could not find the requested resource"
        elif code == HTTPStatus.CONFLICT:
            reason = StatusReason.CONFLICT
            message = "the server reported a conflict"
        elif code == HTTPStatus.UNAUTHORIZED:
            reason = StatusReason.UNAUTHORIZED
            message = "the server has asked for the client to provide credentials"
        elif code == HTTPStatus.FORBIDDEN:
            reason = StatusReason.FORBIDDEN
            message = server_message
        elif code == HTTPStatus.UNPROCESSABLE_ENTITY:
            reason = StatusReason.INVALID
            message = "the server rejected our request due to an error in our request"

        elif code == HTTPStatus.GATEWAY_TIMEOUT:
            reason = StatusReason.TIMEOUT
            message = "the server was unable to return a response in the time allotted, but may still be processing the request"

        elif code >= 500:
            reason = "InternalError"
            message = f'an error on the server ("{server_message}") has prevented the request from succeeding'

        super().__init__(code, reason, message)


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
