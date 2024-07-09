import re

MASK_TOKEN = '******'
RIGHT_N = 2
LEFT_N = 2


def anonymize_url(url):
    """Anonymize a URL by replacing the username and password with a mask token.

    Change the username and password to *** keeping one character before and after each.

    :param url: Database URL
    """
    if not url:
        return url

    pattern = re.compile(r'(?<=://)(.*?)(?=@)')
    username_password = pattern.search(url)
    if not username_password:
        return url
    try:
        username, password = username_password.group().split(':')
        if username:
            username_masked = f'{username[:LEFT_N]}{MASK_TOKEN}{username[-RIGHT_N:]}'
        else:
            username_masked = ''
        if password:
            password_masked = f'{password[:LEFT_N]}{MASK_TOKEN}{password[-RIGHT_N:]}'
        else:
            password_masked = ''
        prefix = url.split('://')[0]
        suffix = url.split('@')[-1]
        return f'{prefix}://{username_masked}:{password_masked}@{suffix}'
    except Exception:
        return url
