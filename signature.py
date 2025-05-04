import hmac
import base64
import hashlib
import time

class SignatureHandler:
    def __init__(self, secret_key, passphrase=None):
        if secret_key is None:
            raise ValueError("API secret key is required")
        self.secret_key = secret_key
        self.passphrase = passphrase  # Add passphrase support

    def get_timestamp(self):
        return int(time.time() * 1000)

    def sign(self, message):
        """
        Generate a signature for the given message using the provided secret key.
        
        Args:
            message (str): The message to be signed.
        
        Returns:
            str: The generated signature.
        """
        if not isinstance(message, str):
            message = str(message)
        mac = hmac.new(bytes(self.secret_key, encoding='utf8'), bytes(message, encoding='utf-8'), digestmod='sha256')
        d = mac.digest()
        return base64.b64encode(d).decode()  # Ensure the output is a string

    def pre_hash(self, timestamp, method, request_path, body):
        return str(timestamp) + str.upper(method) + request_path + body

    def parse_params_to_str(self, params):
        params = [(key, val) for key, val in params.items()]
        params.sort(key=lambda x: x[0])
        url = '?' + self.toQueryWithNoEncode(params)
        if url == '?':
            return ''
        return url

    def toQueryWithNoEncode(self, params):
        url = ''
        for key, value in params:
            url = url + str(key) + '=' + str(value) + '&'
        return url[0:-1]