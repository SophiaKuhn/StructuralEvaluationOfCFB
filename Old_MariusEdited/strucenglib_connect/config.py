from strucenglib_connect.serialize_json import set_whitelist
from strucenglib_connect.whitelist import FEA_WHITE_LIST

WITH_PROXY = True
set_whitelist(FEA_WHITE_LIST)
SERIALIZE_CLIENT_TO_SERVER = 'json'
SERIALIZE_SERVER_TO_CLIENT = 'pickle'
LOG_FILE_SERVER = 'strucenglib_server.log'