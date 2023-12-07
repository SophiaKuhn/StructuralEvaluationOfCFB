import json


def message_to_string(message_type, message):
    msg = {
        'message_type': message_type,
        'message': json.dumps(message)
    }
    return json.dumps(msg)


# returns type, dict
def message_from_string(json_msg):
    unwrap = json.loads(json_msg)
    type = unwrap.get('message_type')
    message = json.loads(unwrap.get('message'))
    return type, message



async def websocket_send(websocket, message_type, obj):
    return await websocket.send(message_to_string(message_type, obj))


async def websocket_receive(websocket):
    message = await websocket.recv()
    return message_from_string(message)
