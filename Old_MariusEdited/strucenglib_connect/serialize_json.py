import json
import sys

from pydoc import locate

#
# This provides an alternative serialize method to pickle.
# Data can dynamically be serialized but solely data no code is sent.
# The receiver has to have the same python packages available in environment
# An optional white listing allows to white list packages to reconstruct
#

class ExtendedEncoder(json.JSONEncoder):
    def default(self, obj):
        name = type(obj).__name__
        encoded = obj.__dict__
        encoded["__json_type__"] = name
        encoded["__json_type_module__"] = classname(obj)
        return encoded


class Empty(object):
    pass


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def assign_dict_to_obj(obj, dict):
    try:
        for key in dict:
            setattr(obj, key, dict[key])
    except Exception as e:
        print(e)
        print("assignment failed")
        print(obj, dict)
        pass
    return obj


def set_whitelist(lx):
    global CLASS_WHITE_LIST
    CLASS_WHITE_LIST = lx


CLASS_WHITE_LIST = None


def locate_class(full_name):
    # print(full_name, full_name not in CLASS_WHITE_LIST)
    if CLASS_WHITE_LIST is not None:
        if full_name not in CLASS_WHITE_LIST:
            print("Unknown class ignored: " + full_name)
            raise Exception("Class " + full_name + " was not added to whitelist")
    return locate(full_name)


class ExtendedDecoder(json.JSONDecoder):
    def __init__(self, **kwargs):
        kwargs["object_hook"] = self.object_hook
        super(ExtendedDecoder, self).__init__(**kwargs)

    def object_hook(self, obj):
        try:
            full_name = obj["__json_type_module__"]
            temp = Empty()
            try:
                temp.__class__ = locate_class(full_name)
            except:
                return {}
            return assign_dict_to_obj(temp, obj)

        except Exception as e:
            if isinstance(obj, list):
                return [self.object_hook(v) for v in obj]
            elif isinstance(obj, dict):
                # XXX: JSON does not support int keys
                # As a workaround we convert all keys to int if they are numeric
                # TODO: Can we optimize this?
                return {int(k) if (str(k).lstrip('-').isdigit()) else k: self.object_hook(v) for k, v in obj.items()}
            else:
                # print(obj)
                # print(e.message)
                return obj


def classname(obj):
    cls = type(obj)
    module = cls.__module__
    try:
        name = cls.__qualname__
    except:
        name = type(obj).__name__
    if module is not None and module != "__builtin__":
        name = module + "." + name
    return name


def obj_to_obj(o):
    return json_to_obj(obj_to_json(o))


def obj_to_json(obj):
    return json.dumps(obj, cls=ExtendedEncoder)


def _make_int_key_dict(d):
    return {int(k): v for k, v in d.items()}


def json_to_obj(s):
    mdl = json.loads(s, cls=ExtendedDecoder)
    return mdl


if __name__ == '__main__':
    pass
