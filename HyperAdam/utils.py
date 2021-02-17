import random
import os
import json

def task_id(task, eid):
    return task + '-' + eid


def random_id():
    return ''.join([chr(random.randint(0, 25) + ord('a')) for i in range(6)])


def random_choice(dict_):
    key_ = random.choice(list(dict_.keys()))
    return key_


def task_generator(dict_):
    for key_ in dict_.keys():
        yield key_


def shuffle_meta_train_set(dict_):
    return random.sample(dict_.keys(), len(dict_))


def get_learner(dict_):
    learner_cls = dict_['learner']
    kwargs = dict_['kwargs']
    return learner_cls(kwargs)


def save_loss():
    pass


def set_model_path(name):
    cwd = os.getcwd()
    parent_path = os.path.split(cwd)[0]
    model_path = os.path.join(parent_path, "MetaModel", name.split('-')[0], name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    log_name = os.path.join(model_path, name + ".txt")
    log_name_avg = os.path.join(model_path, name + "-avg" + ".txt")
    exception_log = os.path.join(model_path, name + "-exception" + ".txt")
    config_log = os.path.join(model_path, name + "-config" + ".json")
    return model_path, log_name, log_name_avg, exception_log, config_log


def log_json(dict, filename):
    jsobj = json.dumps(dict)
    fileobj = open(filename, 'w')
    fileobj.write(jsobj)
    fileobj.close()










