from AudioNetwork import AudioNetwork

def train(epochs=3, validate=True, autosave=False):
    if validate not in [True, "True", False, "False"]:
        raise ValueError("'validate' should be True or False")

    if autosave not in [True, "True", False, "False"]:
        raise ValueError("'autosave' should be True or False")

    AudioNetwork.executable_train(epochs=int(epochs), validate=validate, autosave=autosave)

if __name__ == '__main__':
    train(1,True,False)