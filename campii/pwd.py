def getpwuid(uid):
    class DummyUser:
        pw_name = "windows_user"
    return DummyUser()

def getpwnam(name):
    class DummyUser:
        pw_uid = 1000
    return DummyUser()