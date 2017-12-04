
class PublicKey:
    def __init__(self, pkey):
        self.key = self.get_keytext(pkey)

    def get_keytext(self, keytext):
        return self.split_file(keytext)

    @staticmethod
    def split_file(keytext):
        return keytext.replace('[', '').replace(']', '').replace(' ', '').replace('\n', '').replace('\r', '').split(',')
