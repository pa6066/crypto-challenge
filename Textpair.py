from random import randint
import Plaintext
import Chitext


class Textpair:
    def __init__(self, num, pkey):
        self.plaintext = Plaintext.Plaintext(self.generate_plaintext(num))
        self.chitext = Chitext.Chitext(self.plaintext.generate_chitext(pkey))
        # print self.plaintext.y, self.chitext.x

    @staticmethod
    def generate_plaintext(length):
        # return [1, 0, 1, 0, 0] # For testing purposes
        return [randint(0, 1) for x in xrange(length)]
