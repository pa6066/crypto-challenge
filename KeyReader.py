import re
import Chitext
import PublicKey
import numpy as np
import Textpair
from math import pow
from operator import xor, mul
import random

np.set_printoptions(threshold=np.inf)

class KeyReader:
    """
    Anzahl Relationen 43
    """
    def __init__(self, keyfile, rel):
        self.pkey = PublicKey.PublicKey(self.read_keyfile(keyfile))
        self.rel = rel
        self.num = int(self.get_num())

    @staticmethod
    def read_keyfile(keyfile):
        """
        Read in the public key we are given
        :param keyfile:
        :return:
        """
        with open(keyfile, 'rb') as inputstream:
            keystring = inputstream.read()
        return keystring

    @staticmethod
    def read_chitext(chitextfile):
        """
        Read in the chitext we have to solve
        :param chitextfile:
        :return:
        """
        with open(chitextfile, 'rb') as inputstream:
            keystring = inputstream.read()
        return keystring

    def GEPP(self, A, b, doPricing=True):
        '''
        Gaussian elimination with partial pivoting.

        input: A is an n x n numpy matrix
               b is an n x 1 numpy array
        output: x is the solution of Ax=b
                with the entries permuted in
                accordance with the pivoting
                done by the algorithm
        post-condition: A and b have been modified.
        '''
        n = len(A)
        if b.size != n:
            raise ValueError("Invalid argument: incompatible sizes between" +
                             "A & b.", b.size, n)
        # k represents the current pivot row. Since GE traverses the matrix in the
        # upper right triangle, we also use k for indicating the k-th diagonal
        # column index.

        # Elimination
        for k in range(n - 1):
            if doPricing:
                # Pivot
                maxindex = abs(A[k:, k]).argmax() + k
                if A[maxindex, k] == 0:
                    print "K: %d , MAXINDEX:%d"%(k, maxindex)
                    raise ValueError("Matrix is singular.")
                # Swap
                if maxindex != k:
                    A[[k, maxindex]] = A[[maxindex, k]]
                    b[[k, maxindex]] = b[[maxindex, k]]
            else:
                if A[k, k] == 0:
                    raise ValueError("Pivot element is zero. Try setting doPricing to True.")
            # Eliminate
            for row in range(k + 1, n):
                multiplier = A[row, k] / A[k, k]
                A[row, k:] = A[row, k:] - multiplier * A[k, k:]
                b[row] = b[row] - multiplier * b[k]


        return A

    def gauss(self, A, b):
        n = len(A)
        for k in range(n - 1):
            maxindex = abs(A[k:, k]).argmax() + k  # The column where the first 1 is located
            if A[maxindex, k] == 0:
                raise ValueError("Matrix is singular.")
            if maxindex != k:  # Swap the columns so the pivot element is first
                A[[k, maxindex]] = A[[maxindex, k]]
                b[[k, maxindex]] = b[[maxindex, k]]
            for row in range(k + 1, n):  # Eliminate  fuer jede reihe (Gleichung) unterhalb des pivot elements
                # print "row:%d, col:%d, " % (row, k)
                if A[row, k] == 0:
                    continue
                else:
                    for A_elem_row, A_elem_k, b_elem_row, b_elem_k in zip(A[row, :], A[k, :], b[row, :], b[k, :]):
                        # print "belems:%d, %d"%(b_elem_k, b_elem_row)
                        A[row, k] = xor(int(A_elem_row), int(A_elem_k))
                        b[row, 0] = xor(int(b_elem_row), int(b_elem_k))

        return A, b

    def build_coeff(self, length):
        x = np.zeros((length, 1))
        x[5] = 1
        return x


    def get_num(self):
        """
        :return: The number of coefficients we have
        """
        return max(self.pkey.key[0].replace('+', '').replace('*', '').split('x_'))

    def generate_pairs(self, rel):
        """
        Generate keypairs with randomly generated plaintext and the corresponding chitext
        :param rel:
        :return:
        """
        new_pair = self.generate_pair()
        textpairs = [new_pair]

        while len(textpairs) < rel:
            new_pair = self.generate_pair()
            if new_pair not in textpairs:
                textpairs.append(new_pair)

        return textpairs

    def generate_pair(self):
        return Textpair.Textpair(self.num, self.pkey)

    def create_relation(self, textpair):
        """

        :param textpair:
        :return:
        """
        relation = np.zeros((1, (len(textpair.chitext.x) * len(textpair.plaintext.y))))
        k = 0
        for i in xrange(1, len(textpair.plaintext.y) + 1):
            for j in xrange(1, len(textpair.chitext.x) + 1):
                # print "%d * %d = %d" % (textpair.chitext.x_(j), textpair.plaintext.y_(i), textpair.chitext.x_(j) * textpair.plaintext.y_(i))
                relation[0][k] = (mul(textpair.plaintext.y_(i), textpair.chitext.x_(j)))
                k += 1
        return relation[0]

    def fill_in(self, A, b):
        # Check wether the correct chitext is produced and verify if the key was correct
        pass

    def build_special_rel(self, A, b):
        # Brute Force all possible values for the x_ variables
        for x in xrange(0, (int(pow(2, self.num)) + 1)):


    def start(self):
        """
        Start the Attack
        :return:
        """
        # textpairs = self.generate_pairs(self.rel)
        # relations = []
        # for textpair in textpairs:
        #     relations.append(self.create_relation(textpair))
        # relations = np.asarray(relations)
        # coeffs = self.build_coeff(len(np.asmatrix(relations)))
        # print relations
        # self.GEPP(relations, coeffs)
        A = np.array([
            [0., 0., 1., 0., 1.],
            [1., 1., 0., 1., 1.],
            [1., 1., 1., 1., 1.],
            [0., 1., 0., 0., 1.],
            [1., 0., 1., 1., 1.],
        ])
        b = np.array([
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
        ])
        gauss_matrix, coeffs = self.gauss(A, b)
        self.build_special_rel(gauss_matrix, coeffs)


if __name__ == '__main__':
    reader = KeyReader('C:\Users\Pascal\Desktop\cryptochallenge/test_pkey.txt', 100)
    reader.start()
