import re
import Chitext
import PublicKey
import numpy as np
import Textpair
from math import pow, sqrt
from operator import xor, mul
import random
from random import SystemRandom
from multiprocessing.pool import Pool
# import pathos.pools as pp
import time

np.set_printoptions(threshold=np.inf)


def unwrap_self_pairs(arg, **kwarg):
    return KeyReader.chitext_creation_process(*arg, **kwarg)


def unwrap_self_rel(arg, **kwarg):
    return KeyReader.create_relation(*arg, **kwarg)

gen = SystemRandom()


def generate_plaintext(length):
    # return [1, 0, 1, 0, 0] # For testing purposes
    # return np.asarray([randint(0, 1) for x in xrange(length)])
    return np.asarray([gen.randrange(2) for x in xrange(length)])

class KeyReader:
    """
    Anzahl Relationen 43
    """
    def __init__(self, keyfile, rel):
        self.pkey = PublicKey.PublicKey(self.read_keyfile(keyfile))
        self.rel = rel
        self.num = self.get_num()

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

    def gauss(self, A):
        num_columns = A.shape[1]
        num_rows = A.shape[0]
        # try:
        offset = 0
        for k_col in xrange(0, num_columns):
            k_row = k_col + offset
            if k_row == num_columns:
                print "Matrix has row echelon form"
                print k_col, k_row
                return A
            # if k_col >= 2:
            # print A
            # quit()
            print "K: %d, %d" % (k_col, k_row)
            maxindex = A[k_col:, k_row].argmax() + k_col # Pruefen, wo die erste 1 unterhalb des Pivotelements sich befindet
            while A[maxindex, k_col + offset] == 0:

                # print "K: %d, %d" %( k_col, k_row)
                print "Matrix is singular."
                # print A
                offset += 1
                # print A
                # quit()
                k_row = k_col + offset
                if k_row == num_columns:
                    print "Matrix has row echelon form"
                    return A
                maxindex = A[k_col:, k_row].argmax() + k_col  # Pruefen, wo die erste 1 unterhalb des Pivotelements sich befindet
                # print "MAXINDEX: ", maxindex
            print maxindex, k_col
            if maxindex != k_col:  # Swap the columns so the pivot element is first
                # print "PIVOT"
                A[[maxindex, k_col]] = A[[k_col, maxindex]]

            for row in range(k_col + 1, num_rows):
                # print "ROW: %d" % row
                if A[row, k_col + offset] == 0:
                    continue
                A[row, :] = np.logical_xor(A[row, :], A[k_col, :])
        # return A[:self.num * self.num, :]
        return A
        # return A[:self.]


    def build_coeff(self, length):
        x = np.zeros((length, 1))
        return x

    def get_num(self):
        """
        :return: The number of coefficients we have
        """

        # print max(map(int, self.pkey.key[0].replace('+', '').replace('*', '').split('x_')[1:]))
        return max(map(int, self.pkey.key[0].replace('+', '').replace('*', '').split('x_')[1:]))

    def chitext_creation_process(self, i):
        return self.create_relation(self.generate_pair(generate_plaintext(self.num)))

    def generate_pairs(self, rel):
        """
        Generate keypairs with randomly generated plaintext and the corresponding chitext
        :param rel:
        :return:
        """
        print self.num
        p = Pool(10)
        # plaintexts = [self.generate_plaintext(self.num) for i in xrange(0, self.rel)]
        # while len(plaintexts) < rel:
        #     plaintexts.append()
        i = range(0, self.rel)
        relation = np.array(p.map(unwrap_self_pairs, zip([self]*len(i), i)), dtype=int)
        p.close()
        p.join()
        return relation[:, 0, :]
        # i = 0
        # while len(plaintexts) < rel:
        #     new_plaintext = self.generate_plaintext(self.num)
        #     for plaintext in plaintexts:
        #         same = True
        #         for p_elem, n_elem in zip(plaintext, new_plaintext):
        #             if p_elem != n_elem:
        #                 same = False
        #                 break
        #         if same:
        #             break
        #     if not same:
        #         plaintexts.append(new_plaintext)
        # textpairs = []
        # for plaintext in plaintexts:
        #     textpairs.append(self.generate_pair(plaintext))
        # return textpairs

    def generate_pair(self, plaintext):
        return Textpair.Textpair(plaintext, self.num, self.pkey)



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
                relation[0, k] = (mul(textpair.plaintext.y_(i), textpair.chitext.x_(j)))
                k += 1
        return relation

    def build_special_rel(self, A):
        # print A.shape
        # Brute Force all possible values for the x_ variables
        # length = int(pow(self.num, 2))
        length = 25
        # vars = np.empty((length, 1), dtype=int)
        # vars.fill('2')
        vars = ['2' for i in xrange(length)]
        # print vars
        for i in xrange(A.shape[0] - 1, -1, -1):
            # print i
            if i == 14:
                pass
                # print vars
                # return vars
            unknowns = 0
            cur_unknowns = []
            for j in xrange(0, A.shape[1]):
                if i == 15:
                    pass
                    # print vars[j], j
                if A[i, j] == 1:
                    if vars[j] == '2':
                        unknowns += 1
                        cur_unknowns.append(j)
                    if isinstance(vars[j], list):
                        unknowns += 1
                        cur_unknowns.append(j)
        #             else:
        #                 unknowns += 1
        #
            if unknowns == 0:
                continue
            if unknowns == 1:
                vars[cur_unknowns[0]] = '0'
            elif unknowns > 1:
                # for unknown in xrange(1, len(cur_unknowns)):
                #     print "trets"
                vars[cur_unknowns[0]] = [cur_unknowns[unknown] for unknown in xrange(1, len(cur_unknowns))]
            else:
                # print unknowns, cur_unknowns
                raise ValueError("ERROR")
            # elif unknowns == 2:
        #         # vars[cur_unknowns[0]] = 0
        #         for entry in cur_unknowns:
        #             vars
        #     else:
        #         print i, j, unknowns
        #         raise ValueError
            # print A[i, :]
        # print vars
        return vars

    def check_inputs(self, special_rel, rel):

        def fill_in(possible_val):
            print binary
            print possible_val
            for row in xrange(0, rel.shape[0]):
                if np.count_nonzero(rel[row, :]) == 0:
                    continue
                print rel[row, :]
                print np.asarray(possible_val)
                print "\n"
                values = np.logical_and(rel[row, :], possible_val)
                # print values
                solution = reduce(lambda x, y: int(xor(x, y)), values)
                # print solution
                if solution != 0:
                    print "test False"
                    return False
            print "TEST True"
            return True

        def insert(binary):
            # print binary
            # binary = [1, 0, 1, 0, 1]
            rel_copy = list(special_rel)
            cur_unknown = 0
            for i in xrange(len(rel_copy) - 1, -1, -1):
                if rel_copy[i] == '2':
                    rel_copy[i] = binary[cur_unknown]
                    cur_unknown += 1
                if rel_copy == '0':
                    rel_copy[i] = 0
            for i in xrange(len(rel_copy) - 1, -1, -1):
                if isinstance(rel_copy[i], list):
                    result = rel_copy[rel_copy[i][0]]
                    for j in xrange(1, len(rel_copy[i])):
                        result = xor(result, rel_copy[rel_copy[i][j]])
                    rel_copy[i] = result
            return rel_copy

        def get_bin(index, digit):
            if index >= len(digit):
                return 0
            return int(digit[len(digit) - 1 - index])

        unknowns = 0
        possible_values = []
        for entry in special_rel:
            if entry == '2':
                unknowns += 1
        max_num = int(pow(2, unknowns))
        for i in xrange(1, max_num):
            binary = []
            binary_digit = bin(i).split('b')[1]
            # print binary_digit
            for j in range(unknowns - 1, -1, -1):
                binary.append(get_bin(j, binary_digit))
            possible_value = insert(binary)
            if fill_in(possible_value):
                possible_values.append(possible_value)
        return possible_values

    def determine_plaintext(self,  possible_relations, chitext):
        max_xy_val = sqrt(possible_relations.shape[1]) - 1
        for row in xrange(0, possible_relations.shape[0]):
            print "ROW: ", row
            cur_y_val = 0
            cur_x_val = 0
            i = 0
            for elem in possible_relations[row, :]:
                i += 1
                print cur_x_val, cur_y_val
                if cur_y_val == max_xy_val:
                    cur_x_val += 1
                    cur_y_val = 0
                cur_y_val += 1
            print "I: ", i
            quit()


    def start(self):
        """
        Start the Attack
        :return:
        """
        # textpair = self.generate_pair(self.generate_plaintext(self.num))
        # print textpairs
        # print self.create_relation(textpair)
        # relations = self.create_relations(textpairs)

        # print type(self.generate_plaintext(10))
        # quit()

        # start = time.time()
        # print "Creating Textpairs"
        # relations = self.generate_pairs(self.rel)
        # print "Creating Textpairs done."
        # end = time.time()
        # time_rel = end - start
        # start = time.time()
        # print relations
        # print "Starting Gauss elimination"
        # gauss_matrix = self.gauss(relations)
        # print gauss_matrix
        # tot = 0
        # lala = np.count_nonzero(gauss_matrix, axis=1)
        # for entry in lala:
        #     if entry == 0:
        #         tot += 1
        # print "Laenge: ", len(lala)
        # print "Anzahl zero rows: ", tot
        # print "Gaussian elimination done."
        # end = time.time()
        # time_gauss = end - start
        # start = time.time()
        # print "Starting Back-Substitution"
        # rollup = self.build_special_rel(gauss_matrix)
        # print rollup
        # print "Back-Subnstitution done"
        # # quit()
        # end = time.time()
        # time_spec = end - start
        # start = time.time()
        # print "Starting Brute-Forcing for free variables"
        # solutions = self.check_inputs(rollup, relations)
        # print "Possible solutions created"
        # end = time.time()
        # time_check = end - start
        # print time_rel, time_gauss, time_spec, time_check
        # print np.asarray(solutions)
        # quit()
        # coeffs = self.build_coeff(len(np.asmatrix(relations)))
        A_solved = np.array([
            [1., 1., 1., 1., 1.],
            [0., 1., 0., 0., 1.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 1.],
        ], dtype=int)
        # vars = self.build_special_rel(A_solved)
        # vars = [[3], '0', '0', '2', '2', '2']

        # self.gauss(A)
        # A = np.array([
        #    [1., 0., 1., 0., 0.],
        #    [0., 1., 0., 1., 0.],
        #    [1., 1., 0., 1., 1.],
        #    [0., 0., 0., 0., 0.],
        #    [1., 1., 1., 0., 0.],
        #    [0., 1., 0., 1., 0.],
        #    [0., 1., 1., 1., 1.],
        #    [0., 0., 0., 0., 1.],
        #    [1., 0., 0., 0., 0.],
        #    [0., 1., 1., 0., 1.],
        #    [0., 0., 1., 0., 0.],
        #    [1., 1., 0., 0., 0.],
        #    [0., 0., 0., 0., 0.],
        #    [0., 0., 1., 1., 1.],
        #    [1., 1., 0., 1., 0.],
        #    [0., 0., 1., 0., 0.],
        #    [1., 0., 0., 0., 0.],
        #    [0., 1., 1., 0., 0.],
        #    [1., 0., 0., 1., 0.],
        #    [1., 0., 0., 1., 0.],
        #    [0., 0., 0., 1., 1.],
        #    [0., 1., 0., 0., 1.],
        #    [1., 1., 1., 0., 0.],
        #    [1., 0., 0., 1., 1.],
        #    [1., 1., 0., 1., 0.],
        #    [1., 0., 0., 1., 0.],
        #    [1., 0., 0., 0., 0.],
        #    [1., 0., 1., 1., 1.],
        #    [1., 1., 0., 1., 0.],
        #    [0., 1., 1., 1., 1.],
        #    [0., 1., 0., 1., 1.],
        #    [0., 1., 0., 0., 1.],
        #    [0., 1., 0., 1., 0.],
        #    [1., 1., 0., 0., 1.],
        #    [1., 1., 1., 0., 0.],
        #    [1., 1., 0., 0., 1.],
        #    [0., 1., 1., 1., 0.],
        #    [0., 1., 0., 0., 0.],
        #    [0., 0., 1., 0., 0.],
        #    [0., 1., 0., 0., 0.],
        #    [0., 0., 0., 1., 0.],
        #    [1., 1., 1., 0., 0.],
        #    [1., 0., 1., 0., 0.],
        #    [1., 0., 1., 0., 1.],
        #    [0., 1., 1., 0., 1.],
        #    [0., 0., 1., 0., 0.],
        #    [0., 0., 0., 0., 1.],
        # ])
        # print A[:][1]
        # b = np.array([
        #    [0.],
        #    [0.],
        #    [0.],
        #    [0.],
        #    [0.],
        #    [0.],
        #    [0.],
        # ])
        # print relations
        # print relations
        # self.gauss(A)
        A_relation = np.array([
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            ], dtype=int
        )
        # A_relation = np.array([
        #     [0., 0., 1., 0., 0.],
        #     [1., 1., 0., 1., 1.],
        #     [1., 0., 0., 1., 0.],
        #     [1., 0., 1., 1., 1.],
        #     [0., 0., 1., 1., 0.],
        #     [1., 1., 1., 1., 0.],
        #     [0., 1., 1., 0., 0.],
        # ], dtype=int)
        # A_relation = np.array([
        #     [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        #     [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
        #     [0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0],
        #     [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

        # ], dtype=int)
        A = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])
        b = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 1],
        ])
        solved_relation = self.gauss(A_relation)
        print solved_relation
        lala = np.count_nonzero(solved_relation, axis=1)
        tot = 0
        for entry in lala:
            if entry == 0:
                tot += 1
        # print "Laenge: ", len(lala)
        # print "Anzahl zero rows: ", tot
        special_rel = self.build_special_rel(solved_relation)
        print special_rel
        binaries = self.check_inputs(special_rel, A_relation)
        print np.asarray(binaries)
        print self.determine_plaintext(np.asarray(binaries), [1, 1, 1, 0, 0])

if __name__ == '__main__':
    # reader = KeyReader('C:\Users\Pascal\Desktop/cryptochallenge\\public_key.txt', 2000) # Genug Gleichungen erzeugen, sodass die Matrix nie fehlerhaft singular sein kann.
    reader = KeyReader('C:\Users\Pascal\Desktop/cryptochallenge\\test_pkey.txt', 50) # Genug Gleichungen erzeugen, sodass die Matrix nie fehlerhaft singular sein kann.
    # reader = KeyReader('C:\Users\Pascal-Laptop\Desktop\pkey_full.txt', 2000) # Genug Gleichungen erzeugen, sodass die Matrix nie fehlerhaft singular sein kann.
    reader.start()