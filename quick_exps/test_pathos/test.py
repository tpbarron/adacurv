from pathos.multiprocessing import ProcessingPool as Pool
p = Pool(4)

# class Test(object):
#     def plus(self, x):
#         def f(y):
#             return x*y
#         return f

def plus(x):
    def f(y):
        print ("x*y:", x, y)
        return x*y
    return f

f = plus(3) # = Test()
xs = [1, 2, 3, 4, 5]
print(p.map(f, xs))
