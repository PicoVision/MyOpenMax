import multiprocessing
def spawn(num,e,f=9):
  for i in range(100):
      print("{}-{}:{}->{}".format(num,e, i,f))

if __name__ == '__main__':
  for i in range(25):
    ## right here
    p = multiprocessing.Process(target=spawn, args=(i,15,))
    p.start()