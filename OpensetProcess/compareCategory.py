import numpy as np
import os
import argparse


def main(args):
    pathSource = args.pathSource
    targetSource = args.targetSource
    outputfile = args.outfile

    with open(outputfile,"w") as of:
        listsource = os.listdir(pathSource)
        listtaget = os.listdir(targetSource)
        comparelist = [eachfolder for eachfolder in listtaget if eachfolder not in listsource]

        for i in comparelist:
            of.write(i)
            of.write("\n")


if __name__ == "__main__":
    paser = argparse.ArgumentParser()

    paser.add_argument("--pathSource",default="../Dataset",help="path that have source data")
    paser.add_argument("--targetSource",default="../Datataget",help="path select to compare source data")
    paser.add_argument("--outfile",default="./compareFile.txt",help="compare file name")

    args = paser.parse_args()

    main(args)


